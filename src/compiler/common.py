from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

from ..backend import BackendRuntime
from ..representation import (
    BinaryOp,
    CallOp,
    Const,
    Expression,
    IfThenElse,
    NaryOp,
    PropertyOp,
    SpecialOp,
    StateVariable,
    UnaryOp,
    Var,
)


@dataclass(frozen=True)
class StrategyResult:
    step: Callable[..., Any]
    run: Callable[..., Any]


# Lazy evaluator for ⟦e⟧ with memoization and cycle detection.
# Base values are coordinate tensors S_{x∈X}[s] = s(x); derived values are
# symbolic expressions resolved on demand via ⟦e⟧.evaluate(context).
class ResolutionContext(Mapping[str, Any]):
    def __init__(
        self,
        base_values: Mapping[str, Any],
        derived_values: Mapping[str, Expression | Any],
    ) -> None:
        self._base_values = dict(base_values)
        self._derived_values = dict(derived_values)
        self._cache: dict[str, Any] = {}
        self._active: set[str] = set()  # tracks in-flight resolutions for cycle detection

    def __getitem__(self, key: str) -> Any:
        # Return cached ⟦key⟧ if already evaluated
        if key in self._cache:
            return self._cache[key]
        # Base values (coordinate tensors, constants) resolve immediately
        if key in self._base_values:
            value = self._base_values[key]
            self._cache[key] = value
            return value
        if key not in self._derived_values:
            raise KeyError(key)
        # Cycle detection: if key ∈ _active, we have a circular dependency
        if key in self._active:
            raise ValueError(f"Cyclic expression dependency detected for '{key}'")

        self._active.add(key)
        try:
            value = self._derived_values[key]
            # Recursively evaluate ⟦e⟧ₛ for derived expressions
            if isinstance(value, Expression):
                value = value.evaluate(self)
        finally:
            self._active.remove(key)

        self._cache[key] = value
        return value

    def __iter__(self):
        yield from self.keys()

    def __len__(self) -> int:
        return len(self.keys())

    def keys(self):
        return self._base_values.keys() | self._derived_values.keys() | self._cache.keys()


# ---------------------------------------------------------------------------
# Compilation helpers
# ---------------------------------------------------------------------------

# Build environment for evaluating ⟦e⟧: maps constant/function names → values/expressions.
def _build_symbol_values(
    const_env: Mapping[str, Any],
    func_table: Mapping[str, Expression | None],
) -> dict[str, Expression | Any]:
    values: dict[str, Expression | Any] = {}
    for name, value in const_env.items():
        if value is not None:
            values[name] = value
    for name, value in func_table.items():
        if value is not None:
            values[name] = value
    return values


# Evaluate ⟦e⟧ in the given resolution context (coordinate tensors + symbols).
def _evaluate_expression(expr: Expression, context: ResolutionContext, description: str):
    try:
        return expr.evaluate(context)
    except KeyError as exc:
        raise ValueError(f"Unresolved identifier '{exc.args[0]}' in {description}") from exc


# Evaluate ⟦e⟧ expecting a scalar (0-dimensional) result — used for initial values, constants.
def _evaluate_scalar_expression(
    expr: Expression,
    context: ResolutionContext,
    description: str,
    *,
    backend: BackendRuntime,
):
    value = _evaluate_expression(expr, context, description)
    array = backend.array.asarray(value)
    if array.ndim != 0:
        raise ValueError(f"Expected {description} to resolve to a scalar value")
    return array.item()


# Convert ⟦e⟧ result to boolean indicator tensor: 𝟙{⟦e⟧ ≠ 0} ∈ {0,1}^{|State(X)|}.
def _normalize_boolean_tensor(value, *, tensor_shape: tuple[int, ...], backend: BackendRuntime):
    tensor = backend.array.asarray(value)
    if tensor.ndim == 0:
        tensor = backend.array.broadcast_to(tensor, tensor_shape)
    if tensor.shape != tensor_shape:
        raise ValueError(f"Expected a tensor of shape {tensor_shape}, got {tensor.shape}")
    return backend.put(tensor != 0)


# Convert ⟦e⟧ result to float tensor ∈ ℝ^{|State(X)|} with target shape.
def _normalize_numeric_tensor(value, *, tensor_shape: tuple[int, ...], backend: BackendRuntime):
    tensor = backend.array.asarray(value, dtype=backend.float_dtype)
    if tensor.ndim == 0:
        tensor = backend.array.broadcast_to(tensor, tensor_shape)
    if tensor.shape != tensor_shape:
        raise ValueError(f"Expected a tensor of shape {tensor_shape}, got {tensor.shape}")
    return backend.put(tensor)


# Map scalar value v to its index i in dom(x): i such that dom(x)[i] = v.
# Eager-only: performs a host-side domain-membership check. Not safe to call
# from inside a JAX tracer (uses ``to_python_bool`` / ``to_python_int``).
def _scalar_value_to_index(variable: StateVariable, value: Any, *, backend: BackendRuntime) -> int:
    values_tensor = backend.array.asarray(value)
    domain = backend.array.asarray(variable.domain)
    comparison = backend.array.stack(
        [values_tensor == domain_value for domain_value in domain], axis=0
    )
    match_count = backend.array.sum(comparison.astype(backend.array.int32), axis=0)
    if not backend.to_python_bool(match_count == 1):
        raise ValueError(
            f"value for '{variable.name}' is outside its declared domain"
        )
    return backend.to_python_int(backend.array.argmax(comparison.astype(backend.array.int32), axis=0))


# Encode a tensor of domain values → tensor of indices.
# idx(v) = argmaxᵢ 𝟙{dom(x)[i] = v} — maps each value to its position in dom(x).
# Fully traceable: no host-side sync. Callers are responsible for ensuring
# the inputs are valid domain values (the scalar wrapper above is the only
# place where that's checked at model-compile time).
def _encode_values_to_indices(
    variable: StateVariable,
    values,
    *,
    tensor_shape: tuple[int, ...],
    description: str,
    backend: BackendRuntime,
):
    values_tensor = backend.array.asarray(values)
    if values_tensor.ndim == 0 and tensor_shape:
        values_tensor = backend.array.broadcast_to(values_tensor, tensor_shape)
    if values_tensor.shape != tensor_shape:
        raise ValueError(
            f"Expected {description} to match tensor shape {tensor_shape}, got {values_tensor.shape}"
        )

    # For each v ∈ dom(x), compute 𝟙{values = v}; stack along axis 0
    domain = backend.array.asarray(variable.domain)
    comparison = backend.array.stack([values_tensor == domain_value for domain_value in domain], axis=0)
    # argmax over the one-hot comparison axis → index i where dom(x)[i] = value
    return backend.put(backend.array.argmax(comparison.astype(backend.array.int32), axis=0))


# δᵥ ∈ ℝ^{|dom(x)|}: one-hot basis vector at index v.
# For each state s, produces δ_{indices[s]} — the basis vector for the assigned value.
def _one_hot(indices, num_classes: int, *, backend: BackendRuntime):
    """Create one-hot encoding compatible with both NumPy and JAX."""
    classes = backend.array.arange(num_classes, dtype=backend.array.int32)
    shape = [1] * indices.ndim + [num_classes]
    classes = backend.array.reshape(classes, shape)
    # (indices[..., None] == classes) produces δ_{indices[s]} for each state s
    return (indices[..., None] == classes).astype(backend.float_dtype)


# Compute Yₖ \ Xₖ: state variables referenced in guards/updates of module mₖ.
# Yₖ = {x ∈ X : x appears in ⟦g⟧, θᵢ, or ⟦eⱼ⟧ of some command in mₖ}.
def _analyze_module_dependencies(
    commands: list,
    state_var_names: set[str],
    symbol_values: dict[str, Any],
) -> set[str]:
    """Return the set of state variable names that affect a module's transition."""
    deps: set[str] = set()
    for command in commands:
        # Variables in guard expression ⟦g⟧
        ids = _collect_identifiers(command.guard.expr, symbol_values)
        deps.update(ids & state_var_names)
        for update in command.updates:
            # Variables in update probability θᵢ
            if isinstance(update.prob, Expression):
                ids = _collect_identifiers(update.prob, symbol_values)
                deps.update(ids & state_var_names)
            # Variables in assignment expressions x ≔ ⟦eⱼ⟧
            for expr in update.assignments.values():
                if isinstance(expr, Expression):
                    ids = _collect_identifiers(expr, symbol_values)
                    deps.update(ids & state_var_names)
    return deps


def _compute_processing_order(
    n_modules: int,
    module_owned_var_names: list[set[str]],
    module_deps: list[set[str]],
) -> list[int]:
    """Compute module processing order for sequential variable elimination.

    If module k depends on a variable owned by module m (k != m),
    then k must be processed before m so that m's original values
    are still present in the tensor when k needs them.

    When the dependency graph contains cycles, they are broken by
    selecting the cycle node with the smallest in-degree (fewest
    unsatisfied dependencies), yielding a semi-topological order.
    """
    # Build DAG: edge k → m means mₖ must be processed before mₘ.
    # deps(mₖ) ∩ owned(mₘ) ≠ ∅  ⟹  k must precede m.
    successors: list[set[int]] = [set() for _ in range(n_modules)]
    in_degree = [0] * n_modules

    for k in range(n_modules):
        for m in range(n_modules):
            if k == m:
                continue
            if module_deps[k] & module_owned_var_names[m]:
                if m not in successors[k]:
                    successors[k].add(m)
                    in_degree[m] += 1

    # Kahn's algorithm with cycle breaking: repeatedly remove nodes with
    # in-degree 0.  When all remaining nodes have in-degree > 0 (a cycle),
    # break the cycle by selecting the node with the smallest in-degree
    # among unprocessed nodes — this respects the most dependency edges.
    remaining = set(range(n_modules))
    queue = [i for i in range(n_modules) if in_degree[i] == 0]
    order: list[int] = []
    while remaining:
        if not queue:
            # All remaining nodes are in cycles — break by picking the
            # node with the smallest in-degree (fewest unsatisfied deps).
            cycle_breaker = min(remaining, key=lambda i: in_degree[i])
            queue.append(cycle_breaker)
        while queue:
            u = queue.pop(0)
            remaining.remove(u)
            order.append(u)
            for v in successors[u]:
                if v in remaining:
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        queue.append(v)

    return order


def _contains_unsupported_property_ops(expr: Expression) -> bool:
    if isinstance(expr, (PropertyOp, SpecialOp)):
        return True
    if isinstance(expr, UnaryOp):
        return _contains_unsupported_property_ops(expr.operand)
    if isinstance(expr, BinaryOp):
        return _contains_unsupported_property_ops(expr.left) or _contains_unsupported_property_ops(expr.right)
    if isinstance(expr, NaryOp):
        return any(_contains_unsupported_property_ops(arg) for arg in expr.args)
    if isinstance(expr, IfThenElse):
        return (
            _contains_unsupported_property_ops(expr.cond)
            or _contains_unsupported_property_ops(expr.then_branch)
            or _contains_unsupported_property_ops(expr.else_branch)
        )
    if isinstance(expr, CallOp):
        return any(_contains_unsupported_property_ops(arg) for arg in expr.args)
    return False


# Transitively collect free variables FV(⟦e⟧) — identifiers referenced by expression e.
# Follows derived definitions (constants, functions) to find all transitive dependencies.
def _collect_identifiers(
    expr: Expression,
    derived_values: Mapping[str, Expression | Any],
    *,
    _visited_names: set[str] | None = None,
) -> set[str]:
    if _visited_names is None:
        _visited_names = set()

    identifiers: set[str] = set()
    if isinstance(expr, Var):
        identifiers.add(expr.name)
        if expr.name in derived_values and expr.name not in _visited_names:
            _visited_names.add(expr.name)
            nested = derived_values[expr.name]
            if isinstance(nested, Expression):
                identifiers.update(_collect_identifiers(nested, derived_values, _visited_names=_visited_names))
        return identifiers
    if isinstance(expr, Const):
        return identifiers
    if isinstance(expr, UnaryOp):
        return _collect_identifiers(expr.operand, derived_values, _visited_names=_visited_names)
    if isinstance(expr, BinaryOp):
        identifiers.update(_collect_identifiers(expr.left, derived_values, _visited_names=_visited_names))
        identifiers.update(_collect_identifiers(expr.right, derived_values, _visited_names=_visited_names))
        return identifiers
    if isinstance(expr, NaryOp):
        for arg in expr.args:
            identifiers.update(_collect_identifiers(arg, derived_values, _visited_names=_visited_names))
        return identifiers
    if isinstance(expr, CallOp):
        identifiers.add(expr.function)
        if expr.function in derived_values and expr.function not in _visited_names:
            _visited_names.add(expr.function)
            nested = derived_values[expr.function]
            if isinstance(nested, Expression):
                identifiers.update(_collect_identifiers(nested, derived_values, _visited_names=_visited_names))
        for arg in expr.args:
            identifiers.update(_collect_identifiers(arg, derived_values, _visited_names=_visited_names))
        return identifiers
    if isinstance(expr, IfThenElse):
        identifiers.update(_collect_identifiers(expr.cond, derived_values, _visited_names=_visited_names))
        identifiers.update(_collect_identifiers(expr.then_branch, derived_values, _visited_names=_visited_names))
        identifiers.update(_collect_identifiers(expr.else_branch, derived_values, _visited_names=_visited_names))
        return identifiers
    if isinstance(expr, PropertyOp):
        if isinstance(expr.states, Expression):
            identifiers.update(_collect_identifiers(expr.states, derived_values, _visited_names=_visited_names))
        if isinstance(expr.values, dict):
            for value in expr.values.values():
                if isinstance(value, Expression):
                    identifiers.update(_collect_identifiers(value, derived_values, _visited_names=_visited_names))
        return identifiers
    return identifiers
