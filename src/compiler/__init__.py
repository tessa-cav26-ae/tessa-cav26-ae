from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from ..backend import BackendRuntime, BackendSpec, build_backend
from ..representation import (
    Const,
    ParsedModel,
    RewardVariable,
    StateVariable,
    use_array_api,
)
from .common import (
    ResolutionContext,
    _build_symbol_values,
    _collect_identifiers,
    _contains_unsupported_property_ops,
    _evaluate_expression,
    _evaluate_scalar_expression,
    _normalize_boolean_tensor,
    _scalar_value_to_index,
    _analyze_module_dependencies,
    _compute_processing_order,
)
from .sequential import compile_sequential


# Compiled DTMC M = (S, ι, η, G) represented as a traced JAX run-closure.
@dataclass(frozen=True)
class CompiledReachabilityModel:
    property_name: str
    backend: BackendRuntime
    variables: tuple[StateVariable, ...]     # state variables X = {x₁, ..., xₙ}
    tensor_shape: tuple[int, ...]            # (|dom(x₁)|, ..., |dom(xₙ)|)
    initial_tensor: Any                      # Ξ(δ_{s₀}) — point mass at initial state
    goal_mask: Any                           # ⟦G⟧(s) = 𝟙{⟦e⟧(s) ≠ 0}
    non_goal_mask: Any                       # Δ_{¬G}[s] = 1 − ⟦G⟧(s)
    step: Callable[..., Any]                 # ⟦M⟧ : Tensor(State) → Tensor(State)
    run: Callable[..., Any]                  # Pr_M(T ⊨ ◇≤ⁿ G)


def compile_reachability(
    parsed_model: ParsedModel,
    *,
    property_name: str,
    backend: str | BackendSpec | None = None,
    dtype: str | None = None,
    model_path: str | Path | None = None,
    constants: Mapping[str, Any] | None = None,
) -> CompiledReachabilityModel:
    backend = build_backend(backend, dtype=dtype)

    if backend.spec.kind == "explicit":
        # Delegate to Storm. No Tessa tensor pipeline runs.
        from .explicit import compile_explicit
        if model_path is None:
            raise ValueError(
                "--backend explicit requires the original model path; "
                "compile_reachability was called without model_path."
            )
        explicit_model = compile_explicit(
            model_path=model_path,
            property_name=property_name,
            constants=constants,
            backend=backend,
        )
        # Wrap run/step to reject `constants_override`: the explicit backend
        # bakes constants into stormpy's IR before model construction, so
        # there is no per-call substitution path.
        _explicit_run = explicit_model.run
        _explicit_step = explicit_model.step

        def _run_explicit(horizon, *, constants_override=None):
            if constants_override is not None:
                raise NotImplementedError(
                    "Runtime constant override is not supported by the 'explicit' backend."
                )
            return _explicit_run(horizon)

        def _step_explicit(tensor, *, constants_override=None):
            if constants_override is not None:
                raise NotImplementedError(
                    "Runtime constant override is not supported by the 'explicit' backend."
                )
            return _explicit_step(tensor)

        return CompiledReachabilityModel(
            property_name=explicit_model.property_name,
            backend=explicit_model.backend,
            variables=explicit_model.variables,
            tensor_shape=explicit_model.tensor_shape,
            initial_tensor=explicit_model.initial_tensor,
            goal_mask=explicit_model.goal_mask,
            non_goal_mask=explicit_model.non_goal_mask,
            step=_step_explicit,
            run=_run_explicit,
        )

    with use_array_api(backend.array):
        model = copy.deepcopy(parsed_model)
        prop_table, const_env, func_table, modules = model.properties, model.constants, model.functions, model.modules
        if property_name not in prop_table:
            raise ValueError(f"Unknown property '{property_name}'")

        goal_expr = prop_table[property_name]
        if goal_expr is None:
            raise ValueError(f"Property '{property_name}' does not contain an executable state predicate")
        if _contains_unsupported_property_ops(goal_expr):
            raise ValueError(
                f"Property '{property_name}' is not a direct state predicate supported by bounded reachability"
            )

        # Collect state variables X = {x₁, ..., xₙ} defining State(X) = ∏ₓ dom(x)
        state_variables = tuple(
            variable
            for module in modules
            for variable in module.variables
            if isinstance(variable, StateVariable)
        )
        reward_names = {
            variable.name
            for module in modules
            for variable in module.variables
            if isinstance(variable, RewardVariable)
        }
        if not state_variables:
            raise ValueError("Cannot compile bounded reachability for a model without state variables")

        # Build constant/function environment for expression evaluation ⟦e⟧
        symbol_values = _build_symbol_values(const_env, func_table)
        property_values = {
            name: expr
            for name, expr in prop_table.items()
            if expr is not None
        }
        # Transitively collect all identifiers referenced in goal expression ⟦G⟧
        transitive_identifiers = _collect_identifiers(goal_expr, symbol_values | property_values)
        reward_references = sorted(transitive_identifiers.intersection(reward_names))
        if reward_references:
            raise ValueError(
                "Goal expressions that reference reward variables are not supported: "
                + ", ".join(reward_references)
            )

        # Identify constants that are NOT safe to override at runtime: anything
        # referenced (directly or transitively) by the goal expression bakes into
        # the static goal_mask tensor; anything referenced by a variable's initial
        # expression bakes into the static initial_tensor.
        goal_referenced_constants = transitive_identifiers & set(symbol_values.keys())
        initial_referenced_constants: set[str] = set()
        for variable in state_variables:
            if variable.initial is not None:
                initial_referenced_constants |= (
                    _collect_identifiers(variable.initial, symbol_values)
                    & set(symbol_values.keys())
                )
        non_overridable_constants = goal_referenced_constants | initial_referenced_constants

        for variable in state_variables:
            variable.resolve(const_env)

        # tensor_shape = (|dom(x₁)|, ..., |dom(xₙ)|);  |State(X)| = ∏_{x∈X} |dom(x)|
        tensor_shape = tuple(int(variable.size) for variable in state_variables)
        # Build coordinate tensors S_{x∈X}[s] = s(x) for each state variable x
        value_grids = tuple(
            backend.put(backend.array.asarray(grid))
            for grid in backend.array.meshgrid(
                *[backend.array.asarray(variable.domain) for variable in state_variables],
                indexing="ij",
            )
        )

        # Resolution context: implements lazy ⟦e⟧ₛ evaluation with cycle detection
        state_context = {variable.name: grid for variable, grid in zip(state_variables, value_grids)}
        property_context = ResolutionContext(state_context, symbol_values | property_values)

        # ⟦G⟧(s) = 𝟙{⟦e⟧(s) ≠ 0}  — goal indicator mask (paper: Goal⟦M⟧)
        goal_mask = _normalize_boolean_tensor(
            _evaluate_expression(goal_expr, property_context, f"property '{property_name}'"),
            tensor_shape=tensor_shape,
            backend=backend,
        ).astype(backend.float_dtype)
        goal_mask = backend.put(goal_mask)
        # Δ_{¬G}[s] = 1 − ⟦G⟧(s) — non-goal mask for Hadamard masking
        non_goal_mask = backend.put(backend.array.ones(tensor_shape, dtype=backend.float_dtype) - goal_mask)

        # Ξ(Init⟦M⟧) = Ξ(δ_{s₀}) — point mass at initial state s₀
        initial_tensor = backend.put(backend.array.zeros(tensor_shape, dtype=backend.float_dtype))
        # Evaluate initial value for each variable to determine s₀ = (v₁, ..., vₙ)
        initial_indices = tuple(
            _scalar_value_to_index(
                variable,
                _evaluate_scalar_expression(
                    variable.initial if variable.initial is not None else Const(0),
                    ResolutionContext({}, symbol_values),
                    f"initial value of '{variable.name}'",
                    backend=backend,
                ),
                backend=backend,
            )
            for variable in state_variables
        )
        # T[s₀] = 1, all other entries 0 — Dirac delta δ_{s₀}
        initial_tensor = backend.set_index(initial_tensor, initial_indices, 1.0)

        # Modules m₁, ..., mₖ that define the transition relation η
        behavior_modules = [module for module in modules if module.commands]
        if not behavior_modules:
            raise ValueError("Cannot compile bounded reachability for a model without commands")

        variable_positions = {variable.name: index for index, variable in enumerate(state_variables)}

        # PRISM anonymous command semantics: each unlabeled command (action=None)
        # gets a freshly generated unique action label.  This ensures unlabeled
        # commands do not synchronize across modules and are independently averaged
        # when multiple are enabled at the same state.
        anon_counter = 0
        for module in behavior_modules:
            for command in module.commands:
                if command.action is None:
                    command.action = f"__anon_{anon_counter}"
                    anon_counter += 1

        # Collect Action(M) = ⋃ₖ actions(mₖ)
        all_actions: set[str] = set()
        for module in behavior_modules:
            for command in module.commands:
                all_actions.add(command.action)
        sorted_actions = sorted(all_actions)

        # --- Dependency analysis: for each module mₖ, find Yₖ (dep vars) and Xₖ (owned vars) ---
        state_var_names = {v.name for v in state_variables}
        module_owned_names: list[set[str]] = []
        module_deps: list[set[str]] = []
        for module in behavior_modules:
            # Xₖ = variables declared in module mₖ
            owned = {v.name for v in module.variables if isinstance(v, StateVariable)}
            # Yₖ = variables appearing in guards/updates of mₖ (Xₖ ⊆ Yₖ)
            deps = _analyze_module_dependencies(module.commands, state_var_names, symbol_values)
            deps |= owned
            module_owned_names.append(owned)
            module_deps.append(deps)

        # Topological order via Kahn's algorithm for sequential variable elimination
        processing_order = _compute_processing_order(
            len(behavior_modules), module_owned_names, module_deps,
        )

        # --- Per-module info: positions/sizes of Yₖ and Xₖ within the state tuple ---
        module_dep_vars: list[list] = []
        module_dep_positions: list[tuple[int, ...]] = []
        module_dep_sizes: list[tuple[int, ...]] = []
        module_owned_vars: list[list] = []
        module_owned_positions: list[tuple[int, ...]] = []
        module_owned_sizes: list[tuple[int, ...]] = []
        module_owned_dep_indices: list[list[int]] = []

        for module_idx in range(len(behavior_modules)):
            dep_names_sorted = sorted(module_deps[module_idx], key=lambda n: variable_positions[n])
            dep_pos = tuple(variable_positions[n] for n in dep_names_sorted)
            dep_sz = tuple(tensor_shape[p] for p in dep_pos)
            d_vars = [state_variables[p] for p in dep_pos]

            owned_names_sorted = sorted(module_owned_names[module_idx], key=lambda n: variable_positions[n])
            owned_pos = tuple(variable_positions[n] for n in owned_names_sorted)
            owned_sz = tuple(tensor_shape[p] for p in owned_pos)
            o_vars = [state_variables[p] for p in owned_pos]

            # owned_dep_indices: position of each x ∈ Xₖ within the Yₖ ordering
            o_dep_idx = []
            for op in owned_pos:
                for i, dp in enumerate(dep_pos):
                    if dp == op:
                        o_dep_idx.append(i)
                        break

            module_dep_vars.append(d_vars)
            module_dep_positions.append(dep_pos)
            module_dep_sizes.append(dep_sz)
            module_owned_vars.append(o_vars)
            module_owned_positions.append(owned_pos)
            module_owned_sizes.append(owned_sz)
            module_owned_dep_indices.append(o_dep_idx)

        ndim = len(tensor_shape)

        result = compile_sequential(
            processing_order=processing_order,
            behavior_modules=behavior_modules,
            module_dep_vars=module_dep_vars,
            module_dep_positions=module_dep_positions,
            module_dep_sizes=module_dep_sizes,
            module_owned_vars=module_owned_vars,
            module_owned_positions=module_owned_positions,
            module_owned_sizes=module_owned_sizes,
            module_owned_dep_indices=module_owned_dep_indices,
            sorted_actions=sorted_actions,
            symbol_values=symbol_values,
            reward_names=reward_names,
            tensor_shape=tensor_shape,
            ndim=ndim,
            initial_tensor=initial_tensor,
            goal_mask=goal_mask,
            non_goal_mask=non_goal_mask,
            backend=backend,
            non_overridable_constants=non_overridable_constants,
        )

    return CompiledReachabilityModel(
        property_name=property_name,
        backend=backend,
        variables=state_variables,
        tensor_shape=tensor_shape,
        initial_tensor=initial_tensor,
        goal_mask=goal_mask,
        non_goal_mask=non_goal_mask,
        step=result.step,
        run=result.run,
    )
