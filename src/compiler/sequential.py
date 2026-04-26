from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..backend import BackendRuntime, jit
from ..representation import use_array_api
from .common import (
    ResolutionContext,
    StrategyResult,
    _encode_values_to_indices,
    _evaluate_expression,
    _normalize_boolean_tensor,
    _normalize_numeric_tensor,
    _one_hot,
)


def compile_sequential(
    *,
    processing_order: list[int],
    behavior_modules: list,
    module_dep_vars: list[list],
    module_dep_positions: list[tuple[int, ...]],
    module_dep_sizes: list[tuple[int, ...]],
    module_owned_vars: list[list],
    module_owned_positions: list[tuple[int, ...]],
    module_owned_sizes: list[tuple[int, ...]],
    module_owned_dep_indices: list[list[int]],
    sorted_actions: list,
    symbol_values: dict[str, Any],
    reward_names: set[str],
    tensor_shape: tuple[int, ...],
    ndim: int,
    initial_tensor: Any,
    goal_mask: Any,
    non_goal_mask: Any,
    backend: BackendRuntime,
    non_overridable_constants: set[str] | None = None,
) -> StrategyResult:
    """Code-gen sequential compilation.

    Compile-time does only metadata analysis: processing order, per-step
    broadcast shapes, axes-to-sum, permutations. The actual per-module
    transition tensors ``⟦mₖ⟧(a)`` are **not precomputed** — their
    construction is inlined as JAX-traced code inside the ``@jit``-decorated
    ``run`` closure, so XLA compiles the whole construction + fori_loop
    together on first invocation (warmup).
    """
    n_actions = len(sorted_actions)
    n_proc = len(processing_order)

    # --- Delayed variable elimination bookkeeping (metadata only) ---
    # For each source position, find the last module (in processing order)
    # that reads it. Source dimensions are retained until their last consumer.
    last_user_proc: dict[int, int] = {}
    for proc_idx, module_idx in enumerate(processing_order):
        for pos in module_dep_positions[module_idx]:
            last_user_proc[pos] = proc_idx  # overwrites → keeps last

    # Simulate dimension evolution to precompute per-step operations.
    # Dim ID convention: source dim for position p has ID p (0..ndim-1);
    # destination dim for position p has ID ndim+p.
    current_dim_ids: list[int] = list(range(ndim))

    step_broadcast_shapes: list[tuple[int, ...]] = []
    step_n_expand: list[int] = []
    step_axes_to_sum: list[tuple[int, ...]] = []
    step_perms: list[tuple[int, ...]] = []

    for proc_idx, module_idx in enumerate(processing_order):
        dep_pos = module_dep_positions[module_idx]
        dep_sz = module_dep_sizes[module_idx]
        owned_pos = module_owned_positions[module_idx]
        owned_sz = module_owned_sizes[module_idx]

        n_current = len(current_dim_ids)
        n_owned = len(owned_pos)
        n_expanded = n_current + n_owned

        # Broadcast shape: dep sizes at source-dim axes, owned sizes trailing.
        bshape = [1] * n_expanded
        for i, pos in enumerate(dep_pos):
            axis = current_dim_ids.index(pos)  # source dim is still in tensor
            bshape[axis] = dep_sz[i]
        for i, sz in enumerate(owned_sz):
            bshape[n_current + i] = sz

        # After expansion and multiplication, dim IDs are:
        expanded_ids = list(current_dim_ids) + [ndim + p for p in owned_pos]

        # Sum out source dims whose last consumer is this step.
        to_remove = set()
        for pos in range(ndim):
            if last_user_proc.get(pos) == proc_idx and pos in expanded_ids:
                to_remove.add(pos)

        axes = sorted(
            [expanded_ids.index(d) for d in to_remove],
            reverse=True,  # sum highest axis first to avoid index shifts
        )

        remaining_ids = [d for d in expanded_ids if d not in to_remove]
        desired_ids = sorted(remaining_ids)
        perm = tuple(remaining_ids.index(d) for d in desired_ids)

        step_broadcast_shapes.append(tuple(bshape))
        step_n_expand.append(n_owned)
        step_axes_to_sum.append(tuple(axes))
        step_perms.append(perm)

        current_dim_ids = desired_ids

    # After all steps, only destination dims should remain in position order.
    assert current_dim_ids == list(range(ndim, 2 * ndim)), (
        f"Dimension tracking error: expected dest dims {list(range(ndim, 2 * ndim))}, "
        f"got {current_dim_ids}"
    )

    _broadcast_shapes = tuple(step_broadcast_shapes)
    _n_expands = tuple(step_n_expand)
    _axes_to_sum = tuple(step_axes_to_sum)
    _perms = tuple(step_perms)

    # --- Precompute per-(action, module, command) structural metadata ---
    # Freeze action→module→command lists so the tracing loops below don't
    # re-filter on every invocation.
    action_cmd_lists: list[list[list]] = []
    for action in sorted_actions:
        per_proc: list[list] = []
        for proc_idx, module_idx in enumerate(processing_order):
            module = behavior_modules[module_idx]
            per_proc.append([c for c in module.commands if c.action == action])
        action_cmd_lists.append(per_proc)

    # --- Runtime-overridable constants -------------------------------------
    # Only PRISM `const double` values that flow purely into transition
    # probabilities (not into the goal expression or any initial-value
    # expression) are overridable per-call. Integer/bool constants and
    # symbolic Expression-valued constants are excluded — they're either
    # structural (state-space size) or already derived.
    _non_overridable = set(non_overridable_constants or ())
    _overridable_keys: tuple[str, ...] = tuple(sorted(
        name for name, value in symbol_values.items()
        if isinstance(value, float)
        and not isinstance(value, bool)
        and name not in _non_overridable
    ))

    def _validate_and_pack_override(override: Mapping[str, Any]) -> tuple:
        if not isinstance(override, Mapping):
            raise TypeError(
                f"constants_override must be a mapping, got {type(override).__name__}"
            )
        unknown = sorted(set(override) - set(symbol_values))
        if unknown:
            raise KeyError(
                "constants_override contains identifiers not declared in the "
                f"original load_prism_model constants: {', '.join(unknown)}"
            )
        non_overridable_in_override = sorted(set(override) - set(_overridable_keys))
        if non_overridable_in_override:
            raise ValueError(
                "constants_override may only contain floating-point PRISM "
                "constants that do not appear in the goal expression or in any "
                f"variable's initial value: {', '.join(non_overridable_in_override)}"
            )
        return tuple(
            backend.array.asarray(override[k], dtype=backend.float_dtype)
            if k in override
            else backend.array.asarray(symbol_values[k], dtype=backend.float_dtype)
            for k in _overridable_keys
        )

    # ------------------------------------------------------------------
    # Inlined-construction helpers — traced by JAX, not executed eagerly.
    # ------------------------------------------------------------------

    def _make_dep_grids(
        module_idx: int,
        *,
        active_symbol_values: Mapping[str, Any],
    ) -> tuple[tuple[Any, ...], tuple[Any, ...], ResolutionContext]:
        """Build per-module dependency meshgrids inside the tracer."""
        dep_vars = module_dep_vars[module_idx]
        dep_sizes = module_dep_sizes[module_idx]
        dep_value_grids = tuple(
            backend.array.asarray(grid)
            for grid in backend.array.meshgrid(
                *[backend.array.asarray(v.domain) for v in dep_vars],
                indexing="ij",
            )
        )
        dep_index_grids = tuple(
            backend.array.asarray(grid, dtype=backend.array.int32)
            for grid in backend.array.meshgrid(
                *[backend.array.arange(sz, dtype=backend.array.int32) for sz in dep_sizes],
                indexing="ij",
            )
        )
        dep_context = {v.name: grid for v, grid in zip(dep_vars, dep_value_grids)}
        dep_value_context = ResolutionContext(dep_context, active_symbol_values)
        return dep_value_grids, dep_index_grids, dep_value_context

    def _identity_transition(module_idx: int) -> Any:
        """Identity ⟦mₖ⟧(a): per-owned Kronecker delta, broadcast across dep axes."""
        dep_sizes = module_dep_sizes[module_idx]
        owned_sizes = module_owned_sizes[module_idx]
        owned_dep_indices = module_owned_dep_indices[module_idx]
        full_shape = dep_sizes + owned_sizes
        n_dep = len(dep_sizes)
        trans = backend.array.ones(full_shape, dtype=backend.float_dtype)
        for j, dep_idx in enumerate(owned_dep_indices):
            sz = owned_sizes[j]
            src_shape = [1] * len(full_shape)
            src_shape[dep_idx] = sz
            dst_shape = [1] * len(full_shape)
            dst_shape[n_dep + j] = sz
            src_indices = backend.array.reshape(
                backend.array.arange(sz, dtype=backend.array.int32), src_shape
            )
            dst_indices = backend.array.reshape(
                backend.array.arange(sz, dtype=backend.array.int32), dst_shape
            )
            trans = trans * (src_indices == dst_indices).astype(backend.float_dtype)
        return trans

    def _compressed_transition(
        commands: list,
        module_idx: int,
        dep_grids: tuple[tuple[Any, ...], tuple[Any, ...], ResolutionContext],
    ) -> Any:
        """Compressed transition ⟦mₖ⟧(a): per-command guard × per-update mixture."""
        dep_sizes = module_dep_sizes[module_idx]
        owned_vars = module_owned_vars[module_idx]
        owned_sizes = module_owned_sizes[module_idx]
        owned_dep_indices = module_owned_dep_indices[module_idx]
        _, dep_index_grids, dep_value_context = dep_grids

        n_dep = len(dep_sizes)
        n_owned = len(owned_sizes)
        full_shape = dep_sizes + owned_sizes
        trans = backend.array.zeros(full_shape, dtype=backend.float_dtype)

        for command in commands:
            guard_mask = _normalize_boolean_tensor(
                _evaluate_expression(command.guard.expr, dep_value_context, "command guard"),
                tensor_shape=dep_sizes,
                backend=backend,
            ).astype(backend.float_dtype)
            update_probs = [
                _normalize_numeric_tensor(
                    _evaluate_expression(update.prob, dep_value_context, "update probability"),
                    tensor_shape=dep_sizes,
                    backend=backend,
                )
                for update in command.updates
            ]
            for update, prob in zip(command.updates, update_probs):
                weight = guard_mask * prob
                one_hots: list[Any] = []
                for i, owned_var in enumerate(owned_vars):
                    var_name = owned_var.name
                    if var_name in update.assignments and var_name not in reward_names:
                        assigned_value = _evaluate_expression(
                            update.assignments[var_name],
                            dep_value_context,
                            f"assignment to '{var_name}'",
                        )
                        dst_idx = _encode_values_to_indices(
                            owned_var,
                            assigned_value,
                            tensor_shape=dep_sizes,
                            description=f"assignment to '{var_name}'",
                            backend=backend,
                        )
                    else:
                        dst_idx = dep_index_grids[owned_dep_indices[i]]
                    one_hots.append(_one_hot(dst_idx, owned_sizes[i], backend=backend))

                contribution = weight
                for _ in range(n_owned):
                    contribution = contribution[..., None]
                for i, oh in enumerate(one_hots):
                    target_shape = list(dep_sizes) + [1] * n_owned
                    target_shape[n_dep + i] = owned_sizes[i]
                    contribution = contribution * backend.array.reshape(oh, target_shape)
                trans = trans + contribution
        return trans

    def _build_action_matrices(
        *, active_symbol_values: Mapping[str, Any],
    ) -> tuple[tuple[Any, ...], ...]:
        """Build all ⟦mₖ⟧(a) tensors inside the tracer."""
        # Hoist per-module dep meshgrids + ResolutionContext once (shared
        # across all actions operating on that module).
        module_grids: list[tuple[tuple[Any, ...], tuple[Any, ...], ResolutionContext] | None] = []
        action_set = set(sorted_actions)
        for module_idx in range(len(behavior_modules)):
            module = behavior_modules[module_idx]
            if any(c.action in action_set for c in module.commands):
                module_grids.append(_make_dep_grids(module_idx, active_symbol_values=active_symbol_values))
            else:
                module_grids.append(None)

        out: list[tuple[Any, ...]] = []
        for action_idx in range(n_actions):
            per_proc: list[Any] = []
            for proc_idx, module_idx in enumerate(processing_order):
                commands = action_cmd_lists[action_idx][proc_idx]
                if not commands:
                    per_proc.append(_identity_transition(module_idx))
                else:
                    grids = module_grids[module_idx]
                    assert grids is not None
                    per_proc.append(_compressed_transition(commands, module_idx, grids))
            out.append(tuple(per_proc))
        return tuple(out)

    def _compute_enablement(
        action_matrices: tuple[tuple[Any, ...], ...],
    ) -> tuple[Any, Any]:
        """Compute safe_action_count and deadlock_mask from row sums of the transition tensors."""
        action_count = backend.array.zeros(tensor_shape, dtype=backend.float_dtype)
        for action_idx in range(n_actions):
            en_full = backend.array.ones(tensor_shape, dtype=backend.float_dtype)
            for proc_idx, module_idx in enumerate(processing_order):
                matrix = action_matrices[action_idx][proc_idx]
                en_module = matrix
                for _ in range(len(module_owned_sizes[module_idx])):
                    en_module = backend.array.sum(en_module, axis=-1)
                broadcast_shape = [1] * ndim
                for i, pos in enumerate(module_dep_positions[module_idx]):
                    broadcast_shape[pos] = en_module.shape[i]
                en_module = backend.array.reshape(en_module, broadcast_shape)
                en_full = en_full * en_module
            action_count = action_count + en_full
        safe_action_count = backend.array.maximum(
            action_count, backend.array.ones(tensor_shape, dtype=backend.float_dtype)
        )
        deadlock_mask = (action_count == 0).astype(backend.float_dtype)
        return safe_action_count, deadlock_mask

    # ------------------------------------------------------------------
    # Traced step / run — everything above only runs once via JAX tracing.
    # ------------------------------------------------------------------

    def _apply_one_step(tensor: Any, action_matrices, safe_action_count, deadlock_mask) -> Any:
        """Compute ⟦M⟧(tensor) using sequential variable elimination."""
        result = backend.array.zeros_like(tensor, dtype=backend.float_dtype)
        weighted = tensor / safe_action_count
        for action_idx in range(n_actions):
            action_tensor = weighted
            for proc_idx in range(n_proc):
                comp_matrix = action_matrices[action_idx][proc_idx]
                bshape = _broadcast_shapes[proc_idx]
                n_expand = _n_expands[proc_idx]
                axes = _axes_to_sum[proc_idx]
                perm = _perms[proc_idx]

                reshaped = backend.array.reshape(comp_matrix, bshape)
                expanded = action_tensor
                for _ in range(n_expand):
                    expanded = expanded[..., None]
                expanded = expanded * reshaped
                for axis in axes:
                    expanded = backend.array.sum(expanded, axis=axis)
                action_tensor = backend.array.transpose(expanded, perm)
            result = result + action_tensor
        result = result + tensor * deadlock_mask
        return result

    def _run_body(horizon, current_symbol_values):
        """Shared body of ``_run_baked`` and ``_run_overridden``."""
        with use_array_api(backend.array):
            action_matrices = _build_action_matrices(active_symbol_values=current_symbol_values)
            safe_action_count, deadlock_mask = _compute_enablement(action_matrices)

            # When ``horizon`` arrives as a concrete Python int (override path,
            # static_argnums=(0,) on _run_overridden), pass it directly to
            # fori_loop so reverse-mode autodiff sees concrete loop bounds.
            # The baked path passes a JAX tracer; preserve the original
            # int32 cast to avoid changing the existing XLA graph.
            if isinstance(horizon, int):
                horizon_array = horizon
            else:
                horizon_array = backend.array.asarray(horizon, dtype=backend.array.int32)

            def body(_, carry):
                total_probability, current_tensor = carry
                goal_mass = backend.array.sum(goal_mask * current_tensor)
                next_tensor = _apply_one_step(
                    non_goal_mask * current_tensor,
                    action_matrices,
                    safe_action_count,
                    deadlock_mask,
                )
                return total_probability + goal_mass, next_tensor

            total_probability, final_tensor = backend.lax.fori_loop(
                0,
                horizon_array,
                body,
                (
                    backend.array.array(0.0, dtype=backend.float_dtype),
                    initial_tensor,
                ),
            )
            return backend.array.asarray(
                total_probability + backend.array.sum(goal_mask * final_tensor),
                dtype=backend.float_dtype,
            )

    def _step_body(tensor, current_symbol_values):
        """Shared body of ``_step_baked`` and ``_step_overridden``."""
        with use_array_api(backend.array):
            action_matrices = _build_action_matrices(active_symbol_values=current_symbol_values)
            safe_action_count, deadlock_mask = _compute_enablement(action_matrices)
            return _apply_one_step(tensor, action_matrices, safe_action_count, deadlock_mask)

    @jit(backend)
    def _run_baked(horizon):
        """Compute Pr_M(T ⊨ ◇≤ⁿ G) with constants baked into the graph.

        Transition matrices and enablement tensors are constructed here
        (inside the ``@jit`` scope). Their Python-level assembly happens
        once during tracing; XLA folds them into the fori_loop body.
        """
        return _run_body(horizon, symbol_values)

    @jit(backend, static_argnums=(0,))
    def _run_overridden(horizon, override_values):
        """Same as ``_run_baked`` but with constants supplied as JAX inputs.

        ``horizon`` is marked static because reverse-mode differentiation
        through ``lax.fori_loop`` requires concrete loop bounds. Each unique
        ``int`` horizon triggers a fresh trace of this kernel; in practice
        training loops use a fixed horizon so the cost is paid once.
        """
        merged = dict(symbol_values)
        for key, value in zip(_overridable_keys, override_values):
            merged[key] = value
        return _run_body(horizon, merged)

    @jit(backend)
    def _step_baked(tensor):
        """Single-step ⟦M⟧(tensor) with constants baked into the graph."""
        return _step_body(tensor, symbol_values)

    @jit(backend)
    def _step_overridden(tensor, override_values):
        """Single-step ⟦M⟧(tensor) with constants supplied as JAX inputs."""
        merged = dict(symbol_values)
        for key, value in zip(_overridable_keys, override_values):
            merged[key] = value
        return _step_body(tensor, merged)

    def run(horizon, *, constants_override: Mapping[str, Any] | None = None):
        """Compute Pr_M(T ⊨ ◇≤ⁿ G) — bounded reachability probability.

        With ``constants_override=None`` (the default) the original baked-in
        constants are used and the compiled XLA graph is identical to the
        pre-extension behavior. Supplying a mapping of constant names to
        floats / JAX scalars routes through a separate jit-compiled kernel
        in which those constants flow as function arguments — enabling
        ``jax.grad`` to differentiate the result w.r.t. them.
        """
        if constants_override is None:
            return _run_baked(horizon)
        # Static-argnum on horizon for `_run_overridden` requires a hashable
        # Python int (not a 0-d tensor) so the JAX cache keys correctly.
        return _run_overridden(int(horizon), _validate_and_pack_override(constants_override))

    def step(tensor, *, constants_override: Mapping[str, Any] | None = None):
        """Single-step ⟦M⟧(tensor) — rebuilds matrices per call.

        See :func:`run` for the semantics of ``constants_override``.
        """
        if constants_override is None:
            return _step_baked(tensor)
        return _step_overridden(tensor, _validate_and_pack_override(constants_override))

    return StrategyResult(step=step, run=run)
