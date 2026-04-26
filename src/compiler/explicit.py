"""Explicit backend: delegate bounded reachability to Storm via stormpy.

Unlike the dense backends, nothing in Tessa's tensor pipeline runs here. We
reparse the PRISM file through stormpy, build the sparse DTMC once, and on
each ``run(horizon)`` rebuild only the property formula (``F<=H <label>``)
and invoke ``stormpy.model_checking`` to extract the initial-state
probability.

This is the right choice when reachable states ≪ total states. For models
where reachable ≈ Cartesian product (herman, weather_factory, meeting,
parqueues), the dense path is strictly faster.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable, Mapping

from ..backend import BackendRuntime


def _import_stormpy():
    try:
        return importlib.import_module("stormpy")
    except ImportError as exc:
        raise RuntimeError(
            "Backend 'explicit' requires stormpy. Install it or use the repo's Nix environment."
        ) from exc


def _format_constants(constants: Mapping[str, Any] | None) -> str:
    """Render {name: value} into stormpy's 'a=1,b=2' form."""
    if not constants:
        return ""
    parts = []
    for name, value in constants.items():
        if isinstance(value, bool):
            parts.append(f"{name}={'true' if value else 'false'}")
        else:
            parts.append(f"{name}={value}")
    return ",".join(parts)


def _build_explicit_runner(
    *,
    model_path: str | Path,
    property_name: str,
    constants: Mapping[str, Any] | None,
) -> tuple[Callable[[int], float], Any, Any]:
    """Return ``(run, stormpy_model, stormpy_program)``.

    ``run(horizon)`` model-checks ``P=?[F<=H "<property_name>"]`` against the
    already-built sparse DTMC and returns a Python float at the initial state.
    """
    stormpy = _import_stormpy()

    program = stormpy.parse_prism_program(str(model_path))
    constants_str = _format_constants(constants)
    if constants_str:
        # preprocess_symbolic_input applies -const overrides and returns the
        # residual program. The call signature is (program, properties,
        # constants_string) → (preprocessed_program, formulas).
        preprocessed = stormpy.preprocess_symbolic_input(program, [], constants_str)
        program = preprocessed[0].as_prism_program()

    # Build the sparse DTMC once (stormpy needs at least one property at
    # construction time, so we use a placeholder; the horizon-specific
    # property is re-parsed per ``run()`` call).
    placeholder = stormpy.parse_properties_for_prism_program(
        f'P=? [ F<=0 "{property_name}" ]', program,
    )
    model = stormpy.build_model(program, placeholder)
    initial_state = model.initial_states[0]

    def run(horizon: int) -> float:
        prop_str = f'P=? [ F<={int(horizon)} "{property_name}" ]'
        props = stormpy.parse_properties_for_prism_program(prop_str, program)
        result = stormpy.model_checking(model, props[0])
        return float(result.at(initial_state))

    return run, model, program


def compile_explicit(
    *,
    model_path: str | Path,
    property_name: str,
    constants: Mapping[str, Any] | None,
    backend: BackendRuntime,
):
    """Build a stormpy-backed CompiledReachabilityModel-compatible shim.

    The returned object exposes ``.run(horizon)`` and ``.backend`` — the only
    fields the CLI actually consumes. Dense-tensor fields (``initial_tensor``,
    ``goal_mask``, etc.) are set to ``None`` sentinels because nothing in the
    explicit path materializes them.
    """
    # Imported here to avoid a circular import with compiler/__init__.py.
    from . import CompiledReachabilityModel

    run, model, program = _build_explicit_runner(
        model_path=model_path,
        property_name=property_name,
        constants=constants,
    )

    def step(_tensor):
        raise RuntimeError("step() is not available for --backend explicit")

    return CompiledReachabilityModel(
        property_name=property_name,
        backend=backend,
        variables=(),
        tensor_shape=(),
        initial_tensor=None,
        goal_mask=None,
        non_goal_mask=None,
        step=step,
        run=run,
    )
