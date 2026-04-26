from __future__ import annotations

import argparse
import os
import pprint
import sys
import time
from pathlib import Path
from typing import Any

from .timing import (
    append_jsonl_record,
    build_timing_summary_records,
    coerce_probability,
    execute_timed_runs,
)
from .parser import load_model
from .pretty_print import model_to_data
from .backend import parse_backend, parse_dtype
from .compiler import compile_reachability


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    try:
        args = parser.parse_args(raw_argv)
    except SystemExit as exc:
        return int(exc.code)

    try:
        model_type = args.model_type or infer_model_type(args.model_path)
        constants = parse_constants(args.constants)
        if args.property_name is None and args.horizon is None:
            if any(
                option_was_provided(raw_argv, option)
                for option in ("--backend", "--dtype", "--num-cold-work-runs", "--num-work-runs", "--num-timed-runs", "--time-log")
            ):
                raise ValueError(
                    "--backend, --dtype, --num-cold-work-runs, --num-work-runs, --num-timed-runs, and --time-log "
                    "are only valid together with --property and --horizon"
                )
            parsed_model = load_model(model_type, args.model_path, constants=constants or None)
            pprint.pprint(model_to_data(parsed_model), sort_dicts=False)
            return 0

        if args.property_name is None or args.horizon is None:
            raise ValueError("--property and --horizon must be provided together")
        if args.horizon < 0:
            raise ValueError("--horizon must be non-negative")
        if args.num_work_runs <= 0:
            raise ValueError("--num-work-runs must be positive")
        if args.num_cold_work_runs < 0:
            raise ValueError("--num-cold-work-runs must be non-negative")
        backend_spec = parse_backend(args.backend)

        backend = backend_spec.raw
        dtype = parse_dtype(args.dtype)
        base_metadata = {
            "tool": "tessa",
            "model_path": os.path.relpath(str(args.model_path)),
            "model_type": model_type,
            "property": args.property_name,
            "horizon": args.horizon,
            "backend": backend,
            "dtype": dtype,
            "num_cold_work_runs": args.num_cold_work_runs,
            "num_work_runs": args.num_work_runs,
            "num_timed_runs": args.num_timed_runs,
            "constants": constants,
        }

        load_started = time.perf_counter()
        parsed_model = load_model(model_type, args.model_path, constants=constants or None)
        load_elapsed = time.perf_counter() - load_started
        if args.time_log is not None:
            append_jsonl_record(
                args.time_log,
                {
                    **base_metadata,
                    "phase": "load",
                    "iteration": 1,
                    "phase_iteration": 1,
                    "elapsed_seconds": load_elapsed,
                },
            )

        # Cold loop: discardable compile+warmup pairs run before the measured
        # work loop to absorb one-time Python/JAX setup overhead (module
        # deepcopy caches, stormpy init, JAX lowering caches). Records are
        # emitted with phases 'cold_compile' and 'cold_warmup' and ignored by
        # the summary/CSV layers so work_seconds reflects stable post-warmup
        # compile + warmup_run, not cold-start overhead.
        for cold_iter in range(1, args.num_cold_work_runs + 1):
            compile_started = time.perf_counter()
            cold_compiled = compile_reachability(
                parsed_model,
                property_name=args.property_name,
                backend=backend,
                dtype=dtype,
                model_path=args.model_path,
                constants=constants or None,
            )
            cold_compile_elapsed = time.perf_counter() - compile_started
            if args.time_log is not None:
                append_jsonl_record(
                    args.time_log,
                    {
                        **base_metadata,
                        "phase": "cold_compile",
                        "iteration": cold_iter,
                        "phase_iteration": cold_iter,
                        "elapsed_seconds": cold_compile_elapsed,
                    },
                )

            warmup_started = time.perf_counter()
            cold_raw = cold_compiled.run(args.horizon)
            cold_compiled.backend.block_until_ready(cold_raw)
            cold_warmup_elapsed = time.perf_counter() - warmup_started
            if args.time_log is not None:
                append_jsonl_record(
                    args.time_log,
                    {
                        **base_metadata,
                        "phase": "cold_warmup",
                        "iteration": cold_iter,
                        "phase_iteration": cold_iter,
                        "elapsed_seconds": cold_warmup_elapsed,
                    },
                )

        # Work loop: repeat compile+warmup as a pair so each iteration triggers
        # a fresh JIT compile. Only the first call to a @jit closure triggers
        # XLA compilation; to measure cold-start compile+warmup repeatedly we
        # must call compile_reachability() again each iteration (it produces a
        # new @jit closure) rather than re-running the same jitted function.
        work_records: list[dict[str, Any]] = []
        compiled_model = None
        for work_iter in range(1, args.num_work_runs + 1):
            compile_started = time.perf_counter()
            compiled_model = compile_reachability(
                parsed_model,
                property_name=args.property_name,
                backend=backend,
                dtype=dtype,
                model_path=args.model_path,
                constants=constants or None,
            )
            compile_elapsed = time.perf_counter() - compile_started
            compile_record = {
                **base_metadata,
                "phase": "compile",
                "iteration": work_iter,
                "phase_iteration": work_iter,
                "elapsed_seconds": compile_elapsed,
            }
            work_records.append(compile_record)
            if args.time_log is not None:
                append_jsonl_record(args.time_log, compile_record)

            warmup_started = time.perf_counter()
            raw_probability = compiled_model.run(args.horizon)
            raw_probability = compiled_model.backend.block_until_ready(raw_probability)
            warmup_elapsed = time.perf_counter() - warmup_started
            probability = coerce_probability(
                compiled_model.backend.device_get(raw_probability)
            )
            warmup_record = {
                **base_metadata,
                "phase": "warmup",
                "iteration": work_iter,
                "phase_iteration": work_iter,
                "elapsed_seconds": warmup_elapsed,
                "probability": probability,
            }
            work_records.append(warmup_record)
            if args.time_log is not None:
                append_jsonl_record(args.time_log, warmup_record)

        assert compiled_model is not None  # --num-work-runs >= 1 guaranteed above

        probability, measured_records = execute_timed_runs(
            lambda: compiled_model.run(args.horizon),
            num_timed_runs=args.num_timed_runs,
            synchronize=compiled_model.backend.block_until_ready,
            record_result=lambda raw: coerce_probability(
                compiled_model.backend.device_get(raw)
            ),
            time_log=args.time_log,
            metadata=base_metadata,
        )
        if args.time_log is not None:
            for record in build_timing_summary_records(
                work_records + measured_records,
                metadata=base_metadata,
            ):
                append_jsonl_record(args.time_log, record)
        print(compiled_model.backend)
        print(repr(probability))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def infer_model_type(model_path: str | Path) -> str:
    suffix = Path(model_path).suffix.lower()
    if suffix == ".jani":
        return "jani"
    if suffix in {".prism", ".pm", ".nm"}:
        return "prism"
    raise ValueError("Could not infer model type from file extension. Use --type jani or --type prism.")


def parse_constants(raw_constants: list[str]) -> dict[str, bool | int | float]:
    constants: dict[str, bool | int | float] = {}
    for raw_constant in raw_constants:
        if "=" not in raw_constant:
            raise ValueError(f"Invalid --const value '{raw_constant}'. Expected NAME=VALUE.")
        name, raw_value = raw_constant.split("=", 1)
        name = name.strip()
        raw_value = raw_value.strip()
        if not name:
            raise ValueError(f"Invalid --const value '{raw_constant}'. Expected NAME=VALUE.")
        constants[name] = parse_constant_value(raw_value)
    return constants


def parse_constant_value(raw_value: str) -> bool | int | float:
    lowered = raw_value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        return int(raw_value)
    except ValueError:
        pass

    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid constant value '{raw_value}'. Expected bool, int, or float."
        ) from exc


def option_was_provided(argv: list[str], option: str) -> bool:
    return any(argument == option or argument.startswith(f"{option}=") for argument in argv)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tessa")
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--type", dest="model_type", choices=("jani", "prism"))
    parser.add_argument(
        "--const",
        dest="constants",
        action="append",
        default=[],
        help="Constant override in the form NAME=VALUE",
    )
    parser.add_argument("--property", dest="property_name")
    parser.add_argument("--horizon", type=int)
    parser.add_argument(
        "--backend",
        help="Reachability backend: numpy, explicit, jax:cpu, or jax:cuda:N",
    )
    parser.add_argument(
        "--dtype",
        help="Reachability dtype: float32 or float64",
    )
    parser.add_argument(
        "--num-cold-work-runs",
        type=int,
        default=3,
        help="Number of throwaway compile+warmup pairs run before the measured "
        "work loop to prime JAX/XLA caches. Records are emitted with phases "
        "'cold_compile' and 'cold_warmup' and excluded from summary stats.",
    )
    parser.add_argument(
        "--num-work-runs",
        type=int,
        default=1,
        help="Number of compile+warmup iterations (each pair is timed). "
        "Mean and std of compile_seconds, warmup_avg_seconds, and "
        "work_seconds = stable_compile + warmup_run are emitted to the "
        "time-log summary (means over the warm loop; use "
        "--num-cold-work-runs to prime caches first).",
    )
    parser.add_argument(
        "--num-timed-runs",
        type=int,
        default=1,
        help="Number of measured executions after the work loop",
    )
    parser.add_argument(
        "--time-log",
        type=Path,
        help="Append JSONL timing records to this path",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
