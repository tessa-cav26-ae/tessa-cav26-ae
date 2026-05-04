from __future__ import annotations

import json
import logging
import re
import statistics
import time
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


_SCALAR_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_STORM_RESULT_RE = re.compile(
    r"Result \(for initial states\):\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


def coerce_probability(value: Any) -> float:
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def parse_tessa_probability(stdout: str) -> float:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Could not parse a probability from empty tessa output")

    candidates = list(reversed(lines))
    for candidate in candidates:
        match = _SCALAR_RE.search(candidate)
        if match is not None:
            return float(match.group(0))
    raise ValueError("Could not parse a probability from tessa output")


def parse_storm_probability(stdout: str) -> float:
    matches = _STORM_RESULT_RE.findall(stdout)
    if not matches:
        raise ValueError("Could not parse a probability from storm output")
    return float(matches[-1])


def parse_rubicon_probability(stdout: str) -> float:
    """Parse the trailing probability scalar emitted by ``src.rubicon_runner``.

    The runner echoes dice's full JSON (which contains many scalars) on
    earlier lines, then writes the extracted probability on its own final
    line. We therefore take the *last* scalar on the *last* non-empty line
    rather than scanning backwards across the whole stdout.
    """
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Could not parse a probability from empty rubicon output")
    matches = _SCALAR_RE.findall(lines[-1])
    if not matches:
        raise ValueError("Could not parse a probability from rubicon output")
    return float(matches[-1])


def parse_geni_probability(stdout: str) -> float:
    """Parse the trailing probability scalar emitted by ``src.geni_runner``.

    Gennifer's stdout contains many scalars (per-value Pr lines and the
    --pt timing block). The runner re-emits Pr(true) on its own final line
    after a sentinel newline, so the same last-line-scalar contract used
    for rubicon applies here.
    """
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Could not parse a probability from empty geni output")
    matches = _SCALAR_RE.findall(lines[-1])
    if not matches:
        raise ValueError("Could not parse a probability from geni output")
    return float(matches[-1])


def append_jsonl_record(path: str | Path, record: dict[str, Any]) -> None:
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        json.dump(record, handle, sort_keys=True)
        handle.write("\n")
    logger.debug("Appended timing record to %s", log_path)


def _mean_std(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def build_timing_summary_records(
    records: list[dict[str, Any]],
    *,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Summarize per-phase timings as mean+std.

    Emits one summary record per phase present in ``records``:
    ``compile_summary``, ``warmup_summary``, ``work_summary``
    (compile_i + warmup_i per iteration), ``measured_summary``.
    ``sample_count == 1`` implies ``std_elapsed_seconds == 0.0``.

    Cold-start records (``cold_compile`` / ``cold_warmup``) may also be
    present in the JSONL input; they represent throwaway iterations used
    to prime JAX/XLA caches before the warm loop and are intentionally
    ignored here so ``work_seconds`` reflects stable compile + warmup_run.
    """
    summary_records: list[dict[str, Any]] = []

    def _summary(phase: str, values: list[float]) -> dict[str, Any]:
        mean, std = _mean_std(values)
        record = {
            "phase": phase,
            "mean_elapsed_seconds": mean,
            "std_elapsed_seconds": std,
            "sample_count": len(values),
        }
        if metadata:
            record.update(metadata)
        return record

    compile_records = sorted(
        (r for r in records if r["phase"] == "compile"),
        key=lambda r: r.get("phase_iteration", 0),
    )
    warmup_records = sorted(
        (r for r in records if r["phase"] == "warmup"),
        key=lambda r: r.get("phase_iteration", 0),
    )
    measured_records = [r for r in records if r["phase"] == "measured"]

    if compile_records:
        summary_records.append(
            _summary("compile_summary", [r["elapsed_seconds"] for r in compile_records])
        )
    if warmup_records:
        summary_records.append(
            _summary("warmup_summary", [r["elapsed_seconds"] for r in warmup_records])
        )
    # work_seconds = stable_compile + warmup_run, computed as the paired-sum
    # mean/std of the warm compile and warmup records (cold_* records are
    # excluded upstream). Only meaningful when we have the same number of
    # each (the CLI emits them in lockstep).
    if compile_records and warmup_records and len(compile_records) == len(warmup_records):
        work_values = [
            c["elapsed_seconds"] + w["elapsed_seconds"]
            for c, w in zip(compile_records, warmup_records)
        ]
        summary_records.append(_summary("work_summary", work_values))
    if measured_records:
        summary_records.append(
            _summary("measured_summary", [r["elapsed_seconds"] for r in measured_records])
        )

    return summary_records


def execute_timed_runs(
    run_once: Callable[[], Any],
    *,
    num_timed_runs: int = 1,
    synchronize: Callable[[Any], Any] | None = None,
    record_result: Callable[[Any], float] | None = None,
    time_log: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Run ``run_once`` ``num_timed_runs`` times and record per-iteration walltime.

    ``synchronize``, if provided, is invoked inside the timed region after
    ``run_once`` returns. This is required for JAX backends whose operations
    are asynchronous — ``run_once`` returns a future and the actual work is
    still in flight, so the timer would capture only dispatch time unless
    we block here. Pass ``backend.block_until_ready`` from the CLI.

    This function assumes the callable is already warm (JIT compiled). Warmup
    is handled upstream as part of the compile+warmup work loop.
    """
    if num_timed_runs <= 0:
        raise ValueError("--num-timed-runs must be positive")

    logger.debug("Starting timed runs: %d measured", num_timed_runs)
    records: list[dict[str, Any]] = []
    parsed_result: float | None = None
    raw_result: Any = None

    for index in range(num_timed_runs):
        started = time.perf_counter()
        raw_result = run_once()
        if synchronize is not None:
            raw_result = synchronize(raw_result)
        elapsed = time.perf_counter() - started

        if record_result is not None:
            parsed_result = record_result(raw_result)

        logger.debug("  measured iter %d: %.4fs", index + 1, elapsed)
        record = {
            "phase": "measured",
            "iteration": index + 1,
            "phase_iteration": index + 1,
            "elapsed_seconds": elapsed,
        }
        if parsed_result is not None:
            record["probability"] = parsed_result
        if metadata:
            record.update(metadata)
        records.append(record)

        if time_log is not None:
            append_jsonl_record(time_log, record)

    return raw_result, records


__all__ = [
    "append_jsonl_record",
    "build_timing_summary_records",
    "coerce_probability",
    "execute_timed_runs",
    "parse_rubicon_probability",
    "parse_storm_probability",
    "parse_tessa_probability",
]
