from __future__ import annotations

import csv
import json
import logging
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Optional


import click

from .log_config import log_command_args, setup_logging
from .timing import (
    parse_geni_probability,
    parse_rubicon_probability,
    parse_storm_probability,
    parse_tessa_probability,
)

# Tools that drive a single one-shot subprocess per case (no in-process work
# loop). For these we repeat at the subprocess level, average elapsed_seconds
# across trials, and skip the tessa-only time-log JSONL.
_SUBPROCESS_TOOLS = ("storm", "rubicon", "geni")

# Each tessa phase contributes (mean, std) columns derived from a single
# time-log. work_seconds = stable_compile + warmup_run, paired per warm-loop
# iteration. Cold-start records (cold_compile/cold_warmup) exist in the JSONL
# but are dropped here so the CSV reflects only post-warmup timings.
PHASE_TIMING_KEYS = (
    "load_seconds",
    "compile_seconds",
    "warmup_avg_seconds",
    "measured_avg_seconds",
)
PHASE_TIMING_STD_KEYS = (
    "compile_std",
    "warmup_std",
    "measured_std",
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_ROOT = REPO_ROOT / "benchmarks"


def _rel(p: Path | str) -> str:
    p = Path(p)
    if p.is_absolute():
        try:
            return str(p.relative_to(REPO_ROOT))
        except ValueError:
            return str(p)
    return str(p)


def pretty_path(p: Path | str) -> str:
    p = Path(p).expanduser().resolve()
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        pass
    home = Path.home().resolve()
    try:
        return "~/" + str(p.relative_to(home))
    except ValueError:
        return str(p)


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _mean_std(values: list[float]) -> tuple[float | None, float]:
    if not values:
        return None, 0.0
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def _read_phase_timings(time_log: Path) -> dict[str, float | None]:
    """Read the JSONL time log and extract per-phase timing columns.

    A single tessa invocation with ``--num-work-runs N`` writes N compile
    records and N warmup records (one pair per work iteration), plus
    ``--num-timed-runs M`` measured records. We return the mean and std
    of each phase, plus ``work_seconds`` = mean(compile_i + warmup_i) and
    ``work_std`` = stdev of the same paired sum.
    """
    timings: dict[str, float | None] = {key: None for key in PHASE_TIMING_KEYS}
    for key in PHASE_TIMING_STD_KEYS:
        timings[key] = 0.0
    timings["work_seconds"] = None
    timings["work_std"] = 0.0
    if not time_log.exists():
        return timings
    try:
        records = [json.loads(line) for line in time_log.read_text().splitlines() if line.strip()]
    except (json.JSONDecodeError, OSError):
        return timings

    compile_times: list[float] = []
    warmup_times: list[float] = []
    measured_times: list[float] = []
    for record in records:
        phase = record.get("phase")
        if phase == "load":
            timings["load_seconds"] = record.get("elapsed_seconds")
        elif phase == "compile":
            compile_times.append(record["elapsed_seconds"])
        elif phase == "warmup":
            warmup_times.append(record["elapsed_seconds"])
        elif phase == "measured":
            measured_times.append(record["elapsed_seconds"])

    compile_mean, compile_std = _mean_std(compile_times)
    warmup_mean, warmup_std = _mean_std(warmup_times)
    measured_mean, measured_std = _mean_std(measured_times)

    timings["compile_seconds"] = compile_mean
    timings["compile_std"] = compile_std
    timings["warmup_avg_seconds"] = warmup_mean
    timings["warmup_std"] = warmup_std
    timings["measured_avg_seconds"] = measured_mean
    timings["measured_std"] = measured_std

    # work_seconds pairs compile_i with warmup_i per iteration, then takes
    # mean/std of the sum. Requires equal-length lists (guaranteed by CLI).
    if compile_times and warmup_times and len(compile_times) == len(warmup_times):
        work_values = [c + w for c, w in zip(compile_times, warmup_times)]
        work_mean, work_std = _mean_std(work_values)
        timings["work_seconds"] = work_mean
        timings["work_std"] = work_std

    return timings


@dataclass
class BenchmarkContext:
    tool: str
    backend: str | None
    timeout: int
    num_work_runs: int
    num_timed_runs: int
    dtype: str
    output_dir: Path
    storm_cmd: str
    tessa_cmd: str
    rubicon_cmd_argv: list[str] = field(default_factory=list)
    dice_cmd: str = "dice"
    geni_cmd_argv: list[str] = field(default_factory=list)
    gennifer_cmd: str = "gennifer"
    geni_mode: str = "monolithic"
    model_type: str = "prism"
    storm_extra_args: list[str] = field(default_factory=list)
    dice_extra_args: list[str] = field(default_factory=list)
    engine: str | None = None
    _csv_header_written: bool = field(default=False, init=False)

    @property
    def csv_path(self) -> Path:
        return self.output_dir / f"{self.tool_label}.csv"

    _STORM_ENGINE: ClassVar[dict[str, str]] = {"add": "dd", "spm": "sparse"}

    @property
    def jsonl_tag(self) -> str:
        return self.dtype

    @property
    def tool_label(self) -> str:
        if self.tool == "storm":
            if self.engine:
                return f"storm.{self.engine}"
            return "storm"
        if self.tool == "rubicon":
            return "rubicon"
        if self.tool == "geni":
            return "geni"
        return "tessa"

    @property
    def jsonl_tool_label(self) -> str:
        """Detailed label used in JSONL time-log filenames."""
        if self.tool == "storm":
            if self.engine:
                return f"storm.{self.engine}"
            return "storm"
        if self.tool == "rubicon":
            return "rubicon"
        if self.tool == "geni":
            return "geni"
        backend = self.backend or "numpy"
        return f"tessa.{backend.replace(':', '-')}"

    def run_case(
        self,
        *,
        suite: str,
        case_id: str,
        model_path: Path,
        property_name: str,
        horizon: int,
        constants: dict[str, int | float | bool],
        parameters: dict[str, int | float],
    ) -> dict[str, Any]:
        logger.info("[%s] %s :: %s", suite, case_id, self.tool_label)

        status = "ok"
        probability: float | None = None
        error_message: str | None = None

        # Storm and rubicon have no in-process work-loop notion: we repeat at
        # the subprocess level so elapsed_seconds becomes a mean (with std)
        # across runs. Tessa: single subprocess, --num-work-runs carries the
        # repetition.
        is_subprocess_tool = self.tool in _SUBPROCESS_TOOLS
        subprocess_trials = self.num_work_runs if is_subprocess_tool else 1
        subprocess_elapsed: list[float] = []
        tessa_elapsed: float | None = None
        time_log: Path | None = None

        for trial in range(1, subprocess_trials + 1):
            if is_subprocess_tool:
                if subprocess_trials > 1:
                    logger.info("  trial %d/%d", trial, subprocess_trials)
                if self.tool == "storm":
                    command = self._build_storm_command(model_path, property_name, horizon, constants)
                elif self.tool == "geni":
                    command = self._build_geni_command(parameters)
                else:
                    command = self._build_rubicon_command(model_path, property_name, horizon, constants)
            else:
                time_log = self.output_dir / f"{case_id}.{self.jsonl_tool_label}.{self.jsonl_tag}.time.jsonl"
                # Fresh time-log per invocation to avoid appending to a stale
                # file from a previous run.
                if time_log.exists():
                    time_log.unlink()
                command = self._build_tessa_command(model_path, property_name, horizon, constants, time_log)

            logger.debug("  $ %s", " ".join(_rel(token) for token in command))

            try:
                started = time.perf_counter()
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    check=False,
                    cwd=REPO_ROOT,
                )
                elapsed = time.perf_counter() - started
                if is_subprocess_tool:
                    subprocess_elapsed.append(elapsed)
                else:
                    tessa_elapsed = elapsed
                if completed.stdout:
                    for line in completed.stdout.splitlines():
                        logger.debug("  %s", line)
                if completed.stderr:
                    for line in completed.stderr.splitlines():
                        logger.warning("  %s", line)
                if completed.returncode != 0:
                    status = "failed"
                    error_message = f"Exit code {completed.returncode}"
                    logger.error("  %s: %s", status, error_message)
                    break
                parser = {
                    "storm": parse_storm_probability,
                    "rubicon": parse_rubicon_probability,
                    "geni": parse_geni_probability,
                    "tessa": parse_tessa_probability,
                }[self.tool]
                try:
                    probability = parser(completed.stdout)
                except ValueError as exc:
                    status = "parse_error"
                    error_message = str(exc)
                    logger.error("  %s: %s", status, error_message)
                    break
            except subprocess.TimeoutExpired:
                status = "timeout"
                error_message = f"Timed out after {self.timeout} seconds"
                logger.error("  %s: %s", status, error_message)
                break
            except Exception as exc:
                status = "error"
                error_message = str(exc)
                logger.error("  %s: %s", status, error_message)
                break

        if is_subprocess_tool:
            elapsed_seconds, elapsed_std = _mean_std(subprocess_elapsed)
        else:
            elapsed_seconds = tessa_elapsed
            elapsed_std = 0.0

        phase_timings: dict[str, Any] = {}
        tessa_timings: dict[str, Any] = {}
        if not is_subprocess_tool and time_log is not None:
            phase_timings = _read_phase_timings(time_log)
            # ``work_seconds`` = compile + warmup_avg is Tessa-only. Storm's
            # "work time" is already captured by ``elapsed_seconds``; writing
            # a duplicate column would be redundant and break schema
            # compatibility with historical Storm CSVs that predate the key.
            tessa_timings["work_seconds"] = phase_timings.pop("work_seconds", None)
            tessa_timings["work_std"] = phase_timings.pop("work_std", 0.0)

        if self.tool == "storm":
            tool_meta: dict[str, Any] = {
                "num_work_runs": self.num_work_runs,
                "storm_extra_args": " ".join(self.storm_extra_args) if self.storm_extra_args else "",
            }
        elif self.tool == "rubicon":
            tool_meta = {"num_work_runs": self.num_work_runs}
        elif self.tool == "geni":
            tool_meta = {
                "num_work_runs": self.num_work_runs,
                "geni_mode": self.geni_mode,
            }
        else:
            tool_meta = {
                "backend": self.backend,
                "dtype": self.dtype,
                "num_work_runs": self.num_work_runs,
                "num_timed_runs": self.num_timed_runs,
            }
        result: dict[str, Any] = {
            "suite": suite,
            "case_id": case_id,
            "tool": self.jsonl_tool_label,
            **tool_meta,
            "property": property_name,
            "horizon": horizon,
            "constants": json.dumps(constants) if constants else "",
            **parameters,
            "status": status,
            "probability": probability,
            "elapsed_seconds": elapsed_seconds,
            **({"elapsed_std": elapsed_std} if is_subprocess_tool else {}),
            **tessa_timings,
            **phase_timings,
            "error_message": error_message,
        }
        self._write_csv_row(result)
        self._log_summary(result)
        return result

    def _log_summary(self, result: dict[str, Any]) -> None:
        parts = [f"  => {result['status']}"]
        if result.get("probability") is not None:
            parts.append(f"P={result['probability']:.6e}")
        if result.get("elapsed_seconds") is not None:
            parts.append(f"total={result['elapsed_seconds']:.3f}s")
        if result.get("work_seconds") is not None:
            work_std = result.get("work_std") or 0.0
            parts.append(f"work={result['work_seconds']:.3f}±{work_std:.3f}s")
        if result.get("compile_seconds") is not None:
            parts.append(f"compile={result['compile_seconds']:.3f}s")
        if result.get("measured_avg_seconds") is not None:
            parts.append(f"run={result['measured_avg_seconds']:.3f}s")
        logger.info("%s", "  ".join(parts))

    def _write_csv_row(self, row: dict[str, Any]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.csv_path
        fieldnames = list(row.keys())
        write_header = not self._csv_header_written and not (
            csv_path.exists() and csv_path.stat().st_size > 0
        )
        with csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        self._csv_header_written = True
        logger.debug("Wrote CSV row to %s", pretty_path(csv_path))

    def _build_storm_command(
        self,
        model_path: Path,
        property_name: str,
        horizon: int,
        constants: dict[str, int | float | bool],
    ) -> list[str]:
        input_flag = "--jani" if self.model_type == "jani" else "--prism"
        command = [
            self.storm_cmd,
            input_flag, _rel(model_path),
            "--prop", f'P=? [ F<={horizon} "{property_name}" ]',
        ]
        if self.engine:
            command.extend(["--engine", self._STORM_ENGINE.get(self.engine, self.engine)])
        if constants:
            command.extend(["--constants", ",".join(f"{k}={v}" for k, v in constants.items())])
        command.extend(self.storm_extra_args)
        return command

    def _build_geni_command(self, parameters: dict[str, int | float]) -> list[str]:
        # Geni doesn't read PRISM/JANI: gen_weather_factory_gennifer.py emits
        # a .gir from (N, H, mode) and gennifer interprets it. The runner
        # caches the .gir under <output-dir>/geni-cache so multiple
        # num_work_runs trials skip the generator after the first.
        if "N" not in parameters or "H" not in parameters:
            raise click.UsageError(
                "--tool geni requires the suite to expose N and H parameters "
                "(currently only weather-factory does)"
            )
        cache_dir = self.output_dir / "geni-cache"
        command = [
            *self.geni_cmd_argv,
            "--n", str(parameters["N"]),
            "--h", str(parameters["H"]),
            "--mode", self.geni_mode,
            "--gennifer-cmd", self.gennifer_cmd,
            "--workdir", str(cache_dir),
        ]
        return command

    def _build_rubicon_command(
        self,
        model_path: Path,
        property_name: str,
        horizon: int,
        constants: dict[str, int | float | bool],
    ) -> list[str]:
        # Rubicon requires PRISM input — JANI models are not supported by the
        # transpiler. Surface the mismatch loudly rather than producing a
        # misleading translation error from rubicon itself.
        if self.model_type != "prism":
            raise click.UsageError(
                f"--tool rubicon requires --model-type prism (got {self.model_type})"
            )
        command = [
            *self.rubicon_cmd_argv,
            "--prism", _rel(model_path),
            "--property", property_name,
            "--horizon", str(horizon),
            "--dice-cmd", self.dice_cmd,
        ]
        for name, value in constants.items():
            command.extend(["--const", f"{name}={value}"])
        for token in self.dice_extra_args:
            command.extend(["--dice-extra-arg", token])
        return command

    def _build_tessa_command(
        self,
        model_path: Path,
        property_name: str,
        horizon: int,
        constants: dict[str, int | float | bool],
        time_log: Path,
    ) -> list[str]:
        command = [
            self.tessa_cmd,
            _rel(model_path),
            "--type", self.model_type,
            "--property", property_name,
            "--horizon", str(horizon),
            "--backend", self.backend or "numpy",
            "--dtype", self.dtype,
            "--num-work-runs", str(self.num_work_runs),
            "--num-timed-runs", str(self.num_timed_runs),
            "--time-log", _rel(time_log),
        ]
        for name, value in constants.items():
            command.extend(["--const", f"{name}={value}"])
        return command


@click.group(chain=True)
@click.option("--tool", required=True, type=click.Choice(["storm", "tessa", "rubicon", "geni"]))
@click.option("--backend", default=None, help="Tessa backend: numpy, jax:cpu, jax:cuda:N")
@click.option("--timeout", default=600, type=int)
@click.option(
    "--num-work-runs",
    default=3,
    type=int,
    help="Number of compile+warmup iterations per tessa invocation (mean+std "
    "reported). For storm, the number of subprocess runs averaged into "
    "elapsed_seconds.",
)
@click.option("--num-timed-runs", default=1, type=int)
@click.option("--dtype", default="float32")
@click.option("--output-dir", required=True, type=click.Path(path_type=Path), help="Directory for CSV and time-log outputs")
@click.option("--engine", default=None, type=click.Choice(["add", "spm"]), help="Storm engine: add (MTBDD/dd), spm (sparse)")
@click.option("--model-type", default="prism", type=click.Choice(["prism", "jani"]), help="Model file format to load (selects .prism or .jani siblings under benchmarks/)")
@click.option("--storm-cmd", default="storm")
@click.option("--storm-extra-args", default="", help="Extra arguments passed through to the storm command")
@click.option("--tessa-cmd", default="tessa")
@click.option(
    "--rubicon-cmd",
    default=f"{sys.executable} -m src.rubicon_runner",
    help="Command (with optional args) that runs translate-and-dice; stdout's last line must be the probability",
)
@click.option("--dice-cmd", default="dice", help="Dice binary; passed through to --rubicon-cmd")
@click.option("--dice-extra-args", default="", help="Extra arguments passed through to dice (only effective with --tool rubicon)")
@click.option(
    "--geni-cmd",
    default=f"{sys.executable} -m src.geni_runner",
    help="Command (with optional args) that generates a .gir and runs gennifer; stdout's last line must be the probability",
)
@click.option("--gennifer-cmd", default="gennifer", help="Gennifer binary; passed through to --geni-cmd")
@click.option(
    "--geni-mode",
    default="monolithic",
    type=click.Choice(["monolithic", "sequential"]),
    help="Generator mode for the weather-factory .gir program",
)
@click.option("--log-file", default=None, type=click.Path(path_type=Path), help="Path to write detailed logs")
@click.option(
    "--log-console-level",
    default="INFO",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
)
@click.option(
    "--log-file-level",
    default="DEBUG",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
)
@click.pass_context
def cli(
    ctx,
    tool,
    backend,
    timeout,
    num_work_runs,
    num_timed_runs,
    dtype,
    output_dir,
    engine,
    model_type,
    storm_cmd,
    storm_extra_args,
    tessa_cmd,
    rubicon_cmd,
    dice_cmd,
    dice_extra_args,
    geni_cmd,
    gennifer_cmd,
    geni_mode,
    log_file: Optional[Path],
    log_console_level: str,
    log_file_level: str,
):
    setup_logging(log_console_level, log_file_level, log_file)
    log_command_args(
        "cli",
        tool=tool,
        backend=backend,
        timeout=timeout,
        num_work_runs=num_work_runs,
        num_timed_runs=num_timed_runs,
        dtype=dtype,
        output_dir=output_dir,
        engine=engine,
    )
    if tool in _SUBPROCESS_TOOLS:
        tessa_only = {
            "backend": backend,
            "dtype": dtype,
            "num_timed_runs": num_timed_runs,
        }
        explicit = [
            f"--{name.replace('_', '-')}"
            for name, val in tessa_only.items()
            if ctx.get_parameter_source(name) == click.core.ParameterSource.COMMANDLINE
        ]
        if explicit:
            raise click.UsageError(
                f"The following flags are tessa-only and cannot be used with --tool {tool}: {', '.join(explicit)}"
            )

    ctx.ensure_object(dict)
    resolved_dir = output_dir.resolve()
    resolved_dir.mkdir(parents=True, exist_ok=True)
    ctx.obj = BenchmarkContext(
        tool=tool,
        backend=backend,
        timeout=timeout,
        num_work_runs=num_work_runs,
        num_timed_runs=num_timed_runs,
        dtype=dtype,
        output_dir=resolved_dir,
        storm_cmd=storm_cmd,
        storm_extra_args=shlex.split(storm_extra_args),
        tessa_cmd=tessa_cmd,
        rubicon_cmd_argv=shlex.split(rubicon_cmd),
        dice_cmd=dice_cmd,
        dice_extra_args=shlex.split(dice_extra_args),
        geni_cmd_argv=shlex.split(geni_cmd),
        gennifer_cmd=gennifer_cmd,
        geni_mode=geni_mode,
        engine=engine,
        model_type=model_type,
    )


@cli.command("herman")
@click.option("-N", "n_values", type=str, default="17", help="Comma-separated station counts")
@click.option("-H", "horizons", type=str, default="10", help="Comma-separated horizons")
@click.pass_context
def herman_cmd(ctx, n_values, horizons):
    ext = ctx.obj.model_type
    for N in parse_int_list(n_values):
        for H in parse_int_list(horizons):
            ctx.obj.run_case(
                suite="herman",
                case_id=f"n{N}-h{H}",
                model_path=BENCHMARKS_ROOT / "herman" / f"herman-{N}.{ext}",
                property_name="stable",
                horizon=H,
                constants={},
                parameters={"N": N, "H": H},
            )


@cli.command("meeting")
@click.option("-N", "n_values", type=str, default="10", help="Comma-separated participant counts")
@click.option("-H", "horizons", type=str, default="10", help="Comma-separated horizons")
@click.pass_context
def meeting_cmd(ctx, n_values, horizons):
    ext = ctx.obj.model_type
    for N in parse_int_list(n_values):
        for H in parse_int_list(horizons):
            ctx.obj.run_case(
                suite="meeting",
                case_id=f"n{N}-h{H}",
                model_path=BENCHMARKS_ROOT / "meeting" / f"meeting-{N}.{ext}",
                property_name="goal",
                horizon=H,
                constants={},
                parameters={"N": N, "H": H},
            )


@cli.command("weather-factory")
@click.option("-N", "n_values", type=str, default="13", help="Comma-separated factory counts")
@click.option("-H", "horizons", type=str, default="10", help="Comma-separated horizons")
@click.pass_context
def weather_factory_cmd(ctx, n_values, horizons):
    ext = ctx.obj.model_type
    for N in parse_int_list(n_values):
        for H in parse_int_list(horizons):
            ctx.obj.run_case(
                suite="weather-factory",
                case_id=f"n{N}-h{H}",
                model_path=BENCHMARKS_ROOT / "weather_factory" / f"weatherfactory{N}.{ext}",
                property_name="allStrike",
                horizon=H,
                constants={},
                parameters={"N": N, "H": H},
            )


@cli.command("parqueues")
@click.option("-Q", "queue_values", type=str, default="8", help="Comma-separated queue counts")
@click.option("-N", "capacity_values", type=str, default="3", help="Comma-separated capacities")
@click.option("-H", "horizons", type=str, default="10", help="Comma-separated horizons")
@click.pass_context
def parqueues_cmd(ctx, queue_values, capacity_values, horizons):
    ext = ctx.obj.model_type
    for Q in parse_int_list(queue_values):
        for N in parse_int_list(capacity_values):
            for H in parse_int_list(horizons):
                ctx.obj.run_case(
                    suite="parqueues",
                    case_id=f"q{Q}-n{N}-h{H}",
                    model_path=BENCHMARKS_ROOT / "parqueues" / f"queue-{Q}.{ext}",
                    property_name="target",
                    horizon=H,
                    constants={"N": N},
                    parameters={"Q": Q, "N": N, "H": H},
                )


def main():
    cli()


if __name__ == "__main__":
    main()
