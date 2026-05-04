from __future__ import annotations

import importlib.util
import json
import math
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import compile_reachability, load_model
from src.benchmarks import (
    BenchmarkContext,
    _read_phase_timings,
    parse_int_list,
)
from src.timing import (
    append_jsonl_record,
    build_timing_summary_records,
    execute_timed_runs,
    parse_rubicon_probability,
    parse_storm_probability,
    parse_tessa_probability,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
HAS_STORM = shutil.which("storm") is not None
HAS_STORMPY = importlib.util.find_spec("stormpy") is not None


class BenchmarkingHelpersTests(unittest.TestCase):
    def test_parse_tessa_probability_reads_last_scalar(self) -> None:
        self.assertAlmostEqual(
            parse_tessa_probability("numpy\narray(0.25, dtype=float32)\n"),
            0.25,
        )

    def test_parse_storm_probability_reads_reference_line(self) -> None:
        self.assertAlmostEqual(
            parse_storm_probability('Model checking property "1"\nResult (for initial states): 0.05142680456\n'),
            0.05142680456,
        )

    def test_parse_rubicon_probability_reads_trailing_scalar(self) -> None:
        # rubicon_runner echoes dice's JSON (which contains many scalars)
        # then prints a sentinel newline + the probability on its own line.
        # The parser must take the trailing scalar on the trailing line —
        # not, say, an earlier scalar from inside the JSON.
        stdout = (
            'Translating model.prism to /tmp/model.dice\n'
            '[{"Joint Distribution": [["Value", "Probability"], '
            '["(true, false)", "0.875"], ["(false, false)", "0.125"]]}]\n'
            '\n'
            '0.875\n'
        )
        self.assertAlmostEqual(parse_rubicon_probability(stdout), 0.875)

    def test_append_jsonl_record_writes_one_json_object_per_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "timings.jsonl"
            append_jsonl_record(log_path, {"phase": "measured", "elapsed_seconds": 1.25})
            append_jsonl_record(log_path, {"phase": "measured", "elapsed_seconds": 2.5})

            lines = log_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["elapsed_seconds"], 1.25)
        self.assertEqual(json.loads(lines[1])["elapsed_seconds"], 2.5)

    def test_execute_timed_runs_records_measured_runs(self) -> None:
        values = iter([1.0, 2.0])

        def run_once():
            return next(values)

        result, records = execute_timed_runs(
            run_once,
            num_timed_runs=2,
            record_result=float,
        )

        self.assertEqual(result, 2.0)
        self.assertEqual([record["phase"] for record in records], ["measured", "measured"])
        self.assertEqual([record["phase_iteration"] for record in records], [1, 2])
        self.assertEqual(records[-1]["probability"], 2.0)

    def test_build_timing_summary_records_reports_mean_and_std(self) -> None:
        records = [
            {"phase": "compile", "phase_iteration": 1, "elapsed_seconds": 1.0},
            {"phase": "warmup", "phase_iteration": 1, "elapsed_seconds": 2.0},
            {"phase": "compile", "phase_iteration": 2, "elapsed_seconds": 3.0},
            {"phase": "warmup", "phase_iteration": 2, "elapsed_seconds": 4.0},
            {"phase": "measured", "elapsed_seconds": 0.5},
            {"phase": "measured", "elapsed_seconds": 1.5},
        ]

        summaries = build_timing_summary_records(
            records,
            metadata={"tool": "tessa", "backend": "numpy"},
        )

        phases = [record["phase"] for record in summaries]
        self.assertEqual(phases, ["compile_summary", "warmup_summary", "work_summary", "measured_summary"])

        by_phase = {record["phase"]: record for record in summaries}
        self.assertAlmostEqual(by_phase["compile_summary"]["mean_elapsed_seconds"], 2.0)
        self.assertAlmostEqual(by_phase["warmup_summary"]["mean_elapsed_seconds"], 3.0)
        # work_i = compile_i + warmup_i → [3.0, 7.0], mean=5.0, stdev=sqrt(8)≈2.828
        self.assertAlmostEqual(by_phase["work_summary"]["mean_elapsed_seconds"], 5.0)
        self.assertAlmostEqual(by_phase["work_summary"]["std_elapsed_seconds"], math.sqrt(8), places=6)
        self.assertAlmostEqual(by_phase["measured_summary"]["mean_elapsed_seconds"], 1.0)
        self.assertEqual(by_phase["compile_summary"]["tool"], "tessa")

    def test_build_timing_summary_records_single_sample_has_zero_std(self) -> None:
        summaries = build_timing_summary_records(
            [
                {"phase": "compile", "phase_iteration": 1, "elapsed_seconds": 1.0},
                {"phase": "warmup", "phase_iteration": 1, "elapsed_seconds": 2.0},
                {"phase": "measured", "elapsed_seconds": 0.3},
            ],
        )
        for record in summaries:
            self.assertEqual(record["std_elapsed_seconds"], 0.0)
            self.assertEqual(record["sample_count"], 1)

    def test_read_phase_timings_exposes_mean_std_and_paired_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "time.jsonl"
            for record in [
                {"phase": "load", "elapsed_seconds": 0.3},
                {"phase": "compile", "phase_iteration": 1, "elapsed_seconds": 0.4},
                {"phase": "warmup", "phase_iteration": 1, "elapsed_seconds": 2.0},
                {"phase": "compile", "phase_iteration": 2, "elapsed_seconds": 0.6},
                {"phase": "warmup", "phase_iteration": 2, "elapsed_seconds": 2.4},
                {"phase": "measured", "elapsed_seconds": 0.1},
                {"phase": "measured", "elapsed_seconds": 0.3},
            ]:
                append_jsonl_record(log_path, record)

            timings = _read_phase_timings(log_path)

        self.assertAlmostEqual(timings["load_seconds"], 0.3)
        self.assertAlmostEqual(timings["compile_seconds"], 0.5)
        self.assertAlmostEqual(timings["warmup_avg_seconds"], 2.2)
        self.assertAlmostEqual(timings["measured_avg_seconds"], 0.2)
        # work_i = [0.4+2.0, 0.6+2.4] = [2.4, 3.0] → mean=2.7
        self.assertAlmostEqual(timings["work_seconds"], 2.7)
        self.assertGreater(timings["compile_std"], 0.0)
        self.assertGreater(timings["warmup_std"], 0.0)
        self.assertGreater(timings["work_std"], 0.0)
        self.assertGreater(timings["measured_std"], 0.0)

    def test_cold_phases_are_ignored_by_summary_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "time.jsonl"
            records = [
                {"phase": "load", "elapsed_seconds": 0.3},
                {"phase": "cold_compile", "phase_iteration": 1, "elapsed_seconds": 9.9},
                {"phase": "cold_warmup", "phase_iteration": 1, "elapsed_seconds": 9.9},
                {"phase": "cold_compile", "phase_iteration": 2, "elapsed_seconds": 9.9},
                {"phase": "cold_warmup", "phase_iteration": 2, "elapsed_seconds": 9.9},
                {"phase": "compile", "phase_iteration": 1, "elapsed_seconds": 0.4},
                {"phase": "warmup", "phase_iteration": 1, "elapsed_seconds": 2.0},
                {"phase": "compile", "phase_iteration": 2, "elapsed_seconds": 0.6},
                {"phase": "warmup", "phase_iteration": 2, "elapsed_seconds": 2.4},
                {"phase": "measured", "elapsed_seconds": 0.2},
            ]
            for record in records:
                append_jsonl_record(log_path, record)

            timings = _read_phase_timings(log_path)

        self.assertAlmostEqual(timings["compile_seconds"], 0.5)
        self.assertAlmostEqual(timings["warmup_avg_seconds"], 2.2)
        self.assertAlmostEqual(timings["work_seconds"], 2.7)

        summaries = build_timing_summary_records(records)
        phases = {record["phase"] for record in summaries}
        self.assertNotIn("cold_compile_summary", phases)
        self.assertNotIn("cold_warmup_summary", phases)
        self.assertEqual(
            phases,
            {"compile_summary", "warmup_summary", "work_summary", "measured_summary"},
        )


class BenchmarkRunnerTests(unittest.TestCase):
    def test_parse_int_list(self) -> None:
        self.assertEqual(parse_int_list("1,2,3"), [1, 2, 3])
        self.assertEqual(parse_int_list("17"), [17])

    def test_tool_label_storm(self) -> None:
        ctx = BenchmarkContext(
            tool="storm", backend=None, timeout=600, num_work_runs=1,
            num_timed_runs=1, dtype="float32",
            output_dir=Path("/tmp"), storm_cmd="storm", tessa_cmd="tessa",
        )
        self.assertEqual(ctx.tool_label, "storm")

    def test_tool_label_tessa_numpy(self) -> None:
        ctx = BenchmarkContext(
            tool="tessa", backend="numpy", timeout=600, num_work_runs=1,
            num_timed_runs=1, dtype="float32",
            output_dir=Path("/tmp"), storm_cmd="storm", tessa_cmd="tessa",
        )
        self.assertEqual(ctx.tool_label, "tessa")

    def test_tool_label_tessa_jax_cuda(self) -> None:
        ctx = BenchmarkContext(
            tool="tessa", backend="jax:cuda:0", timeout=600, num_work_runs=1,
            num_timed_runs=1, dtype="float32",
            output_dir=Path("/tmp"), storm_cmd="storm", tessa_cmd="tessa",
        )
        self.assertEqual(ctx.tool_label, "tessa")

    def test_build_storm_command_includes_constants(self) -> None:
        ctx = BenchmarkContext(
            tool="storm", backend=None, timeout=600, num_work_runs=1,
            num_timed_runs=1, dtype="float32",
            output_dir=Path("/tmp"), storm_cmd="storm", tessa_cmd="tessa",
        )
        command = ctx._build_storm_command(
            REPO_ROOT / "benchmarks" / "parqueues" / "queue-3.prism",
            "target", 10, {"N": 2},
        )
        self.assertEqual(command[0], "storm")
        self.assertIn("--constants", command)
        self.assertIn("N=2", command)

    def test_tool_label_rubicon(self) -> None:
        ctx = BenchmarkContext(
            tool="rubicon", backend=None, timeout=600, num_work_runs=1,
            num_timed_runs=1, dtype="float32",
            output_dir=Path("/tmp"), storm_cmd="storm", tessa_cmd="tessa",
            rubicon_cmd_argv=["python", "-m", "src.rubicon_runner"],
        )
        self.assertEqual(ctx.tool_label, "rubicon")

    def test_build_rubicon_command_includes_property_horizon_and_constants(self) -> None:
        ctx = BenchmarkContext(
            tool="rubicon", backend=None, timeout=600, num_work_runs=1,
            num_timed_runs=1, dtype="float32",
            output_dir=Path("/tmp"), storm_cmd="storm", tessa_cmd="tessa",
            rubicon_cmd_argv=["python", "-m", "src.rubicon_runner"],
            dice_cmd="dice",
        )
        command = ctx._build_rubicon_command(
            REPO_ROOT / "benchmarks" / "parqueues" / "queue-3.prism",
            "target", 10, {"N": 2},
        )
        self.assertEqual(command[:3], ["python", "-m", "src.rubicon_runner"])
        self.assertIn("--prism", command)
        self.assertIn("--property", command)
        self.assertIn("target", command)
        self.assertIn("--horizon", command)
        self.assertIn("10", command)
        self.assertIn("--const", command)
        self.assertIn("N=2", command)
        self.assertIn("--dice-cmd", command)
        self.assertIn("dice", command)

    def test_build_rubicon_command_threads_dice_extra_args(self) -> None:
        ctx = BenchmarkContext(
            tool="rubicon", backend=None, timeout=600, num_work_runs=1,
            num_timed_runs=1, dtype="float32",
            output_dir=Path("/tmp"), storm_cmd="storm", tessa_cmd="tessa",
            rubicon_cmd_argv=["python", "-m", "src.rubicon_runner"],
            dice_cmd="dice",
            dice_extra_args=["-show-size", "-num-recursive-calls"],
        )
        command = ctx._build_rubicon_command(
            REPO_ROOT / "benchmarks" / "herman" / "herman-3.prism",
            "stable", 100, {},
        )
        # Each token must be paired with a fresh --dice-extra-arg flag so the
        # inner runner doesn't have to re-shlex the list.
        self.assertEqual(command.count("--dice-extra-arg"), 2)
        flag_idx = [i for i, tok in enumerate(command) if tok == "--dice-extra-arg"]
        self.assertEqual(command[flag_idx[0] + 1], "-show-size")
        self.assertEqual(command[flag_idx[1] + 1], "-num-recursive-calls")

    def test_build_tessa_command_includes_benchmark_flags(self) -> None:
        ctx = BenchmarkContext(
            tool="tessa", backend="numpy", timeout=600, num_work_runs=3,
            num_timed_runs=3, dtype="float64",
            output_dir=Path("/tmp"), storm_cmd="storm", tessa_cmd="tessa",
        )
        command = ctx._build_tessa_command(
            REPO_ROOT / "benchmarks" / "meeting" / "meeting-3.prism",
            "goal", 10, {},
            Path("/tmp/time.jsonl"),
        )
        self.assertIn("--num-work-runs", command)
        self.assertIn("--num-timed-runs", command)
        self.assertIn("--time-log", command)
        self.assertIn("float64", command)
        self.assertNotIn("--num-warmup", command)

    def test_run_case_writes_csv_and_parses_probability(self) -> None:
        class FakeCompleted:
            def __init__(self):
                self.returncode = 0
                self.stdout = "numpy\narray(0.25, dtype=float32)\n"
                self.stderr = ""

        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmpdir:
            output_dir = Path(tmpdir)
            ctx = BenchmarkContext(
                tool="tessa", backend="numpy", timeout=600, num_work_runs=1,
                num_timed_runs=1, dtype="float32",
                output_dir=output_dir, storm_cmd="storm", tessa_cmd="tessa",
            )
            with patch("src.benchmarks.subprocess.run", return_value=FakeCompleted()):
                result = ctx.run_case(
                    suite="meeting",
                    case_id="n3-h10",
                    model_path=REPO_ROOT / "benchmarks" / "meeting" / "meeting-3.prism",
                    property_name="goal",
                    horizon=10,
                    constants={},
                    parameters={"N": 3, "H": 10},
                )

            self.assertEqual(result["status"], "ok")
            self.assertAlmostEqual(result["probability"], 0.25)
            self.assertTrue(ctx.csv_path.exists())


@unittest.skipUnless(HAS_STORM and HAS_STORMPY, "storm and stormpy are required for live benchmark comparison tests")
class BenchmarkReferenceSmokeTests(unittest.TestCase):
    def compare_case_against_storm(
        self,
        model_path: Path,
        property_name: str,
        horizon: int,
        *,
        constants: dict[str, int | float | bool] | None = None,
    ) -> None:
        constants = constants or {}
        parsed_model = load_model("prism", model_path, constants=constants or None)
        compiled = compile_reachability(
            parsed_model,
            property_name=property_name,
            backend="numpy",
            dtype="float64",
        )
        tessa_probability = float(compiled.run(horizon).item())

        storm_command = [
            "storm",
            "--prism", str(model_path),
            "--prop", f'P=? [ F<={horizon} "{property_name}" ]',
        ]
        if constants:
            storm_command.extend(
                ["--constants", ",".join(f"{name}={value}" for name, value in constants.items())]
            )
        completed = subprocess.run(storm_command, capture_output=True, text=True, check=True)
        storm_probability = parse_storm_probability(completed.stdout)

        self.assertTrue(
            math.isclose(storm_probability, tessa_probability, abs_tol=1e-8, rel_tol=1e-6),
            msg=f"storm={storm_probability}, tessa={tessa_probability}",
        )

    def test_herman_matches_storm_on_small_case(self) -> None:
        self.compare_case_against_storm(
            REPO_ROOT / "benchmarks" / "herman" / "herman-3.prism",
            "stable", 3,
        )

    def test_meeting_matches_storm_on_small_case(self) -> None:
        self.compare_case_against_storm(
            REPO_ROOT / "benchmarks" / "meeting" / "meeting-3.prism",
            "goal", 3,
        )

    def test_weather_factory_matches_storm_on_small_case(self) -> None:
        self.compare_case_against_storm(
            REPO_ROOT / "benchmarks" / "weather_factory" / "weatherfactory3.prism",
            "allStrike", 3,
        )

    def test_parqueues_matches_storm_on_small_case(self) -> None:
        self.compare_case_against_storm(
            REPO_ROOT / "benchmarks" / "parqueues" / "queue-3.prism",
            "target", 3,
            constants={"N": 1},
        )


if __name__ == "__main__":
    unittest.main()
