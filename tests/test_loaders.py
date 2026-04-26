from __future__ import annotations

import contextlib
import importlib.util
import io
import tomllib
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import src as tessa
from src import load_model
from src.cli import main as cli_main
from src.parser import parse_prism_expression
from src.reachability import parse_backend, parse_dtype
from src.representation import RewardVariable

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
JANI_MODEL = WORKSPACE_ROOT / "mdp" / "benchmarks" / "resource.jani"
PRISM_MODEL = WORKSPACE_ROOT / "mdp" / "benchmarks" / "resource-gathering.v2.pm"
WEATHER_FACTORY_MODEL = REPO_ROOT / "examples" / "weather_factory_3.prism"
HAS_STORMPY = importlib.util.find_spec("stormpy") is not None
HAS_JAX = importlib.util.find_spec("jax") is not None


class PrintableBackend:
    def __init__(self, label: str) -> None:
        self._label = label

    def __repr__(self) -> str:
        return self._label

    def device_get(self, value):
        return value


class PrintableArray:
    def __init__(self, representation: str) -> None:
        self._representation = representation

    def __repr__(self) -> str:
        return self._representation

    def item(self) -> float:
        if "0.125" in self._representation:
            return 0.125
        if "0.25" in self._representation:
            return 0.25
        raise ValueError(f"Unsupported test representation: {self._representation}")


class LoaderApiTests(unittest.TestCase):
    def test_source_root_reexports_load_model(self) -> None:
        self.assertTrue(callable(tessa.load_model))

    def test_load_model_rejects_unknown_model_type(self) -> None:
        with self.assertRaises(ValueError):
            load_model("unsupported", JANI_MODEL)

    def test_parse_prism_expression_rejects_unsupported_operator(self) -> None:
        with self.assertRaises(NotImplementedError):
            parse_prism_expression("a => b")

    def test_parse_prism_expression_accepts_nested_ternary(self) -> None:
        expression = parse_prism_expression("max(0, ((x = 1) ? 2 : 3))")
        self.assertEqual(
            repr(expression),
            "BinaryOp('max', 0, ite(BinaryOp('==', x, 1), 2, 3))",
        )

    def test_pyproject_exposes_console_script(self) -> None:
        with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
            pyproject = tomllib.load(handle)
        self.assertEqual(pyproject["project"]["scripts"]["tessa"], "tessa.cli:main")
        self.assertEqual(pyproject["project"]["scripts"]["tessa-benchmark"], "tessa.benchmarks:main")

    def test_parse_backend_defaults_to_jax_cpu(self) -> None:
        self.assertEqual(parse_backend().raw, "jax:cpu")

    def test_parse_backend_accepts_numpy_and_jax_variants(self) -> None:
        self.assertEqual(parse_backend("numpy").raw, "numpy")
        self.assertEqual(parse_backend("jax:cpu").raw, "jax:cpu")
        self.assertEqual(parse_backend("jax:cuda:1").raw, "jax:cuda:1")

    def test_parse_backend_rejects_invalid_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid backend"):
            parse_backend("jax")

    def test_parse_dtype_defaults_to_float32(self) -> None:
        self.assertEqual(parse_dtype(), "float32")

    def test_parse_dtype_accepts_supported_values(self) -> None:
        self.assertEqual(parse_dtype("float32"), "float32")
        self.assertEqual(parse_dtype("float64"), "float64")

    def test_parse_dtype_rejects_invalid_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid dtype"):
            parse_dtype("float16")

    def test_infer_model_type_accepts_nm_extension(self) -> None:
        from src.cli import infer_model_type

        self.assertEqual(infer_model_type("queue-8.nm"), "prism")


class JaniLoaderTests(unittest.TestCase):
    def test_load_jani_model_returns_expected_tuple_shape(self) -> None:
        prop_table, const_env, func_table, modules = load_model("jani", JANI_MODEL)

        self.assertIsInstance(prop_table, dict)
        self.assertIsInstance(const_env, dict)
        self.assertIsInstance(func_table, dict)
        self.assertIsInstance(modules, list)
        self.assertIn("__globals__", [module.name for module in modules])
        self.assertIn("robot", [module.name for module in modules])
        self.assertIn("above_of_home", func_table)
        self.assertTrue(any(isinstance(var, RewardVariable) for var in modules[0].variables))


class CliTests(unittest.TestCase):
    def run_cli(self, argv: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            exit_code = cli_main(argv)
        return exit_code, stdout.getvalue(), stderr.getvalue()

    def test_cli_infers_jani_from_extension(self) -> None:
        fake_model = ({}, {}, {}, [])
        with patch("src.cli.load_model", return_value=fake_model) as load_model_mock:
            exit_code, stdout, stderr = self.run_cli([str(JANI_MODEL)])
        self.assertEqual(exit_code, 0)
        self.assertIn("'modules': []", stdout)
        self.assertEqual(stderr, "")
        load_model_mock.assert_called_once_with("jani", Path(JANI_MODEL), constants=None)

    def test_cli_infers_prism_from_prism_extension(self) -> None:
        fake_model = ({}, {}, {}, [])
        with patch("src.cli.load_model", return_value=fake_model) as load_model_mock:
            exit_code, _, _ = self.run_cli([str(WEATHER_FACTORY_MODEL)])
        self.assertEqual(exit_code, 0)
        load_model_mock.assert_called_once_with("prism", Path(WEATHER_FACTORY_MODEL), constants=None)

    def test_cli_infers_prism_from_pm_extension(self) -> None:
        fake_model = ({}, {}, {}, [])
        with patch("src.cli.load_model", return_value=fake_model) as load_model_mock:
            exit_code, _, _ = self.run_cli([str(PRISM_MODEL)])
        self.assertEqual(exit_code, 0)
        load_model_mock.assert_called_once_with("prism", Path(PRISM_MODEL), constants=None)

    def test_cli_rejects_unknown_extension_without_type(self) -> None:
        exit_code, _, stderr = self.run_cli(["model.unknown"])
        self.assertEqual(exit_code, 1)
        self.assertIn("Could not infer model type", stderr)

    def test_cli_accepts_type_override(self) -> None:
        fake_model = ({}, {}, {}, [])
        with patch("src.cli.load_model", return_value=fake_model) as load_model_mock:
            exit_code, _, _ = self.run_cli(["model.unknown", "--type", "jani"])
        self.assertEqual(exit_code, 0)
        load_model_mock.assert_called_once_with("jani", Path("model.unknown"), constants=None)

    def test_cli_parses_repeated_constants(self) -> None:
        fake_model = ({}, {}, {}, [])
        with patch("src.cli.load_model", return_value=fake_model) as load_model_mock:
            exit_code, _, _ = self.run_cli(
                [
                    str(PRISM_MODEL),
                    "--const",
                    "A=1",
                    "--const",
                    "B=true",
                    "--const",
                    "C=1.5e-2",
                ]
            )
        self.assertEqual(exit_code, 0)
        load_model_mock.assert_called_once_with(
            "prism",
            Path(PRISM_MODEL),
            constants={"A": 1, "B": True, "C": 1.5e-2},
        )

    def test_cli_rejects_malformed_constant(self) -> None:
        exit_code, _, stderr = self.run_cli([str(PRISM_MODEL), "--const", "BAD"])
        self.assertEqual(exit_code, 1)
        self.assertIn("Invalid --const value", stderr)

    def test_cli_requires_property_and_horizon_together(self) -> None:
        fake_model = ({}, {}, {}, [])
        with patch("src.cli.load_model", return_value=fake_model):
            exit_code, _, stderr = self.run_cli([str(PRISM_MODEL), "--property", "goal"])
        self.assertEqual(exit_code, 1)
        self.assertIn("--property and --horizon must be provided together", stderr)

    def test_cli_rejects_backend_outside_reachability_mode(self) -> None:
        fake_model = ({}, {}, {}, [])
        with patch("src.cli.load_model", return_value=fake_model):
            exit_code, _, stderr = self.run_cli([str(PRISM_MODEL), "--backend", "numpy"])
        self.assertEqual(exit_code, 1)
        self.assertIn("--backend, --dtype, --num-work-runs, --num-timed-runs, and --time-log", stderr)

    def test_cli_rejects_dtype_outside_reachability_mode(self) -> None:
        fake_model = ({}, {}, {}, [])
        with patch("src.cli.load_model", return_value=fake_model):
            exit_code, _, stderr = self.run_cli([str(PRISM_MODEL), "--dtype", "float64"])
        self.assertEqual(exit_code, 1)
        self.assertIn("--backend, --dtype, --num-work-runs, --num-timed-runs, and --time-log", stderr)

    def test_cli_rejects_num_work_runs_outside_reachability_mode(self) -> None:
        fake_model = ({}, {}, {}, [])
        with patch("src.cli.load_model", return_value=fake_model):
            exit_code, _, stderr = self.run_cli([str(PRISM_MODEL), "--num-work-runs", "3"])
        self.assertEqual(exit_code, 1)
        self.assertIn("--num-work-runs", stderr)

    def test_cli_accepts_implicit_defaults_outside_reachability_mode(self) -> None:
        fake_model = ({}, {}, {}, [])
        with patch("src.cli.load_model", return_value=fake_model) as load_model_mock:
            exit_code, _, stderr = self.run_cli([str(PRISM_MODEL)])
        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr, "")
        load_model_mock.assert_called_once_with("prism", Path(PRISM_MODEL), constants=None)

    def test_cli_rejects_num_timed_runs_outside_reachability_mode(self) -> None:
        fake_model = ({}, {}, {}, [])
        with patch("src.cli.load_model", return_value=fake_model):
            exit_code, _, stderr = self.run_cli([str(PRISM_MODEL), "--num-timed-runs", "2"])
        self.assertEqual(exit_code, 1)
        self.assertIn("--num-timed-runs", stderr)

    def test_cli_rejects_time_log_outside_reachability_mode(self) -> None:
        fake_model = ({}, {}, {}, [])
        with patch("src.cli.load_model", return_value=fake_model):
            exit_code, _, stderr = self.run_cli([str(PRISM_MODEL), "--time-log", "timings.jsonl"])
        self.assertEqual(exit_code, 1)
        self.assertIn("--time-log", stderr)

    def test_cli_rejects_invalid_backend(self) -> None:
        fake_model = ({}, {}, {}, [])
        with patch("src.cli.load_model", return_value=fake_model):
            exit_code, _, stderr = self.run_cli(
                [str(PRISM_MODEL), "--property", "success", "--horizon", "10", "--backend", "jax"]
            )
        self.assertEqual(exit_code, 1)
        self.assertIn("Invalid backend", stderr)

    def test_cli_rejects_invalid_dtype(self) -> None:
        fake_model = ({}, {}, {}, [])
        with patch("src.cli.load_model", return_value=fake_model):
            exit_code, _, stderr = self.run_cli(
                [str(PRISM_MODEL), "--property", "success", "--horizon", "10", "--dtype", "float16"]
            )
        self.assertEqual(exit_code, 1)
        self.assertIn("Invalid dtype", stderr)

    def test_cli_prints_numeric_result_for_reachability_queries(self) -> None:
        fake_model = ({}, {}, {}, [])
        compiled_model = SimpleNamespace(
            run=lambda horizon: PrintableArray("Array(0.125, dtype=float32)"),
            backend=PrintableBackend("cpu"),
        )
        with patch("src.cli.load_model", return_value=fake_model) as load_model_mock:
            with patch("src.cli.compile_reachability", return_value=compiled_model) as compile_mock:
                exit_code, stdout, stderr = self.run_cli(
                    [str(PRISM_MODEL), "--property", "success", "--horizon", "10"]
                )
        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout, "cpu\nArray(0.125, dtype=float32)\n")
        self.assertEqual(stderr, "")
        load_model_mock.assert_called_once_with("prism", Path(PRISM_MODEL), constants=None)
        compile_mock.assert_called_once_with(fake_model, property_name="success", backend="jax:cpu", dtype="float32")

    def test_cli_forwards_explicit_backend_for_reachability_queries(self) -> None:
        fake_model = ({}, {}, {}, [])
        compiled_model = SimpleNamespace(
            run=lambda horizon: PrintableArray("array(0.25)"),
            backend=PrintableBackend("numpy"),
        )
        with patch("src.cli.load_model", return_value=fake_model):
            with patch("src.cli.compile_reachability", return_value=compiled_model) as compile_mock:
                exit_code, stdout, stderr = self.run_cli(
                    [
                        str(PRISM_MODEL),
                        "--property",
                        "success",
                        "--horizon",
                        "10",
                        "--backend",
                        "numpy",
                        "--dtype",
                        "float64",
                    ]
                )
        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout, "numpy\narray(0.25)\n")
        self.assertEqual(stderr, "")
        compile_mock.assert_called_once_with(fake_model, property_name="success", backend="numpy", dtype="float64")

    def test_cli_records_timing_options_for_reachability_queries(self) -> None:
        fake_model = ({}, {}, {}, [])
        compiled_model = SimpleNamespace(
            run=lambda horizon: PrintableArray("array(0.25)"),
            backend=PrintableBackend("numpy"),
        )
        with patch("src.cli.load_model", return_value=fake_model):
            with patch("src.cli.compile_reachability", return_value=compiled_model):
                with patch("src.cli.append_jsonl_record") as cli_append_log_mock:
                    with patch("src.timing.append_jsonl_record") as benchmarking_append_log_mock:
                        exit_code, stdout, stderr = self.run_cli(
                            [
                                str(PRISM_MODEL),
                                "--property",
                                "success",
                                "--horizon",
                                "10",
                                "--backend",
                                "numpy",
                                "--num-work-runs",
                                "1",
                                "--num-timed-runs",
                                "2",
                                "--time-log",
                                "timings.jsonl",
                            ]
                        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout, "numpy\narray(0.25)\n")
        self.assertEqual(stderr, "")
        # Per-iteration records from src.cli: load + compile + warmup + 4 summaries.
        # Measured records are appended by src.timing.
        cli_records = [call.args[1] for call in cli_append_log_mock.call_args_list]
        self.assertEqual(
            [record["phase"] for record in cli_records],
            ["load", "compile", "warmup", "compile_summary", "warmup_summary", "work_summary", "measured_summary"],
        )
        # All compile/warmup summaries have sample_count == --num-work-runs;
        # measured_summary has sample_count == --num-timed-runs.
        for summary_phase in ("compile_summary", "warmup_summary", "work_summary"):
            by_phase = next(r for r in cli_records if r["phase"] == summary_phase)
            self.assertEqual(by_phase["sample_count"], 1)
        measured_summary = next(r for r in cli_records if r["phase"] == "measured_summary")
        self.assertEqual(measured_summary["sample_count"], 2)
        # Measured per-iteration records go through src.timing.
        self.assertEqual(benchmarking_append_log_mock.call_count, 2)

    def test_cli_pretty_prints_structured_output(self) -> None:
        fake_model = (
            {"label": "expr"},
            {"C": 1},
            {"f": "body"},
            [
                type("M", (), {"name": "demo", "variables": [], "commands": []})(),
            ],
        )
        with patch("src.cli.load_model", return_value=fake_model):
            exit_code, stdout, _ = self.run_cli([str(JANI_MODEL)])
        self.assertEqual(exit_code, 0)
        self.assertIn("'properties'", stdout)
        self.assertIn("'constants'", stdout)
        self.assertIn("'functions'", stdout)
        self.assertIn("'modules'", stdout)
        self.assertIn("'name': 'demo'", stdout)

    @unittest.skipUnless(HAS_STORMPY, "stormpy is required for PRISM CLI smoke test")
    def test_cli_smoke_test_for_weather_factory_example(self) -> None:
        exit_code, stdout, stderr = self.run_cli([str(WEATHER_FACTORY_MODEL)])
        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr, "")
        self.assertIn("weathermodule", stdout)
        self.assertIn("factory1", stdout)
        self.assertIn("allStrike", stdout)

    @unittest.skipUnless(HAS_JAX and HAS_STORMPY, "jax and stormpy are required for PRISM reachability CLI smoke test")
    def test_cli_smoke_test_for_weather_factory_reachability(self) -> None:
        exit_code, stdout, stderr = self.run_cli(
            [str(WEATHER_FACTORY_MODEL), "--property", "allStrike", "--horizon", "10"]
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr, "")
        lines = stdout.strip().splitlines()
        self.assertEqual(lines[0], "cpu")
        self.assertEqual(len(lines), 2)
        self.assertIn("Array(", lines[1])
        self.assertIn("dtype=float32", lines[1])

    @unittest.skipUnless(HAS_STORMPY, "stormpy is required for PRISM reachability CLI smoke test")
    def test_cli_smoke_test_for_weather_factory_reachability_numpy(self) -> None:
        exit_code, stdout, stderr = self.run_cli(
            [str(WEATHER_FACTORY_MODEL), "--property", "allStrike", "--horizon", "10", "--backend", "numpy"]
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr, "")
        lines = stdout.strip().splitlines()
        self.assertEqual(lines[0], "numpy")
        self.assertEqual(len(lines), 2)
        self.assertIn("array(", lines[1])
        self.assertIn("dtype=float32", lines[1])

    @unittest.skipUnless(HAS_JAX and HAS_STORMPY, "jax and stormpy are required for PRISM reachability CLI smoke test")
    def test_cli_smoke_test_for_weather_factory_reachability_float64(self) -> None:
        exit_code, stdout, stderr = self.run_cli(
            [str(WEATHER_FACTORY_MODEL), "--property", "allStrike", "--horizon", "10", "--dtype", "float64"]
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr, "")
        lines = stdout.strip().splitlines()
        self.assertEqual(lines[0], "cpu")
        self.assertEqual(len(lines), 2)
        self.assertIn("Array(", lines[1])
        self.assertIn("dtype=float64", lines[1])

    @unittest.skipUnless(HAS_STORMPY, "stormpy is required for PRISM reachability CLI smoke test")
    def test_cli_smoke_test_for_weather_factory_reachability_numpy_float64(self) -> None:
        exit_code, stdout, stderr = self.run_cli(
            [
                str(WEATHER_FACTORY_MODEL),
                "--property",
                "allStrike",
                "--horizon",
                "10",
                "--backend",
                "numpy",
                "--dtype",
                "float64",
            ]
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr, "")
        lines = stdout.strip().splitlines()
        self.assertEqual(lines[0], "numpy")
        self.assertEqual(len(lines), 2)
        self.assertIn("array(", lines[1])


@unittest.skipUnless(HAS_STORMPY, "stormpy is required for PRISM loader tests")
class PrismLoaderTests(unittest.TestCase):
    def test_load_prism_model_returns_expected_structure(self) -> None:
        prop_table, const_env, func_table, modules = load_model(
            "prism",
            PRISM_MODEL,
            constants={"B": 1, "GOLD_TO_COLLECT": 2, "GEM_TO_COLLECT": 3},
        )

        self.assertIn("success", prop_table)
        self.assertIn("left_of_enemy", func_table)
        self.assertEqual(const_env["B"], 1)
        self.assertEqual(const_env["GOLD_TO_COLLECT"], 2)
        self.assertEqual(const_env["GEM_TO_COLLECT"], 3)
        self.assertIn("robot", [module.name for module in modules])
        self.assertIn("goldcounter", [module.name for module in modules])

        reward_names = {
            variable.name
            for variable in modules[0].variables
            if isinstance(variable, RewardVariable)
        }
        self.assertEqual(reward_names, {"attacks", "rew_gold", "rew_gem"})

        robot_module = next(module for module in modules if module.name == "robot")
        self.assertTrue(
            any(
                "attacks" in update.assignments and "rew_gold" in update.assignments
                for command in robot_module.commands
                for update in command.updates
            )
        )

    def test_load_prism_model_requires_missing_constants(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing values for PRISM constants"):
            load_model("prism", PRISM_MODEL, constants={"GOLD_TO_COLLECT": 2})


@unittest.skipIf(HAS_STORMPY, "missing-stormpy behavior is only relevant without stormpy")
class PrismLoaderWithoutStormpyTests(unittest.TestCase):
    def test_load_prism_model_explains_missing_dependency(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "stormpy"):
            load_model(
                "prism",
                PRISM_MODEL,
                constants={"GOLD_TO_COLLECT": 2, "GEM_TO_COLLECT": 3},
            )
