from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from src.backend import build_backend, parse_backend
from src.compiler import compile_reachability
from src.parser import load_model

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples"
COMPLEX_MULTI_ACTION = EXAMPLES / "complex_multi_action.prism"
WEATHER_FACTORY_3 = EXAMPLES / "weather_factory_3.prism"
LEADERSYNC = EXAMPLES / "explicit" / "leadersync.prism"
EGL = EXAMPLES / "explicit" / "egl.prism"
BRP = EXAMPLES / "explicit" / "brp.prism"

HAS_STORMPY = importlib.util.find_spec("stormpy") is not None


@unittest.skipUnless(HAS_STORMPY, "stormpy required for --backend explicit")
class TestExplicitBackendPlumbing(unittest.TestCase):
    def test_backend_parser_accepts_explicit(self):
        spec = parse_backend("explicit")
        self.assertEqual(spec.kind, "explicit")
        self.assertEqual(spec.raw, "explicit")

    def test_backend_parser_accepts_storm_alias(self):
        spec = parse_backend("storm")
        self.assertEqual(spec.kind, "explicit")
        self.assertEqual(spec.raw, "explicit")

    def test_build_backend_explicit_runtime(self):
        runtime = build_backend("explicit", dtype="float64")
        self.assertEqual(runtime.spec.kind, "explicit")
        self.assertIsNone(runtime.jax_module)
        self.assertEqual(repr(runtime), "explicit")

    def test_requires_model_path(self):
        # compile_reachability called without model_path should error
        parsed = load_model("prism", COMPLEX_MULTI_ACTION)
        with self.assertRaises(ValueError) as cm:
            compile_reachability(
                parsed,
                property_name="goal",
                backend="explicit",
            )
        self.assertIn("model_path", str(cm.exception).lower())


@unittest.skipUnless(HAS_STORMPY, "stormpy required for --backend explicit")
class TestExplicitBackendParity(unittest.TestCase):
    """Explicit backend must match the dense backend (where dense fits) and
    Storm's published numbers (where dense OOMs)."""

    def _explicit(self, model_path: Path, prop: str, horizon: int, constants=None) -> float:
        parsed = load_model("prism", model_path, constants=constants)
        compiled = compile_reachability(
            parsed,
            property_name=prop,
            backend="explicit",
            model_path=model_path,
            constants=constants,
        )
        return float(compiled.backend.device_get(compiled.run(horizon)))

    def _dense(self, model_path: Path, prop: str, horizon: int, constants=None) -> float:
        parsed = load_model("prism", model_path, constants=constants)
        compiled = compile_reachability(
            parsed,
            property_name=prop,
            backend="numpy",
            dtype="float64",
        )
        return float(compiled.backend.device_get(compiled.run(horizon)))

    def test_dense_parity_complex_multi_action(self):
        for h in (0, 1, 5, 10):
            dense = self._dense(COMPLEX_MULTI_ACTION, "goal", h)
            explicit = self._explicit(COMPLEX_MULTI_ACTION, "goal", h)
            self.assertAlmostEqual(dense, explicit, places=6,
                msg=f"horizon {h}: dense={dense} explicit={explicit}")

    def test_dense_parity_weather_factory_3(self):
        for h in (0, 1, 5, 10):
            dense = self._dense(WEATHER_FACTORY_3, "allStrike", h)
            explicit = self._explicit(WEATHER_FACTORY_3, "allStrike", h)
            self.assertAlmostEqual(dense, explicit, places=6,
                msg=f"horizon {h}: dense={dense} explicit={explicit}")

    def test_storm_parity_leadersync_N3(self):
        # Storm: P=? [ F<=10 "elected"] on N=3 → 0.999755859375
        p = self._explicit(LEADERSYNC, "elected", 10, {"N": 3})
        self.assertAlmostEqual(p, 0.999755859375, places=10)

    def test_storm_parity_leadersync_N4(self):
        # Storm: P=? [ F<=10 "elected"] on N=4 → 0.9981536865234375
        p = self._explicit(LEADERSYNC, "elected", 10, {"N": 4})
        self.assertAlmostEqual(p, 0.9981536865234375, places=10)

    def test_storm_parity_egl_default(self):
        # Storm: default L=2 N=2 → H=20: 1.0; H=10: 0.0
        self.assertAlmostEqual(self._explicit(EGL, "target", 20), 1.0, places=10)
        self.assertAlmostEqual(self._explicit(EGL, "target", 10), 0.0, places=10)

    def test_storm_parity_brp(self):
        # brp's "target" label = (recv=true): the receiver has accepted at
        # least one frame. Storm: P=? [F<=5 "target"] = 0.9996 on any
        # (N, MAX) because the receive channel's success probability is
        # 0.99, and the 5-step horizon is enough to cover the first frame.
        p = self._explicit(BRP, "target", 5, {"N": 20, "MAX": 8})
        self.assertAlmostEqual(p, 0.9996, places=10)

    def test_horizon_sweep_leadersync(self):
        # Sweep a handful of horizons; monotone non-decreasing in H.
        last = 0.0
        for h in (0, 1, 5, 10, 20):
            p = self._explicit(LEADERSYNC, "elected", h, {"N": 3})
            self.assertGreaterEqual(p + 1e-9, last)
            last = p


@unittest.skipUnless(HAS_STORMPY, "stormpy required for --backend explicit")
class TestExplicitBackendCLI(unittest.TestCase):
    """Smoke test: CLI path accepts --backend explicit without --mode."""

    def test_cli_accepts_explicit_without_mode(self):
        from src.cli import main
        exit_code = main([
            str(LEADERSYNC),
            "--backend", "explicit",
            "--property", "elected",
            "--horizon", "10",
            "--const", "N=3",
        ])
        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
