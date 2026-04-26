"""Test that weather factory bounded reachability scales linearly with horizon H."""
from __future__ import annotations

import time
import unittest

from src.reachability import compile_reachability
from src.representation import (
    BinaryOp,
    Command,
    Const,
    Guard,
    IfThenElse,
    Module,
    NaryOp,
    ParsedModel,
    StateVariable,
    Update,
    Var,
)


def build_weather_factory_3():
    """Build weather_factory_3 model matching the PRISM benchmark.

    Weather module: sun transitions with P(sunny|sunny)=0.7, P(sunny|cloudy)=0.6
    Factory i: binary state, transitions depend on (sun, state_i) and params (p_i, q_i)
    Goal: all factories in state 1 ("allStrike")
    """
    P = [0.1, 0.2, 0.41]
    Q = [0.2, 0.3, 0.45]

    sun_var = StateVariable(name="sun", domain_kind="bounded", lower=Const(0), upper=Const(1), initial=Const(1))
    weather_mod = Module(
        name="weather",
        variables=[sun_var],
        commands=[
            Command(
                action="a",
                guard=Guard(BinaryOp("==", Var("sun"), Const(0))),
                updates=[Update(Const(0.4), {"sun": Const(0)}), Update(Const(0.6), {"sun": Const(1)})],
            ),
            Command(
                action="a",
                guard=Guard(BinaryOp("==", Var("sun"), Const(1))),
                updates=[Update(Const(0.3), {"sun": Const(0)}), Update(Const(0.7), {"sun": Const(1)})],
            ),
        ],
    )

    factory_modules = []
    for i in range(3):
        vname = f"state_{i}"
        v = StateVariable(name=vname, domain_kind="bounded", lower=Const(0), upper=Const(1), initial=Const(0))
        prob_become_1 = IfThenElse(BinaryOp("==", Var("sun"), Const(1)), Const(0.7 * Q[i]), Const(0.4 * Q[i]))
        prob_stay_1 = IfThenElse(BinaryOp("==", Var("sun"), Const(1)), Const(0.3 * P[i]), Const(0.6 * P[i]))
        factory_modules.append(
            Module(
                name=f"factory_{i}",
                variables=[v],
                commands=[
                    Command(
                        action="a",
                        guard=Guard(BinaryOp("==", Var(vname), Const(0))),
                        updates=[
                            Update(prob_become_1, {vname: Const(1)}),
                            Update(BinaryOp("-", Const(1), prob_become_1), {vname: Const(0)}),
                        ],
                    ),
                    Command(
                        action="a",
                        guard=Guard(BinaryOp("==", Var(vname), Const(1))),
                        updates=[
                            Update(prob_stay_1, {vname: Const(1)}),
                            Update(BinaryOp("-", Const(1), prob_stay_1), {vname: Const(0)}),
                        ],
                    ),
                ],
            )
        )

    goal = NaryOp(
        "\u2227",
        [BinaryOp("==", Var("state_0"), Const(1)), BinaryOp("==", Var("state_1"), Const(1)), BinaryOp("==", Var("state_2"), Const(1))],
    )
    return ParsedModel(properties={"allStrike": goal}, constants={}, functions={}, modules=[weather_mod] + factory_modules)


class WeatherFactoryCorrectnessTests(unittest.TestCase):
    def test_matches_storm_reference_h10(self):
        """P(allStrike within 10 steps) should match Storm's reference value."""
        compiled = compile_reachability(build_weather_factory_3(), property_name="allStrike", backend="numpy")
        result = float(compiled.run(10))
        self.assertAlmostEqual(result, 0.05142680456, places=5)

    def test_probability_monotonically_increases(self):
        compiled = compile_reachability(build_weather_factory_3(), property_name="allStrike", backend="numpy")
        prev = 0.0
        for h in range(0, 51):
            r = float(compiled.run(h))
            self.assertGreaterEqual(r, prev - 1e-9, f"monotonicity violated at h={h}")
            prev = r


class WeatherFactoryScalingTests(unittest.TestCase):
    def test_runtime_scales_linearly_with_horizon(self):
        """After compilation, runtime should scale O(H) since each step is O(1)."""
        compiled = compile_reachability(build_weather_factory_3(), property_name="allStrike", backend="numpy")

        # Warmup
        compiled.run(10)

        h_small = 500
        h_large = 2000

        t0 = time.perf_counter()
        compiled.run(h_small)
        t_small = time.perf_counter() - t0

        t0 = time.perf_counter()
        compiled.run(h_large)
        t_large = time.perf_counter() - t0

        h_ratio = h_large / h_small
        t_ratio = t_large / t_small

        # Allow generous tolerance: time ratio should be within 0.5x-3x of horizon ratio
        self.assertGreater(t_ratio, h_ratio * 0.3, f"Scaling too fast: t_ratio={t_ratio:.2f}, h_ratio={h_ratio:.1f}")
        self.assertLess(t_ratio, h_ratio * 3.0, f"Scaling too slow: t_ratio={t_ratio:.2f}, h_ratio={h_ratio:.1f}")


if __name__ == "__main__":
    unittest.main()
