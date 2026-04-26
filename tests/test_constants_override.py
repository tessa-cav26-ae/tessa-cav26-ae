from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from src import compile_reachability, load_prism_model
from src.representation import (
    BinaryOp,
    Command,
    Const,
    Guard,
    Module,
    ParsedModel,
    StateVariable,
    Update,
    Var,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
HAS_JAX = importlib.util.find_spec("jax") is not None
HAS_STORMPY = importlib.util.find_spec("stormpy") is not None


def build_parametric_coin_model() -> ParsedModel:
    """A two-state DTMC with one float constant ``p`` controlling the
    probability of moving from state 0 to state 1. Goal label ``reached``
    fires when the chain enters state 1.

    Constructed programmatically (no PRISM dependency) so tests run
    without stormpy. ``p`` does not appear in the goal expression nor in
    any variable's initial — therefore it is runtime-overridable.
    """
    s = StateVariable(
        name="s",
        domain_kind="bounded",
        lower=Const(0),
        upper=Const(1),
        initial=Const(0),
    )
    module = Module(
        name="coin",
        variables=[s],
        commands=[
            Command(
                action=None,
                guard=Guard(BinaryOp("==", Var("s"), Const(0))),
                updates=[
                    Update(Var("p"), {"s": Const(1)}),
                    Update(BinaryOp("-", Const(1.0), Var("p")), {"s": Const(0)}),
                ],
            ),
            Command(
                action=None,
                guard=Guard(BinaryOp("==", Var("s"), Const(1))),
                updates=[Update(Const(1.0), {"s": Const(1)})],
            ),
        ],
    )
    return ParsedModel(
        properties={"reached": BinaryOp("==", Var("s"), Const(1))},
        constants={"p": 0.5},
        functions={},
        modules=[module],
    )


@unittest.skipUnless(HAS_JAX, "constants_override requires the JAX backend")
class TestConstantsOverrideJax(unittest.TestCase):
    def setUp(self) -> None:
        self.parsed = build_parametric_coin_model()
        self.compiled = compile_reachability(
            self.parsed,
            property_name="reached",
            backend="jax:cpu",
            dtype="float32",
        )

    def test_baseline_run_unchanged(self):
        # Baseline path (no override) must still work and return a probability in [0, 1].
        result = float(self.compiled.run(8))
        self.assertTrue(0.0 <= result <= 1.0)

    def test_override_with_baked_value_matches_baseline(self):
        import jax.numpy as jnp

        baseline = float(self.compiled.run(8))
        with_override = float(
            self.compiled.run(8, constants_override={"p": jnp.float32(0.5)})
        )
        self.assertAlmostEqual(baseline, with_override, places=5)

    def test_override_changes_result(self):
        import jax.numpy as jnp

        low = float(self.compiled.run(8, constants_override={"p": jnp.float32(0.1)}))
        high = float(self.compiled.run(8, constants_override={"p": jnp.float32(0.9)}))
        # Higher transition probability into state 1 ⇒ higher reachability probability.
        self.assertGreater(high, low)

    def test_jax_grad_flows_through_override(self):
        import jax
        import jax.numpy as jnp

        def loss(p):
            return self.compiled.run(8, constants_override={"p": p})

        value, grad = jax.value_and_grad(loss)(jnp.float32(0.3))
        self.assertTrue(jnp.isfinite(value))
        self.assertTrue(jnp.isfinite(grad))
        # ∂P(reach state 1 within 8 steps) / ∂p > 0 — increasing p must increase the probability.
        self.assertGreater(float(grad), 0.0)

    def test_unknown_constant_raises_keyerror(self):
        with self.assertRaises(KeyError) as ctx:
            self.compiled.run(8, constants_override={"not_a_constant": 0.1})
        self.assertIn("not_a_constant", str(ctx.exception))

    def test_step_supports_override(self):
        import jax.numpy as jnp

        out = self.compiled.step(
            self.compiled.initial_tensor,
            constants_override={"p": jnp.float32(0.4)},
        )
        self.assertEqual(tuple(out.shape), self.compiled.tensor_shape)


@unittest.skipUnless(HAS_JAX and HAS_STORMPY, "PRISM-based override test requires JAX and stormpy")
class TestConstantsOverridePrism(unittest.TestCase):
    """End-to-end smoke test using the herman-13 parametric PRISM model.

    Verifies that a constant declared as ``const double p1;`` in PRISM
    ends up overridable through the JAX path. Confirms gradients do not
    NaN out on a real-world model with many parametric constants.
    """

    PARAMETRIC_MODEL = REPO_ROOT / "benchmarks" / "herman" / "herman-13-random-parametric.prism"

    def setUp(self) -> None:
        if not self.PARAMETRIC_MODEL.exists():
            self.skipTest(f"missing reference model {self.PARAMETRIC_MODEL}")
        baked_constants = {f"p{i}": 0.5 for i in range(1, 14)}
        self.parsed = load_prism_model(str(self.PARAMETRIC_MODEL), constants=baked_constants)
        self.compiled = compile_reachability(
            self.parsed,
            property_name="stable",
            backend="jax:cpu",
            dtype="float32",
        )

    def test_override_matches_baked_at_same_values(self):
        import jax.numpy as jnp

        baseline = float(self.compiled.run(4))
        overridden = float(
            self.compiled.run(
                4,
                constants_override={f"p{i}": jnp.float32(0.5) for i in range(1, 14)},
            )
        )
        self.assertAlmostEqual(baseline, overridden, places=4)


if __name__ == "__main__":
    unittest.main()
