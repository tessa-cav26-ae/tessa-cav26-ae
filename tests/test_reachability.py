from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import src as tessa
from src.parser import load_model
from src.reachability import BackendRuntime, build_backend, compile_reachability, jit, parse_backend
from src.representation import BinaryOp, Command, Const, Guard, Module, ParsedModel, RewardVariable, StateVariable, Update, Var

REPO_ROOT = Path(__file__).resolve().parents[1]
WEATHER_FACTORY_MODEL = REPO_ROOT / "examples" / "weather_factory_3.prism"
COMPLEX_MULTI_ACTION_MODEL = REPO_ROOT / "examples" / "complex_multi_action.prism"
HAS_JAX = importlib.util.find_spec("jax") is not None
HAS_STORMPY = importlib.util.find_spec("stormpy") is not None


def build_single_module_model():
    variable = StateVariable(
        name="s",
        domain_kind="bounded",
        lower=Const(0),
        upper=Const(1),
        initial=Const(0),
    )
    module = Module(
        name="coin",
        variables=[variable],
        commands=[
            Command(
                action=None,
                guard=Guard(BinaryOp("==", Var("s"), Const(0))),
                updates=[
                    Update(Const(0.25), {"s": Const(1)}),
                    Update(Const(0.75), {"s": Const(0)}),
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
        properties={"goal": Var("goal_formula")},
        constants={"TARGET": 1},
        functions={"goal_formula": BinaryOp("==", Var("s"), Var("TARGET"))},
        modules=[module],
    )


def build_two_module_model():
    left = StateVariable(
        name="left",
        domain_kind="bounded",
        lower=Const(0),
        upper=Const(1),
        initial=Const(0),
    )
    right = StateVariable(
        name="right",
        domain_kind="bounded",
        lower=Const(0),
        upper=Const(1),
        initial=Const(0),
    )
    modules = [
        Module(
            name="left_module",
            variables=[left],
            commands=[
                Command(
                    action="tick",
                    guard=Guard(Const(True)),
                    updates=[
                        Update(Const(0.5), {"left": Const(1)}),
                        Update(Const(0.5), {"left": Const(0)}),
                    ],
                )
            ],
        ),
        Module(
            name="right_module",
            variables=[right],
            commands=[
                Command(
                    action="tick",
                    guard=Guard(Const(True)),
                    updates=[
                        Update(Const(0.5), {"right": Const(1)}),
                        Update(Const(0.5), {"right": Const(0)}),
                    ],
                )
            ],
        ),
    ]
    goal = BinaryOp("∧", BinaryOp("==", Var("left"), Const(1)), BinaryOp("==", Var("right"), Const(1)))
    return ParsedModel(properties={"goal": goal}, constants={}, functions={}, modules=modules)


def build_ambiguous_model():
    variable = StateVariable(
        name="s",
        domain_kind="bounded",
        lower=Const(0),
        upper=Const(1),
        initial=Const(0),
    )
    module = Module(
        name="ambiguous",
        variables=[variable],
        commands=[
            Command(
                action=None,
                guard=Guard(Const(True)),
                updates=[Update(Const(1.0), {"s": Var("s")})],
            ),
            Command(
                action=None,
                guard=Guard(Const(True)),
                updates=[Update(Const(1.0), {"s": Var("s")})],
            ),
        ],
    )
    return ParsedModel(properties={"goal": BinaryOp("==", Var("s"), Const(1))}, constants={}, functions={}, modules=[module])


def build_reward_property_model():
    variable = StateVariable(
        name="s",
        domain_kind="bounded",
        lower=Const(0),
        upper=Const(1),
        initial=Const(0),
    )
    reward = RewardVariable(name="reward", initial=Const(0))
    module = Module(
        name="coin",
        variables=[variable],
        commands=[
            Command(
                action=None,
                guard=Guard(Const(True)),
                updates=[Update(Const(1.0), {"s": Var("s"), "reward": Const(1)})],
            )
        ],
    )
    return ParsedModel(
        properties={"goal": BinaryOp("==", Var("reward"), Const(1))},
        constants={},
        functions={},
        modules=[Module("__globals__", [reward], []), module],
    )


def build_overlapping_guard_model():
    """Two commands with same action [a] and overlapping guards at s=0.

    [a] s=0 -> (s'=1)
    [a] s=0 -> (s'=2)

    Command-level averaging: at s=0, both enabled -> 0.5*s=1 + 0.5*s=2.
    s=1, s=2 are deadlock (absorbing).

    Pr(goal=s=2 | H=0)=0, H=1)=0.5, H>=1)=0.5
    """
    variable = StateVariable(
        name="s",
        domain_kind="bounded",
        lower=Const(0),
        upper=Const(2),
        initial=Const(0),
    )
    module = Module(
        name="proc",
        variables=[variable],
        commands=[
            Command(
                action="a",
                guard=Guard(BinaryOp("==", Var("s"), Const(0))),
                updates=[Update(Const(1.0), {"s": Const(1)})],
            ),
            Command(
                action="a",
                guard=Guard(BinaryOp("==", Var("s"), Const(0))),
                updates=[Update(Const(1.0), {"s": Const(2)})],
            ),
        ],
    )
    return ParsedModel(
        properties={"goal": BinaryOp("==", Var("s"), Const(2))},
        constants={},
        functions={},
        modules=[module],
    )


def build_multi_anon_model():
    """Single-module model with two overlapping anonymous commands at s=0.

    [] s=0 -> (s'=1)
    [] s=0 -> (s'=2)

    Each anonymous command gets a unique action label. At s=0, both are enabled
    and averaged: 0.5*s=1 + 0.5*s=2. States s=1, s=2 are deadlock (absorbing).

    Pr(goal=s=2 | H=0)=0, H=1)=0.5, H>=1)=0.5
    """
    variable = StateVariable(
        name="s",
        domain_kind="bounded",
        lower=Const(0),
        upper=Const(2),
        initial=Const(0),
    )
    module = Module(
        name="proc",
        variables=[variable],
        commands=[
            Command(
                action=None,
                guard=Guard(BinaryOp("==", Var("s"), Const(0))),
                updates=[Update(Const(1.0), {"s": Const(1)})],
            ),
            Command(
                action=None,
                guard=Guard(BinaryOp("==", Var("s"), Const(0))),
                updates=[Update(Const(1.0), {"s": Const(2)})],
            ),
        ],
    )
    return ParsedModel(
        properties={"goal": BinaryOp("==", Var("s"), Const(2))},
        constants={},
        functions={},
        modules=[module],
    )


def build_multi_action_model():
    """Single-module model with two actions [a] and [b], both enabled at s=0.

    s=0: [a] 0.5->s=1, 0.5->s=2;  [b] 1.0->s=3.  Averaged: 0.25*1+0.25*2+0.5*3
    s=1: [a] 1.0->s=3
    s=2, s=3: deadlock (absorbing)

    Pr(goal=s=3 | H=0)=0, H=1)=0.5, H=2)=0.75, H>=3)=0.75
    """
    variable = StateVariable(
        name="s",
        domain_kind="bounded",
        lower=Const(0),
        upper=Const(3),
        initial=Const(0),
    )
    module = Module(
        name="proc",
        variables=[variable],
        commands=[
            Command(
                action="a",
                guard=Guard(BinaryOp("==", Var("s"), Const(0))),
                updates=[
                    Update(Const(0.5), {"s": Const(1)}),
                    Update(Const(0.5), {"s": Const(2)}),
                ],
            ),
            Command(
                action="b",
                guard=Guard(BinaryOp("==", Var("s"), Const(0))),
                updates=[Update(Const(1.0), {"s": Const(3)})],
            ),
            Command(
                action="a",
                guard=Guard(BinaryOp("==", Var("s"), Const(1))),
                updates=[Update(Const(1.0), {"s": Const(3)})],
            ),
        ],
    )
    return ParsedModel(
        properties={"goal": BinaryOp("==", Var("s"), Const(3))},
        constants={},
        functions={},
        modules=[module],
    )


def build_multi_action_sync_model():
    """Two-module model with actions [a] and [b] synchronizing across modules.

    (0,0): [a] and [b] both enabled → averaged
      [a]: x->1, y->0.5*0+0.5*1  [b]: x->0, y->1
      Average: 0.25*(1,0)+0.25*(1,1)+0.5*(0,1)
    (1,0): only [b] → (1,1)
    (0,1): only [b] → (0,1)  (loops)
    (1,1): only [b] → (1,1)  (absorbing)

    Pr(goal=x=1&y=1 | H=0)=0, H=1)=0.25, H=2)=0.5, H>=3)=0.5
    """
    left = StateVariable(
        name="x",
        domain_kind="bounded",
        lower=Const(0),
        upper=Const(1),
        initial=Const(0),
    )
    right = StateVariable(
        name="y",
        domain_kind="bounded",
        lower=Const(0),
        upper=Const(1),
        initial=Const(0),
    )
    modules = [
        Module(
            name="left",
            variables=[left],
            commands=[
                Command(
                    action="a",
                    guard=Guard(BinaryOp("==", Var("x"), Const(0))),
                    updates=[Update(Const(1.0), {"x": Const(1)})],
                ),
                Command(
                    action="b",
                    guard=Guard(Const(True)),
                    updates=[Update(Const(1.0), {"x": Var("x")})],
                ),
            ],
        ),
        Module(
            name="right",
            variables=[right],
            commands=[
                Command(
                    action="a",
                    guard=Guard(BinaryOp("==", Var("y"), Const(0))),
                    updates=[
                        Update(Const(0.5), {"y": Const(0)}),
                        Update(Const(0.5), {"y": Const(1)}),
                    ],
                ),
                Command(
                    action="b",
                    guard=Guard(Const(True)),
                    updates=[Update(Const(1.0), {"y": Const(1)})],
                ),
            ],
        ),
    ]
    goal = BinaryOp("∧", BinaryOp("==", Var("x"), Const(1)), BinaryOp("==", Var("y"), Const(1)))
    return ParsedModel(properties={"goal": goal}, constants={}, functions={}, modules=modules)


class NumpyReachabilityTests(unittest.TestCase):
    def test_numpy_backend_matches_expected_probabilities(self) -> None:
        compiled = compile_reachability(build_single_module_model(), property_name="goal", backend="numpy")

        self.assertIsInstance(compiled.backend, BackendRuntime)
        self.assertEqual(repr(compiled.backend), "numpy")
        self.assertEqual(compiled.backend.spec.raw, "numpy")
        probability0 = compiled.run(0)
        probability1 = compiled.run(1)
        probability2 = compiled.run(2)
        self.assertEqual(str(probability1.dtype), "float32")
        self.assertAlmostEqual(float(probability0.item()), 0.0, places=6)
        self.assertAlmostEqual(float(probability1.item()), 0.25, places=6)
        self.assertAlmostEqual(float(probability2.item()), 0.4375, places=6)

    def test_numpy_backend_supports_synchronized_modules(self) -> None:
        compiled = compile_reachability(build_two_module_model(), property_name="goal", backend="numpy")

        probability = compiled.run(1)
        self.assertEqual(str(probability.dtype), "float32")
        self.assertAlmostEqual(float(probability.item()), 0.25, places=6)

    def test_numpy_overlapping_guard_averaging(self) -> None:
        compiled = compile_reachability(build_overlapping_guard_model(), property_name="goal", backend="numpy")
        self.assertAlmostEqual(float(compiled.run(0).item()), 0.0, places=6)
        self.assertAlmostEqual(float(compiled.run(1).item()), 0.5, places=6)
        self.assertAlmostEqual(float(compiled.run(5).item()), 0.5, places=6)

    def test_numpy_multi_anon_averaging(self) -> None:
        compiled = compile_reachability(build_multi_anon_model(), property_name="goal", backend="numpy")
        self.assertAlmostEqual(float(compiled.run(0).item()), 0.0, places=6)
        self.assertAlmostEqual(float(compiled.run(1).item()), 0.5, places=6)
        self.assertAlmostEqual(float(compiled.run(5).item()), 0.5, places=6)

    def test_numpy_multi_action_single_module(self) -> None:
        compiled = compile_reachability(build_multi_action_model(), property_name="goal", backend="numpy")
        self.assertAlmostEqual(float(compiled.run(0).item()), 0.0, places=6)
        self.assertAlmostEqual(float(compiled.run(1).item()), 0.5, places=6)
        self.assertAlmostEqual(float(compiled.run(2).item()), 0.75, places=6)
        self.assertAlmostEqual(float(compiled.run(5).item()), 0.75, places=6)

    def test_numpy_complex_multi_action(self) -> None:
        """Multi-module model with overlapping guards, multiple actions, anonymous commands, deadlock."""
        compiled = compile_reachability(
            load_model("prism", COMPLEX_MULTI_ACTION_MODEL),
            property_name="goal", backend="numpy", dtype="float64",
        )
        # Storm reference values
        self.assertAlmostEqual(float(compiled.run(1).item()), 0.25, places=6)
        self.assertAlmostEqual(float(compiled.run(2).item()), 0.6041666667, places=6)
        self.assertAlmostEqual(float(compiled.run(5).item()), 0.7521701389, places=6)
        self.assertAlmostEqual(float(compiled.run(10).item()), 0.7815552386, places=6)

    def test_numpy_multi_action_sync_two_modules(self) -> None:
        compiled = compile_reachability(build_multi_action_sync_model(), property_name="goal", backend="numpy")
        self.assertAlmostEqual(float(compiled.run(0).item()), 0.0, places=6)
        self.assertAlmostEqual(float(compiled.run(1).item()), 0.25, places=6)
        self.assertAlmostEqual(float(compiled.run(2).item()), 0.5, places=6)
        self.assertAlmostEqual(float(compiled.run(5).item()), 0.5, places=6)

    def test_numpy_backend_exposes_backend_repr_and_dtype(self) -> None:
        compiled = compile_reachability(build_single_module_model(), property_name="goal", backend="numpy")

        probability = compiled.run(1)
        host_probability = compiled.backend.device_get(probability)

        self.assertEqual(repr(compiled.backend), "numpy")
        self.assertIn("array(", repr(probability))
        self.assertEqual(str(getattr(probability, "dtype", getattr(host_probability, "dtype", None))), "float32")

    def test_numpy_backend_supports_float64_dtype(self) -> None:
        compiled = compile_reachability(
            build_single_module_model(),
            property_name="goal",
            backend="numpy",
            dtype="float64",
        )

        probability = compiled.run(2)
        self.assertEqual(compiled.backend.dtype_name, "float64")
        self.assertEqual(str(probability.dtype), "float64")
        self.assertIn("array(", repr(probability))
        self.assertAlmostEqual(float(probability.item()), 0.4375, places=10)


@unittest.skipUnless(HAS_JAX, "jax is required for JAX reachability tests")
class JaxReachabilityCompilerTests(unittest.TestCase):
    def test_compile_builds_initial_and_goal_tensors(self) -> None:
        compiled = compile_reachability(build_single_module_model(), property_name="goal", backend="jax:cpu")

        self.assertEqual(compiled.tensor_shape, (2,))
        self.assertEqual(compiled.backend.spec.raw, "jax:cpu")
        self.assertEqual(repr(compiled.backend), "cpu")
        self.assertAlmostEqual(float(compiled.initial_tensor[0]), 1.0)
        self.assertAlmostEqual(float(compiled.initial_tensor[1]), 0.0)
        self.assertAlmostEqual(float(compiled.goal_mask[0]), 0.0)
        self.assertAlmostEqual(float(compiled.goal_mask[1]), 1.0)

    def test_jax_cpu_backend_resolves_constants_and_formulas(self) -> None:
        compiled = compile_reachability(build_single_module_model(), property_name="goal", backend="jax:cpu")

        probability0 = compiled.run(0)
        probability1 = compiled.run(1)
        probability2 = compiled.run(2)
        self.assertEqual(str(compiled.backend.device_get(probability1).dtype), "float32")
        self.assertAlmostEqual(float(compiled.backend.device_get(probability0).item()), 0.0, places=6)
        self.assertAlmostEqual(float(compiled.backend.device_get(probability1).item()), 0.25, places=6)
        self.assertAlmostEqual(float(compiled.backend.device_get(probability2).item()), 0.4375, places=6)

    def test_jax_cpu_backend_matches_numpy_backend(self) -> None:
        numpy_compiled = compile_reachability(build_single_module_model(), property_name="goal", backend="numpy")
        jax_compiled = compile_reachability(build_single_module_model(), property_name="goal", backend="jax:cpu")

        self.assertAlmostEqual(
            float(jax_compiled.backend.device_get(jax_compiled.run(2)).item()),
            float(numpy_compiled.run(2).item()),
            places=6,
        )

    def test_jax_cpu_backend_supports_synchronized_modules(self) -> None:
        compiled = compile_reachability(build_two_module_model(), property_name="goal", backend="jax:cpu")

        probability = compiled.run(1)
        self.assertEqual(str(compiled.backend.device_get(probability).dtype), "float32")
        self.assertAlmostEqual(float(compiled.backend.device_get(probability).item()), 0.25, places=6)

    def test_compile_rejects_unknown_property(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown property"):
            compile_reachability(build_single_module_model(), property_name="missing", backend="jax:cpu")

    def test_anonymous_commands_are_independently_averaged(self) -> None:
        """Two anonymous commands with guard=True and identity updates are averaged.

        Both actions keep s unchanged, so Pr(goal=s==1 | s₀=0) = 0 for all horizons.
        """
        compiled = compile_reachability(build_ambiguous_model(), property_name="goal", backend="jax:cpu")
        probability = compiled.run(5)
        self.assertAlmostEqual(float(compiled.backend.device_get(probability).item()), 0.0, places=6)

    def test_compile_rejects_reward_backed_goal_expression(self) -> None:
        with self.assertRaisesRegex(ValueError, "reward variables"):
            compile_reachability(build_reward_property_model(), property_name="goal", backend="jax:cpu")

    def test_jax_cpu_backend_exposes_backend_repr_and_dtype(self) -> None:
        compiled = compile_reachability(build_single_module_model(), property_name="goal", backend="jax:cpu")

        probability = compiled.run(1)
        host_probability = compiled.backend.device_get(probability)

        self.assertEqual(repr(compiled.backend), "cpu")
        self.assertIn("Array(", repr(probability))
        self.assertEqual(str(getattr(probability, "dtype", getattr(host_probability, "dtype", None))), "float32")

    def test_jax_cpu_backend_supports_float64_dtype(self) -> None:
        compiled = compile_reachability(
            build_single_module_model(),
            property_name="goal",
            backend="jax:cpu",
            dtype="float64",
        )

        probability = compiled.run(2)
        self.assertEqual(compiled.backend.dtype_name, "float64")
        self.assertEqual(str(compiled.backend.device_get(probability).dtype), "float64")
        self.assertIn("Array(", repr(probability))
        self.assertIn("dtype=float64", repr(probability))
        self.assertAlmostEqual(float(compiled.backend.device_get(probability).item()), 0.4375, places=10)


class BackendParsingTests(unittest.TestCase):
    def test_parse_backend_accepts_expected_values(self) -> None:
        self.assertEqual(parse_backend("numpy").raw, "numpy")
        self.assertEqual(parse_backend("jax:cpu").raw, "jax:cpu")
        self.assertEqual(parse_backend("jax:cuda:0").raw, "jax:cuda:0")

    def test_parse_backend_rejects_unexpected_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid backend"):
            parse_backend("jax")


class BackendRuntimeDecoratorTests(unittest.TestCase):
    def test_package_exports_jit(self) -> None:
        self.assertIs(tessa.jit, jit)

    def test_numpy_jit_decorator_is_identity(self) -> None:
        backend = build_backend("numpy")

        @jit(backend)
        def increment(value):
            return value + 1

        self.assertEqual(increment(1), 2)

    def test_numpy_lax_fori_loop_matches_python_loop(self) -> None:
        backend = build_backend("numpy")

        @jit(backend)
        def accumulate(horizon):
            def body(_, carry):
                return carry + 2

            return backend.lax.fori_loop(0, backend.array.asarray(horizon, dtype=backend.array.int32), body, 0)

        self.assertEqual(accumulate(0), 0)
        self.assertEqual(accumulate(3), 6)


@unittest.skipUnless(HAS_JAX, "jax is required for JAX decorator tests")
class JaxBackendRuntimeDecoratorTests(unittest.TestCase):
    def test_jax_cpu_jit_decorator_executes(self) -> None:
        backend = build_backend("jax:cpu")

        @jit(backend)
        def increment(value):
            return value + 1

        self.assertEqual(int(increment(1)), 2)


@unittest.skipUnless(HAS_JAX and HAS_STORMPY, "jax and stormpy are required for PRISM reachability integration")
class PrismReachabilityIntegrationTests(unittest.TestCase):
    def test_weather_factory_matches_embedded_storm_reference(self) -> None:
        parsed_model = load_model("prism", WEATHER_FACTORY_MODEL)
        compiled = compile_reachability(parsed_model, property_name="allStrike", backend="jax:cpu")

        probability = compiled.run(10)
        self.assertEqual(str(compiled.backend.device_get(probability).dtype), "float32")
        self.assertAlmostEqual(float(compiled.backend.device_get(probability).item()), 0.05142680456, places=6)


@unittest.skipUnless(HAS_STORMPY, "stormpy is required for PRISM reachability integration")
class PrismNumpyReachabilityIntegrationTests(unittest.TestCase):
    def test_weather_factory_matches_embedded_storm_reference_with_numpy(self) -> None:
        parsed_model = load_model("prism", WEATHER_FACTORY_MODEL)
        compiled = compile_reachability(parsed_model, property_name="allStrike", backend="numpy")

        probability = compiled.run(10)
        self.assertEqual(str(probability.dtype), "float32")
        self.assertAlmostEqual(float(probability.item()), 0.05142680456, places=6)
