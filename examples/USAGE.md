Using Tessa on Your Own Models
==============================

This guide shows how to run Tessa on PRISM and JANI models beyond the four benchmark suites driven by `reproduce.mk`. It complements the top-level [`README.md`](../README.md): the README is about reproducing the paper figures; this file is about *writing and checking your own model*.

Everything below assumes you are inside the `tessa-shell` (the default `nix develop` shell described in [README §Option 1](../README.md#option-1-nix-recommended) or [README §Option 2](../README.md#option-2-docker)). `tessa --help` should print the CLI usage.

## Contents

* [1. Anatomy of a Tessa-compatible model](#1-anatomy-of-a-tessa-compatible-model)
* [2. Hello world — bounded reachability](#2-hello-world--bounded-reachability)
* [3. Passing constants from the command line](#3-passing-constants-from-the-command-line)
* [4. Parametric (differentiable) models](#4-parametric-differentiable-models)
* [5. Where to go next](#5-where-to-go-next)

## 0. CLI interface

`tessa MODEL --property NAME --horizon H --backend jax:cuda:N` is the standard invocation: it compiles `P=? [ F<=H "NAME" ]` and prints the resolved backend on one line, the scalar probability on the next. `--property` and `--horizon` must be supplied together — passing one without the other is a usage error.

The full flag listing (verbatim from `tessa --help`):

```
$ tessa --help
usage: tessa [-h] [--type {jani,prism}] [--const CONSTANTS] [--property PROPERTY_NAME] [--horizon HORIZON] [--backend BACKEND] [--dtype DTYPE] [--num-cold-work-runs NUM_COLD_WORK_RUNS]
             [--num-work-runs NUM_WORK_RUNS] [--num-timed-runs NUM_TIMED_RUNS] [--time-log TIME_LOG]
             model_path

positional arguments:
  model_path

options:
  -h, --help            show this help message and exit
  --type {jani,prism}
  --const CONSTANTS     Constant override in the form NAME=VALUE
  --property PROPERTY_NAME
  --horizon HORIZON
  --backend BACKEND     Reachability backend: numpy, explicit, jax:cpu, or jax:cuda:N
  --dtype DTYPE         Reachability dtype: float32 or float64
  --num-cold-work-runs NUM_COLD_WORK_RUNS
                        Number of throwaway compile+warmup pairs run before the measured work loop to prime JAX/XLA caches. Records are emitted with phases 'cold_compile' and 'cold_warmup' and excluded
                        from summary stats.
  --num-work-runs NUM_WORK_RUNS
                        Number of compile+warmup iterations (each pair is timed). Mean and std of compile_seconds, warmup_avg_seconds, and work_seconds = stable_compile + warmup_run are emitted to the
                        time-log summary (means over the warm loop; use --num-cold-work-runs to prime caches first).
  --num-timed-runs NUM_TIMED_RUNS
                        Number of measured executions after the work loop
  --time-log TIME_LOG   Append JSONL timing records to this path
```

### Defaults the help text does not show

The flag list above doesn't print defaults:

| Flag | Default | Notes |
|---|---|---|
| `--type` | inferred from extension | `.jani` → JANI; `.prism` / `.pm` / `.nm` → PRISM. Pass explicitly only when the extension lies. |
| `--const` | none (repeatable) | One `--const NAME=VALUE` per constant; values coerce as `bool` → `int` → `float` in that order. |
| `--backend` | **`jax:cpu`** | The paper and `reproduce.mk` pin `jax:cuda:0`. Never omit `--backend` when reproducing results — the smoke test exists specifically because `jax:cpu` is a silent valid fallback. |
| `--dtype` | **`float32`** | Use `float64` for tighter agreement with Storm; `reproduce.mk`'s verify step uses `atol=1e-5, rtol=1e-4`. |
| `--num-cold-work-runs` | **`3`** | Throwaway compile+warmup pairs to prime JAX caches; not in summary stats. |
| `--num-work-runs`      | **`1`** | Timed compile+warmup pairs (the warm loop). |
| `--num-timed-runs`     | **`1`** | Measured executions after the warm loop. |
| `--time-log` | unset | When set, appends JSONL timing records to the given path. |

For one-off interactive checks (not benchmarking), the warmup defaults add ~3× wall-clock; pass `--num-cold-work-runs 0 --num-work-runs 1 --num-timed-runs 1` to skip them. The full reproduction pipeline keeps the defaults because warm-loop statistics are what the paper figures are built on.

### Common invocation patterns

```shell
# Bounded reachability against a named property (→ §2)
tessa examples/weather_factory_3.jani --property allStrike --horizon 10 --backend jax:cuda:0

# Same, with --const overrides for unbound model constants (→ §3)
tessa your_model.jani --property reach --horizon 20 \
    --const N=8 --const p=0.3 --backend jax:cuda:0

# Parametric / differentiable evaluation — drops down to the Python API,
# the CLI does not expose --defer-const (→ §4)
python -c "from src import compile_reachability, load_prism_model; ..."
```

## 1. Anatomy of a Tessa-compatible model

Tessa checks **bounded reachability** (`P=? [ F<=H "label" ]`) on **discrete-time Markov chains (DTMCs)** expressed in either of two input languages:

* **PRISM** — text format. Reference: PRISM language manual at <https://www.prismmodelchecker.org/manual/ThePRISMLanguage/Introduction>. PRISM models loaded by Tessa go through `stormpy`, so any DTMC accepted by Storm is accepted here.
* **JANI** — JSON-based interchange format. Specification at <https://jani-spec.org/>. JANI is the format you should use if you build your model programmatically (e.g. from another tool's exporter), and the only format that works on [Option 3](../README.md#option-3-nix-storm--pip-tessa) (no `stormpy`).

Every benchmark suite under `benchmarks/` and both files in `examples/` ship a `.prism` and a `.jani` form of the same model. You can take either pair as a starting template.

A property is referenced by **name**, not by inline formula. In PRISM that's a `label "name" = expr;` declaration in the model file; in JANI it's an entry of the top-level `properties` array.

For example, [`examples/weather_factory_3.prism`](weather_factory_3.prism) declares:

```prism
label "allStrike" = state1 & state2 & state3;
```

and the equivalent JANI form [`examples/weather_factory_3.jani`](weather_factory_3.jani) lists it under `properties`:

```json
"properties": [
  { "name": "allStrike",
    "expression": { "left": { "left": "state1", "op": "∧", "right": "state2" }, "op": "∧", "right": "state3" } }
]
```

The Tessa CLI takes the property name as `--property allStrike` and wraps it as `P=? [ F<=H "allStrike" ]` internally.

## 2. Hello world — bounded reachability

`examples/weather_factory_3` is a 3-factory weather model: each factory only strikes when conditions align, and `allStrike` is the joint event that all three strike. We ask: what is the probability that `allStrike` becomes true within 10 steps?

```shell
tessa examples/weather_factory_3.jani \
    --property allStrike \
    --horizon 10 \
    --backend jax:cuda:0
```

Expected last two lines of output:

```
gpu
Array(0.05142681, dtype=float32)
```

The first line is `compiled_model.backend` (proof CUDA was actually used — see [README §Confirm CUDA was actually used](../README.md#confirm-cuda-was-actually-used-read-it-from-the-logs-dont-trust-statusok)); the second is the probability as a Python `repr`.

Same model, PRISM form, same answer:

```shell
tessa examples/weather_factory_3.prism --property allStrike --horizon 10 --backend jax:cuda:0
```

> **Defaults to know:** without `--num-cold-work-runs 0`, the CLI performs 3 throwaway compile+warmup pairs before the timed run to prime JAX caches. That's the right default for benchmarking but slow for an interactive single check — pass `--num-cold-work-runs 0 --num-work-runs 1 --num-timed-runs 1` to skip the warmup loop.

## 3. Passing constants from the command line

A model can leave structural constants (`const int N;`, `const double p;`) unbound and have Tessa fill them in at the CLI. [`examples/complex_multi_action.prism`](complex_multi_action.prism) (and its JANI equivalent [`examples/complex_multi_action.jani`](complex_multi_action.jani)) declares a goal label that's reachable through several action-averaging paths:

```prism
label "goal" = x=2 & y=0;
```

The same property in the JANI form ([`complex_multi_action.jani`](complex_multi_action.jani)) is a top-level `properties` entry:

```json
"properties": [
  { "name": "goal",
    "expression": { "left": { "left": "x", "op": "=", "right": 2 },
                    "op": "∧",
                    "right": { "left": "y", "op": "=", "right": 0 } } }
]
```

It has no constants to override, so the command is plain — both forms return the same probability:

```shell
tessa examples/complex_multi_action.prism --property goal --horizon 8 --backend jax:cuda:0
tessa examples/complex_multi_action.jani  --property goal --horizon 8 --backend jax:cuda:0
```

Expected:

```
gpu
Array(0.77976555, dtype=float32)
```

For a model that *does* take constants, repeat `--const NAME=VALUE` once per constant. Values are coerced to `bool` / `int` / `float` in that order, so `true`/`false` are booleans, plain integers stay integers, and anything else parses as a float:

```shell
tessa your_model.jani \
    --property reach \
    --horizon 20 \
    --const N=8 \
    --const p=0.3 \
    --const debug=false \
    --backend jax:cuda:0
```

Constants that the model declares with no default *must* be supplied via `--const` or Storm's loader will reject the model.

## 4. Parametric (differentiable) models

Tessa's parametric mode keeps named constants **symbolic** so the resulting reachability kernel is differentiable through JAX. This is the path used for Figure 8 in the paper; it's also the most useful Tessa-only feature for downstream users (synthesis, sensitivity analysis, optimisation).

The Knuth-Yao die model under [`benchmarks/kydice/kydice.prism`](../benchmarks/kydice/kydice.prism) declares two symbolic coin biases:

```prism
const double x;
const double y;
...
label "die1" = (s=7) & (d=1);
```

Compile once, then evaluate at any `(x, y)` you want:

```python
from src import compile_reachability, load_prism_model

parsed = load_prism_model(
    "benchmarks/kydice/kydice.prism",
    constants={"x": 0.5, "y": 0.5},     # placeholder values, only matters for shape inference
    defer_constants=["x", "y"],         # PRISM only — keep x, y symbolic instead of substituting
)
kernel = compile_reachability(parsed, property_name="die1", backend="jax:cuda:0", dtype="float32")

for (xv, yv) in [(0.5, 0.5), (0.3, 0.7), (0.1, 0.9)]:
    p = kernel.run(100, constants_override={"x": xv, "y": yv})
    print(f"P(die1 | x={xv}, y={yv}) = {float(p):.6f}")
```

Expected:

```
P(die1 | x=0.5, y=0.5) = 0.166667
P(die1 | x=0.3, y=0.7) = 0.186076
P(die1 | x=0.1, y=0.9) = 0.089011
```

At `x = y = 0.5` the die is fair, so each face has probability `1/6 ≈ 0.166667` — a sanity check that the model is wired up correctly.

> **PRISM vs JANI for parametric models:** PRISM models go through Storm's `substitute_constants()` which would bake numeric values into the expression tree, so you need `defer_constants=["x", "y"]` to override that. JANI's parser keeps constant references symbolic natively, so for a JANI model you'd call `load_jani_model(path, constants={"x": 0.5, "y": 0.5})` and skip `defer_constants`.

Because `kernel.run` is a regular JAX-traceable function, you can wrap it in `jax.grad`, `jax.value_and_grad`, `jax.vmap`, or feed it to an optimiser. The full training loop that produces Figure 8 lives in [`benchmarks/kydice/kydice.py`](../benchmarks/kydice/kydice.py) and uses `optax.adam` inside `jax.lax.scan` — recommended reading once the snippet above runs.

## 5. Where to go next

* **Reproduce the paper figures** end-to-end: `make -f reproduce.mk` (see [README §Experimental Evaluation](../README.md#experimental-evaluation)).
* **Larger reference models** to copy from: `benchmarks/herman/`, `benchmarks/meeting/`, `benchmarks/weather_factory/`, `benchmarks/parqueues/`. Each ships paired `.prism` and `.jani` forms plus a generator script.
* **Convert a PRISM model to JANI** for the Option 3 (`pip`) path: `benchmarks/convert_prism_to_jani.sh`.
* **Troubleshooting** Tessa / CUDA / JAX issues: [README §Troubleshooting](../README.md#troubleshooting).
