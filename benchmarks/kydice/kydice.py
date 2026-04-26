"""Knuth-Yao parametric die — train coin biases (x, y) so the resulting die
distribution becomes uniform, then plot the trajectory and the KL landscape.

Usage:
    python kydice.py --model PATH [--output-dir DIR] [--horizon H] [--steps S]

The script imports ``src`` (the tessa package) — the caller is responsible
for ensuring it is on PYTHONPATH (e.g. by running from the repo root or by
exporting ``PYTHONPATH=/path/to/tessa``).

Outputs to ``--output-dir`` (default: cwd):
    loss.csv        — per-step training trajectory (step, loss, x, y)
    loss.png        — loss curve + parameter evolution
    landscape.png   — KL(p || u) contour over (x, y) ∈ [0, 1]²
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from src import compile_reachability, load_jani_model, load_prism_model

FACE_LABELS = [f"die{k}" for k in range(1, 7)]


def build_kernels(model_path: Path, backend: str, dtype: str):
    """Compile six per-face reachability kernels with x, y as JAX inputs.

    Accepts both .prism and .jani inputs. PRISM goes through stormpy's
    ``substitute_constants()`` which would bake the placeholder values into
    the expression tree — ``defer_constants`` keeps x, y symbolic. JANI's
    parser keeps constant references symbolic natively, so no defer needed.
    """
    if model_path.suffix.lower() == ".jani":
        parsed = load_jani_model(str(model_path), constants={"x": 0.5, "y": 0.5})
    else:
        parsed = load_prism_model(
            str(model_path),
            constants={"x": 0.5, "y": 0.5},
            defer_constants=["x", "y"],
        )
    return [
        compile_reachability(parsed, property_name=label, backend=backend, dtype=dtype)
        for label in FACE_LABELS
    ]


def make_loss_fns(kernels, horizon: int):
    def die_distribution(x, y):
        return jnp.stack([
            k.run(horizon, constants_override={"x": x, "y": y}) for k in kernels
        ])

    @jax.jit
    def kl_divergence(x, y):
        p = die_distribution(x, y)
        p = p / jnp.sum(p)
        u = jnp.full_like(p, 1.0 / p.size)
        return jnp.sum(p * (jnp.log(p + 1e-30) - jnp.log(u)))

    @jax.jit
    def loss_unconstrained(unconstrained):
        x = jax.nn.sigmoid(unconstrained[0])
        y = jax.nn.sigmoid(unconstrained[1])
        return kl_divergence(x, y)

    return die_distribution, kl_divergence, loss_unconstrained


def train(loss_fn, steps: int, lr: float = 0.1):
    """Adam (via optax) in jax.lax.scan. Returns (xs, ys, losses, elapsed_s)."""
    grad_fn = jax.value_and_grad(loss_fn)
    optimizer = optax.adam(lr)

    @jax.jit
    def step(state, _):
        unconstrained, opt_state = state
        loss_value, loss_grad = grad_fn(unconstrained)
        updates, opt_state = optimizer.update(loss_grad, opt_state)
        unconstrained = optax.apply_updates(unconstrained, updates)
        params = jax.nn.sigmoid(unconstrained)
        return (unconstrained, opt_state), (params, loss_value)

    key = jax.random.PRNGKey(0)
    unconstrained = jax.random.normal(key, (2,))
    opt_state = optimizer.init(unconstrained)

    t0 = time.perf_counter()
    (_, _), (params, losses) = jax.lax.scan(
        step, (unconstrained, opt_state), None, length=steps + 1,
    )
    losses.block_until_ready()
    elapsed = time.perf_counter() - t0
    return np.asarray(params[:, 0]), np.asarray(params[:, 1]), np.asarray(losses), elapsed


def save_loss_csv(path: Path, xs, ys, losses):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss", "params-x", "params-y"])
        for i, (l, x, y) in enumerate(zip(losses, xs, ys)):
            writer.writerow([i, float(l), float(x), float(y)])


def save_loss_png(path: Path, xs, ys, losses):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.xlabel("step"); plt.ylabel("KL divergence")
    plt.subplot(1, 2, 2)
    plt.plot(xs, label="x"); plt.plot(ys, label="y")
    plt.xlabel("step"); plt.ylabel("param value"); plt.legend()
    plt.tight_layout(); plt.savefig(path); plt.close()


def save_landscape_png(path: Path, kl_divergence, grid_size: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = jnp.linspace(0.0, 1.0, grid_size, dtype=jnp.float32)
    loss_grid = jax.vmap(lambda x: jax.vmap(lambda y: kl_divergence(x, y))(grid))(grid)
    loss_np = np.asarray(loss_grid)

    plt.figure(figsize=(5, 4))
    plt.contourf(np.asarray(grid), np.asarray(grid), loss_np.T, levels=18)
    cbar = plt.colorbar(label="KL divergence")
    cbar.ax.tick_params(labelsize=10)
    plt.xlabel("$x$", fontsize=14); plt.ylabel("$y$", fontsize=14)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=Path, required=True,
                        help="Model path (.jani or .prism); type is inferred from the suffix")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd(),
                        help="Directory for loss.csv, loss.png, landscape.png (default: cwd)")
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--landscape-grid", type=int, default=200)
    parser.add_argument("--backend", default="jax:cuda:0",
                        help="Computation backend (default: jax:cuda:0; pass jax:cpu to run without a GPU)")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--skip-landscape", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"JAX backend: {jax.default_backend()}")
    kernels = build_kernels(args.model, backend=args.backend, dtype=args.dtype)
    _, kl_divergence, loss_fn = make_loss_fns(kernels, args.horizon)

    # Warm the jit cache, then time a clean run.
    train(loss_fn, steps=10)
    xs, ys, losses, elapsed = train(loss_fn, steps=args.steps)
    print(f"Training: {elapsed:.4f}s   final loss: {float(losses[-1]):.6e}   "
          f"final (x, y): ({float(xs[-1]):.4f}, {float(ys[-1]):.4f})")

    loss_csv = args.output_dir / "loss.csv"
    loss_png = args.output_dir / "loss.png"
    landscape_png = args.output_dir / "landscape.png"
    save_loss_csv(loss_csv, xs, ys, losses); print(f"Wrote {os.path.relpath(loss_csv)}")
    save_loss_png(loss_png, xs, ys, losses); print(f"Wrote {os.path.relpath(loss_png)}")
    if not args.skip_landscape:
        save_landscape_png(landscape_png, kl_divergence, args.landscape_grid)
        print(f"Wrote {os.path.relpath(landscape_png)}")


if __name__ == "__main__":
    main()
