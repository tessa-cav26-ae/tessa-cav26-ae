"""Generate a weather-factory .gir program and run gennifer on it.

Invoked by ``BenchmarkContext._build_geni_command`` so the geni path through
``run_case`` is symmetric with rubicon: a single command list whose stdout
ends in a parseable probability scalar.

The generator step calls
``benchmarks/weather_factory/gen_weather_factory_gennifer.py`` and the
interpret step calls ``gennifer -l -i --pt``. Gennifer emits per-value
probability rows (``Pr(true) = 0.123``, ``Pr(false) = 0.877``) followed by a
``Time consumption, ...`` block when ``--pt`` is set; we extract the
``Pr(true)`` marginal — gennifer's translation of the boolean ``allStrike``
property.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import click


# Gennifer prints one Pr(<value>) = <scalar> line per output value. For
# boolean reachability programs (the shape the generator emits) the values
# come out as 0 / 1, *not* true / false; we accept both spellings so the
# parser remains stable if upstream changes its formatter.
_PR_RE = re.compile(
    r"Pr\(\s*(true|false|1|0)\s*\)\s*=\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


def _extract_probability(gennifer_stdout: str) -> float:
    pr_true: float | None = None
    for value, scalar in _PR_RE.findall(gennifer_stdout):
        if value in ("true", "1"):
            pr_true = float(scalar)
    if pr_true is None:
        raise SystemExit(
            "Could not find a 'Pr(true)/Pr(1) = <scalar>' line in gennifer output"
        )
    return pr_true


@click.command()
@click.option("--n", "n", required=True, type=int, help="Factory count")
@click.option("--h", "h", required=True, type=int, help="Horizon")
@click.option("--mode", default="monolithic", type=click.Choice(["monolithic", "sequential"]))
@click.option("--generator", "generator_path", default=None, type=click.Path(path_type=Path),
              help="Path to gen_weather_factory_gennifer.py (default: benchmarks/weather_factory/...)")
@click.option("--gennifer-cmd", default="gennifer")
@click.option("--workdir", default=None, type=click.Path(path_type=Path),
              help="Persistent workdir for the .gir cache; default is a per-invocation tempdir")
def main(n, h, mode, generator_path, gennifer_cmd, workdir):
    if generator_path is None:
        generator_path = (
            Path(__file__).resolve().parents[1]
            / "benchmarks" / "weather_factory" / "gen_weather_factory_gennifer.py"
        )
    if not generator_path.exists():
        raise SystemExit(f"generator not found: {generator_path}")

    workdir_ctx = tempfile.TemporaryDirectory() if workdir is None else None
    work_root = Path(workdir_ctx.name) if workdir_ctx is not None else Path(workdir)
    work_root.mkdir(parents=True, exist_ok=True)
    gir_path = work_root / f"weather-factory-{mode}-{n}-{h}.gir"

    try:
        # Skip generation when a persistent workdir already has the .gir from
        # a prior trial — the generator is deterministic in (n, h, mode), so
        # reusing the file lets multiple num_work_runs trials skip the
        # ~O(N*H) Python work.
        if not gir_path.exists():
            gen = subprocess.run(
                [
                    sys.executable, str(generator_path),
                    "--n", str(n),
                    "--h", str(h),
                    "--mode", mode,
                ],
                capture_output=True, text=True, check=False,
            )
            sys.stderr.write(gen.stderr)
            if gen.returncode != 0:
                raise SystemExit(f"gen_weather_factory_gennifer failed (exit {gen.returncode})")
            gir_path.write_text(gen.stdout)

        gennifer = subprocess.run(
            [gennifer_cmd, "-l", "-i", "--pt", str(gir_path)],
            capture_output=True, text=True, check=False,
        )
        sys.stdout.write(gennifer.stdout)
        sys.stderr.write(gennifer.stderr)
        if gennifer.returncode != 0:
            raise SystemExit(f"gennifer failed (exit {gennifer.returncode})")

        probability = _extract_probability(gennifer.stdout)
    finally:
        if workdir_ctx is not None:
            workdir_ctx.cleanup()

    # Sentinel newline so the probability lands on its own line — gennifer's
    # final `Time consumption, ...` block has no trailing newline guarantee
    # and the outer parser reads only the last line.
    sys.stdout.write("\n")
    print(f"{probability:.17g}")


if __name__ == "__main__":
    main()
