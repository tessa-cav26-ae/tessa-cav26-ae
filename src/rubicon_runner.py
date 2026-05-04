"""Run rubicon (translate) + dice as a single subprocess for benchmarking.

Invoked by ``BenchmarkContext._build_rubicon_command`` so the rubicon path
through ``run_case`` is symmetric with storm: a single command list whose
stdout ends in a parseable probability scalar.

The translate step calls the ``rubicon`` console script (``rubicon.rubicon:
translate_cli``) and the run step calls ``dice -json``. Dice's JSON output
is a single-element list: ``[{"Joint Distribution": [["Value", "Probability"],
[<value>, <prob>], ...]}]``. Rubicon's reachability translation returns a
boolean, so we pull the probability from the row whose value is ``"true"``.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import click


def _echo_command(cmd: list[str]) -> None:
    print("$ " + shlex.join(cmd), flush=True)


def _format_constants(pairs: tuple[str, ...]) -> str:
    return ",".join(pairs)


def _extract_probability(dice_stdout: str) -> float:
    payload = json.loads(dice_stdout)
    # rubicon.translate_prism emits a tuple `(hit_goal, prob0_cond)` — see
    # rubicon/rubicon.py:737-739. Dice's joint-distribution table therefore
    # has rows like ["(true, false)", "1."]; the answer is the marginal
    # P(hit_goal = true) summed across all rows whose first component is
    # "true". For non-tuple outputs (rare; only when rubicon is bypassed)
    # we also accept a plain "true" row.
    table = payload[0]["Joint Distribution"]
    total = 0.0
    for row in table[1:]:  # row 0 is the header
        value = str(row[0]).strip()
        if value == "true" or value.startswith("(true,") or value.startswith("(true "):
            total += float(row[1])
    return total


@click.command()
@click.option("--prism", "prism_path", required=True, type=click.Path(path_type=Path, exists=True))
@click.option("--property", "property_name", required=True)
@click.option("--horizon", required=True, type=int)
@click.option("--const", "constants", multiple=True, help="K=V (repeatable)")
@click.option("--rubicon-cmd", default="rubicon")
@click.option("--dice-cmd", default="dice")
@click.option("--dice-extra-arg", "dice_extra_args", multiple=True, help="Repeatable; appended to the dice invocation")
@click.option("--workdir", default=None, type=click.Path(path_type=Path))
def main(prism_path, property_name, horizon, constants, rubicon_cmd, dice_cmd, dice_extra_args, workdir):
    prop = f'P=? [ F<={horizon} "{property_name}" ]'

    workdir_ctx = tempfile.TemporaryDirectory() if workdir is None else None
    work_root = Path(workdir_ctx.name) if workdir_ctx is not None else Path(workdir)
    work_root.mkdir(parents=True, exist_ok=True)
    dice_path = work_root / (prism_path.stem + ".dice")

    try:
        translate_cmd = [
            rubicon_cmd,
            "--prism", str(prism_path),
            "--prop", prop,
            "--output", str(dice_path),
        ]
        if constants:
            translate_cmd.extend(["--constants", _format_constants(constants)])
        _echo_command(translate_cmd)
        translate = subprocess.run(translate_cmd, capture_output=True, text=True, check=False)
        sys.stdout.write(translate.stdout)
        sys.stderr.write(translate.stderr)
        if translate.returncode != 0:
            raise SystemExit(f"rubicon translate failed (exit {translate.returncode})")

        dice_cmd_list = [dice_cmd, *dice_extra_args, "-json", str(dice_path)]
        _echo_command(dice_cmd_list)
        dice = subprocess.run(
            dice_cmd_list,
            capture_output=True, text=True, check=False,
        )
        sys.stdout.write(dice.stdout)
        sys.stderr.write(dice.stderr)
        if dice.returncode != 0:
            raise SystemExit(f"dice failed (exit {dice.returncode})")

        probability = _extract_probability(dice.stdout)
    finally:
        if workdir_ctx is not None:
            workdir_ctx.cleanup()

    # Sentinel newline: dice's JSON has no trailing newline, so without
    # this the probability would concatenate to the JSON line and the
    # parser would pick up a scalar from inside the JSON instead.
    sys.stdout.write("\n")
    print(f"{probability:.17g}")


if __name__ == "__main__":
    main()
