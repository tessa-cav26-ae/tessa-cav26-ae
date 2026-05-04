"""Verify benchmark results: cross-check tessa vs storm probabilities."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

import click

from .log_config import setup_logging

logger = logging.getLogger(__name__)


def _read_runner_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _probability_map(csv_path: Path) -> dict[str, float]:
    """Build {case_id: probability} from a benchmark runner CSV."""
    result: dict[str, float] = {}
    for row in _read_runner_csv(csv_path):
        if row.get("status") == "ok" and row.get("probability"):
            try:
                result[row["case_id"]] = float(row["probability"])
            except (ValueError, TypeError):
                pass
    return result


@click.group()
@click.option("--log-file", default=None, type=click.Path(path_type=Path), help="Path to write detailed logs")
@click.option(
    "--log-console-level",
    default="INFO",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
)
@click.option(
    "--log-file-level",
    default="DEBUG",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
)
@click.pass_context
def cli(
    ctx,
    log_file: Optional[Path],
    log_console_level: str,
    log_file_level: str,
):
    """Verify benchmark results."""
    setup_logging(log_console_level, log_file_level, log_file)


@cli.command("verify")
@click.option("--tessa-csv", type=click.Path(path_type=Path, exists=True), required=True, multiple=True,
              help="One or more tessa CSVs to verify")
@click.option("--storm-csv", type=click.Path(path_type=Path, exists=True), required=True, multiple=True,
              help="One or more storm CSVs (dd, sparse) to compare against")
@click.option("--rubicon-csv", type=click.Path(path_type=Path, exists=True), default=(), multiple=True,
              help="One or more rubicon CSVs to compare against (compared the same way as storm CSVs)")
@click.option("--geni-csv", type=click.Path(path_type=Path, exists=True), default=(), multiple=True,
              help="One or more geni CSVs to compare against (compared the same way as rubicon CSVs)")
@click.option("--atol", type=float, default=1e-5, help="Absolute tolerance for probability comparison")
@click.option("--rtol", type=float, default=1e-4, help="Relative tolerance for probability comparison")
@click.option("--output-csv", type=click.Path(path_type=Path), default=None,
              help="Write per-case verification results to a CSV file")
def verify_cmd(tessa_csv, storm_csv, rubicon_csv, geni_csv, atol, rtol, output_csv):
    """Verify that tessa and storm/rubicon/geni compute the same probabilities."""
    # All non-tessa runners are treated as baselines: each baseline CSV is
    # cross-checked against every tessa CSV. The CSV's tool field is captured
    # so output rows distinguish storm/rubicon/geni comparisons.
    baseline_csvs = list(storm_csv) + list(rubicon_csv) + list(geni_csv)
    all_ok = True
    total_checked = 0
    csv_rows: list[dict[str, str]] = []

    for tessa_path in tessa_csv:
        tessa_probs = _probability_map(tessa_path)
        tessa_name = Path(tessa_path).name
        logger.info("Tessa file: %s (%d cases)", tessa_path, len(tessa_probs))
        if not tessa_probs:
            logger.warning("  No valid probabilities in %s — skipping", tessa_path)
            continue

        for storm_path in baseline_csvs:
            storm_probs = _probability_map(storm_path)
            storm_name = Path(storm_path).name
            common = sorted(set(tessa_probs) & set(storm_probs))
            if not common:
                logger.info("  vs %s: no common cases to compare", storm_path)
                continue

            mismatches = []
            for cid in common:
                tp, sp = tessa_probs[cid], storm_probs[cid]
                diff = abs(tp - sp)
                match = diff <= atol + rtol * abs(sp)
                status = "ok" if match else "MISMATCH"
                if not match:
                    mismatches.append((cid, tp, sp, diff))
                total_checked += 1
                rel_diff_pct = (diff / abs(sp) * 100) if sp != 0 else float("inf")
                csv_rows.append({
                    "tessa_file": tessa_name,
                    "storm_file": storm_name,
                    "case_id": cid,
                    "tessa_probability": f"{tp:.8e}",
                    "storm_probability": f"{sp:.8e}",
                    "abs_diff": f"{diff:.2e}",
                    "rel_diff": f"{rel_diff_pct:.3f}%",
                    "status": status,
                })

            if mismatches:
                all_ok = False
                logger.info("  vs %s: %d MISMATCH(es) out of %d:", storm_path, len(mismatches), len(common))
                for cid, tp, sp, diff in mismatches[:10]:
                    logger.info("    %s: tessa=%.8e  storm=%.8e  diff=%.2e", cid, tp, sp, diff)
                if len(mismatches) > 10:
                    logger.info("    ... and %d more", len(mismatches) - 10)
            else:
                logger.info("  vs %s: %d cases match (atol=%s, rtol=%s)", storm_path, len(common), atol, rtol)

    if total_checked == 0:
        raise click.ClickException("No cases were compared — check that tessa and storm produced results")

    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["tessa_file", "storm_file", "case_id", "tessa_probability",
                      "storm_probability", "abs_diff", "rel_diff", "status"]
        with output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        logger.info("Wrote %d rows to %s", len(csv_rows), output_csv)

    if all_ok:
        logger.info("Verified %d probability comparisons OK", total_checked)

    if not all_ok:
        raise click.ClickException("Probability verification FAILED")


def main():
    cli()


if __name__ == "__main__":
    main()
