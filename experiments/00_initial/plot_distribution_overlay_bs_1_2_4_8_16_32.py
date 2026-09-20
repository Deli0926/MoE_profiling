#!/usr/bin/env python3
"""
/mnt/data/plot_distribution_overlay_bs_1_2_4_8_16_32.py

Reads saved comparison CSV files and creates a single overlay plot for:
bs = 1, 2, 4, 8, 16, 32

Style:
- real   : solid line
- theory : dashed line
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


TARGET_BS = [1, 2, 4, 8, 16, 32]


def read_summary_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_comparison_csv(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "m": int(row["m"]),
                    "theory_prob": float(row["theory_prob"]),
                    "real_prob": float(row["real_prob"]),
                    "abs_error": float(row["abs_error"]),
                }
            )
    return rows


def build_summary_map(summary_rows: List[Dict[str, str]]) -> Dict[int, Dict[str, str]]:
    result: Dict[int, Dict[str, str]] = {}
    for row in summary_rows:
        try:
            bs = int(row["analysis_bs"])
        except (KeyError, ValueError):
            continue
        result[bs] = row
    return result


def plot_overlay(
    input_dir: Path,
    output_path: Path,
    hide_all_zero_rows: bool,
) -> None:
    summary_path = input_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.csv not found: {summary_path}")

    summary_rows = read_summary_csv(summary_path)
    summary_map = build_summary_map(summary_rows)

    plt.figure(figsize=(12, 8))

    plotted_any = False

    for bs in TARGET_BS:
        if bs not in summary_map:
            print(f"[skip] bs={bs}: not found in summary.csv")
            continue

        row = summary_map[bs]
        status = row.get("status", "ok")
        if status and status != "ok":
            print(f"[skip] bs={bs}: status={status}, message={row.get('message', '')}")
            continue

        comparison_csv = row.get("comparison_csv", "").strip()
        if not comparison_csv:
            print(f"[skip] bs={bs}: empty comparison_csv")
            continue

        comparison_path = Path(comparison_csv)
        if not comparison_path.is_absolute():
            comparison_path = Path(comparison_csv)

        if not comparison_path.exists():
            alt_path = input_dir / Path(comparison_csv).name
            if alt_path.exists():
                comparison_path = alt_path
            else:
                print(f"[skip] bs={bs}: comparison csv not found: {comparison_csv}")
                continue

        rows = read_comparison_csv(comparison_path)
        if hide_all_zero_rows:
            filtered = [
                row for row in rows
                if row["theory_prob"] != 0.0 or row["real_prob"] != 0.0
            ]
            rows = filtered or rows

        xs = [row["m"] for row in rows]
        theory = [row["theory_prob"] for row in rows]
        real = [row["real_prob"] for row in rows]

        plt.plot(xs, real, linestyle="-", marker="o", label=f"real bs={bs}")
        plt.plot(xs, theory, linestyle="--", marker="x", label=f"theory bs={bs}")
        plotted_any = True

    if not plotted_any:
        raise RuntimeError("No valid bs series were plotted.")

    plt.xlabel("m")
    plt.ylabel("Probability")
    plt.title("Theory vs Real Probability Distributions (EP=4)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"[saved] {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay theory(real dashed) vs real(solid) distributions for bs=1,2,4,8,16,32."
    )
    parser.add_argument(
        "--input-dir",
        default="./output/distribution_compare_ep4",
        help="Directory containing summary.csv and comparison CSV files.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output PNG path. Defaults to <input-dir>/plots/distribution_overlay_bs_1_2_4_8_16_32.png",
    )
    parser.add_argument(
        "--show-zero-rows",
        action="store_true",
        help="Include rows where both theory and real are 0.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_path = (
        Path(args.output)
        if args.output
        else input_dir / "plots" / "distribution_overlay_bs_1_2_4_8_16_32.png"
    )

    plot_overlay(
        input_dir=input_dir,
        output_path=output_path,
        hide_all_zero_rows=not args.show_zero_rows,
    )


if __name__ == "__main__":
    main()
