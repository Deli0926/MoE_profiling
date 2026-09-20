#!/usr/bin/env python3
"""
/mnt/data/plot_imbalance_ratio_p95.py

Compute and plot p95 of:
    Imbalance_ratio = max_num_routed_tokens_among_gpu / avg_num_routed_tokens_among_gpu

Data source:
- ./expert_selections/expert_selection_openai_{model_name}_wikitext_103_ep{ep}_bs{bs}_*_layer{layer}.jsonl

Assumption:
- One line == one forward
- Each expert index in the line corresponds to one routed token dispatched to the GPU
  that owns that expert
- If a token is routed to top-k experts, it contributes top-k routed-token dispatches in total
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt


DEFAULT_DIRECTORY = "./expert_selections"
DEFAULT_MODEL_NAME = "gpt-oss-20b"
DEFAULT_NUM_EXPERTS = 32
DEFAULT_EP_SIZE = 4
DEFAULT_OUTPUT_DIR = "./output/imbalance_ratio"


def flatten_expert_indices(data: list) -> list[int]:
    if len(data) > 0 and isinstance(data[0], list):
        return [idx for sublist in data for idx in sublist]
    return data


def percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute percentile of empty data.")
    if q <= 0:
        return float(sorted_values[0])
    if q >= 100:
        return float(sorted_values[-1])

    position = (len(sorted_values) - 1) * (q / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return float(sorted_values[lower])

    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def build_file_pattern(
    directory: str,
    model_name: str,
    ep_size: int,
    bs: int,
    layer: str,
) -> str:
    return os.path.join(
        directory,
        f"expert_selection_openai_{model_name}_wikitext_103_ep{ep_size}_bs{bs}_*_layer{layer}.jsonl",
    )


def discover_bs_values(
    directory: str,
    model_name: str,
    ep_size: int,
) -> list[int]:
    pattern = os.path.join(
        directory,
        f"expert_selection_openai_{model_name}_wikitext_103_ep{ep_size}_bs*_*.jsonl",
    )
    matched = glob.glob(pattern)
    bs_values = set()

    for path in matched:
        name = os.path.basename(path)
        parts = name.split("_")
        for part in parts:
            if part.startswith("bs"):
                try:
                    bs_values.add(int(part[2:]))
                except ValueError:
                    pass

    return sorted(bs_values)


def routed_tokens_per_gpu(
    expert_indices: Iterable[int],
    ep_size: int,
    num_experts: int,
) -> list[int]:
    experts_per_gpu = num_experts // ep_size
    counts = [0] * ep_size

    for idx in expert_indices:
        gpu_id = idx // experts_per_gpu
        if 0 <= gpu_id < ep_size:
            counts[gpu_id] += 1

    return counts


def imbalance_ratio_from_counts(gpu_counts: Sequence[int]) -> float:
    total = sum(gpu_counts)
    if total == 0:
        return 0.0
    avg = total / len(gpu_counts)
    return max(gpu_counts) / avg


def collect_ratios_for_bs(
    directory: str,
    model_name: str,
    ep_size: int,
    num_experts: int,
    bs: int,
    layer: str,
) -> tuple[list[float], list[str], int]:
    pattern = build_file_pattern(directory, model_name, ep_size, bs, layer)
    matched_files = sorted(glob.glob(pattern))

    ratios: list[float] = []
    total_forwards = 0

    for file_path in matched_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)
                expert_indices = flatten_expert_indices(data)
                gpu_counts = routed_tokens_per_gpu(
                    expert_indices=expert_indices,
                    ep_size=ep_size,
                    num_experts=num_experts,
                )
                ratios.append(imbalance_ratio_from_counts(gpu_counts))
                total_forwards += 1

    return ratios, matched_files, total_forwards


def summarize_ratios(ratios: Sequence[float]) -> Dict[str, float]:
    sorted_ratios = sorted(ratios)
    return {
        "count": float(len(sorted_ratios)),
        "mean": mean(sorted_ratios),
        "std": pstdev(sorted_ratios) if len(sorted_ratios) > 1 else 0.0,
        "min": sorted_ratios[0],
        "p50": percentile(sorted_ratios, 50),
        "p90": percentile(sorted_ratios, 90),
        "p95": percentile(sorted_ratios, 95),
        "p99": percentile(sorted_ratios, 99),
        "max": sorted_ratios[-1],
    }


def print_summary_table(rows: Sequence[Dict[str, float | int | str]]) -> None:
    widths = {
        "bs": 6,
        "files": 7,
        "forwards": 10,
        "mean": 10,
        "p50": 10,
        "p90": 10,
        "p95": 10,
        "p99": 10,
        "max": 10,
    }

    print("\n[Imbalance ratio summary]")
    print(
        f"{'bs':>{widths['bs']}} | "
        f"{'files':>{widths['files']}} | "
        f"{'forwards':>{widths['forwards']}} | "
        f"{'mean':>{widths['mean']}} | "
        f"{'p50':>{widths['p50']}} | "
        f"{'p90':>{widths['p90']}} | "
        f"{'p95':>{widths['p95']}} | "
        f"{'p99':>{widths['p99']}} | "
        f"{'max':>{widths['max']}}"
    )
    print(
        f"{'-' * widths['bs']}-+-"
        f"{'-' * widths['files']}-+-"
        f"{'-' * widths['forwards']}-+-"
        f"{'-' * widths['mean']}-+-"
        f"{'-' * widths['p50']}-+-"
        f"{'-' * widths['p90']}-+-"
        f"{'-' * widths['p95']}-+-"
        f"{'-' * widths['p99']}-+-"
        f"{'-' * widths['max']}"
    )

    for row in rows:
        print(
            f"{int(row['bs']):>{widths['bs']}d} | "
            f"{int(row['num_files']):>{widths['files']}d} | "
            f"{int(row['num_forwards']):>{widths['forwards']}d} | "
            f"{float(row['mean']):>{widths['mean']}.6f} | "
            f"{float(row['p50']):>{widths['p50']}.6f} | "
            f"{float(row['p90']):>{widths['p90']}.6f} | "
            f"{float(row['p95']):>{widths['p95']}.6f} | "
            f"{float(row['p99']):>{widths['p99']}.6f} | "
            f"{float(row['max']):>{widths['max']}.6f}"
        )


def write_summary_csv(path: Path, rows: Sequence[Dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "bs",
                "num_files",
                "num_forwards",
                "mean",
                "std",
                "min",
                "p50",
                "p90",
                "p95",
                "p99",
                "max",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "bs": row["bs"],
                    "num_files": row["num_files"],
                    "num_forwards": row["num_forwards"],
                    "mean": row["mean"],
                    "std": row["std"],
                    "min": row["min"],
                    "p50": row["p50"],
                    "p90": row["p90"],
                    "p95": row["p95"],
                    "p99": row["p99"],
                    "max": row["max"],
                }
            )


def plot_p95(rows: Sequence[Dict[str, float | int | str]], out_path: Path, ep_size: int, layer: str) -> None:
    xs = [int(row["bs"]) for row in rows]
    ys = [float(row["p95"]) for row in rows]

    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Batch size (bs)")
    plt.ylabel("p95 of imbalance ratio")
    plt.title(f"Imbalance ratio p95 vs bs (EP={ep_size}, layer={layer})")
    plt.xscale("log", base=2)
    plt.xticks(xs, [str(x) for x in xs])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_distribution_box(rows_by_bs: Dict[int, list[float]], out_path: Path, ep_size: int, layer: str) -> None:
    xs = sorted(rows_by_bs.keys())
    data = [rows_by_bs[bs] for bs in xs]

    plt.figure(figsize=(12, 6))
    plt.boxplot(data, tick_labels=[str(bs) for bs in xs], showfliers=False)
    plt.xlabel("Batch size (bs)")
    plt.ylabel("Imbalance ratio")
    plt.title(f"Imbalance ratio distribution by bs (EP={ep_size}, layer={layer})")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute and plot p95 of imbalance ratio from expert routing jsonl files."
    )
    parser.add_argument("--directory", default=DEFAULT_DIRECTORY, help="Directory containing expert selection jsonl files.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Model name used in file names.")
    parser.add_argument("--ep-size", type=int, default=DEFAULT_EP_SIZE, help="EP degree.")
    parser.add_argument("--num-experts", type=int, default=DEFAULT_NUM_EXPERTS, help="Total number of experts.")
    parser.add_argument("--layer", default="*", help="Layer number (0~23) or * for all layers.")
    parser.add_argument(
        "--bs-list",
        default="",
        help="Comma-separated batch sizes to analyze. If empty, bs values are auto-discovered.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    args = parser.parse_args()

    if args.num_experts % args.ep_size != 0:
        raise ValueError("num_experts must be divisible by ep_size.")

    if args.bs_list.strip():
        bs_values = [int(x.strip()) for x in args.bs_list.split(",") if x.strip()]
    else:
        bs_values = discover_bs_values(
            directory=args.directory,
            model_name=args.model_name,
            ep_size=args.ep_size,
        )

    if not bs_values:
        raise RuntimeError("No bs values found. Check --directory, --model-name, and --ep-size.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[Dict[str, float | int | str]] = []
    ratios_by_bs: Dict[int, list[float]] = {}

    print("=" * 80)
    print("Imbalance ratio analysis")
    print("=" * 80)
    print(f"directory   : {args.directory}")
    print(f"model_name  : {args.model_name}")
    print(f"ep_size     : {args.ep_size}")
    print(f"num_experts : {args.num_experts}")
    print(f"layer       : {args.layer}")
    print(f"bs_values   : {bs_values}")
    print("=" * 80)

    for bs in bs_values:
        ratios, matched_files, total_forwards = collect_ratios_for_bs(
            directory=args.directory,
            model_name=args.model_name,
            ep_size=args.ep_size,
            num_experts=args.num_experts,
            bs=bs,
            layer=args.layer,
        )

        if not ratios:
            print(f"[skip] bs={bs}: no matched data")
            continue

        ratios_by_bs[bs] = ratios
        stats = summarize_ratios(ratios)

        summary_rows.append(
            {
                "bs": bs,
                "num_files": len(matched_files),
                "num_forwards": total_forwards,
                "mean": stats["mean"],
                "std": stats["std"],
                "min": stats["min"],
                "p50": stats["p50"],
                "p90": stats["p90"],
                "p95": stats["p95"],
                "p99": stats["p99"],
                "max": stats["max"],
            }
        )

        print(f"[done] bs={bs}: files={len(matched_files)}, forwards={total_forwards}, p95={stats['p95']:.6f}")

    if not summary_rows:
        raise RuntimeError("No valid data found for any bs.")

    summary_rows.sort(key=lambda row: int(row["bs"]))
    print_summary_table(summary_rows)

    layer_str = "all" if args.layer == "*" else str(args.layer)

    summary_csv_path = output_dir / f"imbalance_ratio_summary_ep{args.ep_size}_layer{layer_str}.csv"
    write_summary_csv(summary_csv_path, summary_rows)
    print(f"\n[saved] {summary_csv_path}")

    p95_plot_path = output_dir / f"imbalance_ratio_p95_ep{args.ep_size}_layer{layer_str}.png"
    plot_p95(summary_rows, p95_plot_path, args.ep_size, args.layer)
    print(f"[saved] {p95_plot_path}")

    box_plot_path = output_dir / f"imbalance_ratio_box_ep{args.ep_size}_layer{layer_str}.png"
    plot_distribution_box(ratios_by_bs, box_plot_path, args.ep_size, args.layer)
    print(f"[saved] {box_plot_path}")


if __name__ == "__main__":
    main()
