#!/usr/bin/env python3
"""
/mnt/data/predict_imbalance_ratio_from_bs1.py

Predict imbalance ratio for larger batch sizes using only bs=1 data.

Model:
- Read bs=1 expert-selection files
- Convert each token's top-k expert choices into a GPU dispatch-count vector
- Estimate the empirical token-level distribution from bs=1
- For each target bs, simulate a forward by summing bs iid token vectors
- Compute imbalance ratio = max_gpu_routed_tokens / avg_gpu_routed_tokens
- Report p50/p90/p95/p99 and plot p95 vs bs

This is a prediction script, not a real-measurement script.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DIRECTORY = "./expert_selections"
DEFAULT_MODEL_NAME = "gpt-oss-20b"
DEFAULT_NUM_EXPERTS = 32
DEFAULT_TOP_K = 4
DEFAULT_EP_SIZE = 4
DEFAULT_NUM_LAYERS = 24
DEFAULT_SOURCE_BS = 1
DEFAULT_BS_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256]
DEFAULT_NUM_SAMPLES = 20000
DEFAULT_SEED = 42
DEFAULT_OUTPUT_DIR = "./output/imbalance_ratio_prediction"


def percentile(values: Sequence[float], q: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, q))


def flatten_to_token_groups(data: list, top_k: int) -> list[list[int]]:
    if not data:
        return []
    if isinstance(data[0], list):
        return [list(group) for group in data]
    return [list(data[i:i + top_k]) for i in range(0, len(data), top_k)]


def expert_to_gpu(expert_idx: int, ep_size: int, num_experts: int) -> int:
    experts_per_gpu = num_experts // ep_size
    return expert_idx // experts_per_gpu


def token_group_to_gpu_count_vector(
    token_group: Sequence[int],
    ep_size: int,
    num_experts: int,
) -> tuple[int, ...]:
    counts = [0] * ep_size
    for expert_idx in token_group:
        gpu_id = expert_to_gpu(expert_idx, ep_size, num_experts)
        if 0 <= gpu_id < ep_size:
            counts[gpu_id] += 1
    return tuple(counts)


def collect_token_gpu_vectors_from_bs1(
    directory: str,
    model_name: str,
    ep_size: int,
    num_experts: int,
    top_k: int,
    source_bs: int,
    num_layers: int,
) -> tuple[np.ndarray, list[str], int]:
    token_vectors: list[tuple[int, ...]] = []
    matched_files_all: list[str] = []

    for layer in range(num_layers):
        pattern = os.path.join(
            directory,
            f"expert_selection_openai_{model_name}_wikitext_103_ep{ep_size}_bs{source_bs}_*_layer{layer}.jsonl",
        )
        matched_files = sorted(glob.glob(pattern))
        if not matched_files:
            continue

        file_path = matched_files[0]
        matched_files_all.append(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)
                token_groups = flatten_to_token_groups(data, top_k)

                for token_group in token_groups:
                    if len(token_group) != top_k:
                        continue
                    token_vectors.append(
                        token_group_to_gpu_count_vector(
                            token_group=token_group,
                            ep_size=ep_size,
                            num_experts=num_experts,
                        )
                    )

    if not token_vectors:
        raise RuntimeError("No token groups found in bs=1 files.")

    return np.asarray(token_vectors, dtype=np.int16), matched_files_all, len(token_vectors)


def summarize_token_vector_distribution(token_vectors: np.ndarray) -> list[tuple[tuple[int, ...], int, float]]:
    counter = Counter(map(tuple, token_vectors.tolist()))
    total = sum(counter.values())
    rows = []
    for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        rows.append((key, count, count / total))
    return rows


def simulate_imbalance_ratios(
    token_vectors: np.ndarray,
    bs: int,
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    num_token_patterns, ep_size = token_vectors.shape
    sampled_idx = rng.integers(0, num_token_patterns, size=(num_samples, bs))
    sampled_vectors = token_vectors[sampled_idx]
    gpu_counts = sampled_vectors.sum(axis=1)
    max_counts = gpu_counts.max(axis=1)
    avg_counts = gpu_counts.mean(axis=1)
    ratios = max_counts / avg_counts
    return ratios.astype(np.float64)


def summarize_ratios(ratios: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(ratios.mean()),
        "std": float(ratios.std()),
        "min": float(ratios.min()),
        "p50": percentile(ratios, 50),
        "p90": percentile(ratios, 90),
        "p95": percentile(ratios, 95),
        "p99": percentile(ratios, 99),
        "max": float(ratios.max()),
    }


def print_token_pattern_table(rows: Sequence[tuple[tuple[int, ...], int, float]], limit: int) -> None:
    print("\n[Top token GPU-dispatch patterns from bs=1]")
    print(" pattern(gpu0,gpu1,...) |    count |     prob")
    print("------------------------+----------+----------")
    for pattern, count, prob in rows[:limit]:
        print(f" {str(pattern):>22} | {count:>8d} | {prob:>8.6f}")


def print_summary_table(rows: Sequence[dict[str, float | int]]) -> None:
    print("\n[Predicted imbalance ratio summary]")
    print("    bs |    mean |     p50 |     p90 |     p95 |     p99 |     max")
    print("-------+---------+---------+---------+---------+---------+---------")
    for row in rows:
        print(
            f"{int(row['bs']):>6d} | "
            f"{float(row['mean']):>7.5f} | "
            f"{float(row['p50']):>7.5f} | "
            f"{float(row['p90']):>7.5f} | "
            f"{float(row['p95']):>7.5f} | "
            f"{float(row['p99']):>7.5f} | "
            f"{float(row['max']):>7.5f}"
        )


def write_summary_csv(path: Path, rows: Sequence[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["bs", "mean", "std", "min", "p50", "p90", "p95", "p99", "max"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_pattern_csv(
    path: Path,
    rows: Sequence[tuple[tuple[int, ...], int, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gpu_dispatch_pattern", "count", "probability"])
        for pattern, count, prob in rows:
            writer.writerow([str(pattern), count, prob])


def plot_p95(rows: Sequence[dict[str, float | int]], out_path: Path, ep_size: int) -> None:
    xs = [int(row["bs"]) for row in rows]
    ys = [float(row["p95"]) for row in rows]

    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Batch size (bs)")
    plt.ylabel("Predicted p95 of imbalance ratio")
    plt.title(f"Predicted imbalance ratio p95 vs bs from bs=1 data (EP={ep_size})")
    plt.xscale("log", base=2)
    plt.xticks(xs, [str(x) for x in xs])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_distribution_box(
    ratios_by_bs: dict[int, np.ndarray],
    out_path: Path,
    ep_size: int,
) -> None:
    xs = sorted(ratios_by_bs.keys())
    data = [ratios_by_bs[bs] for bs in xs]

    plt.figure(figsize=(12, 6))
    plt.boxplot(data, tick_labels=[str(bs) for bs in xs], showfliers=False)
    plt.xlabel("Batch size (bs)")
    plt.ylabel("Predicted imbalance ratio")
    plt.title(f"Predicted imbalance ratio distribution by bs from bs=1 data (EP={ep_size})")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict imbalance ratio for larger batch sizes using only bs=1 data."
    )
    parser.add_argument("--directory", default=DEFAULT_DIRECTORY)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--ep-size", type=int, default=DEFAULT_EP_SIZE)
    parser.add_argument("--num-experts", type=int, default=DEFAULT_NUM_EXPERTS)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--source-bs", type=int, default=DEFAULT_SOURCE_BS)
    parser.add_argument(
        "--bs-list",
        default=",".join(str(x) for x in DEFAULT_BS_LIST),
        help="Comma-separated analysis batch sizes.",
    )
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pattern-topn", type=int, default=12)
    args = parser.parse_args()

    if args.num_experts % args.ep_size != 0:
        raise ValueError("num_experts must be divisible by ep_size.")

    bs_list = [int(x.strip()) for x in args.bs_list.split(",") if x.strip()]
    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)

    print("=" * 80)
    print("Predict imbalance ratio from bs=1 data")
    print("=" * 80)
    print(f"directory    : {args.directory}")
    print(f"model_name   : {args.model_name}")
    print(f"ep_size      : {args.ep_size}")
    print(f"num_experts  : {args.num_experts}")
    print(f"top_k        : {args.top_k}")
    print(f"source_bs    : {args.source_bs}")
    print(f"bs_list      : {bs_list}")
    print(f"num_samples  : {args.num_samples}")
    print(f"seed         : {args.seed}")
    print("=" * 80)

    token_vectors, matched_files, num_tokens = collect_token_gpu_vectors_from_bs1(
        directory=args.directory,
        model_name=args.model_name,
        ep_size=args.ep_size,
        num_experts=args.num_experts,
        top_k=args.top_k,
        source_bs=args.source_bs,
        num_layers=args.num_layers,
    )

    print(f"\n[bs=1 source files] {len(matched_files)} files")
    for file_path in matched_files:
        print(f"  - {file_path}")
    print(f"[collected token groups] {num_tokens}")

    pattern_rows = summarize_token_vector_distribution(token_vectors)
    print_token_pattern_table(pattern_rows, args.pattern_topn)

    summary_rows: list[dict[str, float | int]] = []
    ratios_by_bs: dict[int, np.ndarray] = {}

    for bs in bs_list:
        ratios = simulate_imbalance_ratios(
            token_vectors=token_vectors,
            bs=bs,
            num_samples=args.num_samples,
            rng=rng,
        )
        ratios_by_bs[bs] = ratios
        stats = summarize_ratios(ratios)
        row = {"bs": bs, **stats}
        summary_rows.append(row)
        print(f"[done] bs={bs}: p95={stats['p95']:.6f}, mean={stats['mean']:.6f}")

    print_summary_table(summary_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = output_dir / f"predicted_imbalance_ratio_summary_ep{args.ep_size}_from_bs{args.source_bs}.csv"
    pattern_csv_path = output_dir / f"token_gpu_dispatch_pattern_ep{args.ep_size}_from_bs{args.source_bs}.csv"
    p95_plot_path = output_dir / f"predicted_imbalance_ratio_p95_ep{args.ep_size}_from_bs{args.source_bs}.png"
    box_plot_path = output_dir / f"predicted_imbalance_ratio_box_ep{args.ep_size}_from_bs{args.source_bs}.png"

    write_summary_csv(summary_csv_path, summary_rows)
    write_pattern_csv(pattern_csv_path, pattern_rows)
    plot_p95(summary_rows, p95_plot_path, args.ep_size)
    plot_distribution_box(ratios_by_bs, box_plot_path, args.ep_size)

    print(f"\n[saved] {summary_csv_path}")
    print(f"[saved] {pattern_csv_path}")
    print(f"[saved] {p95_plot_path}")
    print(f"[saved] {box_plot_path}")


if __name__ == "__main__":
    main()
