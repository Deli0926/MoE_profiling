#!/usr/bin/env python3

import csv
import os

import matplotlib.pyplot as plt

from plot_ep_pp_tradeoff_saddlepoint import (
    DEFAULT_GLOBAL_BS,
    DEFAULT_SOURCE_BS,
    DEFAULT_SOURCE_EP,
    NUM_EXPERTS,
    predict_distribution_for_ep_pp,
)


OUTPUT_DIR = "./output/ep_pp_fixed_budget_saddlepoint"


def valid_ep_pp_pairs(fixed_product):
    pairs = []
    for ep in range(1, fixed_product + 1):
        if fixed_product % ep != 0:
            continue
        pp = fixed_product // ep
        if NUM_EXPERTS % ep != 0:
            continue
        pairs.append((ep, pp))
    return pairs


def write_summary_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(
            outfile,
            fieldnames=[
                "fixed_product",
                "target_ep",
                "pp_degree",
                "global_bs",
                "local_bs",
                "source_ep",
                "source_bs",
                "expected_max",
                "p95",
                "imbalance_ratio",
                "avg_load",
                "valid_layers",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_distribution_overlay(results, out_path, fixed_product, global_bs):
    plt.figure(figsize=(11, 7))

    for result in sorted(results, key=lambda item: item["ep"]):
        xs = [idx for idx, prob in enumerate(result["distribution"]) if prob > 0.0]
        ys = [result["distribution"][idx] for idx in xs]
        label = f"EP={result['ep']}, PP={result['pp']}, local bs={result['local_bs']}"
        plt.plot(xs, ys, marker="o", label=label)

    plt.xlabel("Max routed tokens per GPU")
    plt.ylabel("Probability")
    plt.title(f"Distribution Overlay with Fixed EP x PP = {fixed_product} (global bs={global_bs})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_metric(results, metric, ylabel, out_path, fixed_product, global_bs):
    ordered = sorted(results, key=lambda item: item["ep"])
    xs = [result["ep"] for result in ordered]
    ys = [result[metric] for result in ordered]
    labels = [f"PP={result['pp']}" for result in ordered]

    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, marker="o")
    for x, y, label in zip(xs, ys, labels):
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 6), ha="center")
    plt.xlabel("EP Degree")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} with Fixed EP x PP = {fixed_product} (global bs={global_bs})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    global_bs_raw = input(f"global batch size 입력 (기본 {DEFAULT_GLOBAL_BS}): ").strip()
    fixed_product_raw = input("고정할 EP x PP 값 입력 (예: 16): ").strip()
    source_ep_raw = input(f"source 로그의 EP Degree 입력 (기본 {DEFAULT_SOURCE_EP}): ").strip()
    source_bs_raw = input(f"source 로그의 batch size 입력 (기본 {DEFAULT_SOURCE_BS}): ").strip()

    global_bs = int(global_bs_raw) if global_bs_raw else DEFAULT_GLOBAL_BS
    fixed_product = int(fixed_product_raw)
    source_ep = int(source_ep_raw) if source_ep_raw else DEFAULT_SOURCE_EP
    source_bs = int(source_bs_raw) if source_bs_raw else DEFAULT_SOURCE_BS

    pairs = valid_ep_pp_pairs(fixed_product)
    if not pairs:
        raise ValueError(
            f"EP x PP = {fixed_product} 조건을 만족하면서 NUM_EXPERTS={NUM_EXPERTS}를 나누는 EP가 없습니다."
        )

    print(
        "EP x PP를 고정한 상태에서 saddlepoint 근사 분포를 계산 중...\n"
        f"(global bs={global_bs}, fixed product={fixed_product}, source ep={source_ep}, source bs={source_bs})"
    )

    results = []
    summary_rows = []

    for ep, pp in pairs:
        result = predict_distribution_for_ep_pp(
            target_ep=ep,
            global_bs=global_bs,
            pp_degree=pp,
            source_ep=source_ep,
            source_bs=source_bs,
        )
        results.append(result)
        summary_rows.append({
            "fixed_product": fixed_product,
            "target_ep": ep,
            "pp_degree": pp,
            "global_bs": global_bs,
            "local_bs": result["local_bs"],
            "source_ep": source_ep,
            "source_bs": source_bs,
            "expected_max": result["expected_max"],
            "p95": result["p95"],
            "imbalance_ratio": result["imbalance_ratio"],
            "avg_load": result["avg_load"],
            "valid_layers": result["valid_layers"],
        })

        print(
            f"EP={ep:2d}, PP={pp:2d} | "
            f"local bs={result['local_bs']:2d} | "
            f"E[X]={result['expected_max']:.6f} | "
            f"p95={result['p95']:.6f} | "
            f"ratio={result['imbalance_ratio']:.6f}"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_name = f"globalbs{global_bs}_fixedproduct{fixed_product}_source_ep{source_ep}_bs{source_bs}"

    summary_csv = os.path.join(OUTPUT_DIR, f"{base_name}_summary.csv")
    distribution_png = os.path.join(OUTPUT_DIR, f"{base_name}_distribution_overlay.png")
    expected_png = os.path.join(OUTPUT_DIR, f"{base_name}_expected_max.png")
    p95_png = os.path.join(OUTPUT_DIR, f"{base_name}_p95.png")
    ratio_png = os.path.join(OUTPUT_DIR, f"{base_name}_imbalance_ratio.png")

    write_summary_csv(summary_csv, summary_rows)
    plot_distribution_overlay(results, distribution_png, fixed_product, global_bs)
    plot_metric(results, "expected_max", "Expected max routed tokens per GPU", expected_png, fixed_product, global_bs)
    plot_metric(results, "p95", "p95 of max routed tokens per GPU", p95_png, fixed_product, global_bs)
    plot_metric(results, "imbalance_ratio", "Imbalance Ratio", ratio_png, fixed_product, global_bs)

    print("\n완료")
    print(f"summary csv: {summary_csv}")
    print(f"distribution overlay: {distribution_png}")
    print(f"expected max plot: {expected_png}")
    print(f"p95 plot: {p95_png}")
    print(f"imbalance ratio plot: {ratio_png}")


if __name__ == "__main__":
    main()
