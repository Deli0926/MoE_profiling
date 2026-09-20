#!/usr/bin/env python3

import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from plot_actual_vs_gumbel_loadaware import (
    DEFAULT_NUM_SAMPLES,
    DEFAULT_SEED,
    DEFAULT_SOURCE_BS,
    DEFAULT_SOURCE_EP,
    NUM_EXPERTS,
    NUM_LAYERS,
    average_pmfs,
    expected_value_from_pmf,
    find_layer_file,
    load_token_expert_array,
    quantized_pmf,
    sample_loadaware_scores,
)


DEFAULT_EP_SIZE = 4
DEFAULT_BATCH_SIZE = 32
DEFAULT_BIN_DECIMALS = 3
DEFAULT_SCORE_MODE = "combined"
DEFAULT_P_EFF_MODE = "constant"

OUTPUT_DIR = "./output/predicted_gumbel_peff_sweep"


def predict_distribution_predicted_only(
    ep_size,
    batch_size,
    source_ep,
    source_bs,
    num_samples,
    seed,
    base_p_eff,
    p_eff_mode,
    score_mode,
    bin_decimals,
):
    layer_pmfs = []

    for layer in range(NUM_LAYERS):
        source_file = find_layer_file(source_ep, source_bs, layer)
        if not source_file:
            continue

        token_expert_array = load_token_expert_array(source_file)
        if len(token_expert_array) == 0:
            continue

        rng = np.random.default_rng(seed + layer * 1237 + ep_size * 997 + batch_size * 131)
        pred_scores = sample_loadaware_scores(
            token_expert_array=token_expert_array,
            ep_size=ep_size,
            batch_size=batch_size,
            num_samples=num_samples,
            rng=rng,
            base_p_eff=base_p_eff,
            p_eff_mode=p_eff_mode,
            score_mode=score_mode,
        )
        layer_pmf = quantized_pmf(pred_scores, bin_decimals)
        if layer_pmf:
            layer_pmfs.append(layer_pmf)

    if not layer_pmfs:
        raise RuntimeError(f"No valid layers found for source_ep={source_ep}, source_bs={source_bs}")

    return average_pmfs(layer_pmfs)


def main():
    ep_raw = input(f"EP Degree 입력 (기본 {DEFAULT_EP_SIZE}): ").strip()
    batch_size_raw = input(f"batch size 입력 (기본 {DEFAULT_BATCH_SIZE}): ").strip()
    source_ep_raw = input(f"source 로그의 EP Degree 입력 (기본 {DEFAULT_SOURCE_EP}): ").strip()
    source_bs_raw = input(f"source 로그의 batch size 입력 (기본 {DEFAULT_SOURCE_BS}): ").strip()
    num_samples_raw = input(f"Gumbel fit용 샘플 수 입력 (기본 {DEFAULT_NUM_SAMPLES}): ").strip()
    seed_raw = input(f"random seed 입력 (기본 {DEFAULT_SEED}): ").strip()
    score_mode_raw = input(
        f"X 모드 입력 [capacity_only/combined] (기본 {DEFAULT_SCORE_MODE}): "
    ).strip()
    decimals_raw = input(f"PMF bin 소수점 자리수 입력 (기본 {DEFAULT_BIN_DECIMALS}): ").strip()

    ep_size = int(ep_raw) if ep_raw else DEFAULT_EP_SIZE
    batch_size = int(batch_size_raw) if batch_size_raw else DEFAULT_BATCH_SIZE
    source_ep = int(source_ep_raw) if source_ep_raw else DEFAULT_SOURCE_EP
    source_bs = int(source_bs_raw) if source_bs_raw else DEFAULT_SOURCE_BS
    num_samples = int(num_samples_raw) if num_samples_raw else DEFAULT_NUM_SAMPLES
    seed = int(seed_raw) if seed_raw else DEFAULT_SEED
    score_mode = score_mode_raw if score_mode_raw else DEFAULT_SCORE_MODE
    bin_decimals = int(decimals_raw) if decimals_raw else DEFAULT_BIN_DECIMALS

    if NUM_EXPERTS % ep_size != 0:
        raise ValueError(f"NUM_EXPERTS={NUM_EXPERTS} must be divisible by ep_size={ep_size}")

    max_p_eff = NUM_EXPERTS // ep_size
    p_eff_values = list(range(max_p_eff, 0, -1))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stem = (
        f"predicted_peff_sweep_ep{ep_size}_bs{batch_size}"
        f"_source_ep{source_ep}_bs{source_bs}"
        f"_score{score_mode}"
    )
    plot_path = os.path.join(OUTPUT_DIR, f"{stem}.png")
    expected_plot_path = os.path.join(OUTPUT_DIR, f"{stem}_expected_x.png")
    summary_path = os.path.join(OUTPUT_DIR, f"{stem}_summary.csv")

    results = []
    all_xs = set()

    for p_eff in p_eff_values:
        pmf = predict_distribution_predicted_only(
            ep_size=ep_size,
            batch_size=batch_size,
            source_ep=source_ep,
            source_bs=source_bs,
            num_samples=num_samples,
            seed=seed,
            base_p_eff=float(p_eff),
            p_eff_mode=DEFAULT_P_EFF_MODE,
            score_mode=score_mode,
            bin_decimals=bin_decimals,
        )
        xs = sorted(pmf.keys())
        all_xs.update(xs)
        results.append({
            "p_eff": p_eff,
            "pmf": pmf,
            "expected_x": expected_value_from_pmf(pmf),
        })

    ordered_xs = sorted(all_xs)

    plt.figure(figsize=(12, 8))
    for item in results:
        ys = [item["pmf"].get(x, 0.0) for x in ordered_xs]
        plt.plot(ordered_xs, ys, linewidth=2.0, alpha=0.9, label=f"P_eff={item['p_eff']}")
    plt.xlabel("X")
    plt.ylabel("Predicted Probability")
    plt.title(
        "Predicted Distribution While Sweeping P_eff\n"
        f"(EP={ep_size}, bs={batch_size}, source bs={source_bs}, score={score_mode})"
    )
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(
        [item["p_eff"] for item in results],
        [item["expected_x"] for item in results],
        linewidth=2.0,
    )
    plt.xlabel("P_eff")
    plt.ylabel("Predicted Expected X")
    plt.title(
        "Predicted Expected X While Sweeping P_eff\n"
        f"(EP={ep_size}, bs={batch_size}, source bs={source_bs}, score={score_mode})"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(expected_plot_path, dpi=150)
    plt.close()

    with open(summary_path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(
            outfile,
            fieldnames=["p_eff", "expected_x"],
        )
        writer.writeheader()
        for item in results:
            writer.writerow({
                "p_eff": item["p_eff"],
                "expected_x": item["expected_x"],
            })

    print("완료")
    for item in results:
        print(f"P_eff={item['p_eff']}: predicted expected X = {item['expected_x']:.6f}")
    print(f"distribution plot: {plot_path}")
    print(f"expected-x plot: {expected_plot_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
