#!/usr/bin/env python3

import csv
import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gumbel_r

from plot_actual_vs_gumbel import (
    DEFAULT_NUM_SAMPLES,
    DEFAULT_SEED,
    DEFAULT_SOURCE_BS,
    DEFAULT_SOURCE_EP,
    DIRECTORY,
    MODEL_NAME,
    NUM_EXPERTS,
    NUM_LAYERS,
    TOP_K,
    build_token_pattern_pmf,
    extract_token_expert_lists,
    find_layer_file,
    sample_maxima_from_pattern_pmf,
)


OUTPUT_DIR = "./output/actual_vs_gumbel_spike_removed"
TARGET_LAYERS = [0, 10, 23]


def expected_value_from_dist(dist):
    return sum(idx * prob for idx, prob in enumerate(dist))


def tvd_from_dists(dist_a, dist_b):
    size = max(len(dist_a), len(dist_b))
    a = dist_a + [0.0] * (size - len(dist_a))
    b = dist_b + [0.0] * (size - len(dist_b))
    return 0.5 * sum(abs(x - y) for x, y in zip(a, b))


def remove_bin_and_renormalize(dist, spike_x):
    filtered = list(dist)
    if 0 <= spike_x < len(filtered):
        filtered[spike_x] = 0.0

    total = sum(filtered)
    if total <= 0:
        raise RuntimeError(f"All probability mass was removed after dropping spike_x={spike_x}")

    return [value / total for value in filtered]


def discrete_gumbel_pmf(max_value, loc, scale):
    pmf = []
    for m in range(max_value + 1):
        left = m - 0.5
        right = m + 0.5
        if m == 0:
            left = -1e9
        prob = gumbel_r.cdf(right, loc=loc, scale=scale) - gumbel_r.cdf(left, loc=loc, scale=scale)
        pmf.append(max(prob, 0.0))

    total = sum(pmf)
    if total > 0:
        pmf = [value / total for value in pmf]
    return pmf


def layer_label(layer):
    return "all" if layer is None else f"layer{layer}"


def load_real_distribution_for_layer(ep_size, batch_size, layer):
    experts_per_gpu = NUM_EXPERTS // ep_size
    y_counts = Counter()
    total_steps = 0

    file_path = find_layer_file(ep_size, batch_size, layer)
    if not file_path:
        raise FileNotFoundError(
            f"No real data found for EP={ep_size}, batch_size={batch_size}, layer={layer}"
        )

    with open(file_path, "r", encoding="utf-8") as infile:
        for line_idx, raw_line in enumerate(infile):
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            data = json.loads(raw_line)
            token_expert_lists = extract_token_expert_lists(data, TOP_K)
            gpu_routed_counts = [0] * ep_size

            for token_experts in token_expert_lists:
                if len(token_experts) != TOP_K:
                    raise ValueError(
                        f"{file_path}, line {line_idx + 1}: token expert list length "
                        f"{len(token_experts)} != TOP_K {TOP_K}"
                    )

                for exp_idx in token_experts:
                    gpu_id = exp_idx // experts_per_gpu
                    if gpu_id >= ep_size:
                        raise ValueError(
                            f"{file_path}, line {line_idx + 1}: invalid gpu_id={gpu_id} "
                            f"for exp_idx={exp_idx}"
                        )
                    gpu_routed_counts[gpu_id] += 1

            y_counts[max(gpu_routed_counts)] += 1
            total_steps += 1

    if total_steps == 0:
        raise RuntimeError(f"No steps found in {file_path}")

    max_m = batch_size * TOP_K
    return [y_counts[m] / total_steps for m in range(max_m + 1)]


def predict_gumbel_distribution_spike_removed_for_layer(
    ep_size,
    batch_size,
    source_ep,
    source_bs,
    num_samples,
    seed,
    spike_x,
    layer,
):
    max_m = batch_size * TOP_K
    file_path = find_layer_file(source_ep, source_bs, layer)
    if not file_path:
        raise FileNotFoundError(
            f"No source data found for source_ep={source_ep}, source_bs={source_bs}, layer={layer}"
        )

    patterns, probs, total_tokens = build_token_pattern_pmf(
        file_path=file_path,
        top_k=TOP_K,
        num_of_experts=NUM_EXPERTS,
        ep_size=ep_size,
    )
    if total_tokens == 0:
        raise RuntimeError(f"No source tokens found in {file_path}")

    rng = np.random.default_rng(seed + layer * 1237 + ep_size * 997 + batch_size * 131)
    maxima = sample_maxima_from_pattern_pmf(
        patterns=patterns,
        probs=probs,
        batch_size=batch_size,
        num_samples=num_samples,
        rng=rng,
    )
    filtered_maxima = maxima[maxima != spike_x]
    if len(filtered_maxima) == 0:
        raise RuntimeError(f"All sampled maxima were equal to spike_x={spike_x} at layer={layer}")

    loc, scale = gumbel_r.fit(filtered_maxima)
    layer_dist = discrete_gumbel_pmf(max_m, loc=loc, scale=scale)
    return remove_bin_and_renormalize(layer_dist, spike_x=spike_x)


def predict_gumbel_distribution_spike_removed_all_layers(
    ep_size,
    batch_size,
    source_ep,
    source_bs,
    num_samples,
    seed,
    spike_x,
):
    max_m = batch_size * TOP_K
    final_dist_sum = np.zeros(max_m + 1, dtype=np.float64)
    valid_layers = 0

    for layer in range(NUM_LAYERS):
        file_path = find_layer_file(source_ep, source_bs, layer)
        if not file_path:
            continue

        layer_dist = predict_gumbel_distribution_spike_removed_for_layer(
            ep_size=ep_size,
            batch_size=batch_size,
            source_ep=source_ep,
            source_bs=source_bs,
            num_samples=num_samples,
            seed=seed,
            spike_x=spike_x,
            layer=layer,
        )
        final_dist_sum += np.asarray(layer_dist, dtype=np.float64)
        valid_layers += 1

    if valid_layers == 0:
        raise RuntimeError(f"No valid layers found for source_ep={source_ep}, source_bs={source_bs}")

    return (final_dist_sum / valid_layers).tolist()


def plot_distribution_case(real, gumbel_pred, ep_size, batch_size, source_bs, spike_x, case_label, out_path):
    size = max(len(real), len(gumbel_pred))
    real = real + [0.0] * (size - len(real))
    gumbel_pred = gumbel_pred + [0.0] * (size - len(gumbel_pred))

    xs = [idx for idx in range(size) if real[idx] > 0.0 or gumbel_pred[idx] > 0.0]
    real_y = [real[idx] for idx in xs]
    gumbel_y = [gumbel_pred[idx] for idx in xs]

    plt.figure(figsize=(11, 7))
    plt.plot(xs, real_y, marker="s", label=f"Real EP={ep_size}, bs={batch_size} (spike removed)")
    plt.plot(xs, gumbel_y, marker="o", label=f"Gumbel prediction from source bs={source_bs}")
    plt.xlabel("Max routed tokens per GPU")
    plt.ylabel("Probability")
    plt.title(
        "Actual vs Gumbel Distribution After Removing Spike\n"
        f"(case={case_label}, EP={ep_size}, bs={batch_size}, removed X={spike_x})"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ep_raw = input("비교할 EP Degree 입력 (예: 4): ").strip()
    batch_size_raw = input("actual batch size 입력 (예: 32): ").strip()
    source_ep_raw = input(f"source 로그의 EP Degree 입력 (기본 {DEFAULT_SOURCE_EP}): ").strip()
    source_bs_raw = input(f"source 로그의 batch size 입력 (기본 {DEFAULT_SOURCE_BS}): ").strip()
    num_samples_raw = input(f"Gumbel fit용 샘플 수 입력 (기본 {DEFAULT_NUM_SAMPLES}): ").strip()
    seed_raw = input(f"random seed 입력 (기본 {DEFAULT_SEED}): ").strip()
    spike_raw = input("제거할 spike X 입력 (기본 2 * batch size): ").strip()

    ep_size = int(ep_raw)
    batch_size = int(batch_size_raw)
    source_ep = int(source_ep_raw) if source_ep_raw else DEFAULT_SOURCE_EP
    source_bs = int(source_bs_raw) if source_bs_raw else DEFAULT_SOURCE_BS
    num_samples = int(num_samples_raw) if num_samples_raw else DEFAULT_NUM_SAMPLES
    seed = int(seed_raw) if seed_raw else DEFAULT_SEED
    spike_x = int(spike_raw) if spike_raw else 2 * batch_size

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stem = (
        f"actual_vs_gumbel_spike_removed_ep{ep_size}_bs{batch_size}"
        f"_source_ep{source_ep}_bs{source_bs}_spike{spike_x}"
    )
    summary_path = os.path.join(OUTPUT_DIR, f"{stem}_summary.csv")

    cases = []

    real_all = load_real_distribution_for_layer_average(ep_size=ep_size, batch_size=batch_size)
    real_all = remove_bin_and_renormalize(real_all, spike_x=spike_x)
    pred_all = predict_gumbel_distribution_spike_removed_all_layers(
        ep_size=ep_size,
        batch_size=batch_size,
        source_ep=source_ep,
        source_bs=source_bs,
        num_samples=num_samples,
        seed=seed,
        spike_x=spike_x,
    )
    cases.append(("all", real_all, pred_all))

    for layer in TARGET_LAYERS:
        real_layer = load_real_distribution_for_layer(ep_size=ep_size, batch_size=batch_size, layer=layer)
        real_layer = remove_bin_and_renormalize(real_layer, spike_x=spike_x)
        pred_layer = predict_gumbel_distribution_spike_removed_for_layer(
            ep_size=ep_size,
            batch_size=batch_size,
            source_ep=source_ep,
            source_bs=source_bs,
            num_samples=num_samples,
            seed=seed,
            spike_x=spike_x,
            layer=layer,
        )
        cases.append((f"layer{layer}", real_layer, pred_layer))

    summary_rows = []
    for case_label, real, pred in cases:
        plot_path = os.path.join(OUTPUT_DIR, f"{stem}_{case_label}.png")
        plot_distribution_case(
            real=real,
            gumbel_pred=pred,
            ep_size=ep_size,
            batch_size=batch_size,
            source_bs=source_bs,
            spike_x=spike_x,
            case_label=case_label,
            out_path=plot_path,
        )
        summary_rows.append({
            "case": case_label,
            "removed_spike_x": spike_x,
            "actual_expected_max": expected_value_from_dist(real),
            "gumbel_expected_max": expected_value_from_dist(pred),
            "tvd": tvd_from_dists(real, pred),
            "plot_path": plot_path,
        })

    with open(summary_path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(
            outfile,
            fieldnames=[
                "case",
                "removed_spike_x",
                "actual_expected_max",
                "gumbel_expected_max",
                "tvd",
                "plot_path",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("완료")
    print(f"removed spike X = {spike_x}")
    for row in summary_rows:
        print(
            f"{row['case']}: "
            f"actual expected max = {row['actual_expected_max']:.6f}, "
            f"gumbel expected max = {row['gumbel_expected_max']:.6f}, "
            f"TVD = {row['tvd']:.6f}"
        )
        print(f"plot: {row['plot_path']}")
    print(f"summary: {summary_path}")


def load_real_distribution_for_layer_average(ep_size, batch_size):
    max_m = batch_size * TOP_K
    final_dist_sum = np.zeros(max_m + 1, dtype=np.float64)
    valid_layers = 0

    for layer in range(NUM_LAYERS):
        file_path = find_layer_file(ep_size, batch_size, layer)
        if not file_path:
            continue

        layer_dist = load_real_distribution_for_layer(ep_size=ep_size, batch_size=batch_size, layer=layer)
        final_dist_sum += np.asarray(layer_dist, dtype=np.float64)
        valid_layers += 1

    if valid_layers == 0:
        raise RuntimeError(f"No valid real layers found for ep={ep_size}, bs={batch_size}")

    return (final_dist_sum / valid_layers).tolist()


if __name__ == "__main__":
    main()
