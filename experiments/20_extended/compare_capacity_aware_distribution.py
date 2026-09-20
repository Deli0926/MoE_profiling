#!/usr/bin/env python3

import csv
import glob
import json
import math
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


DIRECTORY = "./expert_selections"
MODEL_NAME = "gpt-oss-20b"
NUM_EXPERTS = 32
TOP_K = 4
NUM_LAYERS = 24

DEFAULT_SOURCE_EP = 4
DEFAULT_SOURCE_BS = 1
DEFAULT_NUM_SAMPLES = 10000
DEFAULT_SEED = 42
DEFAULT_P_EFF = 8.0
DEFAULT_OVERHEAD_ALPHA = 0.0
DEFAULT_OVERHEAD_BETA = 0.0
DEFAULT_BIN_DECIMALS = 3

OUTPUT_DIR = "./output/capacity_aware_distribution"


def find_layer_file(file_ep, file_bs, layer):
    pattern = os.path.join(
        DIRECTORY,
        f"expert_selection_openai_{MODEL_NAME}_wikitext_103_ep{file_ep}_bs{file_bs}_*_layer{layer}.jsonl",
    )
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def extract_token_expert_lists(data, top_k):
    if len(data) > 0 and isinstance(data[0], list):
        return data

    if len(data) % top_k != 0:
        raise ValueError(f"flat data length {len(data)} is not divisible by top_k={top_k}")

    return [data[i:i + top_k] for i in range(0, len(data), top_k)]


def load_token_expert_array(file_path):
    rows = []

    with open(file_path, "r", encoding="utf-8") as infile:
        for line_idx, line in enumerate(infile):
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            token_expert_lists = extract_token_expert_lists(data, TOP_K)
            for token_experts in token_expert_lists:
                if len(token_experts) != TOP_K:
                    raise ValueError(
                        f"{file_path}, line {line_idx + 1}: token expert list length "
                        f"{len(token_experts)} != TOP_K {TOP_K}"
                    )
                rows.append(token_experts)

    if not rows:
        return np.empty((0, TOP_K), dtype=np.int16)

    return np.asarray(rows, dtype=np.int16)


def capacity_aware_score_from_local_counts(local_counts, p_eff, overhead_alpha, overhead_beta):
    gpu_total_counts = local_counts.sum(axis=2).astype(np.float64)
    gpu_max_expert = local_counts.max(axis=2).astype(np.float64)
    gpu_capacity_term = gpu_total_counts / p_eff
    gpu_overhead = overhead_alpha + overhead_beta * gpu_total_counts
    gpu_score = gpu_overhead + np.maximum(gpu_max_expert, gpu_capacity_term)
    return gpu_score.max(axis=1)


def sample_capacity_scores(token_expert_array, ep_size, batch_size, num_samples, rng, p_eff, overhead_alpha, overhead_beta):
    experts_per_gpu = NUM_EXPERTS // ep_size

    sampled_indices = rng.integers(0, len(token_expert_array), size=(num_samples, batch_size))
    sampled_tokens = token_expert_array[sampled_indices]

    gpu_idx = sampled_tokens // experts_per_gpu
    local_idx = sampled_tokens % experts_per_gpu

    local_counts = np.zeros((num_samples, ep_size, experts_per_gpu), dtype=np.int16)
    sample_axis = np.broadcast_to(
        np.arange(num_samples, dtype=np.int32)[:, None, None],
        (num_samples, batch_size, TOP_K),
    )
    np.add.at(
        local_counts,
        (sample_axis.ravel(), gpu_idx.ravel(), local_idx.ravel()),
        1,
    )

    return capacity_aware_score_from_local_counts(
        local_counts=local_counts,
        p_eff=p_eff,
        overhead_alpha=overhead_alpha,
        overhead_beta=overhead_beta,
    )


def actual_capacity_scores_for_layer(file_path, ep_size, p_eff, overhead_alpha, overhead_beta):
    experts_per_gpu = NUM_EXPERTS // ep_size
    scores = []

    with open(file_path, "r", encoding="utf-8") as infile:
        for line_idx, line in enumerate(infile):
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            token_expert_lists = extract_token_expert_lists(data, TOP_K)
            local_counts = np.zeros((1, ep_size, experts_per_gpu), dtype=np.int16)

            for token_experts in token_expert_lists:
                if len(token_experts) != TOP_K:
                    raise ValueError(
                        f"{file_path}, line {line_idx + 1}: token expert list length "
                        f"{len(token_experts)} != TOP_K {TOP_K}"
                    )
                for expert_idx in token_experts:
                    gpu_idx = expert_idx // experts_per_gpu
                    local_expert_idx = expert_idx % experts_per_gpu
                    local_counts[0, gpu_idx, local_expert_idx] += 1

            score = capacity_aware_score_from_local_counts(
                local_counts=local_counts,
                p_eff=p_eff,
                overhead_alpha=overhead_alpha,
                overhead_beta=overhead_beta,
            )[0]
            scores.append(float(score))

    return scores


def quantized_pmf(values, decimals):
    counter = Counter(round(float(value), decimals) for value in values)
    total = sum(counter.values())
    if total == 0:
        return {}
    return {key: count / total for key, count in sorted(counter.items())}


def pmf_to_xy(pmf):
    xs = sorted(pmf.keys())
    ys = [pmf[x] for x in xs]
    return xs, ys


def expected_value_from_pmf(pmf):
    return sum(x * prob for x, prob in pmf.items())


def percentile_from_values(values, q):
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q * 100.0))


def tvd(pmf_a, pmf_b):
    keys = set(pmf_a) | set(pmf_b)
    return 0.5 * sum(abs(pmf_a.get(key, 0.0) - pmf_b.get(key, 0.0)) for key in keys)


def main():
    ep_raw = input("비교할 EP Degree 입력 (예: 4): ").strip()
    batch_size_raw = input("비교할 batch size 입력 (예: 32): ").strip()
    source_ep_raw = input(f"source 로그의 EP Degree 입력 (기본 {DEFAULT_SOURCE_EP}): ").strip()
    source_bs_raw = input(f"source 로그의 batch size 입력 (기본 {DEFAULT_SOURCE_BS}): ").strip()
    num_samples_raw = input(f"Monte Carlo 샘플 수 입력 (기본 {DEFAULT_NUM_SAMPLES}): ").strip()
    seed_raw = input(f"random seed 입력 (기본 {DEFAULT_SEED}): ").strip()
    p_eff_raw = input(f"P_eff 입력 (기본 {DEFAULT_P_EFF}): ").strip()
    alpha_raw = input(f"Overhead alpha 입력 (기본 {DEFAULT_OVERHEAD_ALPHA}): ").strip()
    beta_raw = input(f"Overhead beta 입력 (기본 {DEFAULT_OVERHEAD_BETA}): ").strip()
    decimals_raw = input(f"PMF bin 소수점 자리수 입력 (기본 {DEFAULT_BIN_DECIMALS}): ").strip()

    ep_size = int(ep_raw)
    batch_size = int(batch_size_raw)
    source_ep = int(source_ep_raw) if source_ep_raw else DEFAULT_SOURCE_EP
    source_bs = int(source_bs_raw) if source_bs_raw else DEFAULT_SOURCE_BS
    num_samples = int(num_samples_raw) if num_samples_raw else DEFAULT_NUM_SAMPLES
    seed = int(seed_raw) if seed_raw else DEFAULT_SEED
    p_eff = float(p_eff_raw) if p_eff_raw else DEFAULT_P_EFF
    overhead_alpha = float(alpha_raw) if alpha_raw else DEFAULT_OVERHEAD_ALPHA
    overhead_beta = float(beta_raw) if beta_raw else DEFAULT_OVERHEAD_BETA
    bin_decimals = int(decimals_raw) if decimals_raw else DEFAULT_BIN_DECIMALS

    if NUM_EXPERTS % ep_size != 0:
        raise ValueError(f"NUM_EXPERTS={NUM_EXPERTS} must be divisible by ep_size={ep_size}")

    print(
        "capacity-aware 모델\n"
        "X = max_g ( Overhead(L_g) + max(max_e T_{k,g,e}, L_g / P_eff) )\n"
        "에 대한 predicted vs actual PMF를 계산 중..."
    )

    predicted_scores_all = []
    actual_scores_all = []

    for layer in range(NUM_LAYERS):
        source_file = find_layer_file(source_ep, source_bs, layer)
        actual_file = find_layer_file(ep_size, batch_size, layer)
        if not source_file or not actual_file:
            continue

        token_expert_array = load_token_expert_array(source_file)
        if len(token_expert_array) == 0:
            continue

        rng = np.random.default_rng(seed + layer * 1237 + ep_size * 997 + batch_size * 131)
        pred_scores = sample_capacity_scores(
            token_expert_array=token_expert_array,
            ep_size=ep_size,
            batch_size=batch_size,
            num_samples=num_samples,
            rng=rng,
            p_eff=p_eff,
            overhead_alpha=overhead_alpha,
            overhead_beta=overhead_beta,
        )
        actual_scores = actual_capacity_scores_for_layer(
            file_path=actual_file,
            ep_size=ep_size,
            p_eff=p_eff,
            overhead_alpha=overhead_alpha,
            overhead_beta=overhead_beta,
        )

        predicted_scores_all.extend(pred_scores.tolist())
        actual_scores_all.extend(actual_scores)

    if not predicted_scores_all or not actual_scores_all:
        raise RuntimeError("No valid layers found for the requested comparison.")

    pred_pmf = quantized_pmf(predicted_scores_all, bin_decimals)
    actual_pmf = quantized_pmf(actual_scores_all, bin_decimals)

    pred_x, pred_y = pmf_to_xy(pred_pmf)
    actual_x, actual_y = pmf_to_xy(actual_pmf)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stem = (
        f"capacity_aware_ep{ep_size}_bs{batch_size}"
        f"_source_ep{source_ep}_bs{source_bs}"
        f"_peff{str(p_eff).replace('.', 'p')}"
        f"_alpha{str(overhead_alpha).replace('.', 'p')}"
        f"_beta{str(overhead_beta).replace('.', 'p')}"
    )
    plot_path = os.path.join(OUTPUT_DIR, f"{stem}.png")
    summary_path = os.path.join(OUTPUT_DIR, f"{stem}_summary.csv")
    pmf_path = os.path.join(OUTPUT_DIR, f"{stem}_pmf.csv")

    plt.figure(figsize=(11, 7))
    plt.plot(pred_x, pred_y, marker="o", label="Predicted")
    plt.plot(actual_x, actual_y, marker="s", label="Actual")
    plt.xlabel("Capacity-aware X")
    plt.ylabel("Probability")
    plt.title(
        "Predicted vs Actual Capacity-aware Distribution\n"
        f"(EP={ep_size}, bs={batch_size}, P_eff={p_eff}, alpha={overhead_alpha}, beta={overhead_beta})"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    with open(summary_path, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(
            outfile,
            fieldnames=[
                "ep",
                "bs",
                "source_ep",
                "source_bs",
                "num_samples",
                "p_eff",
                "overhead_alpha",
                "overhead_beta",
                "bin_decimals",
                "pred_expected_x",
                "actual_expected_x",
                "pred_p95_x",
                "actual_p95_x",
                "tvd",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "ep": ep_size,
                "bs": batch_size,
                "source_ep": source_ep,
                "source_bs": source_bs,
                "num_samples": num_samples,
                "p_eff": p_eff,
                "overhead_alpha": overhead_alpha,
                "overhead_beta": overhead_beta,
                "bin_decimals": bin_decimals,
                "pred_expected_x": expected_value_from_pmf(pred_pmf),
                "actual_expected_x": expected_value_from_pmf(actual_pmf),
                "pred_p95_x": percentile_from_values(predicted_scores_all, 0.95),
                "actual_p95_x": percentile_from_values(actual_scores_all, 0.95),
                "tvd": tvd(pred_pmf, actual_pmf),
            }
        )

    all_bins = sorted(set(pred_pmf) | set(actual_pmf))
    with open(pmf_path, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["x_bin", "pred_prob", "actual_prob"])
        writer.writeheader()
        for x_bin in all_bins:
            writer.writerow(
                {
                    "x_bin": x_bin,
                    "pred_prob": pred_pmf.get(x_bin, 0.0),
                    "actual_prob": actual_pmf.get(x_bin, 0.0),
                }
            )

    print("완료")
    print(f"pred expected X = {expected_value_from_pmf(pred_pmf):.6f}")
    print(f"actual expected X = {expected_value_from_pmf(actual_pmf):.6f}")
    print(f"pred p95 X = {percentile_from_values(predicted_scores_all, 0.95):.6f}")
    print(f"actual p95 X = {percentile_from_values(actual_scores_all, 0.95):.6f}")
    print(f"TVD = {tvd(pred_pmf, actual_pmf):.6f}")
    print(f"plot: {plot_path}")
    print(f"summary: {summary_path}")
    print(f"pmf: {pmf_path}")


if __name__ == "__main__":
    main()
