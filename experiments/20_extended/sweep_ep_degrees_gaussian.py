#!/usr/bin/env python3

import csv
import glob
import json
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
DEFAULT_NUM_SAMPLES = 200000
DEFAULT_SEED = 42
OUTPUT_DIR = "./output/ep_sweep_gaussian"


def extract_token_expert_lists(data, top_k):
    if len(data) > 0 and isinstance(data[0], list):
        return data

    if len(data) % top_k != 0:
        raise ValueError(f"flat data length {len(data)} is not divisible by top_k={top_k}")

    return [data[i:i + top_k] for i in range(0, len(data), top_k)]


def token_to_gpu_pattern(token_experts, num_of_experts, ep_size):
    experts_per_gpu = num_of_experts // ep_size
    counts = [0] * ep_size

    for expert_idx in token_experts:
        if expert_idx < 0 or expert_idx >= num_of_experts:
            raise ValueError(f"expert index {expert_idx} out of range [0, {num_of_experts - 1}]")
        gpu_idx = expert_idx // experts_per_gpu
        counts[gpu_idx] += 1

    return counts


def find_layer_file(source_ep, source_bs, layer):
    pattern = os.path.join(
        DIRECTORY,
        f"expert_selection_openai_{MODEL_NAME}_wikitext_103_ep{source_ep}_bs{source_bs}_*_layer{layer}.jsonl",
    )
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def load_token_patterns(file_path, ep_size):
    token_patterns = []

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
                        f"{len(token_experts)} != top_k {TOP_K}"
                    )
                token_patterns.append(
                    token_to_gpu_pattern(token_experts, NUM_EXPERTS, ep_size)
                )

    if not token_patterns:
        return np.empty((0, ep_size), dtype=np.float64)

    return np.asarray(token_patterns, dtype=np.float64)


def percentile_from_dist(dist, q):
    cdf = 0.0
    for idx, prob in enumerate(dist):
        cdf += prob
        if cdf >= q:
            return idx
    return len(dist) - 1


def expected_value_from_dist(dist):
    return sum(idx * prob for idx, prob in enumerate(dist))


def gaussian_max_distribution(token_patterns, predicted_bs, ep_size, num_samples, rng):
    mean_vec = token_patterns.mean(axis=0)
    cov_mat = np.cov(token_patterns, rowvar=False)

    if ep_size == 1:
        cov_mat = np.array([[0.0]])

    batch_mean = predicted_bs * mean_vec
    batch_cov = predicted_bs * cov_mat

    samples = rng.multivariate_normal(
        mean=batch_mean,
        cov=batch_cov,
        size=num_samples,
        check_valid="warn",
        method="svd",
    )

    samples = np.rint(samples)
    samples = np.clip(samples, 0, predicted_bs * TOP_K)

    max_values = samples.max(axis=1).astype(int)
    dist = np.bincount(max_values, minlength=predicted_bs * TOP_K + 1).astype(np.float64)
    dist /= dist.sum()

    return dist, mean_vec, cov_mat


def predict_distribution_for_ep(target_ep, predicted_bs, source_ep, source_bs, num_samples, seed):
    if NUM_EXPERTS % target_ep != 0:
        raise ValueError(f"num_of_experts={NUM_EXPERTS} must be divisible by ep_size={target_ep}")

    max_m = predicted_bs * TOP_K
    final_cond_prob_sum = np.zeros(max_m + 1, dtype=np.float64)
    final_expected_max_sum = 0.0
    final_p95_sum = 0.0
    final_imbalance_ratio_sum = 0.0
    valid_layers = 0

    for layer in range(NUM_LAYERS):
        file_path = find_layer_file(source_ep, source_bs, layer)
        if not file_path:
            continue

        token_patterns = load_token_patterns(file_path, target_ep)
        if token_patterns.shape[0] == 0:
            continue

        rng = np.random.default_rng(seed + layer * 1009 + target_ep * 9176 + predicted_bs * 131)
        layer_dist, _, _ = gaussian_max_distribution(
            token_patterns=token_patterns,
            predicted_bs=predicted_bs,
            ep_size=target_ep,
            num_samples=num_samples,
            rng=rng,
        )

        expected_max = expected_value_from_dist(layer_dist)
        p95 = percentile_from_dist(layer_dist.tolist(), 0.95)
        avg_load = (predicted_bs * TOP_K) / target_ep
        imbalance_ratio = expected_max / avg_load if avg_load > 0 else 0.0

        final_cond_prob_sum += layer_dist
        final_expected_max_sum += expected_max
        final_p95_sum += p95
        final_imbalance_ratio_sum += imbalance_ratio
        valid_layers += 1

    if valid_layers == 0:
        raise RuntimeError(f"No valid layers found for source_ep={source_ep}, source_bs={source_bs}")

    avg_dist = (final_cond_prob_sum / valid_layers).tolist()
    return {
        "ep": target_ep,
        "distribution": avg_dist,
        "expected_max": final_expected_max_sum / valid_layers,
        "p95": final_p95_sum / valid_layers,
        "imbalance_ratio": final_imbalance_ratio_sum / valid_layers,
        "avg_load": (predicted_bs * TOP_K) / target_ep,
        "valid_layers": valid_layers,
    }


def write_summary_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(
            outfile,
            fieldnames=[
                "target_ep",
                "predicted_bs",
                "source_ep",
                "source_bs",
                "num_samples",
                "expected_max",
                "p95",
                "imbalance_ratio",
                "avg_load",
                "valid_layers",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_distribution_overlay(results, out_path, predicted_bs, source_ep, source_bs):
    plt.figure(figsize=(11, 7))

    for result in results:
        xs = [idx for idx, prob in enumerate(result["distribution"]) if prob > 0.0]
        ys = [result["distribution"][idx] for idx in xs]
        plt.plot(xs, ys, marker="o", label=f"EP={result['ep']}")

    plt.xlabel("Max routed tokens per GPU")
    plt.ylabel("Probability")
    plt.title(
        f"Gaussian-approximated distributions across EP degrees "
        f"(predicted bs={predicted_bs}, source ep={source_ep}, source bs={source_bs})"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_metric(results, metric, ylabel, out_path):
    ordered = sorted(results, key=lambda item: item["ep"])
    xs = [result["ep"] for result in ordered]
    ys = [result[metric] for result in ordered]

    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("EP Degree")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} across EP degrees (Gaussian approximation)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    predicted_bs = int(input("예측할 배치 사이즈(bs) 입력: "))
    ep_values_raw = input("비교할 EP Degree 목록 입력 (예: 2,4,8,16): ").strip()
    source_ep_raw = input(f"source 로그의 EP Degree 입력 (기본 {DEFAULT_SOURCE_EP}): ").strip()
    source_bs_raw = input(f"source 로그의 batch size 입력 (기본 {DEFAULT_SOURCE_BS}): ").strip()
    num_samples_raw = input(f"Gaussian 샘플 수 입력 (기본 {DEFAULT_NUM_SAMPLES}): ").strip()
    seed_raw = input(f"random seed 입력 (기본 {DEFAULT_SEED}): ").strip()

    if not ep_values_raw:
        raise ValueError("비교할 EP Degree 목록이 필요합니다.")

    target_eps = [int(value.strip()) for value in ep_values_raw.split(",") if value.strip()]
    source_ep = int(source_ep_raw) if source_ep_raw else DEFAULT_SOURCE_EP
    source_bs = int(source_bs_raw) if source_bs_raw else DEFAULT_SOURCE_BS
    num_samples = int(num_samples_raw) if num_samples_raw else DEFAULT_NUM_SAMPLES
    seed = int(seed_raw) if seed_raw else DEFAULT_SEED

    for target_ep in target_eps:
        if NUM_EXPERTS % target_ep != 0:
            raise ValueError(f"target EP {target_ep} must divide NUM_EXPERTS={NUM_EXPERTS}")

    print(
        "고정된 expert selection 로그를 사용해 EP degree별 Gaussian 근사 분포를 계산 중...\n"
        f"(source ep={source_ep}, source bs={source_bs}, predicted bs={predicted_bs}, "
        f"num_samples={num_samples})"
    )

    results = []
    summary_rows = []

    for target_ep in target_eps:
        result = predict_distribution_for_ep(
            target_ep=target_ep,
            predicted_bs=predicted_bs,
            source_ep=source_ep,
            source_bs=source_bs,
            num_samples=num_samples,
            seed=seed,
        )
        results.append(result)
        summary_rows.append({
            "target_ep": target_ep,
            "predicted_bs": predicted_bs,
            "source_ep": source_ep,
            "source_bs": source_bs,
            "num_samples": num_samples,
            "expected_max": result["expected_max"],
            "p95": result["p95"],
            "imbalance_ratio": result["imbalance_ratio"],
            "avg_load": result["avg_load"],
            "valid_layers": result["valid_layers"],
        })

        print("============================================================")
        print(f"Target EP = {target_ep}")
        print("============================================================")
        print(f"valid_layers = {result['valid_layers']}")
        print(f"E[max routed tokens per GPU] = {result['expected_max']:.6f}")
        print(f"p95(max routed tokens per GPU) = {result['p95']:.6f}")
        print(f"avg load per GPU = {result['avg_load']:.6f}")
        print(f"Imbalance Ratio = {result['imbalance_ratio']:.6f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_name = (
        f"gaussian_predbs{predicted_bs}_from_source_ep{source_ep}_bs{source_bs}"
        f"_nsamples{num_samples}"
    )

    summary_csv = os.path.join(OUTPUT_DIR, f"{base_name}_summary.csv")
    overlay_png = os.path.join(OUTPUT_DIR, f"{base_name}_distribution_overlay.png")
    expected_png = os.path.join(OUTPUT_DIR, f"{base_name}_expected_max.png")
    p95_png = os.path.join(OUTPUT_DIR, f"{base_name}_p95.png")
    ratio_png = os.path.join(OUTPUT_DIR, f"{base_name}_imbalance_ratio.png")

    write_summary_csv(summary_csv, summary_rows)
    plot_distribution_overlay(results, overlay_png, predicted_bs, source_ep, source_bs)
    plot_metric(results, "expected_max", "Expected max routed tokens per GPU", expected_png)
    plot_metric(results, "p95", "p95 of max routed tokens per GPU", p95_png)
    plot_metric(results, "imbalance_ratio", "Imbalance Ratio", ratio_png)

    print("\n완료")
    print(f"summary csv: {summary_csv}")
    print(f"distribution plot: {overlay_png}")
    print(f"expected max plot: {expected_png}")
    print(f"p95 plot: {p95_png}")
    print(f"imbalance ratio plot: {ratio_png}")


if __name__ == "__main__":
    main()
