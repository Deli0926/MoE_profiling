#!/usr/bin/env python3

import glob
import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gumbel_r

from compare_token_distribution import load_real_distribution_for_bs


DIRECTORY = "./expert_selections"
MODEL_NAME = "gpt-oss-20b"
NUM_EXPERTS = 32
TOP_K = 4
NUM_LAYERS = 24
DEFAULT_SOURCE_EP = 4
DEFAULT_SOURCE_BS = 1
DEFAULT_NUM_SAMPLES = 50000
DEFAULT_SEED = 42
OUTPUT_DIR = "./output/actual_vs_gumbel"


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

    return tuple(counts)


def find_layer_file(source_ep, source_bs, layer):
    pattern = os.path.join(
        DIRECTORY,
        f"expert_selection_openai_{MODEL_NAME}_wikitext_103_ep{source_ep}_bs{source_bs}_*_layer{layer}.jsonl",
    )
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def build_token_pattern_pmf(file_path, top_k, num_of_experts, ep_size):
    pattern_counter = Counter()
    total_tokens = 0

    with open(file_path, "r", encoding="utf-8") as infile:
        for line_idx, line in enumerate(infile):
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            token_expert_lists = extract_token_expert_lists(data, top_k)

            for token_experts in token_expert_lists:
                if len(token_experts) != top_k:
                    raise ValueError(
                        f"{file_path}, line {line_idx + 1}: token expert list length "
                        f"{len(token_experts)} != top_k {top_k}"
                    )

                pattern = token_to_gpu_pattern(token_experts, num_of_experts, ep_size)
                pattern_counter[pattern] += 1
                total_tokens += 1

    if total_tokens == 0:
        return [], np.array([]), 0

    patterns = np.asarray(list(pattern_counter.keys()), dtype=np.int16)
    probs = np.asarray([count / total_tokens for count in pattern_counter.values()], dtype=np.float64)
    return patterns, probs, total_tokens


def expected_value_from_dist(dist):
    return sum(idx * prob for idx, prob in enumerate(dist))


def sample_maxima_from_pattern_pmf(patterns, probs, batch_size, num_samples, rng):
    sampled_indices = rng.choice(len(patterns), size=(num_samples, batch_size), p=probs)
    sampled_patterns = patterns[sampled_indices]
    load_vectors = sampled_patterns.sum(axis=1)
    return load_vectors.max(axis=1).astype(np.int32)


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


def predict_gumbel_distribution(ep_size, batch_size, source_ep, source_bs, num_samples, seed):
    max_m = batch_size * TOP_K
    final_dist_sum = np.zeros(max_m + 1, dtype=np.float64)
    valid_layers = 0

    for layer in range(NUM_LAYERS):
        file_path = find_layer_file(source_ep, source_bs, layer)
        if not file_path:
            continue

        patterns, probs, total_tokens = build_token_pattern_pmf(
            file_path=file_path,
            top_k=TOP_K,
            num_of_experts=NUM_EXPERTS,
            ep_size=ep_size,
        )
        if total_tokens == 0:
            continue

        rng = np.random.default_rng(seed + layer * 1237 + ep_size * 997 + batch_size * 131)
        maxima = sample_maxima_from_pattern_pmf(
            patterns=patterns,
            probs=probs,
            batch_size=batch_size,
            num_samples=num_samples,
            rng=rng,
        )
        loc, scale = gumbel_r.fit(maxima)
        layer_dist = discrete_gumbel_pmf(max_m, loc=loc, scale=scale)

        final_dist_sum += np.asarray(layer_dist, dtype=np.float64)
        valid_layers += 1

    if valid_layers == 0:
        raise RuntimeError(f"No valid layers found for source_ep={source_ep}, source_bs={source_bs}")

    return (final_dist_sum / valid_layers).tolist()


def main():
    ep_raw = input("비교할 EP Degree 입력 (예: 4): ").strip()
    batch_size_raw = input("actual batch size 입력 (예: 32): ").strip()
    source_ep_raw = input(f"source 로그의 EP Degree 입력 (기본 {DEFAULT_SOURCE_EP}): ").strip()
    source_bs_raw = input(f"source 로그의 batch size 입력 (기본 {DEFAULT_SOURCE_BS}): ").strip()
    num_samples_raw = input(f"Gumbel fit용 샘플 수 입력 (기본 {DEFAULT_NUM_SAMPLES}): ").strip()
    seed_raw = input(f"random seed 입력 (기본 {DEFAULT_SEED}): ").strip()

    ep_size = int(ep_raw)
    batch_size = int(batch_size_raw)
    source_ep = int(source_ep_raw) if source_ep_raw else DEFAULT_SOURCE_EP
    source_bs = int(source_bs_raw) if source_bs_raw else DEFAULT_SOURCE_BS
    num_samples = int(num_samples_raw) if num_samples_raw else DEFAULT_NUM_SAMPLES
    seed = int(seed_raw) if seed_raw else DEFAULT_SEED

    real = load_real_distribution_for_bs(ep_size=ep_size, real_file_bs=batch_size)
    gumbel_pred = predict_gumbel_distribution(
        ep_size=ep_size,
        batch_size=batch_size,
        source_ep=source_ep,
        source_bs=source_bs,
        num_samples=num_samples,
        seed=seed,
    )

    size = max(len(real), len(gumbel_pred))
    real = real + [0.0] * (size - len(real))
    gumbel_pred = gumbel_pred + [0.0] * (size - len(gumbel_pred))

    xs = [idx for idx in range(size) if real[idx] > 0.0 or gumbel_pred[idx] > 0.0]
    real_y = [real[idx] for idx in xs]
    gumbel_y = [gumbel_pred[idx] for idx in xs]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(
        OUTPUT_DIR,
        f"actual_vs_gumbel_ep{ep_size}_bs{batch_size}_source_ep{source_ep}_bs{source_bs}.png",
    )

    plt.figure(figsize=(11, 7))
    plt.plot(xs, real_y, marker="s", label=f"Real EP={ep_size}, bs={batch_size}")
    plt.plot(xs, gumbel_y, marker="o", label=f"Gumbel prediction from source bs={source_bs}")
    plt.xlabel("Max routed tokens per GPU")
    plt.ylabel("Probability")
    plt.title(f"Actual vs Gumbel Distribution (EP={ep_size}, bs={batch_size})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print("완료")
    print(f"actual expected max = {expected_value_from_dist(real):.6f}")
    print(f"gumbel expected max = {expected_value_from_dist(gumbel_pred):.6f}")
    print(f"plot: {out_path}")


if __name__ == "__main__":
    main()
