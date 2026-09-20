#!/usr/bin/env python3

import glob
import json
import math
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gumbel_r


DIRECTORY = "./expert_selections"
MODEL_NAME = "gpt-oss-20b"
NUM_EXPERTS = 32
TOP_K = 4
NUM_LAYERS = 24

DEFAULT_SOURCE_EP = 4
DEFAULT_SOURCE_BS = 1
DEFAULT_NUM_SAMPLES = 50000
DEFAULT_SEED = 42
DEFAULT_P_EFF = 8.0
DEFAULT_BIN_DECIMALS = 3
DEFAULT_P_EFF_MODE = "constant"
DEFAULT_SCORE_MODE = "combined"

OUTPUT_DIR = "./output/actual_vs_gumbel_loadaware"


def extract_token_expert_lists(data, top_k):
    if len(data) > 0 and isinstance(data[0], list):
        return data

    if len(data) % top_k != 0:
        raise ValueError(f"flat data length {len(data)} is not divisible by top_k={top_k}")

    return [data[i:i + top_k] for i in range(0, len(data), top_k)]


def find_layer_file(file_ep, file_bs, layer):
    pattern = os.path.join(
        DIRECTORY,
        f"expert_selection_openai_{MODEL_NAME}_wikitext_103_ep{file_ep}_bs{file_bs}_*_layer{layer}.jsonl",
    )
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


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


def loadaware_score_from_local_counts(local_counts, base_p_eff, p_eff_mode, score_mode):
    gpu_total_counts = local_counts.sum(axis=2).astype(np.float64)
    gpu_max_expert = local_counts.max(axis=2).astype(np.float64)
    gpu_active_experts = (local_counts > 0).sum(axis=2).astype(np.float64)

    if p_eff_mode == "constant":
        gpu_p_eff = np.full_like(gpu_total_counts, float(base_p_eff), dtype=np.float64)
    elif p_eff_mode == "capped_by_load":
        gpu_p_eff = np.maximum(1.0, np.minimum(gpu_total_counts, float(base_p_eff)))
    elif p_eff_mode == "perfect_parallel":
        gpu_p_eff = np.maximum(1.0, gpu_active_experts)
    elif p_eff_mode == "no_parallel":
        gpu_p_eff = np.ones_like(gpu_total_counts, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported p_eff mode: {p_eff_mode}")

    gpu_capacity_term = np.divide(
        gpu_total_counts,
        gpu_p_eff,
        out=np.zeros_like(gpu_total_counts, dtype=np.float64),
        where=gpu_p_eff > 0,
    )

    if score_mode == "capacity_only":
        gpu_score = gpu_capacity_term
    elif score_mode == "combined":
        gpu_score = np.maximum(gpu_max_expert, gpu_capacity_term)
    else:
        raise ValueError(f"Unsupported score mode: {score_mode}")

    return gpu_score.max(axis=1)


def sample_loadaware_scores(token_expert_array, ep_size, batch_size, num_samples, rng, base_p_eff, p_eff_mode, score_mode):
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
    np.add.at(local_counts, (sample_axis.ravel(), gpu_idx.ravel(), local_idx.ravel()), 1)

    return loadaware_score_from_local_counts(
        local_counts=local_counts,
        base_p_eff=base_p_eff,
        p_eff_mode=p_eff_mode,
        score_mode=score_mode,
    )


def actual_loadaware_scores_for_layer(file_path, ep_size, base_p_eff, p_eff_mode, score_mode):
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

            score = loadaware_score_from_local_counts(
                local_counts=local_counts,
                base_p_eff=base_p_eff,
                p_eff_mode=p_eff_mode,
                score_mode=score_mode,
            )[0]
            scores.append(float(score))

    return scores


def quantized_pmf(values, decimals):
    counter = Counter(round(float(value), decimals) for value in values)
    total = sum(counter.values())
    if total == 0:
        return {}
    return {key: count / total for key, count in sorted(counter.items())}


def average_pmfs(pmfs):
    if not pmfs:
        return {}

    accum = Counter()
    for pmf in pmfs:
        for x, prob in pmf.items():
            accum[x] += prob

    scale = 1.0 / len(pmfs)
    return {x: accum[x] * scale for x in sorted(accum)}


def expected_value_from_pmf(pmf):
    return sum(x * prob for x, prob in pmf.items())


def gumbel_quantized_pmf(loc, scale, support, decimals):
    if scale <= 0:
        raise ValueError(f"Gumbel scale must be positive, got {scale}")

    step = 10.0 ** (-decimals)
    half_step = step / 2.0
    pmf = {}

    for x in support:
        left = x - half_step
        right = x + half_step
        prob = gumbel_r.cdf(right, loc=loc, scale=scale) - gumbel_r.cdf(left, loc=loc, scale=scale)
        pmf[x] = max(float(prob), 0.0)

    total = sum(pmf.values())
    if total <= 0:
        return {x: 0.0 for x in support}

    return {x: pmf[x] / total for x in support}


def quantized_support(values, decimals):
    return sorted(quantized_pmf(values, decimals).keys())


def predict_gumbel_distribution_loadaware(
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
        actual_file = find_layer_file(ep_size, batch_size, layer)
        if not source_file or not actual_file:
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
        actual_scores = actual_loadaware_scores_for_layer(
            file_path=actual_file,
            ep_size=ep_size,
            base_p_eff=base_p_eff,
            p_eff_mode=p_eff_mode,
            score_mode=score_mode,
        )

        support = sorted(set(quantized_support(pred_scores, bin_decimals)) | set(quantized_support(actual_scores, bin_decimals)))
        if not support:
            continue

        loc, scale = gumbel_r.fit(pred_scores)
        layer_pmfs.append(gumbel_quantized_pmf(loc, scale, support, bin_decimals))

    if not layer_pmfs:
        raise RuntimeError(f"No valid layers found for source_ep={source_ep}, source_bs={source_bs}")

    return average_pmfs(layer_pmfs)


def actual_distribution_loadaware(ep_size, batch_size, base_p_eff, p_eff_mode, score_mode, bin_decimals):
    layer_pmfs = []

    for layer in range(NUM_LAYERS):
        file_path = find_layer_file(ep_size, batch_size, layer)
        if not file_path:
            continue

        actual_scores = actual_loadaware_scores_for_layer(
            file_path=file_path,
            ep_size=ep_size,
            base_p_eff=base_p_eff,
            p_eff_mode=p_eff_mode,
            score_mode=score_mode,
        )
        if actual_scores:
            layer_pmfs.append(quantized_pmf(actual_scores, bin_decimals))

    if not layer_pmfs:
        raise RuntimeError(f"No valid layers found for ep={ep_size}, bs={batch_size}")

    return average_pmfs(layer_pmfs)


def main():
    ep_raw = input("비교할 EP Degree 입력 (예: 4): ").strip()
    batch_size_raw = input("actual batch size 입력 (예: 32): ").strip()
    source_ep_raw = input(f"source 로그의 EP Degree 입력 (기본 {DEFAULT_SOURCE_EP}): ").strip()
    source_bs_raw = input(f"source 로그의 batch size 입력 (기본 {DEFAULT_SOURCE_BS}): ").strip()
    num_samples_raw = input(f"Gumbel fit용 샘플 수 입력 (기본 {DEFAULT_NUM_SAMPLES}): ").strip()
    seed_raw = input(f"random seed 입력 (기본 {DEFAULT_SEED}): ").strip()
    p_eff_raw = input(f"base P_eff 입력 (기본 {DEFAULT_P_EFF}): ").strip()
    p_eff_mode_raw = input(
        "P_eff(L_g) 모드 입력 "
        f"[constant/capped_by_load/perfect_parallel/no_parallel] (기본 {DEFAULT_P_EFF_MODE}): "
    ).strip()
    score_mode_raw = input(
        f"X 모드 입력 [capacity_only/combined] (기본 {DEFAULT_SCORE_MODE}): "
    ).strip()
    decimals_raw = input(f"PMF bin 소수점 자리수 입력 (기본 {DEFAULT_BIN_DECIMALS}): ").strip()

    ep_size = int(ep_raw)
    batch_size = int(batch_size_raw)
    source_ep = int(source_ep_raw) if source_ep_raw else DEFAULT_SOURCE_EP
    source_bs = int(source_bs_raw) if source_bs_raw else DEFAULT_SOURCE_BS
    num_samples = int(num_samples_raw) if num_samples_raw else DEFAULT_NUM_SAMPLES
    seed = int(seed_raw) if seed_raw else DEFAULT_SEED
    base_p_eff = float(p_eff_raw) if p_eff_raw else DEFAULT_P_EFF
    p_eff_mode = p_eff_mode_raw if p_eff_mode_raw else DEFAULT_P_EFF_MODE
    score_mode = score_mode_raw if score_mode_raw else DEFAULT_SCORE_MODE
    bin_decimals = int(decimals_raw) if decimals_raw else DEFAULT_BIN_DECIMALS

    if NUM_EXPERTS % ep_size != 0:
        raise ValueError(f"NUM_EXPERTS={NUM_EXPERTS} must be divisible by ep_size={ep_size}")

    actual_pmf = actual_distribution_loadaware(
        ep_size=ep_size,
        batch_size=batch_size,
        base_p_eff=base_p_eff,
        p_eff_mode=p_eff_mode,
        score_mode=score_mode,
        bin_decimals=bin_decimals,
    )
    gumbel_pmf = predict_gumbel_distribution_loadaware(
        ep_size=ep_size,
        batch_size=batch_size,
        source_ep=source_ep,
        source_bs=source_bs,
        num_samples=num_samples,
        seed=seed,
        base_p_eff=base_p_eff,
        p_eff_mode=p_eff_mode,
        score_mode=score_mode,
        bin_decimals=bin_decimals,
    )

    xs = sorted(set(actual_pmf) | set(gumbel_pmf))
    actual_y = [actual_pmf.get(x, 0.0) for x in xs]
    gumbel_y = [gumbel_pmf.get(x, 0.0) for x in xs]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stem = (
        f"actual_vs_gumbel_loadaware_ep{ep_size}_bs{batch_size}"
        f"_source_ep{source_ep}_bs{source_bs}"
        f"_peff{str(base_p_eff).replace('.', 'p')}"
        f"_mode{p_eff_mode}"
        f"_score{score_mode}"
    )
    out_path = os.path.join(OUTPUT_DIR, f"{stem}.png")

    plt.figure(figsize=(11, 7))
    plt.plot(xs, actual_y, marker="s", label=f"Actual EP={ep_size}, bs={batch_size}")
    plt.plot(xs, gumbel_y, marker="o", label=f"Gumbel prediction from source bs={source_bs}")
    plt.xlabel("X")
    plt.ylabel("Probability")
    plt.title(
        "Actual vs Gumbel Distribution with Load-aware X\n"
        f"(EP={ep_size}, bs={batch_size}, P_eff mode={p_eff_mode}, score={score_mode})"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print("완료")
    print(f"actual expected X = {expected_value_from_pmf(actual_pmf):.6f}")
    print(f"gumbel expected X = {expected_value_from_pmf(gumbel_pmf):.6f}")
    print(f"plot: {out_path}")


if __name__ == "__main__":
    main()
