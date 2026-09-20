#!/usr/bin/env python3

import csv
import glob
import json
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

DEFAULT_GLOBAL_BS = 32
DEFAULT_PRODUCT = 4
DEFAULT_SOURCE_EP = 4
DEFAULT_SOURCE_BS = 1
DEFAULT_NUM_SAMPLES = 50000
DEFAULT_SEED = 42

OUTPUT_DIR = "./output/gumbel_expert_max_fixed_product"


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


def sample_expert_maxima(token_expert_array, batch_size, num_samples, rng):
    sampled_indices = rng.integers(0, len(token_expert_array), size=(num_samples, batch_size))
    sampled_tokens = token_expert_array[sampled_indices]

    expert_counts = np.zeros((num_samples, NUM_EXPERTS), dtype=np.int16)
    sample_axis = np.broadcast_to(
        np.arange(num_samples, dtype=np.int32)[:, None, None],
        (num_samples, batch_size, TOP_K),
    )
    np.add.at(expert_counts, (sample_axis.ravel(), sampled_tokens.ravel()), 1)
    return expert_counts.max(axis=1).astype(np.int32)


def discrete_gumbel_pmf(max_value, loc, scale):
    pmf = []
    for x_value in range(max_value + 1):
        cdf_x = gumbel_r.cdf(x_value, loc=loc, scale=scale)
        cdf_prev = gumbel_r.cdf(x_value - 1, loc=loc, scale=scale)
        prob = max(cdf_x - cdf_prev, 0.0)
        pmf.append(prob)

    total = sum(pmf)
    if total > 0.0:
        pmf = [value / total for value in pmf]
    return pmf


def predict_gumbel_distribution_from_source(batch_size, source_ep, source_bs, num_samples, seed):
    max_x = batch_size * TOP_K
    all_maxima = []

    for layer in range(NUM_LAYERS):
        file_path = find_layer_file(source_ep, source_bs, layer)
        if not file_path:
            continue

        token_expert_array = load_token_expert_array(file_path)
        if len(token_expert_array) == 0:
            continue

        rng = np.random.default_rng(seed + layer * 1237 + batch_size * 131)
        maxima = sample_expert_maxima(
            token_expert_array=token_expert_array,
            batch_size=batch_size,
            num_samples=num_samples,
            rng=rng,
        )
        all_maxima.append(maxima)

    if not all_maxima:
        raise RuntimeError(f"No valid layers found for source_ep={source_ep}, source_bs={source_bs}")

    pooled_maxima = np.concatenate(all_maxima)
    loc, scale = gumbel_r.fit(pooled_maxima)
    return discrete_gumbel_pmf(max_x, loc, scale)


def load_actual_expert_max_distribution(file_ep, file_bs):
    x_counter = Counter()
    total_steps = 0

    for layer in range(NUM_LAYERS):
        file_path = find_layer_file(file_ep, file_bs, layer)
        if not file_path:
            continue

        with open(file_path, "r", encoding="utf-8") as infile:
            for line_idx, line in enumerate(infile):
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)
                token_expert_lists = extract_token_expert_lists(data, TOP_K)

                expert_counts = [0] * NUM_EXPERTS
                for token_experts in token_expert_lists:
                    if len(token_experts) != TOP_K:
                        raise ValueError(
                            f"{file_path}, line {line_idx + 1}: token expert list length "
                            f"{len(token_experts)} != TOP_K {TOP_K}"
                        )
                    for expert_idx in token_experts:
                        expert_counts[expert_idx] += 1

                x_counter[max(expert_counts)] += 1
                total_steps += 1

    if total_steps == 0:
        raise RuntimeError(f"No actual data found for file_ep={file_ep}, file_bs={file_bs}")

    max_x = file_bs * TOP_K
    return [x_counter.get(x_value, 0) / total_steps for x_value in range(max_x + 1)]


def expected_value_from_dist(dist):
    return sum(idx * prob for idx, prob in enumerate(dist))


def tvd(dist_a, dist_b):
    size = max(len(dist_a), len(dist_b))
    a = dist_a + [0.0] * (size - len(dist_a))
    b = dist_b + [0.0] * (size - len(dist_b))
    return 0.5 * sum(abs(x - y) for x, y in zip(a, b))


def main():
    global_bs_raw = input(f"고정 global bs 입력 (기본 {DEFAULT_GLOBAL_BS}): ").strip()
    product_raw = input(f"고정할 PP x EP 값 입력 (기본 {DEFAULT_PRODUCT}): ").strip()
    source_ep_raw = input(f"source 로그의 EP Degree 입력 (기본 {DEFAULT_SOURCE_EP}): ").strip()
    source_bs_raw = input(f"source 로그의 batch size 입력 (기본 {DEFAULT_SOURCE_BS}): ").strip()
    num_samples_raw = input(f"Gumbel fit용 샘플 수 입력 (기본 {DEFAULT_NUM_SAMPLES}): ").strip()
    seed_raw = input(f"random seed 입력 (기본 {DEFAULT_SEED}): ").strip()

    global_bs = int(global_bs_raw) if global_bs_raw else DEFAULT_GLOBAL_BS
    fixed_product = int(product_raw) if product_raw else DEFAULT_PRODUCT
    source_ep = int(source_ep_raw) if source_ep_raw else DEFAULT_SOURCE_EP
    source_bs = int(source_bs_raw) if source_bs_raw else DEFAULT_SOURCE_BS
    num_samples = int(num_samples_raw) if num_samples_raw else DEFAULT_NUM_SAMPLES
    seed = int(seed_raw) if seed_raw else DEFAULT_SEED

    # Only compare PP values that divide global bs, satisfy EP*PP=fixed_product, and skip PP=4 as requested.
    pp_values = []
    for pp_degree in range(1, fixed_product + 1):
        if fixed_product % pp_degree != 0:
            continue
        if pp_degree == 4:
            continue
        if global_bs % pp_degree != 0:
            continue
        ep_degree = fixed_product // pp_degree
        if NUM_EXPERTS % max(ep_degree, 1) != 0:
            continue
        pp_values.append(pp_degree)

    if not pp_values:
        raise RuntimeError("No valid PP values found under the given constraints.")

    results = []
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.figure(figsize=(11, 7))

    for pp_degree in pp_values:
        ep_degree = fixed_product // pp_degree
        local_bs = global_bs // pp_degree

        predicted = predict_gumbel_distribution_from_source(
            batch_size=local_bs,
            source_ep=source_ep,
            source_bs=source_bs,
            num_samples=num_samples,
            seed=seed + pp_degree * 1000,
        )
        # Actual logs only exist for ep4 in this workspace; use the real token tuples for the corresponding local bs.
        actual = load_actual_expert_max_distribution(file_ep=4, file_bs=local_bs)

        size = max(len(predicted), len(actual))
        pred_pad = predicted + [0.0] * (size - len(predicted))
        act_pad = actual + [0.0] * (size - len(actual))

        xs = [idx for idx in range(size) if pred_pad[idx] > 0.0 or act_pad[idx] > 0.0]
        plt.plot(
            xs,
            [pred_pad[idx] for idx in xs],
            marker="o",
            label=f"Gumbel pred PP={pp_degree}, EP={ep_degree}, local bs={local_bs}",
        )
        plt.plot(
            xs,
            [act_pad[idx] for idx in xs],
            marker="s",
            linestyle="--",
            label=f"Actual tuples PP={pp_degree}, EP={ep_degree}, local bs={local_bs}",
        )

        results.append(
            {
                "pp": pp_degree,
                "ep": ep_degree,
                "local_bs": local_bs,
                "pred_expected_x": expected_value_from_dist(predicted),
                "actual_expected_x": expected_value_from_dist(actual),
                "tvd": tvd(predicted, actual),
            }
        )

    plt.xlabel("X = max expert token count")
    plt.ylabel("Probability")
    plt.title(f"Gumbel vs Actual Expert-Max Distribution (global bs={global_bs}, EP*PP={fixed_product})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(
        OUTPUT_DIR,
        f"gumbel_expert_max_globalbs{global_bs}_product{fixed_product}.png",
    )
    plt.savefig(plot_path, dpi=150)
    plt.close()

    csv_path = os.path.join(
        OUTPUT_DIR,
        f"gumbel_expert_max_globalbs{global_bs}_product{fixed_product}_summary.csv",
    )
    with open(csv_path, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(
            outfile,
            fieldnames=["pp", "ep", "local_bs", "pred_expected_x", "actual_expected_x", "tvd"],
        )
        writer.writeheader()
        writer.writerows(results)

    print("완료")
    print("주의: 현재 workspace에는 ep4 실제 로그만 있어서, actual 비교는 ep4 로그의 token tuples를 local bs 기준으로 사용했습니다.")
    print(f"plot: {plot_path}")
    print(f"summary: {csv_path}")
    for row in results:
        print(
            f"PP={row['pp']}, EP={row['ep']}, local_bs={row['local_bs']}: "
            f"pred_E[X]={row['pred_expected_x']:.6f}, "
            f"actual_E[X]={row['actual_expected_x']:.6f}, TVD={row['tvd']:.6f}"
        )


if __name__ == "__main__":
    main()
