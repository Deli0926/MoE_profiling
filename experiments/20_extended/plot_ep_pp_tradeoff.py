#!/usr/bin/env python3

import csv
import glob
import json
import math
import os
from collections import Counter, defaultdict

import matplotlib.pyplot as plt


DIRECTORY = "./expert_selections"
MODEL_NAME = "gpt-oss-20b"
NUM_EXPERTS = 32
TOP_K = 4
NUM_LAYERS = 24
DEFAULT_SOURCE_EP = 4
DEFAULT_SOURCE_BS = 1
DEFAULT_GLOBAL_BS = 32
OUTPUT_DIR = "./output/ep_pp_tradeoff"


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
        return {}, 0

    pattern_pmf = {
        pattern: count / total_tokens
        for pattern, count in pattern_counter.items()
    }
    return pattern_pmf, total_tokens


def convolve_distributions(dist_a, dist_b, ep_size):
    result = defaultdict(float)

    for load_vec_a, prob_a in dist_a.items():
        for load_vec_b, prob_b in dist_b.items():
            new_vec = tuple(load_vec_a[g] + load_vec_b[g] for g in range(ep_size))
            result[new_vec] += prob_a * prob_b

    return dict(result)


def power_pattern_distribution(pattern_pmf, batch_size, ep_size):
    zero_vec = tuple([0] * ep_size)
    result = {zero_vec: 1.0}
    base = dict(pattern_pmf)
    exponent = batch_size

    while exponent > 0:
        if exponent & 1:
            result = convolve_distributions(result, base, ep_size)
        exponent >>= 1
        if exponent > 0:
            base = convolve_distributions(base, base, ep_size)

    return result


def load_vec_dist_to_max_dist(load_vec_dist, batch_size):
    max_total = batch_size * TOP_K
    max_dist = [0.0] * (max_total + 1)

    for load_vec, prob in load_vec_dist.items():
        max_dist[max(load_vec)] += prob

    return max_dist


def expected_value_from_dist(dist):
    return sum(idx * prob for idx, prob in enumerate(dist))


def percentile_from_dist(dist, q):
    cdf = 0.0
    for idx, prob in enumerate(dist):
        cdf += prob
        if cdf >= q:
            return idx
    return len(dist) - 1


def find_layer_file(source_ep, source_bs, layer):
    pattern = os.path.join(
        DIRECTORY,
        f"expert_selection_openai_{MODEL_NAME}_wikitext_103_ep{source_ep}_bs{source_bs}_*_layer{layer}.jsonl",
    )
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def effective_micro_batch(global_bs, pp_degree):
    return math.ceil(global_bs / pp_degree)


def predict_distribution_for_ep_pp(target_ep, global_bs, pp_degree, source_ep, source_bs):
    if NUM_EXPERTS % target_ep != 0:
        raise ValueError(f"num_of_experts={NUM_EXPERTS} must be divisible by ep_size={target_ep}")

    local_bs = effective_micro_batch(global_bs, pp_degree)
    max_m = local_bs * TOP_K
    final_cond_prob_sum = [0.0] * (max_m + 1)
    final_expected_max_sum = 0.0
    final_p95_sum = 0.0
    final_imbalance_ratio_sum = 0.0
    valid_layers = 0

    for layer in range(NUM_LAYERS):
        file_path = find_layer_file(source_ep, source_bs, layer)
        if not file_path:
            continue

        pattern_pmf, total_tokens = build_token_pattern_pmf(
            file_path=file_path,
            top_k=TOP_K,
            num_of_experts=NUM_EXPERTS,
            ep_size=target_ep,
        )
        if total_tokens == 0:
            continue

        load_vec_dist = power_pattern_distribution(
            pattern_pmf=pattern_pmf,
            batch_size=local_bs,
            ep_size=target_ep,
        )
        layer_dist = load_vec_dist_to_max_dist(load_vec_dist, local_bs)
        expected_max = expected_value_from_dist(layer_dist)
        p95 = percentile_from_dist(layer_dist, 0.95)
        avg_load = (local_bs * TOP_K) / target_ep
        imbalance_ratio = expected_max / avg_load if avg_load > 0 else 0.0

        for idx in range(max_m + 1):
            final_cond_prob_sum[idx] += layer_dist[idx]
        final_expected_max_sum += expected_max
        final_p95_sum += p95
        final_imbalance_ratio_sum += imbalance_ratio
        valid_layers += 1

    if valid_layers == 0:
        raise RuntimeError(f"No valid layers found for source_ep={source_ep}, source_bs={source_bs}")

    avg_dist = [value / valid_layers for value in final_cond_prob_sum]
    return {
        "ep": target_ep,
        "pp": pp_degree,
        "global_bs": global_bs,
        "local_bs": local_bs,
        "distribution": avg_dist,
        "expected_max": final_expected_max_sum / valid_layers,
        "p95": final_p95_sum / valid_layers,
        "imbalance_ratio": final_imbalance_ratio_sum / valid_layers,
        "avg_load": (local_bs * TOP_K) / target_ep,
        "valid_layers": valid_layers,
    }


def write_summary_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(
            outfile,
            fieldnames=[
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


def plot_metric_by_ep(rows, metric, ylabel, out_path):
    plt.figure(figsize=(10, 6))
    pp_values = sorted(set(row["pp_degree"] for row in rows))

    for pp_degree in pp_values:
        selected = sorted(
            [row for row in rows if row["pp_degree"] == pp_degree],
            key=lambda row: row["target_ep"],
        )
        xs = [row["target_ep"] for row in selected]
        ys = [row[metric] for row in selected]
        local_bs = selected[0]["local_bs"] if selected else "?"
        plt.plot(xs, ys, marker="o", label=f"PP={pp_degree} (local bs={local_bs})")

    plt.xlabel("EP Degree")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} across EP / PP choices")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_distribution_grid(results, out_path):
    pp_values = sorted(set(result["pp"] for result in results))
    fig, axes = plt.subplots(len(pp_values), 1, figsize=(11, 4 * len(pp_values)), squeeze=False)

    for row_idx, pp_degree in enumerate(pp_values):
        ax = axes[row_idx][0]
        selected = sorted(
            [result for result in results if result["pp"] == pp_degree],
            key=lambda result: result["ep"],
        )
        for result in selected:
            xs = [idx for idx, prob in enumerate(result["distribution"]) if prob > 0.0]
            ys = [result["distribution"][idx] for idx in xs]
            ax.plot(xs, ys, marker="o", label=f"EP={result['ep']}")

        local_bs = selected[0]["local_bs"] if selected else "?"
        ax.set_title(f"PP={pp_degree}, local bs={local_bs}")
        ax.set_xlabel("Max routed tokens per GPU")
        ax.set_ylabel("Probability")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    global_bs_raw = input(f"global batch size 입력 (기본 {DEFAULT_GLOBAL_BS}): ").strip()
    ep_values_raw = input("비교할 EP Degree 목록 입력 (예: 2,4,8,16): ").strip()
    pp_values_raw = input("비교할 PP Degree 목록 입력 (예: 1,2,4,8): ").strip()
    source_ep_raw = input(f"source 로그의 EP Degree 입력 (기본 {DEFAULT_SOURCE_EP}): ").strip()
    source_bs_raw = input(f"source 로그의 batch size 입력 (기본 {DEFAULT_SOURCE_BS}): ").strip()

    global_bs = int(global_bs_raw) if global_bs_raw else DEFAULT_GLOBAL_BS
    source_ep = int(source_ep_raw) if source_ep_raw else DEFAULT_SOURCE_EP
    source_bs = int(source_bs_raw) if source_bs_raw else DEFAULT_SOURCE_BS

    if not ep_values_raw:
        raise ValueError("비교할 EP Degree 목록이 필요합니다.")
    if not pp_values_raw:
        raise ValueError("비교할 PP Degree 목록이 필요합니다.")

    target_eps = [int(value.strip()) for value in ep_values_raw.split(",") if value.strip()]
    pp_degrees = [int(value.strip()) for value in pp_values_raw.split(",") if value.strip()]

    for target_ep in target_eps:
        if NUM_EXPERTS % target_ep != 0:
            raise ValueError(f"target EP {target_ep} must divide NUM_EXPERTS={NUM_EXPERTS}")
    for pp_degree in pp_degrees:
        if pp_degree <= 0:
            raise ValueError("PP Degree는 양수여야 합니다.")

    print(
        "global bs를 고정하고 EP / PP 조합별 max routed-token load를 계산 중...\n"
        f"(global bs={global_bs}, source ep={source_ep}, source bs={source_bs})"
    )
    print(
        "가정: PP를 적용하면 각 stage에서 한 번에 처리하는 micro-batch 크기는\n"
        "ceil(global_bs / PP) 입니다."
    )

    results = []
    summary_rows = []

    for pp_degree in pp_degrees:
        local_bs = effective_micro_batch(global_bs, pp_degree)
        print("============================================================")
        print(f"PP = {pp_degree}, local bs = {local_bs}")
        print("============================================================")

        for target_ep in target_eps:
            result = predict_distribution_for_ep_pp(
                target_ep=target_ep,
                global_bs=global_bs,
                pp_degree=pp_degree,
                source_ep=source_ep,
                source_bs=source_bs,
            )
            results.append(result)
            summary_rows.append({
                "target_ep": target_ep,
                "pp_degree": pp_degree,
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
                f"EP={target_ep:2d} | "
                f"E[X]={result['expected_max']:.6f} | "
                f"p95={result['p95']:.6f} | "
                f"avg={result['avg_load']:.6f} | "
                f"ratio={result['imbalance_ratio']:.6f}"
            )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_name = f"globalbs{global_bs}_source_ep{source_ep}_bs{source_bs}"

    summary_csv = os.path.join(OUTPUT_DIR, f"{base_name}_summary.csv")
    expected_png = os.path.join(OUTPUT_DIR, f"{base_name}_expected_max.png")
    p95_png = os.path.join(OUTPUT_DIR, f"{base_name}_p95.png")
    ratio_png = os.path.join(OUTPUT_DIR, f"{base_name}_imbalance_ratio.png")
    dist_png = os.path.join(OUTPUT_DIR, f"{base_name}_distribution_grid.png")

    write_summary_csv(summary_csv, summary_rows)
    plot_metric_by_ep(summary_rows, "expected_max", "Expected max routed tokens per GPU", expected_png)
    plot_metric_by_ep(summary_rows, "p95", "p95 of max routed tokens per GPU", p95_png)
    plot_metric_by_ep(summary_rows, "imbalance_ratio", "Imbalance Ratio", ratio_png)
    plot_distribution_grid(results, dist_png)

    print("\n완료")
    print(f"summary csv: {summary_csv}")
    print(f"expected max plot: {expected_png}")
    print(f"p95 plot: {p95_png}")
    print(f"imbalance ratio plot: {ratio_png}")
    print(f"distribution grid: {dist_png}")


if __name__ == "__main__":
    main()
