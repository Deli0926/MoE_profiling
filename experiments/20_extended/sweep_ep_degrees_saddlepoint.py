#!/usr/bin/env python3

import csv
import glob
import json
import math
import os
from collections import Counter

import matplotlib.pyplot as plt
from scipy.optimize import brentq


DIRECTORY = "./expert_selections"
MODEL_NAME = "gpt-oss-20b"
NUM_EXPERTS = 32
TOP_K = 4
NUM_LAYERS = 24
DEFAULT_SOURCE_EP = 4
DEFAULT_SOURCE_BS = 1
OUTPUT_DIR = "./output/ep_sweep_saddlepoint"


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


def find_layer_file(source_ep, source_bs, layer):
    pattern = os.path.join(
        DIRECTORY,
        f"expert_selection_openai_{MODEL_NAME}_wikitext_103_ep{source_ep}_bs{source_bs}_*_layer{layer}.jsonl",
    )
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def expected_value_from_dist(dist):
    return sum(idx * prob for idx, prob in enumerate(dist))


def percentile_from_dist(dist, q):
    cdf = 0.0
    for idx, prob in enumerate(dist):
        cdf += prob
        if cdf >= q:
            return idx
    return len(dist) - 1


def marginal_count_pmf(pattern_pmf, gpu_idx):
    pmf = [0.0] * (TOP_K + 1)
    for pattern, prob in pattern_pmf.items():
        pmf[pattern[gpu_idx]] += prob
    return pmf


class SaddlepointMarginal:
    def __init__(self, single_token_pmf, batch_size):
        self.single_token_pmf = single_token_pmf
        self.batch_size = batch_size
        self.support = [value for value, prob in enumerate(single_token_pmf) if prob > 0.0]
        self.min_single = min(self.support)
        self.max_single = max(self.support)
        self.min_sum = self.batch_size * self.min_single
        self.max_sum = self.batch_size * self.max_single

    def _tilted_stats(self, t):
        support = [(value, prob) for value, prob in enumerate(self.single_token_pmf) if prob > 0.0]
        shift = max(t * value for value, _ in support)

        weighted_sum = 0.0
        weighted_first = 0.0
        weighted_second = 0.0

        for value, prob in support:
            weight = prob * math.exp(t * value - shift)
            weighted_sum += weight
            weighted_first += value * weight
            weighted_second += (value ** 2) * weight

        mean_tilted = weighted_first / weighted_sum
        second_moment_tilted = weighted_second / weighted_sum
        return shift, weighted_sum, mean_tilted, second_moment_tilted

    def kgf(self, t):
        shift, weighted_sum, _, _ = self._tilted_stats(t)
        return self.batch_size * (shift + math.log(weighted_sum))

    def kgf_prime(self, t):
        _, _, mean_tilted, _ = self._tilted_stats(t)
        return self.batch_size * mean_tilted

    def kgf_double_prime(self, t):
        _, _, mean_tilted, second_moment_tilted = self._tilted_stats(t)
        return self.batch_size * (second_moment_tilted - mean_tilted ** 2)

    def solve_saddlepoint(self, target_sum):
        mean = self.kgf_prime(0.0)
        if abs(target_sum - mean) < 1e-12:
            return 0.0

        def objective(t):
            return self.kgf_prime(t) - target_sum

        if target_sum < mean:
            left, right = -20.0, 0.0
            while objective(left) > 0:
                left *= 2.0
                if not math.isfinite(left):
                    break
        else:
            left, right = 0.0, 20.0
            while objective(right) < 0:
                right *= 2.0
                if not math.isfinite(right):
                    break

        return brentq(objective, left, right, maxiter=200)

    def point_mass(self, target_sum):
        if target_sum < self.min_sum or target_sum > self.max_sum:
            return 0.0

        if target_sum == self.min_sum:
            return self.single_token_pmf[self.min_single] ** self.batch_size

        if target_sum == self.max_sum:
            return self.single_token_pmf[self.max_single] ** self.batch_size

        t_hat = self.solve_saddlepoint(target_sum)
        k2 = self.kgf_double_prime(t_hat)
        if k2 <= 0:
            return 0.0

        log_prob = self.kgf(t_hat) - t_hat * target_sum - 0.5 * math.log(2.0 * math.pi * k2)
        return math.exp(log_prob)


def normalize_distribution(dist):
    total = sum(dist)
    if total <= 0:
        return dist
    return [value / total for value in dist]


def marginal_saddlepoint_distribution(pattern_pmf, predicted_bs, ep_size):
    max_load = predicted_bs * TOP_K
    marginal_pmfs = []

    for gpu_idx in range(ep_size):
        single_token_pmf = marginal_count_pmf(pattern_pmf, gpu_idx)
        saddle = SaddlepointMarginal(single_token_pmf, predicted_bs)
        marginal = [saddle.point_mass(total) for total in range(max_load + 1)]
        marginal_pmfs.append(normalize_distribution(marginal))

    return marginal_pmfs


def approximate_max_distribution_from_marginals(marginal_pmfs):
    max_load = len(marginal_pmfs[0]) - 1
    cdfs = []
    for pmf in marginal_pmfs:
        cdf = []
        running = 0.0
        for prob in pmf:
            running += prob
            cdf.append(min(max(running, 0.0), 1.0))
        cdfs.append(cdf)

    dist = [0.0] * (max_load + 1)
    prev = 0.0
    for load in range(max_load + 1):
        cdf_product = 1.0
        for cdf in cdfs:
            cdf_product *= cdf[load]
        dist[load] = max(cdf_product - prev, 0.0)
        prev = cdf_product

    return normalize_distribution(dist)


def predict_distribution_for_ep(target_ep, predicted_bs, source_ep, source_bs):
    if NUM_EXPERTS % target_ep != 0:
        raise ValueError(f"num_of_experts={NUM_EXPERTS} must be divisible by ep_size={target_ep}")

    max_m = predicted_bs * TOP_K
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

        marginal_pmfs = marginal_saddlepoint_distribution(
            pattern_pmf=pattern_pmf,
            predicted_bs=predicted_bs,
            ep_size=target_ep,
        )
        layer_dist = approximate_max_distribution_from_marginals(marginal_pmfs)
        expected_max = expected_value_from_dist(layer_dist)
        p95 = percentile_from_dist(layer_dist, 0.95)
        avg_load = (predicted_bs * TOP_K) / target_ep
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
        f"Saddlepoint-approximated distributions across EP degrees "
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
    plt.title(f"{ylabel} across EP degrees (saddlepoint approximation)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    predicted_bs = int(input("예측할 배치 사이즈(bs) 입력: "))
    ep_values_raw = input("비교할 EP Degree 목록 입력 (예: 2,4,8,16): ").strip()
    source_ep_raw = input(f"source 로그의 EP Degree 입력 (기본 {DEFAULT_SOURCE_EP}): ").strip()
    source_bs_raw = input(f"source 로그의 batch size 입력 (기본 {DEFAULT_SOURCE_BS}): ").strip()

    if not ep_values_raw:
        raise ValueError("비교할 EP Degree 목록이 필요합니다.")

    target_eps = [int(value.strip()) for value in ep_values_raw.split(",") if value.strip()]
    source_ep = int(source_ep_raw) if source_ep_raw else DEFAULT_SOURCE_EP
    source_bs = int(source_bs_raw) if source_bs_raw else DEFAULT_SOURCE_BS

    for target_ep in target_eps:
        if NUM_EXPERTS % target_ep != 0:
            raise ValueError(f"target EP {target_ep} must divide NUM_EXPERTS={NUM_EXPERTS}")

    print(
        "고정된 expert selection 로그를 사용해 EP degree별 saddlepoint 근사 분포를 계산 중...\n"
        f"(source ep={source_ep}, source bs={source_bs}, predicted bs={predicted_bs})"
    )
    print(
        "주의: 현재 구현은 GPU별 marginal load에 saddlepoint를 적용한 뒤,\n"
        "max 분포는 GPU 간 independence approximation으로 조합합니다."
    )

    results = []
    summary_rows = []

    for target_ep in target_eps:
        result = predict_distribution_for_ep(
            target_ep=target_ep,
            predicted_bs=predicted_bs,
            source_ep=source_ep,
            source_bs=source_bs,
        )
        results.append(result)
        summary_rows.append({
            "target_ep": target_ep,
            "predicted_bs": predicted_bs,
            "source_ep": source_ep,
            "source_bs": source_bs,
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
    base_name = f"saddle_predbs{predicted_bs}_from_source_ep{source_ep}_bs{source_bs}"

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
