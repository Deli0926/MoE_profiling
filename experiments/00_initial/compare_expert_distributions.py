
import csv
import glob
import json
import math
import os
from collections import Counter
from typing import Dict, List, Tuple

DIRECTORY = "./expert_selections"
MODEL_NAME = "gpt-oss-20b"
NUM_EXPERTS = 32
TOP_K = 4
NUM_LAYERS = 24

EP_SIZE = 4
THEORY_SOURCE_FILE_BS = 1
TARGET_BS_LIST = [1, 2, 4, 8, 16, 32]

OUTPUT_DIR = "./output/distribution_compare_ep4"


def calculate_probability_distribution(m: int, coef: List[List[float]], w_total: List[float], top_k_total: int) -> float:
    ep_size = len(coef)
    max_idx_per_gpu = len(coef[0]) - 1

    def get_sum_up_to(limit: int) -> float:
        if limit < 0:
            return 0.0
        dp = [0.0] * (top_k_total + 1)
        dp[0] = 1.0

        for g in range(ep_size):
            new_dp = [0.0] * (top_k_total + 1)
            upper_c = min(limit, max_idx_per_gpu, top_k_total)
            for c in range(upper_c + 1):
                weight = coef[g][c]
                if weight == 0:
                    continue
                for s in range(c, top_k_total + 1):
                    new_dp[s] += dp[s - c] * weight
            dp = new_dp
        return dp[top_k_total]

    numerator_m = get_sum_up_to(m)
    numerator_m_minus_1 = get_sum_up_to(m - 1)
    prob_sum = numerator_m - numerator_m_minus_1
    return prob_sum / w_total[top_k_total] if w_total[top_k_total] != 0 else 0.0


def smape(a: float, b: float) -> float:
    denom = abs(a) + abs(b)
    if denom == 0:
        return 0.0
    return 200.0 * abs(a - b) / denom


def find_layer_file(ep_size: int, file_bs: int, layer: int) -> str | None:
    pattern = os.path.join(
        DIRECTORY,
        f"expert_selection_openai_{MODEL_NAME}_wikitext_103_ep{ep_size}_bs{file_bs}_*_layer{layer}.jsonl",
    )
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def parse_line_to_expert_indices(data) -> List[int]:
    if len(data) > 0 and isinstance(data[0], list):
        return [expert for sublist in data for expert in sublist]
    return data


def load_theory_distribution_for_bs(ep_size: int, analysis_bs: int, theory_source_file_bs: int) -> List[float]:
    max_m = min(TOP_K * analysis_bs, NUM_EXPERTS)
    final_cond_prob_sum = [0.0] * (max_m + 1)
    valid_layers = 0

    for layer in range(NUM_LAYERS):
        file_path = find_layer_file(ep_size, theory_source_file_bs, layer)
        if not file_path:
            continue

        layer_counts = [0] * NUM_EXPERTS
        layer_tokens = 0

        with open(file_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                data = json.loads(raw_line)
                expert_indices = parse_line_to_expert_indices(data)
                batch_tokens = len(expert_indices) // TOP_K
                layer_tokens += batch_tokens
                for expert in expert_indices:
                    layer_counts[expert] += 1

        layer_total_count = sum(layer_counts)
        if layer_total_count == 0 or layer_tokens == 0:
            continue

        valid_layers += 1
        probabilities = []
        for count in layer_counts:
            p = count / (layer_tokens * TOP_K)
            q = 1 - (1 - p) ** analysis_bs
            probabilities.append(q)

        experts_per_gpu = NUM_EXPERTS // ep_size
        coef = [[1.0] + [0.0] * experts_per_gpu for _ in range(ep_size)]
        coef_total = [1.0] + [0.0] * NUM_EXPERTS

        for gpu in range(ep_size):
            for i in range(experts_per_gpu):
                expert_idx = experts_per_gpu * gpu + i
                prob_q = probabilities[expert_idx]

                for j in range(i + 1, 0, -1):
                    coef[gpu][j] = coef[gpu][j] + coef[gpu][j - 1] * prob_q

                for j in range(expert_idx + 1, 0, -1):
                    coef_total[j] = coef_total[j] + coef_total[j - 1] * prob_q

        sum_of_coef = sum(coef_total[j] for j in range(TOP_K, max_m + 1))
        layer_cond_prob = [0.0] * (max_m + 1)

        for k in range(TOP_K, max_m + 1):
            prob_k = coef_total[k] / sum_of_coef if sum_of_coef != 0 else 0.0
            for m in range(0, max_m + 1):
                prob = calculate_probability_distribution(m, coef, coef_total, k)
                layer_cond_prob[m] += prob_k * prob

        for i in range(max_m + 1):
            final_cond_prob_sum[i] += layer_cond_prob[i]

    if valid_layers == 0:
        raise FileNotFoundError(
            f"No valid theory layers found for EP={ep_size}, source file bs={theory_source_file_bs}"
        )

    return [value / valid_layers for value in final_cond_prob_sum]


def load_real_distribution_for_bs(ep_size: int, real_file_bs: int) -> List[float]:
    experts_per_gpu = NUM_EXPERTS // ep_size
    y_counts: Counter[int] = Counter()
    total_steps = 0

    for layer in range(NUM_LAYERS):
        file_path = find_layer_file(ep_size, real_file_bs, layer)
        if not file_path:
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                data = json.loads(raw_line)
                all_indices = parse_line_to_expert_indices(data)

                unique_experts = set(all_indices)
                gpu_unique_counts = [0] * ep_size
                for exp_idx in unique_experts:
                    gpu_id = exp_idx // experts_per_gpu
                    if gpu_id < ep_size:
                        gpu_unique_counts[gpu_id] += 1

                y = max(gpu_unique_counts)
                y_counts[y] += 1
                total_steps += 1

    if total_steps == 0:
        raise FileNotFoundError(f"No real data found for EP={ep_size}, real file bs={real_file_bs}")

    max_m = min(TOP_K * real_file_bs, NUM_EXPERTS)
    return [y_counts[m] / total_steps for m in range(max_m + 1)]


def summarize_errors(theory: List[float], real: List[float]) -> Dict[str, float]:
    size = max(len(theory), len(real))
    t = theory + [0.0] * (size - len(theory))
    r = real + [0.0] * (size - len(real))
    abs_errors = [abs(a - b) for a, b in zip(t, r)]
    squared_errors = [(a - b) ** 2 for a, b in zip(t, r)]
    smapes = [smape(a, b) for a, b in zip(t, r)]

    l1 = sum(abs_errors)
    return {
        "L1": l1,
        "TVD": 0.5 * l1,
        "MAE": sum(abs_errors) / size if size else 0.0,
        "RMSE": math.sqrt(sum(squared_errors) / size) if size else 0.0,
        "mean_sMAPE_percent": sum(smapes) / size if size else 0.0,
        "max_abs_error": max(abs_errors) if abs_errors else 0.0,
    }


def print_distribution_table(title: str, probs: List[float]) -> None:
    print(title)
    for m, prob in enumerate(probs):
        if prob > 1e-12:
            print(f"  P(Y = {m:2d}) = {prob:.6f} ({prob * 100:7.4f}%)")
    print()


def print_comparison_table(theory: List[float], real: List[float]) -> None:
    size = max(len(theory), len(real))
    t = theory + [0.0] * (size - len(theory))
    r = real + [0.0] * (size - len(real))

    print("  m | theory       | real         | abs_error    | sMAPE(%)")
    print("----+--------------+--------------+--------------+----------")
    for m in range(size):
        print(f"{m:3d} | {t[m]:.8f}   | {r[m]:.8f}   | {abs(t[m]-r[m]):.8f}   | {smape(t[m], r[m]):8.3f}")
    print()


def write_comparison_csv(path: str, theory: List[float], real: List[float]) -> None:
    size = max(len(theory), len(real))
    t = theory + [0.0] * (size - len(theory))
    r = real + [0.0] * (size - len(real))

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["m", "theory_prob", "real_prob", "abs_error", "sMAPE_percent"])
        for m in range(size):
            writer.writerow([m, t[m], r[m], abs(t[m] - r[m]), smape(t[m], r[m])])


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print("EP=4 비교: theory는 bs=1 데이터 고정, real은 각 bs 데이터 사용")
    print("=" * 80)
    print()

    summary_rows = []

    for bs in TARGET_BS_LIST:
        print("#" * 80)
        print(f"[bs = {bs}] 시작")
        print("#" * 80)

        theory = load_theory_distribution_for_bs(
            ep_size=EP_SIZE,
            analysis_bs=bs,
            theory_source_file_bs=THEORY_SOURCE_FILE_BS,
        )
        real = load_real_distribution_for_bs(
            ep_size=EP_SIZE,
            real_file_bs=bs,
        )

        print_distribution_table(
            f"[Theory distribution] source file bs={THEORY_SOURCE_FILE_BS}, analysis bs={bs}",
            theory,
        )
        print_distribution_table(
            f"[Real distribution] real file bs={bs}",
            real,
        )

        print("[Comparison by m]")
        print_comparison_table(theory, real)

        summary = summarize_errors(theory, real)
        print("[Summary errors]")
        for key, value in summary.items():
            print(f"  {key:18s}: {value:.8f}")
        print()

        comparison_csv = os.path.join(
            OUTPUT_DIR,
            f"ep{EP_SIZE}_theorybs{THEORY_SOURCE_FILE_BS}_vs_realbs{bs}.csv",
        )
        write_comparison_csv(comparison_csv, theory, real)

        summary_rows.append({
            "ep": EP_SIZE,
            "theory_source_file_bs": THEORY_SOURCE_FILE_BS,
            "analysis_bs": bs,
            **summary,
            "comparison_csv": comparison_csv,
        })

    summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "ep",
            "theory_source_file_bs",
            "analysis_bs",
            "L1",
            "TVD",
            "MAE",
            "RMSE",
            "mean_sMAPE_percent",
            "max_abs_error",
            "comparison_csv",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("=" * 80)
    print("완료")
    print(f"요약 파일: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
