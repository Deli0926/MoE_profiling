import csv
import glob
import json
import math
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

DIRECTORY = "./expert_selections"
MODEL_NAME = "gpt-oss-20b"
NUM_EXPERTS = 32
TOP_K = 4
NUM_LAYERS = 24

EP_SIZE = 4
THEORY_SOURCE_FILE_BS = 1
TARGET_BS_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256]

OUTPUT_DIR = "./output/distribution_compare_ep4_bs1_routed_tokens"


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


def extract_token_expert_lists(data, top_k: int) -> List[List[int]]:
    """
    경우 1) [[e1,e2,e3,e4], [...], ...] 형태면 그대로 반환
    경우 2) [e1,e2,e3,e4,e5,...] flat 형태면 top_k씩 끊어서 복원
    """
    if len(data) > 0 and isinstance(data[0], list):
        return data

    if len(data) % top_k != 0:
        raise ValueError(f"flat data length {len(data)} is not divisible by top_k={top_k}")

    return [data[i:i + top_k] for i in range(0, len(data), top_k)]


def token_to_gpu_pattern(token_experts: List[int], num_experts: int, ep_size: int) -> Tuple[int, ...]:
    """
    token 하나의 top-k expert 선택을 GPU별 routed-token load pattern으로 변환
    예: EP=4, num_experts=32, experts_per_gpu=8
        [3, 7, 10, 28] -> (2,1,0,1)
    """
    experts_per_gpu = num_experts // ep_size
    counts = [0] * ep_size

    for e in token_experts:
        if e < 0 or e >= num_experts:
            raise ValueError(f"expert index {e} out of range [0, {num_experts - 1}]")
        gpu_id = e // experts_per_gpu
        counts[gpu_id] += 1

    return tuple(counts)


def build_token_pattern_pmf(file_path: str, top_k: int, num_experts: int, ep_size: int) -> Tuple[Dict[Tuple[int, ...], float], int]:
    """
    bs=1 소스 파일 하나에서 token-level GPU routed-load pattern PMF를 구축
    """
    pattern_counter: Counter[Tuple[int, ...]] = Counter()
    total_tokens = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line_idx, raw_line in enumerate(f):
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            data = json.loads(raw_line)
            token_expert_lists = extract_token_expert_lists(data, top_k)

            for token_experts in token_expert_lists:
                if len(token_experts) != top_k:
                    raise ValueError(
                        f"{file_path}, line {line_idx + 1}: token expert list length "
                        f"{len(token_experts)} != top_k {top_k}"
                    )

                pattern = token_to_gpu_pattern(token_experts, num_experts, ep_size)
                pattern_counter[pattern] += 1
                total_tokens += 1

    if total_tokens == 0:
        return {}, 0

    pattern_pmf = {
        pattern: cnt / total_tokens
        for pattern, cnt in pattern_counter.items()
    }
    return pattern_pmf, total_tokens


def convolve_token_patterns(pattern_pmf: Dict[Tuple[int, ...], float], bs: int, ep_size: int) -> Dict[Tuple[int, ...], float]:
    """
    token-level GPU pattern PMF를 bs번 합성하여
    batch 전체 GPU routed-token load vector 분포를 계산
    """
    zero_vec = tuple([0] * ep_size)
    dp: Dict[Tuple[int, ...], float] = {zero_vec: 1.0}

    for _ in range(bs):
        new_dp = defaultdict(float)
        for load_vec, prob1 in dp.items():
            for pattern, prob2 in pattern_pmf.items():
                new_vec = tuple(load_vec[g] + pattern[g] for g in range(ep_size))
                new_dp[new_vec] += prob1 * prob2
        dp = dict(new_dp)

    return dp


def load_vec_dist_to_max_dist(load_vec_dist: Dict[Tuple[int, ...], float], bs: int, top_k: int) -> List[float]:
    """
    load vector distribution -> max routed tokens per GPU distribution
    Y = max_g L_g
    """
    max_total = bs * top_k
    max_dist = [0.0] * (max_total + 1)

    for load_vec, prob in load_vec_dist.items():
        m = max(load_vec)
        max_dist[m] += prob

    return max_dist


def load_theory_distribution_for_bs(ep_size: int, analysis_bs: int, theory_source_file_bs: int) -> List[float]:
    """
    bs=1 이론 소스 데이터로부터 token-level GPU pattern PMF를 만들고,
    analysis_bs에 대해 max routed tokens per GPU 분포를 예측
    """
    max_m = analysis_bs * TOP_K
    final_cond_prob_sum = [0.0] * (max_m + 1)
    valid_layers = 0

    for layer in range(NUM_LAYERS):
        file_path = find_layer_file(ep_size, theory_source_file_bs, layer)
        if not file_path:
            continue

        pattern_pmf, total_tokens = build_token_pattern_pmf(
            file_path=file_path,
            top_k=TOP_K,
            num_experts=NUM_EXPERTS,
            ep_size=ep_size,
        )

        if total_tokens == 0:
            continue

        load_vec_dist = convolve_token_patterns(
            pattern_pmf=pattern_pmf,
            bs=analysis_bs,
            ep_size=ep_size,
        )

        layer_cond_prob = load_vec_dist_to_max_dist(
            load_vec_dist=load_vec_dist,
            bs=analysis_bs,
            top_k=TOP_K,
        )

        for i in range(max_m + 1):
            final_cond_prob_sum[i] += layer_cond_prob[i]

        valid_layers += 1

    if valid_layers == 0:
        raise FileNotFoundError(
            f"No valid theory layers found for EP={ep_size}, source file bs={theory_source_file_bs}"
        )

    return [value / valid_layers for value in final_cond_prob_sum]


def load_real_distribution_for_bs(ep_size: int, real_file_bs: int) -> List[float]:
    """
    실제 bs 데이터에서 step마다 GPU별 routed token 수를 직접 세고,
    max routed tokens per GPU의 empirical 분포를 계산
    """
    experts_per_gpu = NUM_EXPERTS // ep_size
    y_counts: Counter[int] = Counter()
    total_steps = 0

    for layer in range(NUM_LAYERS):
        file_path = find_layer_file(ep_size, real_file_bs, layer)
        if not file_path:
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, raw_line in enumerate(f):
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
                        if gpu_id < ep_size:
                            gpu_routed_counts[gpu_id] += 1
                        else:
                            raise ValueError(
                                f"{file_path}, line {line_idx + 1}: invalid gpu_id={gpu_id} "
                                f"for exp_idx={exp_idx}"
                            )

                y = max(gpu_routed_counts)
                y_counts[y] += 1
                total_steps += 1

    if total_steps == 0:
        raise FileNotFoundError(f"No real data found for EP={ep_size}, real file bs={real_file_bs}")

    max_m = real_file_bs * TOP_K
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


def expected_value_from_dist(dist: List[float]) -> float:
    return sum(i * p for i, p in enumerate(dist))


def percentile_from_dist(dist: List[float], q: float) -> int:
    cdf = 0.0
    for i, p in enumerate(dist):
        cdf += p
        if cdf >= q:
            return i
    return len(dist) - 1


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
        print(f"{m:3d} | {t[m]:.8f} | {r[m]:.8f} | {abs(t[m]-r[m]):.8f} | {smape(t[m], r[m]):8.3f}")
    print()


def write_comparison_csv(path: str, theory: List[float], real: List[float]) -> None:
    size = max(len(theory), len(real))
    t = theory + [0.0] * (size - len(theory))
    r = real + [0.0] * (size - len(real))

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "m",
            "theory_prob",
            "real_prob",
            "abs_error",
            "sMAPE_percent",
        ])
        for m in range(size):
            writer.writerow([m, t[m], r[m], abs(t[m] - r[m]), smape(t[m], r[m])])


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 90)
    print("EP=4 비교: theory는 bs=1 데이터 기반 예측, real은 각 bs 실제 데이터")
    print("비교 대상: max routed tokens per GPU distribution")
    print("=" * 90)
    print()

    summary_rows = []

    for bs in TARGET_BS_LIST:
        print("#" * 90)
        print(f"[bs = {bs}] 시작")
        print("#" * 90)

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

        theory_mean = expected_value_from_dist(theory)
        real_mean = expected_value_from_dist(real)
        theory_p95 = percentile_from_dist(theory, 0.95)
        real_p95 = percentile_from_dist(real, 0.95)

        avg_load = (bs * TOP_K) / EP_SIZE
        theory_ir = theory_mean / avg_load if avg_load > 0 else 0.0
        real_ir = real_mean / avg_load if avg_load > 0 else 0.0

        print("[Comparison by m]")
        print_comparison_table(theory, real)

        summary = summarize_errors(theory, real)
        print("[Summary errors]")
        for key, value in summary.items():
            print(f"  {key:18s}: {value:.8f}")

        print("[Additional stats]")
        print(f"  theory E[max]     : {theory_mean:.8f}")
        print(f"  real   E[max]     : {real_mean:.8f}")
        print(f"  theory p95        : {theory_p95}")
        print(f"  real   p95        : {real_p95}")
        print(f"  theory imbalance  : {theory_ir:.8f}")
        print(f"  real   imbalance  : {real_ir:.8f}")
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
            "theory_E_max": theory_mean,
            "real_E_max": real_mean,
            "theory_p95": theory_p95,
            "real_p95": real_p95,
            "theory_imbalance_ratio": theory_ir,
            "real_imbalance_ratio": real_ir,
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
            "theory_E_max",
            "real_E_max",
            "theory_p95",
            "real_p95",
            "theory_imbalance_ratio",
            "real_imbalance_ratio",
            "comparison_csv",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("=" * 90)
    print("완료")
    print(f"요약 파일: {summary_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
