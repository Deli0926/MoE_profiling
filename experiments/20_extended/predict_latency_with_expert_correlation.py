#!/usr/bin/env python3

import csv
import glob
import json
import os
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np


DIRECTORY = "./expert_selections"
MODEL_NAME = "gpt-oss-20b"
NUM_EXPERTS = 32
TOP_K = 4
NUM_LAYERS = 24

DEFAULT_SOURCE_EP = 4
DEFAULT_SOURCE_BS = 1
DEFAULT_NUM_SAMPLES = 50000
DEFAULT_SEED = 42

DEFAULT_ALPHA = 0.0
DEFAULT_BETA = 1.0
DEFAULT_GAMMA = 0.0
DEFAULT_DELTA = 0.0
DEFAULT_ETA = 0.0

OUTPUT_DIR = "./output/latency_with_expert_correlation"


def find_layer_file(ep_size, file_bs, layer):
    pattern = os.path.join(
        DIRECTORY,
        f"expert_selection_openai_{MODEL_NAME}_wikitext_103_ep{ep_size}_bs{file_bs}_*_layer{layer}.jsonl",
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


def analyze_batch_from_token_experts(token_experts_batch, ep_size, alpha, beta, gamma, delta, eta):
    experts_per_gpu = NUM_EXPERTS // ep_size
    local_counts = np.zeros((ep_size, experts_per_gpu), dtype=np.int32)

    for token_experts in token_experts_batch:
        for expert_idx in token_experts:
            gpu_idx = int(expert_idx) // experts_per_gpu
            local_idx = int(expert_idx) % experts_per_gpu
            local_counts[gpu_idx, local_idx] += 1

    gpu_total_counts = local_counts.sum(axis=1)
    gpu_active_experts = (local_counts > 0).sum(axis=1)
    gpu_max_local = local_counts.max(axis=1)

    gpu_concentration = np.zeros(ep_size, dtype=np.float64)
    for gpu_idx in range(ep_size):
        total = int(gpu_total_counts[gpu_idx])
        if total > 0:
            gpu_concentration[gpu_idx] = float(np.square(local_counts[gpu_idx]).sum()) / float(total * total)

    gpu_latency = (
        alpha
        + beta * gpu_total_counts.astype(np.float64)
        + gamma * gpu_active_experts.astype(np.float64)
        + delta * gpu_max_local.astype(np.float64)
        + eta * gpu_concentration
    )

    max_x = int(gpu_total_counts.max())
    candidate_indices = np.flatnonzero(gpu_total_counts == max_x)
    if len(candidate_indices) == 1:
        chosen_gpu = int(candidate_indices[0])
    else:
        best_idx = int(candidate_indices[0])
        best_key = (
            float(gpu_latency[best_idx]),
            int(gpu_max_local[best_idx]),
            int(gpu_active_experts[best_idx]),
            -best_idx,
        )
        for gpu_idx in candidate_indices[1:]:
            key = (
                float(gpu_latency[gpu_idx]),
                int(gpu_max_local[gpu_idx]),
                int(gpu_active_experts[gpu_idx]),
                -int(gpu_idx),
            )
            if key > best_key:
                best_key = key
                best_idx = int(gpu_idx)
        chosen_gpu = best_idx

    x_value = int(gpu_total_counts[chosen_gpu])
    a_value = int(gpu_active_experts[chosen_gpu])
    m_value = int(gpu_max_local[chosen_gpu])
    c_value = float(gpu_concentration[chosen_gpu])
    latency_value = float(gpu_latency[chosen_gpu])

    return {
        "x": x_value,
        "a": a_value,
        "m": m_value,
        "c": c_value,
        "latency": latency_value,
        "chosen_gpu": chosen_gpu,
        "gpu_total_counts": gpu_total_counts.tolist(),
        "gpu_active_experts": gpu_active_experts.tolist(),
        "gpu_max_local": gpu_max_local.tolist(),
        "gpu_concentration": gpu_concentration.tolist(),
    }


def monte_carlo_predict_for_layer(
    token_expert_array,
    ep_size,
    batch_size,
    num_samples,
    rng,
    alpha,
    beta,
    gamma,
    delta,
    eta,
):
    x_counter = Counter()
    latency_values = []
    conditional = defaultdict(lambda: {"count": 0, "a_sum": 0.0, "m_sum": 0.0, "c_sum": 0.0, "lat_sum": 0.0})

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

    gpu_total_counts = local_counts.sum(axis=2).astype(np.int32)
    gpu_active_experts = (local_counts > 0).sum(axis=2).astype(np.int32)
    gpu_max_local = local_counts.max(axis=2).astype(np.int32)

    total_sq = np.square(local_counts, dtype=np.int32).sum(axis=2).astype(np.float64)
    denom = np.square(gpu_total_counts.astype(np.float64))
    gpu_concentration = np.divide(
        total_sq,
        denom,
        out=np.zeros_like(total_sq, dtype=np.float64),
        where=denom > 0.0,
    )

    gpu_latency = (
        alpha
        + beta * gpu_total_counts.astype(np.float64)
        + gamma * gpu_active_experts.astype(np.float64)
        + delta * gpu_max_local.astype(np.float64)
        + eta * gpu_concentration
    )

    max_x = gpu_total_counts.max(axis=1)
    candidate_mask = gpu_total_counts == max_x[:, None]

    tie_score = (
        gpu_latency
        + 1e-6 * gpu_max_local.astype(np.float64)
        + 1e-9 * gpu_active_experts.astype(np.float64)
        - 1e-12 * np.arange(ep_size, dtype=np.float64)[None, :]
    )
    masked_score = np.where(candidate_mask, tie_score, -np.inf)
    chosen_gpu = masked_score.argmax(axis=1)

    row_idx = np.arange(num_samples, dtype=np.int32)
    x_values = gpu_total_counts[row_idx, chosen_gpu]
    a_values = gpu_active_experts[row_idx, chosen_gpu]
    m_values = gpu_max_local[row_idx, chosen_gpu]
    c_values = gpu_concentration[row_idx, chosen_gpu]
    latency_arr = gpu_latency[row_idx, chosen_gpu]

    for x_value, a_value, m_value, c_value, latency_value in zip(
        x_values.tolist(),
        a_values.tolist(),
        m_values.tolist(),
        c_values.tolist(),
        latency_arr.tolist(),
    ):
        x_counter[x_value] += 1
        latency_values.append(latency_value)

        bucket = conditional[x_value]
        bucket["count"] += 1
        bucket["a_sum"] += a_value
        bucket["m_sum"] += m_value
        bucket["c_sum"] += c_value
        bucket["lat_sum"] += latency_value

    return x_counter, conditional, latency_values


def load_actual_stats_for_layer(file_path, ep_size, alpha, beta, gamma, delta, eta):
    x_counter = Counter()
    latency_values = []
    conditional = defaultdict(lambda: {"count": 0, "a_sum": 0.0, "m_sum": 0.0, "c_sum": 0.0, "lat_sum": 0.0})

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

            stats = analyze_batch_from_token_experts(
                token_experts_batch=token_expert_lists,
                ep_size=ep_size,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                delta=delta,
                eta=eta,
            )

            x_value = stats["x"]
            x_counter[x_value] += 1
            latency_values.append(stats["latency"])

            bucket = conditional[x_value]
            bucket["count"] += 1
            bucket["a_sum"] += stats["a"]
            bucket["m_sum"] += stats["m"]
            bucket["c_sum"] += stats["c"]
            bucket["lat_sum"] += stats["latency"]

    return x_counter, conditional, latency_values


def counter_to_distribution(counter, max_value):
    total = sum(counter.values())
    if total == 0:
        return [0.0] * (max_value + 1)
    return [counter.get(idx, 0) / total for idx in range(max_value + 1)]


def conditional_to_rows(conditional):
    rows = []
    for x_value in sorted(conditional.keys()):
        item = conditional[x_value]
        count = item["count"]
        if count == 0:
            continue
        rows.append(
            {
                "x": x_value,
                "count": count,
                "E_A_given_X": item["a_sum"] / count,
                "E_M_given_X": item["m_sum"] / count,
                "E_C_given_X": item["c_sum"] / count,
                "E_L_given_X": item["lat_sum"] / count,
            }
        )
    return rows


def expected_from_distribution(dist):
    return sum(idx * prob for idx, prob in enumerate(dist))


def percentile_from_distribution(dist, q):
    cdf = 0.0
    for idx, prob in enumerate(dist):
        cdf += prob
        if cdf >= q:
            return float(idx)
    return float(len(dist) - 1)


def percentile_from_values(values, q):
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q * 100.0))


def save_conditional_csv(path, rows_pred, rows_actual):
    actual_by_x = {row["x"]: row for row in rows_actual}
    pred_by_x = {row["x"]: row for row in rows_pred}
    all_x = sorted(set(actual_by_x) | set(pred_by_x))

    with open(path, "w", encoding="utf-8", newline="") as outfile:
        fieldnames = [
            "x",
            "pred_E_A_given_X",
            "actual_E_A_given_X",
            "pred_E_M_given_X",
            "actual_E_M_given_X",
            "pred_E_C_given_X",
            "actual_E_C_given_X",
            "pred_E_L_given_X",
            "actual_E_L_given_X",
        ]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for x_value in all_x:
            pred_row = pred_by_x.get(x_value, {})
            actual_row = actual_by_x.get(x_value, {})
            writer.writerow(
                {
                    "x": x_value,
                    "pred_E_A_given_X": pred_row.get("E_A_given_X", ""),
                    "actual_E_A_given_X": actual_row.get("E_A_given_X", ""),
                    "pred_E_M_given_X": pred_row.get("E_M_given_X", ""),
                    "actual_E_M_given_X": actual_row.get("E_M_given_X", ""),
                    "pred_E_C_given_X": pred_row.get("E_C_given_X", ""),
                    "actual_E_C_given_X": actual_row.get("E_C_given_X", ""),
                    "pred_E_L_given_X": pred_row.get("E_L_given_X", ""),
                    "actual_E_L_given_X": actual_row.get("E_L_given_X", ""),
                }
            )


def plot_distribution_overlay(path, pred_dist, actual_dist, ep_size, batch_size):
    size = max(len(pred_dist), len(actual_dist))
    pred = pred_dist + [0.0] * (size - len(pred_dist))
    actual = actual_dist + [0.0] * (size - len(actual_dist))

    xs = [idx for idx in range(size) if pred[idx] > 0.0 or actual[idx] > 0.0]
    pred_y = [pred[idx] for idx in xs]
    actual_y = [actual[idx] for idx in xs]

    plt.figure(figsize=(11, 7))
    plt.plot(xs, pred_y, marker="o", label="Predicted X distribution")
    plt.plot(xs, actual_y, marker="s", label="Actual X distribution")
    plt.xlabel("X = max routed tokens per GPU")
    plt.ylabel("Probability")
    plt.title(f"Predicted vs Actual X Distribution (EP={ep_size}, bs={batch_size})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_conditional_curves(path, rows_pred, rows_actual, ylabel, pred_key, actual_key, title):
    pred_x = [row["x"] for row in rows_pred if pred_key in row]
    pred_y = [row[pred_key] for row in rows_pred if pred_key in row]
    actual_x = [row["x"] for row in rows_actual if actual_key in row]
    actual_y = [row[actual_key] for row in rows_actual if actual_key in row]

    plt.figure(figsize=(11, 7))
    if pred_x:
        plt.plot(pred_x, pred_y, marker="o", label="Predicted")
    if actual_x:
        plt.plot(actual_x, actual_y, marker="s", label="Actual")
    plt.xlabel("X = max routed tokens per GPU")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_latency_histogram(path, pred_latency, actual_latency, ep_size, batch_size):
    plt.figure(figsize=(11, 7))
    plt.hist(pred_latency, bins=50, alpha=0.5, density=True, label="Predicted latency")
    plt.hist(actual_latency, bins=50, alpha=0.5, density=True, label="Actual-formula latency")
    plt.xlabel("Latency")
    plt.ylabel("Density")
    plt.title(f"Predicted vs Actual Formula Latency (EP={ep_size}, bs={batch_size})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    ep_raw = input("비교할 EP Degree 입력 (예: 4): ").strip()
    batch_size_raw = input("target batch size 입력 (예: 32): ").strip()
    source_ep_raw = input(f"source 로그의 EP Degree 입력 (기본 {DEFAULT_SOURCE_EP}): ").strip()
    source_bs_raw = input(f"source 로그의 batch size 입력 (기본 {DEFAULT_SOURCE_BS}): ").strip()
    num_samples_raw = input(f"Monte Carlo 샘플 수 입력 (기본 {DEFAULT_NUM_SAMPLES}): ").strip()
    seed_raw = input(f"random seed 입력 (기본 {DEFAULT_SEED}): ").strip()
    alpha_raw = input(f"alpha 입력 (기본 {DEFAULT_ALPHA}): ").strip()
    beta_raw = input(f"beta 입력 (기본 {DEFAULT_BETA}): ").strip()
    gamma_raw = input(f"gamma 입력 (기본 {DEFAULT_GAMMA}): ").strip()
    delta_raw = input(f"delta 입력 (기본 {DEFAULT_DELTA}): ").strip()
    eta_raw = input(f"eta 입력 (기본 {DEFAULT_ETA}): ").strip()

    ep_size = int(ep_raw)
    batch_size = int(batch_size_raw)
    source_ep = int(source_ep_raw) if source_ep_raw else DEFAULT_SOURCE_EP
    source_bs = int(source_bs_raw) if source_bs_raw else DEFAULT_SOURCE_BS
    num_samples = int(num_samples_raw) if num_samples_raw else DEFAULT_NUM_SAMPLES
    seed = int(seed_raw) if seed_raw else DEFAULT_SEED
    alpha = float(alpha_raw) if alpha_raw else DEFAULT_ALPHA
    beta = float(beta_raw) if beta_raw else DEFAULT_BETA
    gamma = float(gamma_raw) if gamma_raw else DEFAULT_GAMMA
    delta = float(delta_raw) if delta_raw else DEFAULT_DELTA
    eta = float(eta_raw) if eta_raw else DEFAULT_ETA

    if NUM_EXPERTS % ep_size != 0:
        raise ValueError(f"NUM_EXPERTS={NUM_EXPERTS} must be divisible by ep_size={ep_size}")

    print(
        "bs=1 source routing에서 token-level expert tuple을 직접 샘플링해\n"
        "X, active experts, local concentration, formula latency를 예측 중..."
    )

    max_x = batch_size * TOP_K
    pred_dist_sum = np.zeros(max_x + 1, dtype=np.float64)
    actual_dist_sum = np.zeros(max_x + 1, dtype=np.float64)
    pred_latency_all = []
    actual_latency_all = []
    pred_conditional_sum = defaultdict(lambda: {"count": 0, "a_sum": 0.0, "m_sum": 0.0, "c_sum": 0.0, "lat_sum": 0.0})
    actual_conditional_sum = defaultdict(lambda: {"count": 0, "a_sum": 0.0, "m_sum": 0.0, "c_sum": 0.0, "lat_sum": 0.0})

    valid_pred_layers = 0
    valid_actual_layers = 0

    for layer in range(NUM_LAYERS):
        source_file = find_layer_file(source_ep, source_bs, layer)
        if source_file:
            token_expert_array = load_token_expert_array(source_file)
            if len(token_expert_array) > 0:
                rng = np.random.default_rng(seed + layer * 1237 + ep_size * 997 + batch_size * 131)
                pred_counter, pred_conditional, pred_latency_values = monte_carlo_predict_for_layer(
                    token_expert_array=token_expert_array,
                    ep_size=ep_size,
                    batch_size=batch_size,
                    num_samples=num_samples,
                    rng=rng,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    delta=delta,
                    eta=eta,
                )
                pred_dist_sum += np.asarray(counter_to_distribution(pred_counter, max_x), dtype=np.float64)
                pred_latency_all.extend(pred_latency_values)
                for x_value, item in pred_conditional.items():
                    bucket = pred_conditional_sum[x_value]
                    bucket["count"] += item["count"]
                    bucket["a_sum"] += item["a_sum"]
                    bucket["m_sum"] += item["m_sum"]
                    bucket["c_sum"] += item["c_sum"]
                    bucket["lat_sum"] += item["lat_sum"]
                valid_pred_layers += 1

        actual_file = find_layer_file(ep_size, batch_size, layer)
        if actual_file:
            actual_counter, actual_conditional, actual_latency_values = load_actual_stats_for_layer(
                file_path=actual_file,
                ep_size=ep_size,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                delta=delta,
                eta=eta,
            )
            actual_dist_sum += np.asarray(counter_to_distribution(actual_counter, max_x), dtype=np.float64)
            actual_latency_all.extend(actual_latency_values)
            for x_value, item in actual_conditional.items():
                bucket = actual_conditional_sum[x_value]
                bucket["count"] += item["count"]
                bucket["a_sum"] += item["a_sum"]
                bucket["m_sum"] += item["m_sum"]
                bucket["c_sum"] += item["c_sum"]
                bucket["lat_sum"] += item["lat_sum"]
            valid_actual_layers += 1

    if valid_pred_layers == 0:
        raise RuntimeError(f"No valid source layers found for source_ep={source_ep}, source_bs={source_bs}")
    if valid_actual_layers == 0:
        raise RuntimeError(f"No valid actual layers found for ep={ep_size}, bs={batch_size}")

    pred_dist = (pred_dist_sum / valid_pred_layers).tolist()
    actual_dist = (actual_dist_sum / valid_actual_layers).tolist()
    pred_cond_rows = conditional_to_rows(pred_conditional_sum)
    actual_cond_rows = conditional_to_rows(actual_conditional_sum)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stem = (
        f"latency_corr_ep{ep_size}_bs{batch_size}"
        f"_source_ep{source_ep}_bs{source_bs}"
        f"_nsamples{num_samples}"
    )

    summary_csv = os.path.join(OUTPUT_DIR, f"{stem}_summary.csv")
    conditional_csv = os.path.join(OUTPUT_DIR, f"{stem}_conditional.csv")
    x_plot = os.path.join(OUTPUT_DIR, f"{stem}_x_distribution.png")
    a_plot = os.path.join(OUTPUT_DIR, f"{stem}_cond_active_experts.png")
    m_plot = os.path.join(OUTPUT_DIR, f"{stem}_cond_max_local.png")
    c_plot = os.path.join(OUTPUT_DIR, f"{stem}_cond_concentration.png")
    l_plot = os.path.join(OUTPUT_DIR, f"{stem}_cond_latency.png")
    hist_plot = os.path.join(OUTPUT_DIR, f"{stem}_latency_histogram.png")

    with open(summary_csv, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(
            outfile,
            fieldnames=[
                "ep",
                "bs",
                "source_ep",
                "source_bs",
                "num_samples",
                "alpha",
                "beta",
                "gamma",
                "delta",
                "eta",
                "pred_expected_X",
                "actual_expected_X",
                "pred_p95_X",
                "actual_p95_X",
                "pred_expected_latency",
                "actual_formula_expected_latency",
                "pred_p95_latency",
                "actual_formula_p95_latency",
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
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "delta": delta,
                "eta": eta,
                "pred_expected_X": expected_from_distribution(pred_dist),
                "actual_expected_X": expected_from_distribution(actual_dist),
                "pred_p95_X": percentile_from_distribution(pred_dist, 0.95),
                "actual_p95_X": percentile_from_distribution(actual_dist, 0.95),
                "pred_expected_latency": float(np.mean(pred_latency_all)) if pred_latency_all else 0.0,
                "actual_formula_expected_latency": float(np.mean(actual_latency_all)) if actual_latency_all else 0.0,
                "pred_p95_latency": percentile_from_values(pred_latency_all, 0.95),
                "actual_formula_p95_latency": percentile_from_values(actual_latency_all, 0.95),
            }
        )

    save_conditional_csv(conditional_csv, pred_cond_rows, actual_cond_rows)
    plot_distribution_overlay(x_plot, pred_dist, actual_dist, ep_size, batch_size)
    plot_conditional_curves(
        a_plot,
        pred_cond_rows,
        actual_cond_rows,
        "E[A | X=x]",
        "E_A_given_X",
        "E_A_given_X",
        f"Conditional Active Experts (EP={ep_size}, bs={batch_size})",
    )
    plot_conditional_curves(
        m_plot,
        pred_cond_rows,
        actual_cond_rows,
        "E[M | X=x]",
        "E_M_given_X",
        "E_M_given_X",
        f"Conditional Max Local Expert Load (EP={ep_size}, bs={batch_size})",
    )
    plot_conditional_curves(
        c_plot,
        pred_cond_rows,
        actual_cond_rows,
        "E[C | X=x]",
        "E_C_given_X",
        "E_C_given_X",
        f"Conditional Concentration (EP={ep_size}, bs={batch_size})",
    )
    plot_conditional_curves(
        l_plot,
        pred_cond_rows,
        actual_cond_rows,
        "E[L | X=x]",
        "E_L_given_X",
        "E_L_given_X",
        f"Conditional Formula Latency (EP={ep_size}, bs={batch_size})",
    )
    plot_latency_histogram(hist_plot, pred_latency_all, actual_latency_all, ep_size, batch_size)

    print("완료")
    print(f"pred expected X = {expected_from_distribution(pred_dist):.6f}")
    print(f"actual expected X = {expected_from_distribution(actual_dist):.6f}")
    print(f"pred expected latency = {float(np.mean(pred_latency_all)):.6f}")
    print(f"actual formula expected latency = {float(np.mean(actual_latency_all)):.6f}")
    print(f"summary csv: {summary_csv}")
    print(f"conditional csv: {conditional_csv}")
    print(f"x distribution plot: {x_plot}")
    print(f"conditional A plot: {a_plot}")
    print(f"conditional M plot: {m_plot}")
    print(f"conditional C plot: {c_plot}")
    print(f"conditional latency plot: {l_plot}")
    print(f"latency histogram: {hist_plot}")


if __name__ == "__main__":
    main()
