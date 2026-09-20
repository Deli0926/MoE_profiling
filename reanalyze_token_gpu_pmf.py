#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import math
import os
import random
from collections import Counter, defaultdict

import numpy as np


NUM_LAYERS = 24
NUM_EXPERTS = 32
TOPK = 4
EP = 4
EXPERTS_PER_GPU = NUM_EXPERTS // EP


def load_rows(data_dir, bs, layer):
    pattern = os.path.join(
        data_dir,
        f"expert_selection_openai_gpt-oss-20b_wikitext_103_ep4_bs{bs}_*_layer{layer}.jsonl",
    )
    matches = sorted(glob.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one match for {pattern}, got {matches}")
    rows = []
    with open(matches[0]) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def token_to_gpu_tuple(experts):
    loads = [0] * EP
    for e in experts:
        loads[int(e) // EXPERTS_PER_GPU] += 1
    return tuple(loads)


def row_to_tokens(row, bs):
    if len(row) != bs * TOPK:
        raise ValueError(f"row length {len(row)} != bs*topk {bs * TOPK}")
    return [row[i * TOPK : (i + 1) * TOPK] for i in range(bs)]


def row_to_gpu_load(row, bs):
    loads = [0] * EP
    for tok in row_to_tokens(row, bs):
        t = token_to_gpu_tuple(tok)
        for i, v in enumerate(t):
            loads[i] += v
    return tuple(loads)


def normalize(counter):
    total = sum(counter.values())
    return {k: v / total for k, v in counter.items()}


def convolve_pmf(pmf_a, pmf_b):
    out = defaultdict(float)
    for a, pa in pmf_a.items():
        for b, pb in pmf_b.items():
            out[tuple(x + y for x, y in zip(a, b))] += pa * pb
    return dict(out)


def pmf_power(base, n):
    result = {(0, 0, 0, 0): 1.0}
    # The vector state space is only O((bs * topk)^3), while repeated squaring
    # creates huge intermediate cross products. Sequential DP is cheaper here.
    for _ in range(n):
        result = convolve_pmf(result, base)
    return result


def max_distribution(vec_pmf):
    out = defaultdict(float)
    for vec, p in vec_pmf.items():
        out[max(vec)] += p
    return dict(out)


def fft_convolve_trim(a, b, limit):
    full_shape = tuple(min(a.shape[i] + b.shape[i] - 1, limit + 1) for i in range(3))
    axes = (0, 1, 2)
    conv = np.fft.irfftn(
        np.fft.rfftn(a, full_shape, axes=axes)
        * np.fft.rfftn(b, full_shape, axes=axes),
        full_shape,
        axes=axes,
    ).real
    conv[np.abs(conv) < 1e-15] = 0.0
    return conv


def max_distribution_from_token_pmf_fft(token_pmf, bs):
    limit = TOPK * bs
    kernel = np.zeros((TOPK + 1, TOPK + 1, TOPK + 1), dtype=np.float64)
    for vec, p in token_pmf.items():
        kernel[vec[0], vec[1], vec[2]] += p

    result = np.ones((1, 1, 1), dtype=np.float64)
    base = kernel
    exp = bs
    while exp:
        if exp & 1:
            result = fft_convolve_trim(result, base, limit)
        exp >>= 1
        if exp:
            base = fft_convolve_trim(base, base, limit)

    total = result.sum()
    if total:
        result /= total

    i, j, k = np.indices(result.shape)
    g3 = limit - i - j - k
    mask = g3 >= 0
    max_load = np.maximum.reduce([i, j, k, g3])
    bins = np.bincount(
        max_load[mask].ravel(),
        weights=result[mask].ravel(),
        minlength=limit + 1,
    )
    bins /= bins.sum()
    return {x: float(p) for x, p in enumerate(bins) if p > 1e-14}


def tvd(p, q):
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def expected(dist):
    return sum(x * p for x, p in dist.items())


def p95(dist):
    acc = 0.0
    for x in sorted(dist):
        acc += dist[x]
        if acc >= 0.95:
            return x
    return max(dist) if dist else 0


def sample_cont_batch_dist(token_patterns, bs, samples, rng):
    counts = Counter()
    n = len(token_patterns)
    for _ in range(samples):
        loads = [0] * EP
        for _ in range(bs):
            t = token_patterns[rng.randrange(n)]
            for i, v in enumerate(t):
                loads[i] += v
        counts[max(loads)] += 1
    return normalize(counts)


def write_dist_rows(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "bs",
                "layer",
                "source",
                "x",
                "probability",
            ],
        )
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="expert_selections")
    parser.add_argument("--out-dir", default="output/reanalysis_token_gpu_pmf_20260528")
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--bs", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    summary_rows = []
    dist_rows = []

    for bs in args.bs:
        layer_summaries = []
        for layer in range(NUM_LAYERS):
            bs1_rows = load_rows(args.data_dir, 1, layer)
            bs1_tokens = [token_to_gpu_tuple(row) for row in bs1_rows]
            token_pmf = normalize(Counter(bs1_tokens))
            pred = max_distribution_from_token_pmf_fft(token_pmf, bs)

            fixed_rows = load_rows(args.data_dir, bs, layer)
            fixed = normalize(Counter(max(row_to_gpu_load(row, bs)) for row in fixed_rows))

            cont = sample_cont_batch_dist(bs1_tokens, bs, args.samples, rng)

            for source, dist in [("predicted", pred), ("fixed_actual", fixed), ("cont_sim", cont)]:
                for x, prob in sorted(dist.items()):
                    dist_rows.append(
                        {
                            "bs": bs,
                            "layer": layer,
                            "source": source,
                            "x": x,
                            "probability": prob,
                        }
                    )

            row = {
                "bs": bs,
                "layer": layer,
                "pred_expected_x": expected(pred),
                "fixed_expected_x": expected(fixed),
                "cont_expected_x": expected(cont),
                "pred_p95": p95(pred),
                "fixed_p95": p95(fixed),
                "cont_p95": p95(cont),
                "tvd_fixed_vs_pred": tvd(fixed, pred),
                "tvd_cont_vs_pred": tvd(cont, pred),
                "num_fixed_batches": len(fixed_rows),
                "num_cont_samples": args.samples,
            }
            summary_rows.append(row)
            layer_summaries.append(row)

        mean_row = {
            "bs": bs,
            "layer": "layer_avg",
            "pred_expected_x": sum(r["pred_expected_x"] for r in layer_summaries) / NUM_LAYERS,
            "fixed_expected_x": sum(r["fixed_expected_x"] for r in layer_summaries) / NUM_LAYERS,
            "cont_expected_x": sum(r["cont_expected_x"] for r in layer_summaries) / NUM_LAYERS,
            "pred_p95": "",
            "fixed_p95": "",
            "cont_p95": "",
            "tvd_fixed_vs_pred": sum(r["tvd_fixed_vs_pred"] for r in layer_summaries) / NUM_LAYERS,
            "tvd_cont_vs_pred": sum(r["tvd_cont_vs_pred"] for r in layer_summaries) / NUM_LAYERS,
            "num_fixed_batches": "",
            "num_cont_samples": args.samples,
        }
        summary_rows.append(mean_row)

    summary_path = os.path.join(args.out_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    write_dist_rows(os.path.join(args.out_dir, "distributions.csv"), dist_rows)

    print(f"Wrote {summary_path}")
    for row in summary_rows:
        if row["layer"] == "layer_avg":
            print(
                f"bs={row['bs']} layer_avg "
                f"TVD fixed-vs-pred={row['tvd_fixed_vs_pred']:.6f} "
                f"TVD cont-vs-pred={row['tvd_cont_vs_pred']:.6f} "
                f"E[pred]={row['pred_expected_x']:.3f} "
                f"E[fixed]={row['fixed_expected_x']:.3f} "
                f"E[cont]={row['cont_expected_x']:.3f}"
            )


if __name__ == "__main__":
    main()
