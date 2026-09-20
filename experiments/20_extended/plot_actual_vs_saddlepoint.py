#!/usr/bin/env python3

import os

import matplotlib.pyplot as plt

from compare_token_distribution import load_real_distribution_for_bs
from plot_ep_pp_tradeoff_saddlepoint import (
    DEFAULT_SOURCE_BS,
    DEFAULT_SOURCE_EP,
    predict_distribution_for_ep_pp,
)


OUTPUT_DIR = "./output/actual_vs_saddlepoint"


def main():
    ep_raw = input("비교할 EP Degree 입력 (예: 4): ").strip()
    global_bs_raw = input("global/real batch size 입력 (예: 32): ").strip()
    pp_raw = input("비교할 PP Degree 입력 (기본 1): ").strip()
    source_ep_raw = input(f"source 로그의 EP Degree 입력 (기본 {DEFAULT_SOURCE_EP}): ").strip()
    source_bs_raw = input(f"source 로그의 batch size 입력 (기본 {DEFAULT_SOURCE_BS}): ").strip()

    ep_size = int(ep_raw)
    global_bs = int(global_bs_raw)
    pp_degree = int(pp_raw) if pp_raw else 1
    source_ep = int(source_ep_raw) if source_ep_raw else DEFAULT_SOURCE_EP
    source_bs = int(source_bs_raw) if source_bs_raw else DEFAULT_SOURCE_BS

    predicted = predict_distribution_for_ep_pp(
        target_ep=ep_size,
        global_bs=global_bs,
        pp_degree=pp_degree,
        source_ep=source_ep,
        source_bs=source_bs,
    )

    # Real routed-token distribution exists only for the actual batch size that was measured.
    if pp_degree != 1:
        raise ValueError(
            "현재 real 로그는 PP가 적용되지 않은 actual batch 기준입니다. "
            "actual과 직접 비교하려면 PP=1로 두어야 합니다."
        )

    real = load_real_distribution_for_bs(ep_size=ep_size, real_file_bs=global_bs)

    size = max(len(real), len(predicted["distribution"]))
    real = real + [0.0] * (size - len(real))
    theory = predicted["distribution"] + [0.0] * (size - len(predicted["distribution"]))

    xs = [idx for idx in range(size) if real[idx] > 0.0 or theory[idx] > 0.0]
    real_y = [real[idx] for idx in xs]
    theory_y = [theory[idx] for idx in xs]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(
        OUTPUT_DIR,
        f"actual_vs_saddle_ep{ep_size}_bs{global_bs}_pp{pp_degree}_source_ep{source_ep}_bs{source_bs}.png",
    )

    plt.figure(figsize=(11, 7))
    plt.plot(xs, real_y, marker="s", label=f"Real EP={ep_size}, bs={global_bs}")
    plt.plot(
        xs,
        theory_y,
        marker="o",
        label=f"Saddlepoint EP={ep_size}, PP={pp_degree}, global bs={global_bs}",
    )
    plt.xlabel("Max routed tokens per GPU")
    plt.ylabel("Probability")
    plt.title(f"Actual vs Saddlepoint Distribution (EP={ep_size}, bs={global_bs}, PP={pp_degree})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print("완료")
    print(f"local bs used by saddlepoint = {predicted['local_bs']}")
    print(f"actual expected max = {sum(idx * prob for idx, prob in enumerate(real)):.6f}")
    print(f"saddlepoint expected max = {predicted['expected_max']:.6f}")
    print(f"plot: {out_path}")


if __name__ == "__main__":
    main()
