#!/usr/bin/env python3

from plot_actual_vs_gumbel_layerwise import (
    DEFAULT_NUM_SAMPLES,
    DEFAULT_SEED,
    DEFAULT_SOURCE_BS,
    DEFAULT_SOURCE_EP,
    TARGET_LAYERS,
    expected_value_from_dist,
    load_real_distribution_for_layer,
    load_real_distribution_for_layer_average,
    plot_distribution_case as base_plot_distribution_case,
    predict_gumbel_distribution_all_layers,
    predict_gumbel_distribution_for_layer,
    tvd_from_dists,
)

import csv
import os


OUTPUT_DIR = "./output/actual_vs_gumbel_max_lg_layerwise"


def plot_distribution_case(real, gumbel_pred, ep_size, batch_size, source_bs, case_label, out_path):
    base_plot_distribution_case(
        real=real,
        gumbel_pred=gumbel_pred,
        ep_size=ep_size,
        batch_size=batch_size,
        source_bs=source_bs,
        case_label=case_label,
        out_path=out_path,
    )


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

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stem = (
        f"actual_vs_gumbel_max_lg_layerwise_ep{ep_size}_bs{batch_size}"
        f"_source_ep{source_ep}_bs{source_bs}"
    )
    summary_path = os.path.join(OUTPUT_DIR, f"{stem}_summary.csv")

    cases = []

    real_all = load_real_distribution_for_layer_average(ep_size=ep_size, batch_size=batch_size)
    pred_all = predict_gumbel_distribution_all_layers(
        ep_size=ep_size,
        batch_size=batch_size,
        source_ep=source_ep,
        source_bs=source_bs,
        num_samples=num_samples,
        seed=seed,
    )
    cases.append(("all", real_all, pred_all))

    for layer in TARGET_LAYERS:
        real_layer = load_real_distribution_for_layer(ep_size=ep_size, batch_size=batch_size, layer=layer)
        pred_layer = predict_gumbel_distribution_for_layer(
            ep_size=ep_size,
            batch_size=batch_size,
            source_ep=source_ep,
            source_bs=source_bs,
            num_samples=num_samples,
            seed=seed,
            layer=layer,
        )
        cases.append((f"layer{layer}", real_layer, pred_layer))

    summary_rows = []
    for case_label, real, pred in cases:
        plot_path = os.path.join(OUTPUT_DIR, f"{stem}_{case_label}.png")
        plot_distribution_case(
            real=real,
            gumbel_pred=pred,
            ep_size=ep_size,
            batch_size=batch_size,
            source_bs=source_bs,
            case_label=case_label,
            out_path=plot_path,
        )
        summary_rows.append({
            "case": case_label,
            "x_definition": "max(L_g)",
            "actual_expected_x": expected_value_from_dist(real),
            "gumbel_expected_x": expected_value_from_dist(pred),
            "tvd": tvd_from_dists(real, pred),
            "plot_path": plot_path,
        })

    with open(summary_path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(
            outfile,
            fieldnames=[
                "case",
                "x_definition",
                "actual_expected_x",
                "gumbel_expected_x",
                "tvd",
                "plot_path",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("완료")
    print("X 정의: max(L_g)")
    for row in summary_rows:
        print(
            f"{row['case']}: "
            f"actual expected X = {row['actual_expected_x']:.6f}, "
            f"gumbel expected X = {row['gumbel_expected_x']:.6f}, "
            f"TVD = {row['tvd']:.6f}"
        )
        print(f"plot: {row['plot_path']}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
