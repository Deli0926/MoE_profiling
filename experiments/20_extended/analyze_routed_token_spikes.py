import glob
import json
import os
from collections import Counter, defaultdict


DIRECTORY = "./expert_selections"
MODEL_NAME = "gpt-oss-20b"
NUM_EXPERTS = 32
TOP_K = 4
NUM_LAYERS = 24


def extract_token_expert_lists(data, top_k):
    if len(data) > 0 and isinstance(data[0], list):
        return data

    if len(data) % top_k != 0:
        raise ValueError(f"flat data length {len(data)} is not divisible by top_k={top_k}")

    return [data[i:i + top_k] for i in range(0, len(data), top_k)]


def token_to_gpu_pattern(token_experts, num_experts, ep_size):
    experts_per_gpu = num_experts // ep_size
    counts = [0] * ep_size

    for expert_idx in token_experts:
        if expert_idx < 0 or expert_idx >= num_experts:
            raise ValueError(f"expert index {expert_idx} out of range [0, {num_experts - 1}]")

        gpu_idx = expert_idx // experts_per_gpu
        counts[gpu_idx] += 1

    return tuple(counts)


def find_layer_file(ep_size, bs, layer):
    pattern = os.path.join(
        DIRECTORY,
        f"expert_selection_openai_{MODEL_NAME}_wikitext_103_ep{ep_size}_bs{bs}_*_layer{layer}.jsonl",
    )
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def build_step_record(line_idx, layer, token_expert_lists, ep_size):
    token_patterns = [
        token_to_gpu_pattern(token_experts, NUM_EXPERTS, ep_size)
        for token_experts in token_expert_lists
    ]

    load_vec = [0] * ep_size
    dominant_gpu_pattern_counter = Counter()

    for pattern in token_patterns:
        for gpu_idx, count in enumerate(pattern):
            load_vec[gpu_idx] += count

        max_count = max(pattern)
        dominant_gpus = tuple(
            gpu_idx for gpu_idx, count in enumerate(pattern)
            if count == max_count and count > 0
        )
        dominant_gpu_pattern_counter[dominant_gpus] += 1

    max_load = max(load_vec)
    argmax_gpus = tuple(
        gpu_idx for gpu_idx, count in enumerate(load_vec)
        if count == max_load
    )

    return {
        "layer": layer,
        "line_idx": line_idx,
        "load_vec": tuple(load_vec),
        "max_load": max_load,
        "argmax_gpus": argmax_gpus,
        "token_patterns": token_patterns,
        "dominant_gpu_pattern_counter": dominant_gpu_pattern_counter,
    }


def summarize_token_patterns(token_patterns, top_n):
    counter = Counter(token_patterns)
    total = len(token_patterns)
    rows = []
    for pattern, count in counter.most_common(top_n):
        rows.append((pattern, count, count / total if total > 0 else 0.0))
    return rows


def main():
    ep_size = int(input("EP Degree 입력: "))
    bs = int(input("분석할 실제 batch size(bs) 입력: "))
    target_y = int(input("추적할 spike 값 Y 입력 (예: 64): "))
    top_n = input("상위 몇 개 패턴/step을 볼지 입력 (기본 10): ").strip()
    if top_n == "":
        top_n = "10"
    top_n = int(top_n)

    max_load_counter = Counter()
    load_vec_counter = Counter()
    argmax_gpu_counter = Counter()
    token_pattern_counter = Counter()
    per_layer_y_counter = Counter()
    spike_records = []

    for layer in range(NUM_LAYERS):
        file_path = find_layer_file(ep_size, bs, layer)
        if not file_path:
            continue

        with open(file_path, "r", encoding="utf-8") as infile:
            for line_idx, raw_line in enumerate(infile):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue

                data = json.loads(raw_line)
                token_expert_lists = extract_token_expert_lists(data, TOP_K)

                record = build_step_record(
                    line_idx=line_idx,
                    layer=layer,
                    token_expert_lists=token_expert_lists,
                    ep_size=ep_size,
                )

                y = record["max_load"]
                max_load_counter[y] += 1
                load_vec_counter[record["load_vec"]] += 1
                argmax_gpu_counter[record["argmax_gpus"]] += 1
                per_layer_y_counter[(layer, y)] += 1
                token_pattern_counter.update(record["token_patterns"])

                if y == target_y:
                    spike_records.append(record)

    total_steps = sum(max_load_counter.values())
    spike_count = len(spike_records)

    print("\n============================================================")
    print("Spike 분석 결과")
    print("============================================================")
    print(f"전체 step 수: {total_steps}")
    print(f"Y = {target_y} step 수: {spike_count}")
    print(f"P(Y = {target_y}) = {spike_count / total_steps:.8f}" if total_steps > 0 else "P(Y) 계산 불가")

    print("\n[전체 max-load 상위 빈도]")
    for y, count in max_load_counter.most_common(top_n):
        print(f"Y = {y:3d}: {count:6d} ({count / total_steps:.6%})")

    if spike_count == 0:
        print(f"\nY = {target_y}인 step이 없습니다.")
        return

    spike_load_vec_counter = Counter(record["load_vec"] for record in spike_records)
    spike_argmax_gpu_counter = Counter(record["argmax_gpus"] for record in spike_records)
    spike_layer_counter = Counter(record["layer"] for record in spike_records)
    spike_token_pattern_counter = Counter()
    spike_dominant_gpu_pattern_counter = Counter()

    for record in spike_records:
        spike_token_pattern_counter.update(record["token_patterns"])
        spike_dominant_gpu_pattern_counter.update(record["dominant_gpu_pattern_counter"])

    print("\n[Spike step의 load vector 상위]")
    for load_vec, count in spike_load_vec_counter.most_common(top_n):
        print(f"{load_vec}: {count:6d} ({count / spike_count:.6%})")

    print("\n[Spike step의 argmax GPU 상위]")
    for argmax_gpus, count in spike_argmax_gpu_counter.most_common(top_n):
        print(f"{argmax_gpus}: {count:6d} ({count / spike_count:.6%})")

    print("\n[Spike step이 많이 나온 layer 상위]")
    for layer, count in spike_layer_counter.most_common(top_n):
        print(f"layer {layer:2d}: {count:6d} ({count / spike_count:.6%})")

    print("\n[Spike step 내부 token GPU pattern 상위]")
    for pattern, count, prob in summarize_token_patterns(
        [pattern for record in spike_records for pattern in record["token_patterns"]],
        top_n=top_n,
    ):
        print(f"{pattern}: {count:6d} ({prob:.6%})")

    print("\n[Spike step 내부 'token별 dominant GPU 집합' 상위]")
    spike_total_tokens = sum(spike_dominant_gpu_pattern_counter.values())
    for dominant_gpus, count in spike_dominant_gpu_pattern_counter.most_common(top_n):
        print(f"{dominant_gpus}: {count:6d} ({count / spike_total_tokens:.6%})")

    print("\n[Spike step 예시 상위]")
    for record in spike_records[:top_n]:
        token_counter = Counter(record["token_patterns"])
        top_patterns = ", ".join(
            f"{pattern} x{count}"
            for pattern, count in token_counter.most_common(4)
        )
        print(
            f"layer={record['layer']:2d}, line={record['line_idx']:4d}, "
            f"load_vec={record['load_vec']}, argmax={record['argmax_gpus']}, "
            f"top_token_patterns=[{top_patterns}]"
        )


if __name__ == "__main__":
    main()
