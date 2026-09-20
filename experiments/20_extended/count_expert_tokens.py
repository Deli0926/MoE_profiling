import glob
import json
import os
from collections import Counter, defaultdict


def extract_token_expert_lists(data, top_k):
    """
    입력 json line에서 token별 expert 선택 리스트를 복원한다.
    경우 1) [[e1,e2,e3,e4], [..], ...] 형태면 그대로 사용
    경우 2) [e1,e2,e3,e4,e5,...] flat 형태면 top_k씩 끊어서 복원
    """
    if len(data) > 0 and isinstance(data[0], list):
        return data

    if len(data) % top_k != 0:
        raise ValueError(f"flat data length {len(data)} is not divisible by top_k={top_k}")

    return [data[i:i + top_k] for i in range(0, len(data), top_k)]


def token_to_gpu_pattern(token_experts, num_of_experts, ep_size):
    """
    token 하나의 top-k expert 선택을 GPU별 routed-token load pattern으로 변환한다.
    예: top_k=4, EP=4일 때 [3,7,10,28] -> (2,1,0,1)
    """
    experts_per_gpu = num_of_experts // ep_size
    counts = [0] * ep_size

    for expert_idx in token_experts:
        if expert_idx < 0 or expert_idx >= num_of_experts:
            raise ValueError(f"expert index {expert_idx} out of range [0, {num_of_experts - 1}]")

        gpu_idx = expert_idx // experts_per_gpu
        counts[gpu_idx] += 1

    return tuple(counts)


def build_token_pattern_pmf(file_path, top_k, num_of_experts, ep_size):
    """
    source bs=1 실험 데이터 파일 하나를 읽어서
    token-level GPU routed-load pattern empirical PMF를 만든다.
    """
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
    """
    두 load-vector 분포를 합성한다.
    dist[load_vec] = probability
    """
    result = defaultdict(float)

    for load_vec_a, prob_a in dist_a.items():
        for load_vec_b, prob_b in dist_b.items():
            new_vec = tuple(load_vec_a[g] + load_vec_b[g] for g in range(ep_size))
            result[new_vec] += prob_a * prob_b

    return dict(result)


def power_pattern_distribution(pattern_pmf, bs, ep_size):
    """
    token-level pattern PMF의 bs-fold convolution을 binary exponentiation으로 계산한다.
    bs=1 데이터만으로 큰 배치의 exact load-vector 분포를 예측할 때 사용한다.
    """
    zero_vec = tuple([0] * ep_size)
    result = {zero_vec: 1.0}
    base = dict(pattern_pmf)
    exponent = bs

    while exponent > 0:
        if exponent & 1:
            result = convolve_distributions(result, base, ep_size)
        exponent >>= 1
        if exponent > 0:
            base = convolve_distributions(base, base, ep_size)

    return result


def load_vec_dist_to_max_dist(load_vec_dist, bs, top_k):
    """
    GPU load vector 분포를 max routed tokens per GPU 분포로 변환한다.
    Y = max_g L_g
    """
    max_total = bs * top_k
    max_dist = [0.0] * (max_total + 1)

    for load_vec, prob in load_vec_dist.items():
        max_dist[max(load_vec)] += prob

    return max_dist


def percentile_from_dist(dist, q):
    """
    분포 dist에서 q-quantile 값을 반환한다.
    예: q=0.95 -> p95
    """
    cdf = 0.0
    for idx, prob in enumerate(dist):
        cdf += prob
        if cdf >= q:
            return idx
    return len(dist) - 1


def expected_value_from_dist(dist):
    return sum(idx * prob for idx, prob in enumerate(dist))


def print_distribution_table(dist, title=None):
    if title:
        print(title)
    for idx, prob in enumerate(dist):
        print(f"P(Y = {idx:2}) = {prob:.8f} ({prob * 100:>7.4f}%)")


def write_distribution_table(outfile, dist, title=None):
    if title:
        outfile.write(title + "\n")
    for idx, prob in enumerate(dist):
        outfile.write(f"P(Y = {idx:2}) = {prob:.8f} ({prob * 100:>7.4f}%)\n")
    outfile.write("\n")


def print_pattern_pmf(pattern_pmf, title=None):
    if title:
        print(title)
    for pattern, prob in sorted(pattern_pmf.items()):
        print(f"{pattern}: {prob:.8f}")


def write_pattern_pmf(outfile, pattern_pmf, title=None):
    if title:
        outfile.write(title + "\n")
    for pattern, prob in sorted(pattern_pmf.items()):
        outfile.write(f"{pattern}: {prob:.8f}\n")
    outfile.write("\n")


def format_support_size(load_vec_dist):
    return len(load_vec_dist)


def main():
    directory = "./expert_selections"
    num_of_experts = 32
    top_k = 4
    num_of_layers = 24
    model_name = "gpt-oss-20b"

    ep_size = int(input("EP Degree 입력: "))
    bs = int(input("예측할 배치 사이즈(bs) 입력: "))

    source_bs = input("PMF를 추정할 source batch size (기본 1): ").strip()
    if source_bs == "":
        source_bs = "1"

    if num_of_experts % ep_size != 0:
        raise ValueError(f"num_of_experts={num_of_experts} must be divisible by ep_size={ep_size}")

    output_file_path = "output/routed_token_probability_results.txt"
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    print("bs=1 기반 token-level PMF로 큰 배치의 exact load 분포를 예측 중...")

    max_m = bs * top_k
    final_cond_prob_sum = [0.0] * (max_m + 1)
    final_expected_max_sum = 0.0
    final_p95_sum = 0.0
    final_imbalance_ratio_sum = 0.0
    valid_layers = 0

    for layer in range(num_of_layers):
        file_pattern = os.path.join(
            directory,
            f"expert_selection_openai_{model_name}_wikitext_103_ep{ep_size}_bs{source_bs}_*_layer{layer}.jsonl",
        )
        matched_files = glob.glob(file_pattern)

        if not matched_files:
            continue

        file_path = matched_files[0]

        pattern_pmf, total_tokens = build_token_pattern_pmf(
            file_path=file_path,
            top_k=top_k,
            num_of_experts=num_of_experts,
            ep_size=ep_size,
        )

        if total_tokens == 0:
            continue

        valid_layers += 1

        load_vec_dist = power_pattern_distribution(
            pattern_pmf=pattern_pmf,
            bs=bs,
            ep_size=ep_size,
        )

        layer_cond_prob = load_vec_dist_to_max_dist(
            load_vec_dist=load_vec_dist,
            bs=bs,
            top_k=top_k,
        )

        expected_max = expected_value_from_dist(layer_cond_prob)
        p95 = percentile_from_dist(layer_cond_prob, 0.95)
        avg_load = (bs * top_k) / ep_size
        imbalance_ratio = expected_max / avg_load if avg_load > 0 else 0.0
        support_size = format_support_size(load_vec_dist)

        for idx in range(max_m + 1):
            final_cond_prob_sum[idx] += layer_cond_prob[idx]
        final_expected_max_sum += expected_max
        final_p95_sum += p95
        final_imbalance_ratio_sum += imbalance_ratio

        print("\n==========================================")
        print(f"Layer {layer} 계산 결과")
        print("==========================================")
        print(f"PMF source file batch size: {source_bs}")
        print(f"총 token 수 (source bs={source_bs} 데이터 기준): {total_tokens}")
        print(f"예측된 load-vector support 크기: {support_size}")

        print_pattern_pmf(
            pattern_pmf,
            title="[Token-level GPU routed-load pattern PMF from source data]"
        )

        print_distribution_table(
            layer_cond_prob,
            title="[Predicted Max routed tokens per GPU Distribution]"
        )

        print(f"E[max routed tokens per GPU] = {expected_max:.6f}")
        print(f"p95(max routed tokens per GPU) = {p95}")
        print(f"Imbalance Ratio = E[max] / avg = {expected_max:.6f} / {avg_load:.6f} = {imbalance_ratio:.6f}")

        with open(output_file_path, "a", encoding="utf-8") as outfile:
            outfile.write("==========================================\n")
            outfile.write(f"Layer {layer} 계산 결과\n")
            outfile.write("==========================================\n")
            outfile.write(f"PMF source file batch size: {source_bs}\n")
            outfile.write(f"총 token 수 (source bs={source_bs} 데이터 기준): {total_tokens}\n")
            outfile.write(f"예측된 load-vector support 크기: {support_size}\n\n")

            write_pattern_pmf(
                outfile,
                pattern_pmf,
                title="[Token-level GPU routed-load pattern PMF from source data]"
            )

            write_distribution_table(
                outfile,
                layer_cond_prob,
                title="[Predicted Max routed tokens per GPU Distribution]"
            )

            outfile.write(f"E[max routed tokens per GPU] = {expected_max:.6f}\n")
            outfile.write(f"p95(max routed tokens per GPU) = {p95}\n")
            outfile.write(
                f"Imbalance Ratio = E[max] / avg = {expected_max:.6f} / "
                f"{avg_load:.6f} = {imbalance_ratio:.6f}\n\n"
            )

    if valid_layers > 0:
        avg_dist = [value / valid_layers for value in final_cond_prob_sum]
        avg_expected_max = final_expected_max_sum / valid_layers
        avg_p95 = final_p95_sum / valid_layers
        avg_imbalance_ratio = final_imbalance_ratio_sum / valid_layers
        avg_load = (bs * top_k) / ep_size

        print("\n############################################################")
        print(f"EP = {ep_size}, predicted bs = {bs}, source bs = {source_bs}, valid_layers = {valid_layers}")
        print("전체 레이어 평균 최종 확률분포표")
        print("############################################################")
        print_distribution_table(avg_dist)
        print(f"\n평균 E[max routed tokens per GPU] = {avg_expected_max:.6f}")
        print(f"평균 p95(max routed tokens per GPU) = {avg_p95:.6f}")
        print(f"평균 Imbalance Ratio = {avg_imbalance_ratio:.6f}")
        print(f"(참고) avg load per GPU = {avg_load:.6f}")

        with open(output_file_path, "a", encoding="utf-8") as outfile:
            outfile.write("############################################################\n")
            outfile.write(
                f"EP = {ep_size}, predicted bs = {bs}, source bs = {source_bs}, "
                f"valid_layers = {valid_layers}\n"
            )
            outfile.write("전체 레이어 평균 최종 확률분포표\n")
            outfile.write("############################################################\n")
            write_distribution_table(outfile, avg_dist)
            outfile.write(f"평균 E[max routed tokens per GPU] = {avg_expected_max:.6f}\n")
            outfile.write(f"평균 p95(max routed tokens per GPU) = {avg_p95:.6f}\n")
            outfile.write(f"평균 Imbalance Ratio = {avg_imbalance_ratio:.6f}\n")
            outfile.write(f"(참고) avg load per GPU = {avg_load:.6f}\n")
    else:
        print("\n유효한 데이터가 포함된 레이어가 없습니다.")


if __name__ == "__main__":
    main()
