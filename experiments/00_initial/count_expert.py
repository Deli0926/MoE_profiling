import json
import os
import glob
import itertools

def calculate_probability_distribution(m, coef, W_total, top_k):
    ep_size = len(coef)
    max_idx_per_gpu = len(coef[0]) - 1

    def get_sum_up_to(limit):
        if limit < 0: return 0
        dp = [0.0] * (top_k + 1)
        dp[0] = 1.0
        
        for g in range(ep_size):
            new_dp = [0.0] * (top_k + 1)
            upper_c = min(limit, max_idx_per_gpu, top_k)
            for c in range(upper_c + 1):
                weight = coef[g][c]
                for s in range(c, top_k + 1):
                    new_dp[s] += dp[s - c] * weight
            dp = new_dp
        return dp[top_k]

    numerator_m = get_sum_up_to(m)
    numerator_m_minus_1 = get_sum_up_to(m - 1)
    
    prob_sum = numerator_m - numerator_m_minus_1
    final_probability = prob_sum / W_total[top_k] if W_total[top_k] != 0 else 0.0
    
    return final_probability, []

def main():
    # ==========================================
    # 1. 초기 설정 및 데이터 준비
    # ==========================================
    directory = "./expert_selections"
    num_of_experts = 32
    top_k = 4
    
    model_name = "gpt-oss-20b"
    ep_size = int(input("EP Degree 입력: "))
    bs = int(input("배치 사이즈 입력: "))
    
    num_of_layers = 24  # 레이어 0 ~ 23
    layer_counts = [[0] * num_of_experts for _ in range(num_of_layers)]
    layer_tokens = [0] * num_of_layers 

    output_file_path = "output/expert_probability_results.txt"

    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    print("데이터 파싱 및 레이어별 연산 중...")

    # 전체 레이어의 최종 확률분포 평균을 내기 위한 변수
    max_m = min(top_k * bs, num_of_experts)
    final_cond_prob_sum = [0.0] * (max_m + 1)
    final_best_sum = 0.0
    valid_layers = 0
    
    # ==========================================
    # 2. 레이어별 순회하며 독립적인 확률분포 연산
    # ==========================================
    target_bs = input("실험 데이터 batch size (1이 기본): ")
    if target_bs == "":
        target_bs = "1"
    for layer in range(num_of_layers):
        file_pattern = os.path.join(directory, f"expert_selection_openai_{model_name}_wikitext_103_ep{ep_size}_bs{target_bs}_*_layer{layer}.jsonl")
        matched_files = glob.glob(file_pattern)

        if not matched_files:
            continue

        file_path = matched_files[0]
        
        # --- A. 데이터 파싱 ---
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                data = json.loads(line)
                
                if len(data) > 0 and isinstance(data[0], list):
                    expert_indices = [expert for sublist in data for expert in sublist]
                else:
                    expert_indices = data

                batch_tokens = len(expert_indices) // top_k
                layer_tokens[layer] += batch_tokens

                for expert in expert_indices:
                    layer_counts[layer][expert] += 1

        layer_total_count = sum(layer_counts[layer])
        
        # 유효한 토큰이 있는 레이어만 처리
        if layer_total_count > 0:
            valid_layers += 1
            print(f"\n==========================================")
            print(f" Layer {layer} 데이터 연산 시작")
            print(f"==========================================")
            print(f"Layer {layer} Probability")
            [print(f"{expert}: {layer_counts[layer][expert] / layer_total_count:.5f}%", end=" ") for expert in range(num_of_experts)]
            print() 

            with open(output_file_path, 'a', encoding='utf-8') as outfile:
                outfile.write(f"Layer {layer} Probability\n")
                for expert in range(num_of_experts):
                    outfile.write(f"{expert}: {layer_counts[layer][expert] / layer_total_count:.5f}% ")
                outfile.write("\n\n")

            # --- B. 개별 레이어 기반 전문가별 확률(p, q) 계산 ---
            probabilities = []
            #print(f"\n[Layer {layer}] 전문가 선택 확률:")
            for expert_idx, count in enumerate(layer_counts[layer]):
                p = count / (layer_tokens[layer] * top_k) 
                q = 1 - (1 - p) ** bs
                probabilities.append(q)
                #print(f"Expert {expert_idx:2}: p = {p:.6f}, q(bs={bs}) = {q:.5f}")

            # --- C. 다항식(DP) 전개를 통한 가중치 계산 ---
            N = num_of_experts // ep_size
            coef = [[1.0] + [0.0] * N for _ in range(ep_size)]
            coef_total = [1.0] + [0.0] * num_of_experts

            print(f"\n[Layer {layer}] 다항식 가중치 연산 중...")
            for gpus in range(ep_size):
                #print(f'========== GPU {gpus} ===========')
                for i in range(N):
                    for j in range(i + 1, 0, -1):
                        coef[gpus][j] = coef[gpus][j] + coef[gpus][j - 1] * probabilities[N * gpus + i]
                        #print(f'{coef[gpus][j]:.3f}x^{j} + ', end="")
                    #print('1.0000')
                    
                    for j in range(N * gpus + i + 1, 0, -1):
                        coef_total[j] = coef_total[j] + coef_total[j - 1] * probabilities[N * gpus + i]
                #print(f'==============================\n')

            sum_of_coef = sum(coef_total[j] for j in range(top_k, max_m + 1))
            
            # --- D. 레이어별 확률분포표 P(Y = m) 도출 및 누적 ---
            layer_cond_prob = [0.0] * (max_m + 1)
            layer_best = 0.0
            
            for k in range(top_k, max_m + 1):
                prob_k = coef_total[k] / sum_of_coef if sum_of_coef != 0 else 0.0
                for m in range(0, max_m + 1):
                    prob, _ = calculate_probability_distribution(m, coef, coef_total, k)
                    layer_cond_prob[m] += prob_k * prob
                    if m < (k / ep_size) + 1 and m >= k / ep_size:
                        layer_best += prob_k * prob

            # 최종 평균 계산을 위해 해당 레이어의 분포 결과를 전역 변수에 누적
            for i in range(0, max_m + 1):
                final_cond_prob_sum[i] += layer_cond_prob[i]
            final_best_sum += layer_best

    # ==========================================
    # 3. 누적된 결과를 바탕으로 최종 평균 확률분포표 출력
    # ==========================================
    if valid_layers > 0:
        print(f'\n##### EP = {ep_size}, bs = {bs}에서 {valid_layers}개 레이어 평균 최종 확률분포표 #####')
        for i in range(0, max_m + 1):
            avg_prob = final_cond_prob_sum[i] / valid_layers
            print(f"P(Y = {i:2}) = {avg_prob:.5f} ({avg_prob * 100:>5.2f}%)")
        
        avg_best = final_best_sum / valid_layers
        print(f'\033[31m평균 BEST CASE Probability: {avg_best * 100:.3f}%\033[0m')
    else:
        print("\n유효한 데이터가 포함된 레이어가 없습니다.")

if __name__ == "__main__":
    main()
