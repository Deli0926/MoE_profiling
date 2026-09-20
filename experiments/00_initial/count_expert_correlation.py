import json
import os
import glob

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
    directory = "./expert_selections"
    num_of_experts = 32
    top_k = 4
    
    model_name = "gpt-oss-20b"
    ep_size = int(input("EP Degree 입력: "))
    bs = int(input("배치 사이즈 입력: "))
    
    num_of_layers = 24
    output_file_path = "output/expert_probability_results.txt"

    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    print("데이터 파싱 및 Batch 기반 실측 확률 연산 중...")

    max_m = min(top_k * bs, num_of_experts)
    final_cond_prob_sum = [0.0] * (max_m + 1)
    final_best_sum = 0.0
    valid_layers = 0

    target_bs = input("실험 데이터 batch size: ")
    if target_bs == "":
        target_bs = "1"    
    for layer in range(num_of_layers):
        file_pattern = os.path.join(directory, f"expert_selection_openai_{model_name}_wikitext_103_ep{ep_size}_bs{target_bs}_*_layer{layer}.jsonl")
        matched_files = glob.glob(file_pattern)

        if not matched_files:
            continue

        file_path = matched_files[0]
        
        # 전체 토큰을 순서대로 담을 리스트
        all_tokens_in_layer = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                data = json.loads(line)
                
                if len(data) > 0 and isinstance(data[0], list):
                    all_tokens_in_layer.extend(data)
                else:
                    all_tokens_in_layer.extend([data[i:i+top_k] for i in range(0, len(data), top_k)])

        total_tokens = len(all_tokens_in_layer)
        num_batches = total_tokens // bs  # 주어진 bs 크기만큼 나눈 실제 배치 개수
        
        if num_batches > 0:
            valid_layers += 1
            print(f"\n==========================================")
            print(f" Layer {layer} 데이터 연산 시작")
            print(f"==========================================")
            
            # --- B. 데이터 기반 Batch 확률(q) 직접 측정 ---
            expert_batch_counts = [0] * num_of_experts
            
            # 토큰들을 bs 단위로 묶어서, 해당 배치에서 전문가가 한 번이라도 켜졌는지 카운트
            for i in range(num_batches):
                batch_tokens = all_tokens_in_layer[i*bs : (i+1)*bs]
                
                batch_experts = set()
                for tok_group in batch_tokens:
                    batch_experts.update(tok_group)
                
                for expert in batch_experts:
                    expert_batch_counts[expert] += 1

            # 기존 수식 q = 1 - (1-p)**bs 대신 실측 q 사용
            probabilities = [count / num_batches for count in expert_batch_counts]
            
            with open(output_file_path, 'a', encoding='utf-8') as outfile:
                outfile.write(f"Layer {layer} Empirical Batch Probability (bs={bs})\n")
                for expert in range(num_of_experts):
                    outfile.write(f"{expert}: {probabilities[expert]:.5f} ")
                outfile.write("\n\n")

            # --- C. 다항식(DP) 전개를 통한 가중치 계산 (기존 알고리즘 유지) ---
            N = num_of_experts // ep_size
            coef = [[1.0] + [0.0] * N for _ in range(ep_size)]
            coef_total = [1.0] + [0.0] * num_of_experts

            print(f"[Layer {layer}] 다항식 가중치 연산 중...")
            for gpus in range(ep_size):
                for i in range(N):
                    expert_idx = N * gpus + i
                    prob_q = probabilities[expert_idx] # 수식이 아닌 실측 q 사용
                    
                    for j in range(i + 1, 0, -1):
                        coef[gpus][j] = coef[gpus][j] + coef[gpus][j - 1] * prob_q
                    
                    for j in range(expert_idx + 1, 0, -1):
                        coef_total[j] = coef_total[j] + coef_total[j - 1] * prob_q

            sum_of_coef = sum(coef_total[j] for j in range(top_k, max_m + 1))
            
            # --- D. 확률분포표 P(Y = m) 도출 ---
            layer_cond_prob = [0.0] * (max_m + 1)
            layer_best = 0.0
            
            for k in range(top_k, max_m + 1):
                prob_k = coef_total[k] / sum_of_coef if sum_of_coef != 0 else 0.0
                for m in range(0, max_m + 1):
                    prob, _ = calculate_probability_distribution(m, coef, coef_total, k)
                    layer_cond_prob[m] += prob_k * prob
                    if m < (k / ep_size) + 1 and m >= k / ep_size:
                        layer_best += prob_k * prob

            for i in range(0, max_m + 1):
                final_cond_prob_sum[i] += layer_cond_prob[i]
            final_best_sum += layer_best

    if valid_layers > 0:
        print(f'\n##### EP = {ep_size}, bs = {bs}에서 {valid_layers}개 레이어 평균 최종 확률분포표 #####')
        for i in range(0, max_m + 1):
            avg_prob = final_cond_prob_sum[i] / valid_layers
            if avg_prob > 0.000001:
                print(f"P(Y = {i:2}) = {avg_prob:.5f} ({avg_prob * 100:>5.2f}%)")
        
        avg_best = final_best_sum / valid_layers
        print(f'\033[31m평균 BEST CASE Probability: {avg_best * 100:.3f}%\033[0m')
    else:
        print("\n유효한 데이터가 포함된 레이어가 없습니다.")

if __name__ == "__main__":
    main()
