import json
import os
import glob
import math
from collections import Counter

def run_actual_analysis():
    # --- 설정 ---
    directory = "./expert_selections"
    model_name = "gpt-oss-20b"
    num_of_experts = 32
    top_k = 4
    
    print("--- [실측 데이터 전수 조사] MoE 고유 전문가 부하 분석 ---")
    ep_size = int(input("EP Degree 입력 (예: 4): "))
    # 실제 파일명에 적힌 bs를 입력 (예: bs1 파일이면 1, bs2 파일이면 2)
    target_bs = int(input("분석할 파일의 배치 사이즈(bs) 입력: "))
    
    experts_per_gpu = num_of_experts // ep_size
    
    y_counts = Counter()        # P(Y=m) 카운트
    k_counts = Counter()        # P(K) 카운트
    best_case_counts = Counter() # P(Best|K) 카운트
    total_steps = 0

    # 파일 매칭 (예: gpt-oss-20b_..._ep4_bs1_layer_*_...)
    target_layer = input("레이어 번호 입력(0~23, 모든 레이어는 *): ")
    file_pattern = os.path.join(directory, f"expert_selection_openai_{model_name}_wikitext_103_ep{ep_size}_bs{target_bs}_*_layer{target_layer}.jsonl")
    matched_files = glob.glob(file_pattern)

    if not matched_files:
        print(f"조건에 맞는 파일을 찾을 수 없습니다: EP={ep_size}, BS={target_bs}")
        return

    print(f"총 {len(matched_files)}개 레이어 파일을 읽는 중...")

    for file_path in matched_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                # 1. 중복 제거 (중복되는 전문가는 하나로 체크)
                if len(data) > 0 and isinstance(data[0], list):
                    all_indices = [idx for sublist in data for idx in sublist]
                else:
                    all_indices = data
                
                unique_experts = set(all_indices)
                K = len(unique_experts) # 해당 스텝의 고유 전문가 총수
                
                # 2. 각 GPU가 담당하는 고유 전문가 수(c_i) 계산
                gpu_unique_counts = [0] * ep_size
                for exp_idx in unique_experts:
                    gpu_id = exp_idx // experts_per_gpu
                    if gpu_id < ep_size:
                        gpu_unique_counts[gpu_id] += 1
                
                # 3. 실제 Max Load (Y) 계산
                Y = max(gpu_unique_counts)
                
                # 통계 누적
                y_counts[Y] += 1
                k_counts[K] += 1
                total_steps += 1
                
                # 4. Best Case 판별: 현재 K개 전문가가 최적으로 분산되었는가?
                # 예: K=7, EP=4이면 Best Max는 2. 실제 Y가 2이면 Best Case.
                best_possible_max = math.ceil(K / ep_size)
                if Y == best_possible_max:
                    best_case_counts[K] += 1

    if total_steps == 0: return

    # --- 결과 출력 ---
    print(f'\n##### EP = {ep_size}, bs = {target_bs}에서 확률분포표 P(Y = m) 최종 결과 #####')
    # Y는 한 GPU의 고유 전문가 수이므로 max는 experts_per_gpu(8)이지만, 출력 양식상 32까지 표시
    for m in range(min(top_k*target_bs, num_of_experts) + 1):
        prob = y_counts[m] / total_steps
        print(f"P(Y = {m:2}) = {prob:.5f} ({prob * 100:>5.2f}%)")

    # Best Case Probability 계산 (이미지 공식 준수)
    # Sum( P(K) * P(Best|K) )
    final_best_prob = 0
    for k in sorted(k_counts.keys()):
        p_k = k_counts[k] / total_steps
        p_best_given_k = best_case_counts[k] / k_counts[k]
        final_best_prob += p_k * p_best_given_k

    print(f'\033[31mBEST CASE Probability: {final_best_prob * 100:.3f}%\033[0m')

if __name__ == "__main__":
    run_actual_analysis()
