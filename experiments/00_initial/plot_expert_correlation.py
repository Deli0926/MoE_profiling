import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # --- 설정 ---
    directory = "./expert_selections"
    model_name = "gpt-oss-20b"
    num_of_experts = 32
    
    print("--- MoE 전문가 간 상관관계(Correlation) 분석 ---")
    ep_size = int(input("분석할 파일의 EP Degree 입력 (예: 4): "))
    target_bs = int(input("분석할 파일의 배치 사이즈(bs) 입력 (권장: 1): "))
    target_layer = input("레이어 번호 입력(0~23, 모든 레이어는 *): ")
    
    # 최신 코드의 파일명 양식 적용
    file_pattern = os.path.join(directory, f"expert_selection_openai_{model_name}_wikitext_103_ep{ep_size}_bs{target_bs}_*_layer{target_layer}.jsonl")
    matched_files = glob.glob(file_pattern)

    if not matched_files:
        print(f"조건에 맞는 파일을 찾을 수 없습니다: {file_pattern}")
        return

    print(f"총 {len(matched_files)}개 파일 데이터를 바탕으로 상관계수를 분석합니다...")

    # 데이터를 담을 리스트 (각 행은 하나의 배치/스텝, 열은 32개 전문가의 활성화 여부 0 또는 1)
    activation_matrix = []

    for file_path in matched_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                data = json.loads(line)
                
                # 리스트의 리스트인지 단일 리스트인지 판별하여 Flatten
                if len(data) > 0 and isinstance(data[0], list):
                    expert_indices = [idx for sublist in data for idx in sublist]
                else:
                    expert_indices = data
                
                # 32차원 One-hot (또는 Multi-hot) 벡터 생성
                vector = [0] * num_of_experts
                for idx in expert_indices:
                    vector[idx] = 1 # 활성화된 전문가는 1로 표시
                
                activation_matrix.append(vector)

    if not activation_matrix:
        print("데이터가 비어있습니다.")
        return

    print("행렬 생성 완료. 상관계수를 계산합니다...")
    # Pandas DataFrame으로 변환 후 피어슨 상관계수 행렬 계산
    df = pd.DataFrame(activation_matrix, columns=[f"{i}" for i in range(num_of_experts)])
    corr_matrix = df.corr()

    # 히트맵 시각화
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_matrix, 
                cmap='coolwarm',     # 양(+)은 빨강, 음(-)은 파랑
                center=0,            # 0을 기준으로 색상 대비
                annot=False,         # 숫자가 많아 지저분해지므로 생략
                fmt=".2f",
                linewidths=0.5,
                vmin=-1, vmax=1)
    
    plt.title(f'Expert Activation Correlation Heatmap (Layer: {target_layer}, bs: {target_bs})', fontsize=16)
    plt.xlabel('Expert Index', fontsize=12)
    plt.ylabel('Expert Index', fontsize=12)
    
    if not os.path.exists('./output'):
        os.makedirs('./output')
    
    # 레이어 번호가 * 인 경우를 대비해 파일명 정제
    layer_str = "all" if target_layer == "*" else target_layer
    save_path = f'./output/expert_correlation_heatmap_layer{layer_str}.png'
    plt.savefig(save_path)
    
    print(f"\n✅ 상관계수 분석 완료! 히트맵이 '{save_path}'에 저장되었습니다.")
    plt.show()

if __name__ == "__main__":
    main()
