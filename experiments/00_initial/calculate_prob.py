import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 파일 및 폴더 설정
input_file_path = "output/expert_probability_results.txt"
output_dir = "output"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("데이터 읽기 및 파싱 중...")

# 2. 파일 읽기
with open(input_file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 'Layer 숫자 Probability'를 기준으로 텍스트 블록 나누기
layer_blocks = re.split(r'Layer \d+ Probability', content)
layer_blocks = [block.strip() for block in layer_blocks if block.strip()]

expert_probs = [] 

# 3. 데이터 파싱
for block_idx, block in enumerate(layer_blocks):
    # 💡 정규표현식 수정: "숫자: 숫자%" 형태를 정확히 잡아냅니다. (예: "0: 0.00760%")
    matches = re.findall(r"(\d+):\s+([\d.]+)%", block)
    
    if not matches:
        continue
        
    # 전문가 번호(0~63) 순서대로 정렬
    matches.sort(key=lambda x: int(x[0]))
    probs = [float(m[1]) for m in matches]
    
    if len(probs) > 0:
        expert_probs.append(probs)

# 2차원 numpy 배열로 변환
expert_probs = np.array(expert_probs)

# 배열 형태 확인 (예: 16 레이어 x 64 전문가 = (16, 64)가 나와야 정상입니다)
print(f"✅ 파싱 성공! 데이터 차원(Shape): {expert_probs.shape}")
num_layers, num_experts = expert_probs.shape

# 4. 통계 계산 (각 전문가별 평균 및 표준편차)
means = np.mean(expert_probs, axis=0)
stds = np.std(expert_probs, axis=0)

# 5. 콘솔에 통계 결과 출력
print(f"\n총 {num_layers}개 레이어, {num_experts}명의 전문가 데이터를 분석했습니다.")
print("="*45)
print(f"{'Expert ID':<16} | {'Mean (%)':<12} | {'Std Dev (%)':<12}")
print("-" * 45)
for i in range(num_experts):
    print(f"Expert {i:<9} | {means[i]:<12.5f} | {stds[i]:<12.5f}")
print("="*45)

# 6. 시각화 (Plotting)
print("\n그래프를 생성하고 있습니다...")

# --- Plot 1: 전체 히트맵 (Heatmap) ---
plt.figure(figsize=(16, 8))
# y축(레이어)이 위에서 아래로(0~15) 자연스럽게 보이도록 출력
sns.heatmap(expert_probs, cmap='viridis', cbar_kws={'label': 'Probability (%)'})
plt.title('Expert Routing Probabilities Across All Layers', fontsize=16, pad=15)
plt.xlabel('Expert ID', fontsize=12)
plt.ylabel('Layer Number', fontsize=12)
plt.tight_layout()
heatmap_path = os.path.join(output_dir, 'expert_heatmap.png')
plt.savefig(heatmap_path, dpi=300)
print(f"히트맵 저장 완료: {heatmap_path}")

# --- Plot 2: 레이어별 활성도 편차가 가장 큰 Top 5 전문가 꺾은선 그래프 ---
if num_experts >= 0:
    # 표준편차가 큰 순서대로 정렬하여 상위 5명 인덱스 추출
    top_varying_experts = np.argsort(stds)[::-1][:64]
    
    plt.figure(figsize=(12, 6))
    for exp_id in top_varying_experts:
        plt.plot(range(num_layers), expert_probs[:, exp_id], marker='o', linewidth=2, label=f'Expert {exp_id}')
        
    plt.title('Top 5 Experts with the Highest Variance Across Layers', fontsize=16, pad=15)
    plt.xlabel('Layer Number', fontsize=12)
    plt.ylabel('Probability (%)', fontsize=12)
    plt.xticks(range(num_layers)) # x축 눈금 단위를 레이어 번호 정수로 고정
    plt.legend(title='Expert ID')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    lineplot_path = os.path.join(output_dir, 'expert_variance_lineplot.png')
    plt.savefig(lineplot_path, dpi=300)
    print(f"꺾은선 그래프 저장 완료: {lineplot_path}")

# 화면에 그래프 띄우기 (터미널 환경에 따라 생략 가능)
plt.show()
