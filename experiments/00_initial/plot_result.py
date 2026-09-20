import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1. 결합 확률 데이터 (P(K) * P(Y=m | K))
data = {
    8:  [0.00000, 0.00000, 0.00317, 0.03794, 0.02132, 0.00461, 0.00055, 0.00003, 0.00000, 0.0, 0.0],
    9:  [0.00000, 0.00000, 0.00000, 0.04214, 0.04186, 0.01211, 0.00197, 0.00018, 0.00001, 0.0, 0.0],
    10: [0.00000, 0.00000, 0.00000, 0.03198, 0.06353, 0.02479, 0.00517, 0.00065, 0.00005, 0.0, 0.0],
    11: [0.00000, 0.00000, 0.00000, 0.01642, 0.07476, 0.04085, 0.01063, 0.00172, 0.00017, 0.00001, 0.0],
    12: [0.00000, 0.00000, 0.00000, 0.00447, 0.06759, 0.05512, 0.01777, 0.00354, 0.00045, 0.00004, 0.0],
    13: [0.00000, 0.00000, 0.00000, 0.00000, 0.04595, 0.06135, 0.02477, 0.00592, 0.00093, 0.00009, 0.00001],
    14: [0.00000, 0.00000, 0.00000, 0.00000, 0.02257, 0.05626, 0.02923, 0.00828, 0.00155, 0.00019, 0.00002],
    15: [0.00000, 0.00000, 0.00000, 0.00000, 0.00779, 0.04212, 0.02945, 0.00989, 0.00217, 0.00032, 0.00003],
    16: [0.00000, 0.00000, 0.00000, 0.00000, 0.00148, 0.02531, 0.02542, 0.01021, 0.00259, 0.00045, 0.00005]
}

y_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# DataFrame 변환 (행: K, 열: Y)
#df = pd.DataFrame.from_dict(data, orient='index', columns=y_labels)
#df = pd.DataFrame.from_dict(data, orient='index')
#df.columns = y_labels
# 1. 버그 방지를 위해 모든 데이터를 순수 파이썬 float 리스트로 강제 변환 (안전장치)
safe_data = [list(map(float, val)) for val in data.values()]

# 2. from_dict를 쓰지 않고 정석대로 DataFrame 조립
df = pd.DataFrame(safe_data, index=list(data.keys()), columns=y_labels)

# ==========================================
# 검증 1: 전체 합계 100% 확인
# ==========================================
total_sum = df.sum().sum()
print(f"🔥 모든 확률의 총합: {total_sum * 100:.2f}% (100%가 맞습니다!)")

# ==========================================
# 검증 2: 최종 한계 확률 P(Y=m) 도출
# ==========================================
# 각 열(Y)의 모든 K에 대한 확률을 세로로 다 더합니다 (시그마 연산)
final_P_Y = df.sum()

print("\n====== 최종 병목 확률 P(Y=m) ======")
for m in y_labels:
    if m >= 2:  # 의미 있는 2 이상만 출력
        print(f"P(Y = {m}): {final_P_Y[m]*100:>5.2f}%")

# ==========================================
# 시각화: 최종 P(Y=m) 막대 그래프
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# 1. 히트맵 (전체 우주의 확률 분포)
sns.heatmap(df.loc[:, 2:10] * 100, annot=True, fmt=".2f", cmap="Blues", ax=ax1, 
            cbar_kws={'label': 'Joint Probability (%)'}, linewidths=0.5)
ax1.set_title("Joint Probability Distribution: $P(K) \\times P(Y=m | K)$", fontsize=15, fontweight='bold', pad=15)
ax1.set_xlabel("Max Bottleneck Experts on a Single GPU ($Y=m$)", fontsize=12)
ax1.set_ylabel("Total Active Experts ($K$)", fontsize=12)

# 2. 막대 그래프 (최종 시그마 합산 결과)
x_values = range(2, 11)
y_values = final_P_Y.loc[2:10] * 100

bars = ax2.bar(x_values, y_values, color=sns.color_palette("magma", len(x_values)), edgecolor='black')

# 막대 위에 % 숫자 표기
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{height:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax2.set_title("Final System Bottleneck Probability $P(Y=m)$", fontsize=15, fontweight='bold', pad=15)
ax2.set_xlabel("Max Bottleneck Experts on a Single GPU ($Y=m$)", fontsize=12)
ax2.set_ylabel("Total Probability (%)", fontsize=12)
ax2.set_xticks(x_values)
ax2.set_ylim(0, max(y_values) + 5) # y축 여백 확보

plt.tight_layout(pad=3.0)
plt.show()
