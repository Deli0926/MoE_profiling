import subprocess
import re
import matplotlib.pyplot as plt
import os

def run_simulation(script_name, ep, bs):
    """
    특정 스크립트를 실행하여 BEST CASE Probability를 추출하는 함수
    """
    # 입력값 준비 (실측 코드는 레이어 번호 '*' 추가 입력 필요)
    if script_name == 'calculate_real_expert_prob.py':
        input_data = f"{ep}\n{bs}\n*\n"
    else:
        input_data = f"{ep}\n{bs}\n1\n"

    try:
        process = subprocess.run(
            ['python', script_name], 
            input=input_data, 
            capture_output=True, 
            text=True,
            timeout=60 # 무한 루프 방지
        )
        output = process.stdout
        
        # 'BEST CASE Probability: XX.XXX%' 또는 '평균 BEST CASE Probability: XX.XXX%' 모두 매칭
        match = re.search(r'(?:평균 )?BEST CASE Probability:\s*([\d\.]+)%', output)
        if match:
            return float(match.group(1))
        else:
            print(f"  [!] {script_name} (EP={ep}, BS={bs}) 확률값을 찾지 못했습니다.")
            # 디버깅을 위해 출력이 어떻게 나왔는지 확인하고 싶다면 아래 주석을 해제하세요.
            # print("--- 스크립트 출력 ---")
            # print(output)
            
    except Exception as e:
        print(f"  [!] {script_name} 실행 중 오류 발생: {e}")
    
    return None

def main():
    # 테스트 환경 설정
    ep_list = [4]
    bs_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    
    # 데이터 저장 구조
    results_theory = {ep: [] for ep in ep_list}
    results_actual = {ep: [] for ep in ep_list}

    print("📊 이론 vs 실측 시뮬레이션 통합 비교를 시작합니다...\n")

    for ep in ep_list:
        print(f"=== EP Degree {ep} 분석 중 ===")
        for bs in bs_list:
            print(f"  > Batch Size {bs} 진행 중...", end="\r")
            
            # 1. 이론값 추출 (count_expert.py)
            prob_t = run_simulation('count_expert_correlation.py', ep, bs)
            results_theory[ep].append(prob_t if prob_t is not None else 0.0)
            
            # 2. 실측값 추출 (calculate_real_expert_prob.py)
            prob_a = run_simulation('calculate_real_expert_prob.py', ep, bs)
            results_actual[ep].append(prob_a if prob_a is not None else 0.0)

        print(f"\n  └ EP {ep} 완료: 이론/실측 데이터 수집 성공")

    # 3. 시각화 (Plotting)
    plt.figure(figsize=(12, 7))
    
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    for i, ep in enumerate(ep_list):
        # 이론값 점선(Dashed) + 세모 마커
        plt.plot(bs_list, results_theory[ep], linestyle='--', marker='^', 
                 color=colors[i], label=f'Theory (EP {ep})', alpha=0.7)
        
        # 실측값 실선(Solid) + 동그라미 마커
        plt.plot(bs_list, results_actual[ep], linestyle='-', marker='o', 
                 color=colors[i], label=f'Actual (EP {ep})', linewidth=2)
        
        # 수치 표시 (가독성을 위해 실측값 위주로 표시)
        for x, y in zip(bs_list, results_actual[ep]):
            if y > 0: # 유효한 값이 있을 때만 텍스트 표시
                plt.text(x, y + 1.0, f'{y:.1f}%', ha='center', va='bottom', fontsize=8, color=colors[i])

    plt.xscale('log', base=2)
    plt.xticks(bs_list, bs_list)
    plt.ylim(0, 105) # 확률이므로 0~100 범위 고정
    
    plt.xlabel('Batch Size (bs)', fontsize=12)
    plt.ylabel('Best Case Probability (%)', fontsize=12)
    plt.title('Theory vs Actual: Best Case Probability Comparison', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 결과 저장
    if not os.path.exists('./output'):
        os.makedirs('./output')
        
    plt.tight_layout()
    save_path = './output/theory_vs_actual_plot.png'
    plt.savefig(save_path)
    
    print(f"\n✅ 분석 완료!")
    print(f"결과 그래프가 '{save_path}'에 저장되었습니다.")
    plt.show()

if __name__ == "__main__":
    main()
