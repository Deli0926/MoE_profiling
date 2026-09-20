# MoE Profiling

`MoE_profiling`은 MoE 라우팅 결과를 바탕으로 expert load 분포를 분석하고,
배치 크기 변화에 따른 imbalance, routed-token 분포, Gumbel 근사, saddlepoint/gaussian 근사,
latency proxy까지 비교하는 실험 작업 디렉터리다.

이 디렉터리의 핵심은 아래 세 부분이다.

- `experiments/`
  - 실험 스크립트를 성격별로 분류해 둔 인덱스 디렉터리
  - 내부 파일은 대부분 루트의 원본 `.py`를 가리키는 링크다
- `expert_selections/`
  - 실제 프로파일링 로그 데이터
  - 레이어별 expert selection 결과가 `.jsonl`로 저장되어 있다
- `output/`
  - 실험 스크립트가 생성한 그림, CSV, 텍스트 요약 결과

루트에도 원본 Python 스크립트들이 그대로 남아 있다.
기존 실행 경로를 깨지 않기 위해 원본은 루트에 두고,
`experiments/`는 찾아보기 쉽도록 재분류한 보기용 디렉터리로 사용한다.

## 현재 구조

- [experiments](/home/kms20201444/MoE_profiling/experiments)
- [expert_selections](/home/kms20201444/MoE_profiling/expert_selections)
- [output](/home/kms20201444/MoE_profiling/output)
- [README.md](/home/kms20201444/MoE_profiling/README.md)

## 데이터: `expert_selections/`

이 디렉터리는 실험의 입력 데이터다.
파일명 패턴은 아래와 같다.

```text
expert_selection_openai_gpt-oss-20b_wikitext_103_ep{EP}_bs{BS}_{TIMESTAMP}_layer{LAYER}.jsonl
```

예시:

```text
expert_selection_openai_gpt-oss-20b_wikitext_103_ep4_bs1_20260401_214455_layer0.jsonl
expert_selection_openai_gpt-oss-20b_wikitext_103_ep4_bs32_20260402_085953_layer23.jsonl
```

파일명에 담긴 의미:

- `gpt-oss-20b`: 프로파일링에 사용한 모델명
- `wikitext_103`: 입력 데이터셋 이름
- `ep4`: expert parallel degree
- `bs32`: 로그를 수집한 실제 batch size
- `20260402_085953`: 수집 시각 또는 런 식별자
- `layer23`: 해당 레이어의 라우팅 결과

현재 확인된 데이터 범위:

- 모델: `gpt-oss-20b`
- 데이터셋: `wikitext_103`
- EP: `4`
- batch size: `1, 2, 4, 8, 16, 32`
- layer: `0`부터 `23`까지 총 24개

즉, 현재 데이터는 "EP=4인 24개 레이어에 대해 batch size별 라우팅 로그를 모아둔 상태"라고 보면 된다.

### JSONL 포맷

샘플 파일을 보면 각 줄은 JSON 배열 하나다.
현재 데이터는 주로 아래처럼 token 하나당 `top_k=4`개의 expert index가 기록된 형태다.

```json
[31, 21, 22, 6]
[4, 3, 5, 9]
[30, 27, 9, 3]
```

스크립트들은 두 포맷을 모두 처리하도록 작성되어 있다.

- 중첩형: `[[e1, e2, e3, e4], [e1, e2, e3, e4], ...]`
- 평탄형: `[e1, e2, e3, e4, e5, ...]`

여기서 공통 가정은 다음과 같다.

- 전체 expert 수: `32`
- `top_k`: `4`
- layer 수: `24`
- EP가 4이면 GPU당 expert 수는 `32 / 4 = 8`

많은 스크립트는 token별 expert 선택을 GPU load pattern으로 바꿔서 분석한다.
예를 들어 `EP=4`일 때 `[3, 7, 10, 28]`은 GPU별 routed token 수 패턴 `(2, 1, 0, 1)`로 변환된다.

## 결과물: `output/`

이 디렉터리는 실험 결과 저장소다.
PNG, CSV, TXT, HTML, TeX/Markdown 메모가 섞여 있으며,
보통 스크립트 이름과 유사한 하위 디렉터리로 결과가 정리된다.

현재 확인된 주요 결과 그룹은 아래와 같다.

- `actual_vs_gumbel/`
  - 실제 분포와 Gumbel 예측 분포의 전체 비교 플롯
- `actual_vs_gumbel_layerwise/`
  - 레이어별 Gumbel 비교 플롯과 summary CSV
- `actual_vs_gumbel_loadaware/`
  - load-aware score를 넣은 Gumbel 변형 실험 결과
- `actual_vs_gumbel_spike_removed/`
  - 특정 spike를 제거한 뒤 Gumbel fitting을 다시 한 결과
- `capacity_aware_distribution/`
  - capacity-aware 분포 비교 결과와 PMF/summary CSV
- `distribution_compare_ep4/`
  - 이론 분포와 실제 분포를 batch size별로 비교한 CSV와 플롯
- `distribution_compare_ep4_bs1_routed_tokens/`
  - routed-token exact convolution 기반 비교 결과
- `gumbel_expert_max_fixed_product/`
  - `EP * PP` 고정 조건에서 Gumbel max 분포 비교
- `gumbel_expert_max_per_layer/`
  - 레이어별 expert max 분포와 요약 CSV
- `imbalance_ratio/`
  - 실제 imbalance ratio boxplot, p95, summary CSV
- `imbalance_ratio_prediction/`
  - `bs=1` 데이터 기반 imbalance 예측 결과
- `latency_with_expert_correlation/`
  - expert correlation을 반영한 latency proxy 분석 결과

루트에 직접 생성된 결과도 있다.

- `expert_probability_results.txt`
- `routed_token_probability_results.txt`
- `expert_heatmap.png`
- `expert_variance_lineplot.png`
- `expert_correlation_heatmap_layerall.png`
- `result_plot.png`
- `theory_vs_actual_plot.png`
- `token_expert_table_white.html`
- `count_expert_tokens_formulas.md`
- `count_expert_tokens_formulas.tex`

파일명에서 자주 나오는 표현:

- `source_ep4_bs1`
  - 예측의 기준 PMF 또는 fitting source가 `EP=4`, `bs=1` 로그라는 뜻
- `summary.csv`
  - 여러 레이어나 여러 배치 결과를 요약한 표
- `layerXX.png`
  - 개별 레이어 시각화
- `all.png`
  - 여러 레이어를 합치거나 평균낸 결과
- `peff8p0`
  - effective parameter 같은 실험 파라미터
- `alpha`, `beta`, `mode...`
  - 변형 scoring/penalty 규칙에 대한 설정값

## 스크립트: 원본과 분류 인덱스

루트에는 실제 원본 스크립트가 있고,
`experiments/`는 이를 분류해서 보여주는 인덱스다.

### 루트의 원본 스크립트

현재 확인된 원본 스크립트:

- `analyze_routed_token_spikes.py`
- `calculate_prob.py`
- `calculate_real_expert_prob.py`
- `compare_capacity_aware_distribution.py`
- `compare_expert_distributions.py`
- `compare_token_distribution.py`
- `count_expert.py`
- `count_expert_correlation.py`
- `count_expert_tokens.py`
- `plot_actual_vs_gumbel.py`
- `plot_actual_vs_gumbel_layerwise.py`
- `plot_actual_vs_gumbel_loadaware.py`
- `plot_actual_vs_gumbel_spike_removed.py`
- `plot_actual_vs_saddlepoint.py`
- `plot_best_case_prob.py`
- `plot_distribution_comparisons.py`
- `plot_distribution_overlay_bs_1_2_4_8_16_32.py`
- `plot_ep_pp_tradeoff.py`
- `plot_ep_pp_tradeoff_saddlepoint.py`
- `plot_expert_correlation.py`
- `plot_fixed_ep_pp_budget_saddlepoint.py`
- `plot_gumbel_expert_max_fixed_product.py`
- `plot_gumbel_expert_max_per_layer.py`
- `plot_imbalance_ratio_p95.py`
- `plot_result.py`
- `predict_imbalance_ratio_from_bs1.py`
- `predict_latency_with_expert_correlation.py`
- `sweep_ep_degrees_from_fixed_routing.py`
- `sweep_ep_degrees_gaussian.py`
- `sweep_ep_degrees_saddlepoint.py`

### `experiments/`의 의미

`experiments/`는 아래처럼 실험의 성격별로 분류되어 있다.

- [experiments/00_initial](/home/kms20201444/MoE_profiling/experiments/00_initial)
  - 초기에 만든 확률/분포/상관 분석 성격의 스크립트
- [experiments/10_gumbel](/home/kms20201444/MoE_profiling/experiments/10_gumbel)
  - Gumbel fitting 또는 Gumbel 기반 예측을 직접 수행하는 스크립트
- [experiments/20_extended](/home/kms20201444/MoE_profiling/experiments/20_extended)
  - routed-token exact convolution, latency, capacity-aware 등 확장 실험

여기 있는 파일은 원본을 가리키는 링크이므로,
실제 수정은 루트 원본이나 링크 어느 쪽에서 열어도 같은 파일이 편집된다.

## `experiments/00_initial`

초기 탐색용 또는 베이스라인 성격의 스크립트다.
주로 expert 선택 확률, 분포 시각화, 상관관계, batch size 증가에 따른 변화 관찰에 초점이 있다.

- `calculate_prob.py`
  - 초기 확률 계산 및 heatmap/line plot 계열 결과 생성에 사용된 코드로 보인다
- `calculate_real_expert_prob.py`
  - 실제 로그에서 expert 선택 확률을 직접 계산
- `count_expert.py`
  - 레이어별 expert 확률과 조합 기반 분포를 계산하는 초기 코드
- `count_expert_correlation.py`
  - expert 간 동시 선택 상관 구조 계산
- `plot_expert_correlation.py`
  - 상관관계 결과를 heatmap으로 시각화
- `compare_expert_distributions.py`
  - batch size나 조건별 expert 분포 비교
- `compare_token_distribution.py`
  - `bs=1` 소스에서 token-level GPU pattern PMF를 만들고 exact convolution으로 큰 batch 분포를 예측하는 핵심 스크립트
- `plot_distribution_overlay_bs_1_2_4_8_16_32.py`
  - 여러 batch size 분포를 한 그림에 겹쳐서 시각화
- `predict_imbalance_ratio_from_bs1.py`
  - `bs=1` 데이터에서 imbalance ratio를 예측
- `plot_imbalance_ratio_p95.py`
  - imbalance ratio와 p95 지표 시각화
- `plot_best_case_prob.py`
  - best-case 확률 계열 결과를 정리하는 플롯

`plot_result.py`는 현재 `experiments/` 분류에는 넣지 않았지만,
루트에 남아 있는 초기 결과 시각화 보조 스크립트다.

## `experiments/10_gumbel`

Gumbel 근사를 사용하는 실험 스크립트다.
공통적으로 `scipy.stats.gumbel_r`를 이용해 max load 분포를 fitting하거나 예측한다.

- `plot_actual_vs_gumbel.py`
  - 전체 평균 분포 기준 actual vs Gumbel 비교
- `plot_actual_vs_gumbel_layerwise.py`
  - 레이어별 Gumbel fitting과 비교 결과 저장
- `plot_actual_vs_gumbel_spike_removed.py`
  - 이상 spike를 제거한 뒤 다시 Gumbel fitting
- `plot_actual_vs_gumbel_loadaware.py`
  - 단순 max가 아니라 load-aware scoring을 섞은 Gumbel 변형
- `plot_gumbel_expert_max_fixed_product.py`
  - `EP * PP` 고정 조건에서 expert max 분포 비교
- `plot_gumbel_expert_max_per_layer.py`
  - 레이어별 expert max 분포의 Gumbel 적합 결과

## `experiments/20_extended`

초기 exact 분석이나 Gumbel 이후의 확장 실험들이다.

- `count_expert_tokens.py`
  - token-level GPU dispatch pattern PMF와 exact convolution 기반 분포 계산
- `analyze_routed_token_spikes.py`
  - routed token max 분포의 spike 현상 분석
- `compare_capacity_aware_distribution.py`
  - capacity-aware scoring 또는 penalty를 포함한 분포 비교
- `plot_distribution_comparisons.py`
  - 여러 비교 결과 CSV를 읽어 일괄 플로팅
- `plot_ep_pp_tradeoff.py`
  - EP/PP tradeoff 시각화
- `predict_latency_with_expert_correlation.py`
  - expert correlation 조건부 latency proxy 분석
- `sweep_ep_degrees_from_fixed_routing.py`
  - 고정 routing 가정에서 EP를 sweep

현재 루트에는 아래 확장 스크립트도 존재하지만 `experiments/20_extended` 링크에는 아직 포함되지 않았다.

- `plot_actual_vs_saddlepoint.py`
- `plot_ep_pp_tradeoff_saddlepoint.py`
- `plot_fixed_ep_pp_budget_saddlepoint.py`
- `sweep_ep_degrees_gaussian.py`
- `sweep_ep_degrees_saddlepoint.py`

즉, `experiments/20_extended`는 확장 실험의 대표 인덱스이고,
루트 원본이 전체 집합이라고 보는 편이 정확하다.

## 실험 흐름을 어떻게 이해하면 좋은가

이 디렉터리의 실험 흐름은 대체로 아래 순서로 읽힌다.

1. `expert_selections/`의 layer별 라우팅 로그를 읽는다.
2. token별 expert 선택을 GPU load pattern으로 변환한다.
3. 실제 분포를 직접 세거나, `bs=1`에서 얻은 PMF를 큰 batch로 합성한다.
4. exact 분포와 실제 분포를 비교한다.
5. Gumbel, Gaussian, Saddlepoint 같은 근사 모델을 붙여 비교한다.
6. imbalance ratio, p95, latency proxy 같은 파생 지표를 본다.
7. 결과를 `output/`에 PNG/CSV/TXT 형태로 저장한다.

## 실행/수정 시 참고

- 대부분 스크립트는 경로를 상대경로로 가정한다.
  - 예: `./expert_selections`, `./output/...`
- 따라서 보통 `MoE_profiling/` 루트에서 실행하는 것이 안전하다.
- 원본 스크립트끼리 직접 import하는 경우가 있다.
  - 예: `plot_actual_vs_gumbel.py`는 `compare_token_distribution.py`를 사용한다
- 파일 이동보다는 현재처럼 "루트 원본 유지 + `experiments/`에서 분류" 방식이 실행 안정성에 유리하다.

## 추천 읽기 순서

- 데이터 구조부터 이해하려면
  - `expert_selections/` 샘플 파일
  - `compare_token_distribution.py`
  - `count_expert_tokens.py`
- 초기 베이스라인 흐름을 보려면
  - `count_expert.py`
  - `calculate_real_expert_prob.py`
  - `compare_expert_distributions.py`
- Gumbel 계열만 보려면
  - `plot_actual_vs_gumbel.py`
  - `plot_actual_vs_gumbel_layerwise.py`
  - `plot_actual_vs_gumbel_loadaware.py`
- 결과물부터 훑고 싶으면
  - `output/distribution_compare_ep4/`
  - `output/actual_vs_gumbel_layerwise/`
  - `output/latency_with_expert_correlation/`
