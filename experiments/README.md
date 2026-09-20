# MoE Profiling Experiment Index

이 디렉터리는 `MoE_profiling` 안의 실험 스크립트를 주제별로 다시 묶어 둔 인덱스다.
실제 입력 데이터는 `../expert_selections/`,
실험 결과물은 `../output/`,
프로젝트 전체 설명은 [../README.md](/home/kms20201444/MoE_profiling/README.md)에 정리되어 있다.

원본 Python 파일은 `MoE_profiling/` 루트에 그대로 남아 있고,
이 디렉터리의 파일들은 대부분 그 원본을 가리키는 심볼릭 링크다.
즉, 여기서는 "어떤 실험이 어떤 계열인지 빠르게 찾는 것"이 목적이다.

## 이 디렉터리를 왜 만들었는가

루트에는 실험 스크립트가 한곳에 모여 있지만,
작성 시기와 목적이 섞여 있어서 처음 보면 흐름을 파악하기 어렵다.
그래서 이 디렉터리에서는 스크립트를 아래 세 묶음으로 나눴다.

- `00_initial`
  - 초기 베이스라인 실험
  - 실제 로그를 직접 세고, expert 분포와 imbalance를 관찰하는 성격
- `10_gumbel`
  - Gumbel 근사 실험
  - max routed-token load 분포를 Gumbel로 fitting하거나 예측하는 성격
- `20_extended`
  - 이후 확장 실험
  - routed-token exact convolution, capacity-aware, latency proxy 등 파생 분석

## 사용 방법

- 스크립트를 찾을 때는 이 디렉터리에서 주제별로 훑는다.
- 실제 실행은 보통 `MoE_profiling/` 루트에서 하는 편이 안전하다.
  - 많은 스크립트가 `./expert_selections`, `./output/...` 같은 상대경로를 사용한다.
  - 일부 스크립트는 루트의 다른 `.py`를 import한다.
- 여기 있는 링크를 열어 수정해도 실제로는 루트 원본이 수정된다.

## 디렉터리 구성

### `00_initial`

초기 확률 계산, empirical counting, 분포 비교, imbalance 관찰 스크립트를 모아 둔 영역이다.
이 그룹은 "실제 로그가 어떻게 생겼는지", "batch size가 커지면 max load가 어떻게 변하는지",
"bs=1 데이터만으로 큰 batch를 어느 정도 예측할 수 있는지"를 이해하는 출발점에 가깝다.

현재 포함된 스크립트:

- `calculate_prob.py`
- `calculate_real_expert_prob.py`
- `compare_expert_distributions.py`
- `compare_token_distribution.py`
- `count_expert.py`
- `count_expert_correlation.py`
- `plot_best_case_prob.py`
- `plot_distribution_overlay_bs_1_2_4_8_16_32.py`
- `plot_expert_correlation.py`
- `plot_imbalance_ratio_p95.py`
- `predict_imbalance_ratio_from_bs1.py`

대표적으로 보면 좋은 파일:

- `count_expert.py`
  - 레이어별 expert 선택 확률을 세고, 초기 조합 기반 분포 계산을 수행하는 초반 코드
- `compare_token_distribution.py`
  - token-level GPU dispatch pattern PMF를 만든 뒤 batch 크기만큼 합성해서 exact max-load 분포를 예측하는 핵심 코드
- `predict_imbalance_ratio_from_bs1.py`
  - `bs=1` 로그에서 imbalance ratio를 예측하는 파생 실험
- `plot_imbalance_ratio_p95.py`
  - 실제 imbalance와 p95 지표를 정리해 시각화하는 코드

이 그룹이 답하려는 질문:

- expert 선택 확률은 레이어별로 어떻게 다른가
- batch size가 커지면 GPU별 max routed token load는 어떻게 변하는가
- `bs=1` 로그만으로 큰 batch의 분포를 예측할 수 있는가
- imbalance ratio나 p95 같은 지표가 실제로 얼마나 커지는가

### `10_gumbel`

이 그룹은 max load 분포를 Gumbel 분포로 근사하는 스크립트를 모아 둔 영역이다.
공통적으로 `scipy.stats.gumbel_r`를 사용하고,
실제 분포와 근사 분포를 비교하거나,
레이어별 fitting 결과를 summary CSV와 plot으로 저장한다.

현재 포함된 스크립트:

- `plot_actual_vs_gumbel.py`
- `plot_actual_vs_gumbel_layerwise.py`
- `plot_actual_vs_gumbel_loadaware.py`
- `plot_actual_vs_gumbel_spike_removed.py`
- `plot_gumbel_expert_max_fixed_product.py`
- `plot_gumbel_expert_max_per_layer.py`

대표적으로 보면 좋은 파일:

- `plot_actual_vs_gumbel.py`
  - 가장 기본적인 actual vs Gumbel 비교
- `plot_actual_vs_gumbel_layerwise.py`
  - 레이어 단위로 분포를 따로 fitting해서 비교
- `plot_actual_vs_gumbel_spike_removed.py`
  - spike가 fitting 품질에 미치는 영향을 확인
- `plot_actual_vs_gumbel_loadaware.py`
  - 단순 max count 대신 load-aware score를 넣은 변형 실험

이 그룹이 답하려는 질문:

- max routed-token load 분포를 Gumbel로 근사해도 되는가
- fitting 오차는 전체 평균 기준과 레이어별 기준에서 얼마나 다른가
- spike 제거가 근사 품질을 개선하는가
- load-aware scoring을 넣으면 더 그럴듯한 모형이 되는가

### `20_extended`

이 그룹은 초기 counting과 Gumbel 이후에 붙은 확장 실험을 모아 둔 영역이다.
기본 분포 자체보다는,
"왜 이런 spike가 생기는지",
"capacity-aware 보정을 넣으면 어떤 변화가 나는지",
"expert correlation이 latency proxy에 어떤 영향을 주는지"처럼
조금 더 해석적이고 응용적인 질문을 다룬다.

현재 포함된 스크립트:

- `analyze_routed_token_spikes.py`
- `compare_capacity_aware_distribution.py`
- `count_expert_tokens.py`
- `plot_distribution_comparisons.py`
- `plot_ep_pp_tradeoff.py`
- `predict_latency_with_expert_correlation.py`
- `sweep_ep_degrees_from_fixed_routing.py`

대표적으로 보면 좋은 파일:

- `count_expert_tokens.py`
  - token-level pattern PMF와 exact convolution을 좀 더 체계적으로 정리한 코드
- `analyze_routed_token_spikes.py`
  - max-load 분포에 나타나는 spike를 직접 분석
- `compare_capacity_aware_distribution.py`
  - capacity-aware 보정을 넣은 분포 비교
- `predict_latency_with_expert_correlation.py`
  - expert correlation을 조건부로 반영한 latency proxy 분석

이 그룹이 답하려는 질문:

- routed-token 분포의 구조적 spike는 왜 생기는가
- capacity-aware penalty나 scoring이 분포 형태를 설명하는 데 도움이 되는가
- expert concentration과 latency proxy는 어떻게 연결되는가
- 고정된 routing 가정에서 EP sweep을 하면 어떤 tradeoff가 보이는가

## 현재 인덱스에 없는 루트 스크립트

루트 원본 중 일부는 아직 이 디렉터리의 링크 인덱스에 넣지 않았다.
예를 들어 아래 파일들은 루트에는 있지만 현재 여기에는 링크가 없다.

- `plot_actual_vs_saddlepoint.py`
- `plot_ep_pp_tradeoff_saddlepoint.py`
- `plot_fixed_ep_pp_budget_saddlepoint.py`
- `plot_result.py`
- `sweep_ep_degrees_gaussian.py`
- `sweep_ep_degrees_saddlepoint.py`

즉, 이 디렉터리는 "완전한 원본 저장소"라기보다
"핵심 실험 흐름을 보기 좋게 묶어 둔 작업용 인덱스"라고 이해하는 편이 맞다.

## 추천 읽기 순서

처음 보는 경우에는 아래 순서가 가장 자연스럽다.

1. `00_initial/count_expert.py`
   - 데이터가 어떻게 집계되는지 감을 잡기 좋다.
2. `00_initial/compare_token_distribution.py`
   - 이 프로젝트의 핵심 아이디어 중 하나인 token-level PMF 기반 exact prediction을 이해할 수 있다.
3. `00_initial/predict_imbalance_ratio_from_bs1.py`
   - exact 분포에서 imbalance 지표로 넘어가는 흐름을 볼 수 있다.
4. `10_gumbel/plot_actual_vs_gumbel.py`
   - exact/empirical 분석 뒤 근사 모델이 어디 붙는지 알 수 있다.
5. `10_gumbel/plot_actual_vs_gumbel_layerwise.py`
   - 레이어별 편차를 보는 흐름으로 이어진다.
6. `20_extended/analyze_routed_token_spikes.py`
   - 분포 모양 자체를 해석하는 방향으로 확장된다.
7. `20_extended/predict_latency_with_expert_correlation.py`
   - 최종적으로 시스템 성능 지표와 연결되는 응용 분석으로 이어진다.

## 결과 디렉터리와의 대응

실험 스크립트 이름과 `../output/` 하위 디렉터리 이름은 대체로 대응된다.
예를 들어:

- `plot_actual_vs_gumbel.py`
  - `../output/actual_vs_gumbel/`
- `plot_actual_vs_gumbel_layerwise.py`
  - `../output/actual_vs_gumbel_layerwise/`
- `plot_actual_vs_gumbel_loadaware.py`
  - `../output/actual_vs_gumbel_loadaware/`
- `plot_actual_vs_gumbel_spike_removed.py`
  - `../output/actual_vs_gumbel_spike_removed/`
- `compare_capacity_aware_distribution.py`
  - `../output/capacity_aware_distribution/`
- `predict_latency_with_expert_correlation.py`
  - `../output/latency_with_expert_correlation/`

그래서 "코드에서 결과로", 또는 "결과에서 원본 코드로" 역추적할 때 이 디렉터리가 중간 인덱스 역할을 한다.

## 주의사항

- 이 디렉터리의 파일은 대부분 심볼릭 링크다.
- import 경로와 상대경로 때문에 실행은 보통 루트에서 하는 편이 안전하다.
- 서버로 옮길 때는 `MoE_profiling/` 디렉터리 전체를 유지하는 편이 좋다.
  - `expert_selections/`, `output/`, 루트 원본 `.py`, `experiments/` 링크 구조가 함께 유지되어야 한다.
