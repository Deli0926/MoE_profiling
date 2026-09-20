# MoE Profiling Experiment

This directory organizes the experiment scripts in `MoE_profiling` by topic.

The input profiling data is stored in `../expert_selections/`, experiment outputs are stored in `../output/`, and the project-level overview is available in [`../README.md`](../README.md).

The experiments are grouped into three categories:

- `00_initial`
  - Initial baseline experiments
  - Empirical counting, expert distributions, correlation analysis, and imbalance analysis
- `10_gumbel`
  - Gumbel-based approximation experiments
  - Fits or predicts max routed-token or expert-load distributions using Gumbel models
- `20_extended`
  - Extended experiments
  - Routed-token exact convolution, capacity-aware analysis, Gaussian/saddlepoint approximations, EP/PP tradeoffs, and latency proxies

## Running the Experiments

Most scripts assume paths relative to the `MoE_profiling/` repository root, such as:

```text
./expert_selections
./output/...
```

It is therefore generally safest to run the scripts from the repository root.

For example:

```bash
cd MoE_profiling
python experiments/00_initial/compare_token_distribution.py
```

or:

```bash
python experiments/10_gumbel/plot_actual_vs_gumbel.py
```

Some scripts may import functions from other experiment scripts, so keeping the current directory structure unchanged is recommended.

## Directory Structure

### `00_initial`

This directory contains the initial probability, empirical counting, distribution comparison, correlation, and imbalance experiments.

It is a good starting point for understanding:

- how routing logs are aggregated,
- how expert selections are converted into GPU-level load patterns,
- how the maximum routed-token load changes with batch size,
- and how larger-batch behavior can be predicted from `bs=1` data.

Included scripts:

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
- `plot_result.py`
- `predict_imbalance_ratio_from_bs1.py`

Recommended files to inspect first:

- `count_expert.py`
  - Computes layer-wise expert selection probabilities and combination-based distributions
- `compare_token_distribution.py`
  - Builds a token-level GPU dispatch-pattern PMF and uses exact convolution to predict larger-batch max-load distributions
- `predict_imbalance_ratio_from_bs1.py`
  - Predicts imbalance ratios using `bs=1` routing logs
- `plot_imbalance_ratio_p95.py`
  - Visualizes actual imbalance and p95 statistics
- `count_expert_correlation.py`
  - Measures correlation among expert selections
- `plot_expert_correlation.py`
  - Visualizes expert correlation results

Main questions addressed by this group:

- How do expert selection probabilities vary across layers?
- How does the maximum routed-token load change as batch size increases?
- Can routing distributions for larger batches be predicted using only `bs=1` logs?
- How large are the resulting imbalance ratios and p95 values?
- How strongly are expert selections correlated?

### `10_gumbel`

This directory contains experiments that approximate max-load distributions using Gumbel models.

The scripts mainly use `scipy.stats.gumbel_r` to compare empirical distributions against Gumbel fits, evaluate layer-wise fitting quality, analyze spikes, and explore alternative max-load formulations.

Included scripts:

- `plot_actual_vs_gumbel.py`
- `plot_actual_vs_gumbel_layerwise.py`
- `plot_actual_vs_gumbel_loadaware.py`
- `plot_actual_vs_gumbel_max_lg_layerwise.py`
- `plot_actual_vs_gumbel_max_lg_spike_removed.py`
- `plot_actual_vs_gumbel_spike_removed.py`
- `plot_gumbel_expert_max_fixed_product.py`
- `plot_gumbel_expert_max_per_layer.py`
- `plot_predicted_gumbel_peff_sweep.py`

Recommended files to inspect first:

- `plot_actual_vs_gumbel.py`
  - Basic comparison between actual and Gumbel distributions
- `plot_actual_vs_gumbel_layerwise.py`
  - Performs Gumbel fitting separately for each layer
- `plot_actual_vs_gumbel_spike_removed.py`
  - Evaluates how removing distribution spikes affects fitting quality
- `plot_actual_vs_gumbel_loadaware.py`
  - Uses a load-aware score rather than a simple maximum count
- `plot_actual_vs_gumbel_max_lg_layerwise.py`
  - Performs layer-wise analysis using the max-load/Gumbel variant
- `plot_actual_vs_gumbel_max_lg_spike_removed.py`
  - Evaluates the same variant after removing spikes
- `plot_predicted_gumbel_peff_sweep.py`
  - Sweeps the effective parameter used in the predicted Gumbel model

Main questions addressed by this group:

- Can the max routed-token load distribution be approximated with a Gumbel distribution?
- How much does fitting quality vary across layers?
- Do distribution spikes explain a significant part of the fitting error?
- Does a load-aware or modified max-load formulation improve the approximation?
- How sensitive is the predicted Gumbel model to its effective parameter?

### `20_extended`

This directory contains experiments that extend the initial exact-distribution and Gumbel analyses.

These experiments focus on routed-token spikes, capacity-aware modeling, Gaussian and saddlepoint approximations, EP/PP tradeoffs, and latency proxies.

Included scripts:

- `analyze_routed_token_spikes.py`
- `compare_capacity_aware_distribution.py`
- `count_expert_tokens.py`
- `plot_actual_vs_saddlepoint.py`
- `plot_distribution_comparisons.py`
- `plot_ep_pp_tradeoff.py`
- `plot_ep_pp_tradeoff_saddlepoint.py`
- `plot_fixed_ep_pp_budget_saddlepoint.py`
- `predict_latency_with_expert_correlation.py`
- `sweep_ep_degrees_from_fixed_routing.py`
- `sweep_ep_degrees_gaussian.py`
- `sweep_ep_degrees_saddlepoint.py`

Recommended files to inspect first:

- `count_expert_tokens.py`
  - Computes token-level dispatch-pattern PMFs and exact-convolution-based distributions
- `analyze_routed_token_spikes.py`
  - Analyzes structural spikes in max routed-token distributions
- `compare_capacity_aware_distribution.py`
  - Compares distributions using capacity-aware scoring or penalties
- `plot_actual_vs_saddlepoint.py`
  - Compares actual distributions against saddlepoint approximations
- `sweep_ep_degrees_gaussian.py`
  - Sweeps expert parallel degree using a Gaussian approximation
- `sweep_ep_degrees_saddlepoint.py`
  - Sweeps expert parallel degree using a saddlepoint approximation
- `plot_ep_pp_tradeoff.py`
  - Visualizes EP/PP tradeoffs
- `plot_ep_pp_tradeoff_saddlepoint.py`
  - Evaluates EP/PP tradeoffs using saddlepoint approximations
- `plot_fixed_ep_pp_budget_saddlepoint.py`
  - Studies fixed EP/PP budgets under the saddlepoint model
- `predict_latency_with_expert_correlation.py`
  - Connects routing concentration and expert correlation to a latency proxy

Main questions addressed by this group:

- Why do structural spikes appear in routed-token max-load distributions?
- Can capacity-aware scoring better explain the observed distributions?
- How well do Gaussian and saddlepoint approximations match empirical behavior?
- How does changing EP affect the expected load distribution?
- What tradeoffs appear between EP and PP?
- How are expert concentration and correlation related to latency?

## Recommended Reading Order

For a first pass through the project, the following order is recommended:

1. `00_initial/count_expert.py`
   - Understand how expert selections are counted.
2. `00_initial/compare_token_distribution.py`
   - Understand token-level PMFs and exact prediction.
3. `00_initial/predict_imbalance_ratio_from_bs1.py`
   - See how predicted distributions are converted into imbalance metrics.
4. `10_gumbel/plot_actual_vs_gumbel.py`
   - Introduce the Gumbel approximation on top of the empirical/exact analysis.
5. `10_gumbel/plot_actual_vs_gumbel_layerwise.py`
   - Examine layer-wise variation.
6. `10_gumbel/plot_predicted_gumbel_peff_sweep.py`
   - Inspect sensitivity of the predicted Gumbel model to its effective parameter.
7. `20_extended/analyze_routed_token_spikes.py`
   - Move from fitting to interpreting the shape of the distribution.
8. `20_extended/plot_actual_vs_saddlepoint.py`
   - Compare against a different approximation method.
9. `20_extended/predict_latency_with_expert_correlation.py`
   - Connect routing behavior to a system-level latency proxy.

## Mapping Scripts to Output Directories

Experiment scripts and subdirectories under `../output/` generally correspond to each other.

Examples:

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

This makes it easier to trace an experiment from source code to generated results, or from a result directory back to the corresponding script.

## Notes

- Most scripts should be run from the `MoE_profiling/` repository root because they rely on relative paths.
- Keep `experiments/`, `expert_selections/`, and `output/` together when moving or archiving the repository.

