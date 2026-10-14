# Prespecified capacity-matched operator-order control

This is a new, separate experiment. It does not change the 120-run component
study or the official LargeST-2019 accuracy ranking. The primary question is
whether the studied transport → diffusion → source/sink composition performs
better than a prespecified reverse source/sink → diffusion → transport order
when the **same three learned modules** are present and trained.

## Fixed comparison before new outcomes are inspected

- Datasets: METR-LA, PEMS-BAY, and PEMS08 under their original 5-minute
  component-study protocols. Seeds: 13, 42, and 97. These three datasets have
  time-aligned, unpartitioned per-window test errors in the archived study.
- Reference: the completed `full` run for each dataset/seed under
  `artifacts/v3/studies/dagnn_pistnet_v3_ablation`. Reference source commit:
  `f8865ab01454dc40e552d6aa79141043c07302fd`.
- New control: `operator_order: [source_sink, diffusion, transport]` with all
  modules active once per substep. No width, depth, number of regimes,
  pretraining checkpoint, objective, optimizer, seed, data split, graph,
  rollout curriculum, or selection rule changes. The baseline model code is
  unchanged between the archived source and this study except for the new
  configurable call order and unrelated split-policy validation. A preflight
  must verify normalized configuration equality and parameter-count equality.
- The existing `no_operator_splitting` ablation is **not** the control here:
  it bypasses the operators, reducing effective compute/capacity even though
  their parameters remain allocated.
- Primary endpoint: 60-minute MAE of the declared primary target (speed for
  METR-LA and PEMS-BAY; flow for PEMS08), in original units. Define paired
  difference as reverse minus reference; positive favors the original order.
  Selection of 60 minutes and these three datasets is fixed before the new
  reverse-order runs. Other horizons and distributional scores are secondary.
- Inference: report all nine seed-paired differences, dataset-level mean
  differences, 95% seed-stratified circular moving-block bootstrap intervals
  on aligned test-window errors (12-window blocks, 2000 resamples), and
  Holm-adjusted two-sided block sign-flip p-values across the three dataset
  primary comparisons. Because there are only three independent training
  seeds, also report an exact seed-level sign-flip sensitivity result and do
  not overstate precision from windows alone.

The archived PEMS08 `window_errors.npz` interleaves the three features within
each evaluation batch. Analysis must deinterleave the primary flow series;
concatenating all entries would not test the declared endpoint. Differences
between the unweighted per-window mean and the sensor-weighted global MAE
should be labeled explicitly when nonzero masks vary.

The new study is resumable: completed `result.json` files are skipped only
after manifest verification; interrupted runs resume from their own
`last_training_state_v3.pt`. No second GPU job is launched while the GPU is
occupied. The worktree and common data/artifact links preserve the original
main checkout and avoid duplicating benchmark data.
