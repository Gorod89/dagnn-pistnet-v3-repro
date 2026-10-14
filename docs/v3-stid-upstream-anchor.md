# Adapted upstream-architecture STID anchor

This code-only release includes an adapter for the original STID model at
[`GestaltCogTeam/STID` commit `e8b313bc591bdd0101a1619962c9b503e75127c0`](https://github.com/GestaltCogTeam/STID/tree/e8b313bc591bdd0101a1619962c9b503e75127c0).
The upstream repository is not vendored here. Obtain it separately under its
own Apache-2.0 license and verify the commit before using the adapter.

The adapter tests a specific, separate protocol: LargeST-CA 2019, 15-minute
sampling, 8600 sensors, 12 input steps and 12 forecast steps, canonical
sliding-window train/validation/test arrays, and 7004 test windows. It uses
the pinned upstream California architecture with flow, time-of-day, and
day-of-week channels, a global train-flow scaler, original-unit outputs, and
the shared nonzero-target evaluator. Seeds are 13, 42, and 97. The protocol
hash is `6e302d95e1fd2cb1125b1f2ad8fc6c360e864384f3d8cdf5e087d04393187251`.

The source files for this check are:

- `scripts/v3/competitors/preflight_stid_upstream.py`: CPU architecture and
  input-shape check against the separately obtained upstream checkout.
- `scripts/v3/competitors/preflight_stid_alignment.py`: CPU calendar, split,
  and target-alignment check against the canonical data bundle.
- `scripts/v3/competitors/run_stid_upstream_anchor.py`: one seed per invocation;
  records run configuration, source/environment hashes, learning curve,
  best-validation checkpoint, original-unit predictions, and errors. It
  refuses to overwrite an existing or partial run.
- `scripts/v3/competitors/create_external_prediction_view.py`: zero-copy HDF5
  view of a completed run's prediction descriptor.
- `scripts/v3/competitors/analyze_stid_upstream_anchor.py`: paired analysis
  using separately generated run and reference artifacts.

The runner requires `--repo`, `--native-dir`, `--protocol`, `--seed`, and
`--output-dir`. Run the CPU preflights first; run only one CUDA training job at
a time. For each completed seed, create the external prediction view and call
`dagnn-pistnet-v3 evaluate-predictions` with the frozen configuration,
canonical protocol, and a *separate* evaluator output directory. No raw data,
model weights, predictions, run artifacts, or journal manuscript are shipped.

This is an adapted upstream-*architecture* anchor, **not** an exact
replication of the published PatchSTG STID baseline. The upstream California
generator uses a 60/20/20 timeline split, whereas this adapter uses the
canonical sliding-window split; the published PatchSTG runner and its exact
mask/evaluator correspondence have not been reconstructed. Do not infer the
causal contribution of any one architecture or preprocessing change from the
difference between this anchor and the earlier BasicTS-based local STID run.
