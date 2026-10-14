# DAGNN-PISTNet-v3 reproducibility code

This repository contains the original model implementation and the protocol/evaluation code used for a traffic-sensor forecasting study. It is a code release, **not** the manuscript or a redistribution of benchmark datasets, pretrained weights, model predictions, or third-party model source trees.

## Contents

- `src/dagnn_pistnet` and `src/dagnn_pistnet_v3`: model, data adapters, training, metrics, and study logic.
- `configs/v3`: selected frozen study and benchmark configurations, including the LargeST-CA 2019/15-minute protocol, the separate 5-minute capacity-matched operator-order control, and source commit lock.
- `scripts/v3/competitors`: selected canonical-data, model-runner, evaluation, paired-analysis, and figure-generation scripts, including the adapted upstream-architecture STID anchor. Figure generation requires separately generated analysis artifacts and does not ship article artwork.
- `docs/v3-operator-order-capacity-matched.md` and `docs/v3-stid-upstream-anchor.md`: protocols and scope limits for the two follow-up checks used in the article.

The competitor scripts require separately obtained upstream checkouts at the commits recorded in `configs/v3/competitor_sources.lock.json`. Their source code is **not** vendored here. Each dataset must be obtained from its respective provider and used under its own terms. Configuration paths are examples relative to a local workspace; no data or output files are included.

## Setup

Use Python 3.12 with a CUDA-enabled PyTorch build matching your system. Install the local package with `pip install -e .`. For the competitor runners, use separate environments matching the upstream projects; do not assume the core environment is compatible with all baselines.

The 5-minute component study and the LargeST-CA 2019/15-minute comparison are separate protocols and must not be mixed in an accuracy ranking. The capacity-matched operator-order control belongs to the former. The adapted upstream-architecture STID anchor belongs to the latter, but is not an exact replication of the published PatchSTG STID row: the canonical sliding-window split and shared evaluator differ from the upstream CA timeline split, and the PatchSTG runner was not reconstructed. In particular, the flow-only experiments do not test hard conservation or event-response behavior. The release does not assert that DAGNN-PISTNet-v3 is the best model on the official LargeST benchmark.

## Provenance and scope

The code is released from a reviewed snapshot of the project workspace. The original study source commit was `f8865ab01454dc40e552d6aa79141043c07302fd`; the later benchmark and analysis utilities came from the main development line. Per-run checkpoints, predictions, and large derived arrays are not included in this small code repository. Any result reproduction must record newly generated artifact hashes, source commits, environment versions, seeds, and evaluator configuration.

## License

The original code and documentation in this repository are licensed under the [MIT License](LICENSE). Preserve the copyright and permission notice when reusing substantial portions. This license does not cover third-party datasets, upstream competitor implementations fetched separately, pretrained weights, or materials absent from this repository (including the manuscript and journal template); those remain subject to their respective owners' terms. A code archive identifier has not yet been assigned.
