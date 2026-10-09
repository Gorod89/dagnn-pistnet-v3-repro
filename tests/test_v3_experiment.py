from __future__ import annotations

from pathlib import Path

import numpy as np

from dagnn_pistnet_v3.config import V3ExperimentConfig
from dagnn_pistnet_v3.experiment import V3Experiment


def test_v3_one_epoch_end_to_end(tmp_path: Path) -> None:
    time = np.arange(72, dtype=np.float32)[:, None, None]
    nodes = np.arange(4, dtype=np.float32)[None, :, None]
    values = 20 + np.sin(time / 4) + nodes
    data_path = tmp_path / "pems08.npz"
    np.savez(data_path, data=values.astype(np.float32))
    output = tmp_path / "artifacts"
    config = V3ExperimentConfig.from_dict(
        {
            "data": {
                "dataset": "PEMS08",
                "data_path": str(data_path),
                "feature_names": ["flow"],
                "units": ["vehicles_per_step"],
                "history_steps": 4,
                "forecast_steps": 3,
                "train_ratio": 0.6,
                "val_ratio": 0.2,
                "graph_top_k": 2,
            },
            "model": {
                "hidden_dim": 8,
                "temporal_layers": 1,
                "num_regimes": 2,
                "min_clusters": 2,
                "max_clusters": 2,
            },
            "training": {
                "epochs": 1,
                "batch_size": 4,
                "learning_rate": 0.001,
                "patience": 1,
                "device": "cpu",
                "amp": False,
                "curriculum_start_steps": 3,
                "semigroup_weight": 0.01,
                "masked_pretrain_weight": 0.01,
                "output_dir": str(output),
            },
            "study": {"seeds": [1, 2, 3], "bootstrap_samples": 100},
        }
    )
    result = V3Experiment(config, workspace=tmp_path).train()
    assert result.best_epoch == 1
    assert result.test_metrics["5min"]["flow"]["mae"] >= 0
    assert result.test_metrics["15min"]["flow"]["crps"] >= 0
    assert Path(result.checkpoint_path).is_file()
    assert (output / "manifest.json").is_file()
    assert (output / "window_errors.npz").is_file()
