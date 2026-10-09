import pytest

from dagnn_pistnet.config import ExperimentConfig


def test_config_rejects_invalid_split() -> None:
    with pytest.raises(ValueError, match="less than 1"):
        ExperimentConfig.from_dict(
            {
                "data": {
                    "dataset": "PEMS07",
                    "data_path": "unused.npz",
                    "train_ratio": 0.8,
                    "val_ratio": 0.2,
                }
            }
        )


def test_config_converts_feature_metadata_to_tuples() -> None:
    config = ExperimentConfig.from_dict(
        {
            "data": {
                "dataset": "PEMS07",
                "data_path": "unused.npz",
                "feature_names": ["flow"],
                "units": ["vehicles_per_5_minutes"],
            }
        }
    )
    assert config.data.feature_names == ("flow",)
