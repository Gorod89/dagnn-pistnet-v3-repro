from __future__ import annotations

import numpy as np

from scripts.v3.analyze_operator_order import exact_seed_sign_flip_p, holm_adjust, primary_windows


def test_primary_windows_deinterleaves_features_per_batch(tmp_path) -> None:
    path = tmp_path / "windows.npz"
    np.savez(path, h12=np.array([1, 2, 3, 4, 11, 12, 13, 14, 21, 22, 23, 24, 5, 15, 25]))
    assert primary_windows(path, feature_index=0, feature_count=3, batch_size=4).tolist() == [1, 2, 3, 4, 5]
    assert primary_windows(path, feature_index=1, feature_count=3, batch_size=4).tolist() == [11, 12, 13, 14, 15]


def test_holm_adjustment_is_monotone() -> None:
    adjusted = holm_adjust({"a": 0.04, "b": 0.01, "c": 0.03})
    assert adjusted == {"b": 0.03, "c": 0.06, "a": 0.06}


def test_three_seed_exact_sign_flip_cannot_resolve_five_percent() -> None:
    assert exact_seed_sign_flip_p([1.0, 1.0, 1.0]) == 0.25
