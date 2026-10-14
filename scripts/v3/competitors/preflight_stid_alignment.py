"""CPU-only alignment check between canonical LargeST bundle and STID native splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from canonical_bundle import calendar_features, metadata, target_indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path, required=True)
    args = parser.parse_args()

    meta = metadata(args.bundle)
    manifest = json.loads(
        (args.native_dir / "native_manifest.json").read_text(encoding="utf-8")
    )
    if manifest["protocol_sha256"] != meta.protocol_sha256:
        raise ValueError("Native split and canonical bundle protocol hashes differ")
    if manifest["split_policy"] != "official_sliding_windows":
        raise ValueError("Unexpected native split policy")
    anchors = target_indices(meta, manifest["split_policy"])
    report = {"protocol_sha256": meta.protocol_sha256, "splits": {}}

    with h5py.File(args.bundle, "r") as bundle:
        tod, dow = calendar_features(bundle, meta.sampling_seconds)
        for split, indices in anchors.items():
            native_values = np.load(
                args.native_dir / f"{split}_data.npy", mmap_mode="r"
            )
            native_times = np.load(
                args.native_dir / f"{split}_timestamps.npy", mmap_mode="r"
            )
            start = int(indices[0] - meta.history_steps)
            stop = int(indices[-1] + meta.forecast_steps)
            if native_values.shape != (stop - start, meta.num_nodes):
                raise ValueError(f"{split}: native value shape mismatch")
            if native_times.shape != (stop - start, 2):
                raise ValueError(f"{split}: native calendar shape mismatch")

            # Both endpoints and the first/last forecast boundaries must agree.
            offsets = sorted(
                {0, meta.history_steps, stop - start - meta.forecast_steps, stop - start - 1}
            )
            for offset in offsets:
                source = start + offset
                np.testing.assert_allclose(
                    native_values[offset],
                    bundle["values"][source, :, 0],
                    rtol=0,
                    atol=0,
                    err_msg=f"{split}: flow mismatch at native offset {offset}",
                )
                np.testing.assert_allclose(
                    native_times[offset],
                    (tod[source], dow[source] / 7.0),
                    rtol=0,
                    atol=1e-7,
                    err_msg=f"{split}: calendar mismatch at offset {offset}",
                )
            report["splits"][split] = {
                "first_target_global_index": int(indices[0]),
                "last_target_global_index": int(indices[-1]),
                "num_windows": int(len(indices)),
                "native_time_steps": int(len(native_values)),
                "checked_offsets": offsets,
            }

    if report["splits"]["test"]["num_windows"] != 7004:
        raise ValueError("Test window count differs from canonical evaluator")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
