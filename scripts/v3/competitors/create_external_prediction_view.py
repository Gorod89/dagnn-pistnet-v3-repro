"""Create a zero-copy HDF5 view over a canonical raw prediction memmap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    descriptor = json.loads(args.descriptor.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    raw_path = (args.descriptor.parent / descriptor["path"]).resolve()
    dtype = np.dtype(descriptor["dtype"])
    source_shape = tuple(int(value) for value in descriptor["shape"])
    layout = str(descriptor["layout"]).upper()
    expected_shape = tuple(int(value) for value in protocol["prediction_shape"])

    if descriptor.get("format") != "raw_memmap":
        raise ValueError("descriptor must identify a raw_memmap prediction")
    if dtype != np.dtype("float32"):
        raise ValueError(f"expected float32 predictions, got {dtype}")
    if layout == "SHN":
        resolved_shape = (*source_shape, 1)
    elif layout == "SHNF":
        resolved_shape = source_shape
    else:
        raise ValueError(f"unsupported contiguous raw layout: {layout}")
    if resolved_shape != expected_shape:
        raise ValueError(
            f"descriptor shape/layout resolves to {resolved_shape}, expected {expected_shape}"
        )
    expected_bytes = int(np.prod(source_shape, dtype=np.int64)) * dtype.itemsize
    if raw_path.stat().st_size != expected_bytes:
        raise ValueError(
            f"raw prediction has {raw_path.stat().st_size} bytes, expected {expected_bytes}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    with h5py.File(temporary, "w") as handle:
        handle.attrs["schema"] = "dagnn-pistnet-predictions-v1"
        handle.attrs["protocol_sha256"] = protocol["protocol_sha256"]
        handle.attrs["model_name"] = args.model
        handle.attrs["seed"] = args.seed
        handle.attrs["zero_copy_external_storage"] = True
        handle.create_dataset(
            "prediction",
            shape=expected_shape,
            dtype="float32",
            external=[(raw_path.as_posix(), 0, expected_bytes)],
        )
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
