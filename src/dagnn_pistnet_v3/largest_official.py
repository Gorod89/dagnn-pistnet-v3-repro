from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def aggregate_official_15min(frame: Any) -> Any:
    """Apply the official LargeST 2019 preprocessing operation verbatim.

    The official notebook uses ``resample('15T').mean().round(0)`` followed by
    ``fillna(0)``. ``15min`` is pandas' non-deprecated spelling of the same
    fixed-frequency rule.
    """

    import pandas as pd

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("official LargeST input must have a pandas DatetimeIndex")
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError("official LargeST timestamps must be sorted and unique")
    return frame.resample("15min").mean().round(0).fillna(0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_official_largest_2019(
    source: str | Path,
    output: str | Path,
    manifest_path: str | Path | None = None,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Create the isolated 2019/15-minute CA file used by the official protocol."""

    import pandas as pd

    source_path = Path(source)
    output_path = Path(output)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if output_path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")

    try:
        raw = pd.read_hdf(source_path)
    except ImportError as error:
        raise RuntimeError(
            "preparing the official pandas HDF requires the PyTables 'tables' package"
        ) from error
    if not isinstance(raw.index, pd.DatetimeIndex):
        raise ValueError("official LargeST raw file must have a pandas DatetimeIndex")
    years = sorted(set(int(year) for year in raw.index.year))
    if years != [2019]:
        raise ValueError(f"expected only calendar year 2019, found {years}")

    prepared = aggregate_official_15min(raw)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        prepared.to_hdf(temporary, key="t", mode="w")
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    manifest = {
        "schema": "largest-official-2019-preprocessing-v1",
        "source": str(source_path),
        "source_sha256": _sha256(source_path),
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
        "source_shape": list(raw.shape),
        "output_shape": list(prepared.shape),
        "source_years": years,
        "source_first_timestamp": raw.index[0].isoformat(),
        "source_last_timestamp": raw.index[-1].isoformat(),
        "output_first_timestamp": prepared.index[0].isoformat(),
        "output_last_timestamp": prepared.index[-1].isoformat(),
        "official_formula": "resample('15T').mean().round(0).fillna(0)",
    }
    target_manifest = (
        Path(manifest_path)
        if manifest_path is not None
        else output_path.with_suffix(output_path.suffix + ".manifest.json")
    )
    target_manifest.parent.mkdir(parents=True, exist_ok=True)
    target_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the official LargeST CA 2019/15-minute benchmark file"
    )
    parser.add_argument(
        "--source", default="data/LargeST/ca_his_raw_2019.h5", help="official raw HDF5"
    )
    parser.add_argument(
        "--output",
        default="data/LargeST/official_2019_15min/ca_his_2019.h5",
        help="isolated processed HDF5",
    )
    parser.add_argument("--manifest")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = prepare_official_largest_2019(
        args.source, args.output, args.manifest, force=args.force
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
