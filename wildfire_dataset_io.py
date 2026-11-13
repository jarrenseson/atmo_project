"""Read wildfire months from one Parquet file or monthly Parquet parts."""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

LOCAL_DEPS = Path(__file__).resolve().parent / ".deps"
if LOCAL_DEPS.is_dir():
    sys.path.insert(0, str(LOCAL_DEPS))

import pandas as pd
import pyarrow.parquet as pq


METADATA_KEY = b"wildfire_metadata_json"


def metadata(dataset: Path) -> dict:
    if dataset.is_file():
        value = (pq.ParquetFile(dataset).schema_arrow.metadata or {}).get(METADATA_KEY)
        if value is None:
            raise ValueError(f"No wildfire metadata in {dataset}")
        return json.loads(value)
    return json.loads((dataset / "_dataset_metadata.json").read_text(encoding="utf-8"))


def months(dataset: Path) -> list[tuple[dt.date, Path]]:
    if dataset.is_file():
        names = metadata(dataset).get("months_present", [])
        if not names or len(names) != len(set(names)):
            raise ValueError(f"Missing or repeated month names in {dataset}")
        return [(dt.date.fromisoformat(name + "-01"), dataset) for name in sorted(names)]
    if not dataset.is_dir():
        raise FileNotFoundError(f"No wildfire dataset at {dataset}")
    found = []
    for path in dataset.glob("year=*/month_num=*/data.parquet"):
        year = int(path.parent.parent.name.removeprefix("year="))
        month = int(path.parent.name.removeprefix("month_num="))
        found.append((dt.date(year, month, 1), path))
    return sorted(found)


def read_month(path: Path, month: dt.date, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns,
                           filters=[("month", "==", month.strftime("%Y-%m"))])
