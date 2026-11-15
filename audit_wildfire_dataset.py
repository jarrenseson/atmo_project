"""Check the 2-mile wildfire table and write a source-feature audit.

Usage: python audit_wildfire_dataset.py wildfire_dataset.parquet
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import wildfire_dataset_io as dataset_io
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


FEATURES = {
    "gridMET": ["tmax_c", "tmin_c", "precip_mm", "wind_m_s"],
    "Annual NLCD": ["land_cover_previous_year"],
    "MOD13Q1": ["ndvi_prior", "ndvi_age_days"],
    "USGS 3DEP": ["elevation_m"],
    "USGS Transportation": ["road_distance_m"],
    "USGS Hydrography": ["stream_distance_m"],
    "USDA FPA-FOD": ["fire_count", "wildfire_recorded"],
}
REMOVED_TERRAIN_COLUMNS = frozenset({
    "slope_deg", "aspect_deg", "curvature_per_m", "aspect_sin", "aspect_cos",
})


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python audit_wildfire_dataset.py DATASET_PATH")
    dataset = Path(sys.argv[1]).resolve()
    metadata = dataset_io.metadata(dataset)
    if metadata["grid_crs"] != "EPSG:5070" or metadata["cell_width_m"] != 3218.688:
        raise ValueError("Grid is not the expected equal-area 2-mile grid")
    expected_ids = None
    if dataset.is_dir():
        cells = pd.read_parquet(dataset / "_cells.parquet", columns=["cell_id"])
        if cells.cell_id.duplicated().any():
            raise ValueError("Static grid has duplicate cell IDs")
        expected_ids = cells.cell_id.to_numpy()
    found = dataset_io.months(dataset)
    if not found:
        raise FileNotFoundError("No completed wildfire months")
    feature_names = [name for group in FEATURES.values() for name in group]
    nonmissing = {name: 0 for name in feature_names}
    month_reports = []
    complete_cases = total_rows = positive = 0
    for month, path in found:
        expected_month = month.strftime("%Y-%m")
        frame = dataset_io.read_month(path, month)
        stale = REMOVED_TERRAIN_COLUMNS.intersection(frame.columns)
        if stale:
            raise ValueError(f"{path}: removed terrain columns remain: {sorted(stale)}")
        missing = set(feature_names + ["cell_id", "month"]) - set(frame.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        if expected_ids is None:
            expected_ids = frame.cell_id.to_numpy().copy()
            if pd.Series(expected_ids).duplicated().any():
                raise ValueError(f"{path}: duplicate cell IDs")
        if (len(frame) != len(expected_ids)
                or not np.array_equal(frame.cell_id.to_numpy(), expected_ids)
                or not frame.month.eq(expected_month).all()):
            raise ValueError(f"{path}: invalid cell-month key or incomplete grid")
        if not frame.wildfire_recorded.isin((0, 1)).all():
            raise ValueError(f"{path}: target is not binary")
        if not np.array_equal((frame.fire_count.to_numpy() > 0),
                              (frame.wildfire_recorded.to_numpy() == 1)):
            raise ValueError(f"{path}: fire count and target disagree")
        counts = frame[feature_names].notna().sum()
        for name in feature_names:
            nonmissing[name] += int(counts[name])
        valid_cover = frame.land_cover_previous_year.ne(0)
        valid_features = frame[feature_names[:-2]].notna().all(axis=1)
        valid_features &= valid_cover
        monthly_complete = int(valid_features.sum())
        monthly_positive = int(frame.wildfire_recorded.sum())
        total_rows += len(frame)
        positive += monthly_positive
        complete_cases += monthly_complete
        month_reports.append({"month": expected_month, "rows": len(frame),
                              "positive_cells": monthly_positive,
                              "complete_feature_rows": monthly_complete})
        print(f"Checked {expected_month}: {len(frame):,} rows, "
              f"{monthly_complete:,} complete features", flush=True)
    if not metadata.get("proximity_skipped") and (nonmissing["road_distance_m"] == 0
                                                   or nonmissing["stream_distance_m"] == 0):
        raise ValueError("A required national proximity feature is entirely missing")
    if dataset.is_file() and pq.read_metadata(dataset).num_rows != total_rows:
        raise ValueError("The single file has rows outside the listed months")
    report = {"dataset": str(dataset), "grid_crs": metadata["grid_crs"],
              "cell_width_m": metadata["cell_width_m"],
              "key": ["cell_id", "month"], "cells": len(expected_ids),
              "months": len(found), "rows": total_rows,
              "positive_cells": positive, "complete_feature_rows": complete_cases,
              "nonmissing_by_feature": nonmissing, "source_to_feature": FEATURES,
              "per_month": month_reports}
    report_path = dataset.with_suffix(".audit.json") if dataset.is_file() else dataset / "_audit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Audit passed: {len(found)} months, {total_rows:,} rows", flush=True)


if __name__ == "__main__":
    main()
