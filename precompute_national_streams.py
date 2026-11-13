"""Cache NHD stream distances for the national 2-mile grid."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import build_wildfire_dataset as wildfire


HERE = Path(__file__).resolve().parent
RAW = HERE / "raw_data"
FOLDER = RAW / "USGS_Hydrography"
CACHE = FOLDER / "stream_distance_2mi_50mi.parquet"
RECORD = FOLDER / "stream_distance_2mi_50mi.json"
GRID_CELLS = HERE / "wildfire_dataset" / "_cells.parquet"
PARTS = FOLDER / "stream_cache_parts"


def compute_part(index: int, start: int, end: int, cells_path: Path,
                 fingerprint: str) -> str:
    cells = pd.read_parquet(cells_path).iloc[start:end].copy()
    output = PARTS / f"part_{index}.parquet"
    record_path = PARTS / f"part_{index}.json"
    if output.is_file() and record_path.is_file() and json.loads(record_path.read_text(encoding="utf-8")).get("source_fingerprint") == fingerprint:
        existing = pd.read_parquet(output, columns=["cell_id"])
        if np.array_equal(existing.cell_id.to_numpy(), cells.cell_id.to_numpy()):
            return f"Part {index} already complete: {len(cells):,} cells"
    distances = wildfire.distances_to_state_streams(RAW, cells,
                                                    50 * 1609.344, 64, None)
    frame = pd.DataFrame({"cell_id": cells.cell_id,
                          "stream_distance_m": distances})
    temporary = output.with_suffix(".parquet.part")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, output)
    record_path.write_text(json.dumps({"source_fingerprint": fingerprint,
                                       "cells": len(cells)}) + "\n", encoding="utf-8")
    return f"Part {index} complete: {len(cells):,} cells"


def main() -> None:
    if CACHE.is_file() and RECORD.is_file():
        print(f"Already complete: {CACHE}", flush=True)
        return
    wildfire.nhd_state_archives(RAW)
    PARTS.mkdir(parents=True, exist_ok=True)
    if GRID_CELLS.is_file():
        cells_path = GRID_CELLS
        cells = pd.read_parquet(cells_path)
    else:
        cells, _, _ = wildfire.make_grid(RAW, None)
        cells_path = PARTS / "grid_cells.parquet"
        cells.to_parquet(cells_path, index=False, compression="zstd")
    if cells.cell_id.duplicated().any():
        raise ValueError("National grid contains duplicate cell IDs")
    fingerprint = wildfire.nhd_state_fingerprint(RAW)
    edges = np.linspace(0, len(cells), 5, dtype=int)
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(compute_part, index, int(edges[index]),
                               int(edges[index + 1]), cells_path, fingerprint)
                   for index in range(4)]
        for future in as_completed(futures):
            print(future.result(), flush=True)
    frame = pd.concat([pd.read_parquet(PARTS / f"part_{index}.parquet")
                       for index in range(4)], ignore_index=True)
    if not np.array_equal(frame.cell_id.to_numpy(), cells.cell_id.to_numpy()):
        raise ValueError("Stream cache cell IDs differ from the national grid")
    distances = frame.stream_distance_m.to_numpy()
    finite_count = int(np.isfinite(distances).sum())
    if finite_count == 0:
        raise ValueError("The USGS state stream sources returned no distances")
    part = CACHE.with_suffix(".parquet.part")
    frame.to_parquet(part, index=False, compression="zstd")
    os.replace(part, CACHE)
    record = {"source_fingerprint": fingerprint,
              "source_catalog": "USGS_Hydrography/nhd_conus_state_archives.json",
              "grid_crs": wildfire.CRS, "cell_width_m": wildfire.CELL_M,
              "search_miles": 50, "cells": len(cells),
              "cells_with_stream_distance": finite_count}
    RECORD.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
