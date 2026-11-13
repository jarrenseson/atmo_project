"""Cache road distances for the national 2-mile grid.

This optional cache lets the full dataset build work on stream distances while
the road distances, already computed from the national USGS GeoPackage, are
read from a checked cell-id table.
"""

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
FOLDER = RAW / "USGS_Transportation"
SOURCE = FOLDER / "Transportation_National_GPKG.gpkg"
SOURCE_RECORD = FOLDER / "Transportation_National_GPKG.gpkg.json"
CACHE = FOLDER / "road_distance_2mi_50mi.parquet"
RECORD = FOLDER / "road_distance_2mi_50mi.json"
GRID_CELLS = HERE / "wildfire_dataset" / "_cells.parquet"
PARTS = FOLDER / "road_cache_parts"


def compute_part(index: int, start: int, end: int, cells_path: Path,
                 source_sha256: str) -> str:
    cells = pd.read_parquet(cells_path).iloc[start:end].copy()
    output = PARTS / f"part_{index}.parquet"
    record_path = PARTS / f"part_{index}.json"
    if output.is_file() and record_path.is_file() and json.loads(record_path.read_text(encoding="utf-8")).get("source_sha256") == source_sha256:
        existing = pd.read_parquet(output, columns=["cell_id"])
        if np.array_equal(existing.cell_id.to_numpy(), cells.cell_id.to_numpy()):
            return f"Part {index} already complete: {len(cells):,} cells"
    path = wildfire.open_vector_source(SOURCE)
    layer = wildfire.select_layer(path, None, ("Trans_RoadSegment", "RoadSegment", "Road"))
    distances = wildfire.distances_to_lines(path, layer, cells,
                                             50 * 1609.344, 64, False)
    frame = pd.DataFrame({"cell_id": cells.cell_id,
                          "road_distance_m": distances})
    temporary = output.with_suffix(".parquet.part")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, output)
    record_path.write_text(json.dumps({"source_sha256": source_sha256,
                                       "cells": len(cells)}) + "\n", encoding="utf-8")
    return f"Part {index} complete: {len(cells):,} cells"


def main() -> None:
    if CACHE.is_file() and RECORD.is_file():
        print(f"Already complete: {CACHE}", flush=True)
        return
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
    source_record = json.loads(SOURCE_RECORD.read_text(encoding="utf-8"))
    edges = np.linspace(0, len(cells), 5, dtype=int)
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(compute_part, index, int(edges[index]),
                               int(edges[index + 1]), cells_path,
                               source_record["sha256"])
                   for index in range(4)]
        for future in as_completed(futures):
            print(future.result(), flush=True)
    frame = pd.concat([pd.read_parquet(PARTS / f"part_{index}.parquet")
                       for index in range(4)], ignore_index=True)
    if not np.array_equal(frame.cell_id.to_numpy(), cells.cell_id.to_numpy()):
        raise ValueError("Road cache cell IDs differ from the national grid")
    distances = frame.road_distance_m.to_numpy()
    finite_count = int(np.isfinite(distances).sum())
    if finite_count == 0:
        raise ValueError("The national USGS road source returned no distances")
    part = CACHE.with_suffix(".parquet.part")
    frame.to_parquet(part, index=False, compression="zstd")
    os.replace(part, CACHE)
    record = {"source": str(SOURCE.relative_to(RAW)),
              "source_sha256": source_record["sha256"],
              "grid_crs": wildfire.CRS, "cell_width_m": wildfire.CELL_M,
              "search_miles": 50, "cells": len(cells),
              "cells_with_road_distance": finite_count}
    RECORD.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
