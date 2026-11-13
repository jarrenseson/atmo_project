"""Build a monthly, 2-mile CONUS wildfire feature table from ``raw_data``.

Each output row is one equal-area grid cell and one calendar month. The target
is 1 when FPA-FOD records at least one ignition in that cell and month, else 0.
Weather and vegetation features are from before the target month. Monthly
Parquet parts support resume; the final table is one Parquet file.

Example (a small first run)::

    python build_wildfire_dataset.py --start 2024-07 --end 2024-07 --bbox -106 39 -105 40 --output wildfire_dataset_pilot

National run (2023-2024)::

    python build_wildfire_dataset.py --start 2023-01 --end 2024-12 --output wildfire_dataset --resume

Recreate only the single file from completed monthly parts::

    python build_wildfire_dataset.py --combine-only

Run ``python build_wildfire_dataset.py --help`` for all options. Complete ZIPs
and NetCDFs are required; files ending in .part are never used.

Install Python 3.11+ and the required packages: numpy, pandas, pyarrow,
rasterio, netCDF4, h5py, pyproj, shapely, pyogrio, and geopandas. The source
files must be placed in the local ``raw_data`` folder before a full rebuild.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import sys
import warnings
import zipfile
from contextlib import closing
from pathlib import Path

# Project packages are installed locally without changing the user's system
# Python. A normal virtual environment also works.
LOCAL_DEPS = Path(__file__).resolve().parent / ".deps"
if LOCAL_DEPS.is_dir():
    sys.path.insert(0, str(LOCAL_DEPS))

import h5py
import netCDF4
import numpy as np
import pandas as pd
import pyogrio
import pyarrow.parquet as pq
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds, transform as window_transform
from shapely import STRtree, points


# Proximity uses horizontal coordinates; USGS measure (M) values are unused.
warnings.filterwarnings("ignore", message=r"Measured \(M\) geometry types are not supported")


HERE = Path(__file__).resolve().parent
RAW_DEFAULT = HERE / "raw_data"
METADATA_KEY = b"wildfire_metadata_json"
CRS = "EPSG:5070"  # CONUS Albers; metres and approximately equal area.
CELL_M = 2 * 1609.344
MODIS_CRS = "+proj=sinu +R=6371007.181 +nadgrids=@null +wktext"
GRID = "/MODIS_Grid_16DAY_250m_500m_VI/Data_Fields/"
MODIS_NAME = re.compile(r"MOD13Q1\.A(\d{4})(\d{3})\.(h\d{2}v\d{2})\.")
NLCD_NAME = "Annual_NLCD_LndCov_{year}_CU_C1V2.zip"
CLIMATE = {"tmmx": "tmax_c", "tmmn": "tmin_c", "pr": "precip_mm", "vs": "wind_m_s"}


def months(first: dt.date, last: dt.date):
    current = first
    while current <= last:
        yield current
        current = next_month(current)


def next_month(day: dt.date) -> dt.date:
    return dt.date(day.year + (day.month == 12), day.month % 12 + 1, 1)


def previous_month(day: dt.date) -> dt.date:
    return dt.date(day.year - (day.month == 1), (day.month - 2) % 12 + 1, 1)


def parse_month(value: str) -> dt.date:
    try:
        result = dt.date.fromisoformat(value + "-01")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use YYYY-MM") from exc
    return result


def zip_member_path(archive: Path, suffix: str) -> tuple[str, zipfile.ZipInfo]:
    with zipfile.ZipFile(archive) as package:
        items = [item for item in package.infolist() if item.filename.lower().endswith(suffix.lower())]
    if len(items) != 1:
        raise ValueError(f"Expected one {suffix} in {archive}, found {len(items)}")
    member = items[0]
    return f"/vsizip/{archive.resolve().as_posix()}/{member.filename}", member


def nlcd_source(raw: Path, year: int) -> str:
    archive = raw / "NLCD" / NLCD_NAME.format(year=year)
    if not archive.is_file():
        raise FileNotFoundError(f"Missing complete land-cover ZIP: {archive}")
    return zip_member_path(archive, ".tif")[0]


def bounds_for_grid(nlcd_path: str, bbox: tuple[float, ...] | None):
    with rasterio.open(nlcd_path) as source:
        if bbox:
            left, bottom, right, top = transform_bounds("EPSG:4326", CRS, *bbox, densify_pts=21)
        else:
            left, bottom, right, top = transform_bounds(source.crs, CRS, *source.bounds, densify_pts=21)
    c0, c1 = math.floor(left / CELL_M), math.ceil(right / CELL_M)
    r0, r1 = math.floor(bottom / CELL_M), math.ceil(top / CELL_M)
    width, height = c1 - c0, r1 - r0
    if width <= 0 or height <= 0:
        raise ValueError("The selected bounding box has no grid cells")
    transform = from_origin(c0 * CELL_M, r1 * CELL_M, CELL_M, CELL_M)
    return c0, r1, width, height, transform


def warp_band(source, shape, transform, method: Resampling, nodata=np.nan):
    output = np.full(shape, nodata, dtype=np.float32)
    reproject(
        source=rasterio.band(source, 1), destination=output,
        src_transform=source.transform, src_crs=source.crs,
        src_nodata=source.nodata, dst_transform=transform, dst_crs=CRS,
        dst_nodata=nodata, resampling=method,
        num_threads=min(4, os.cpu_count() or 1),
    )
    return output


def make_grid(raw: Path, bbox: tuple[float, ...] | None):
    # A current CONUS NLCD mask excludes ocean and land outside its footprint.
    source_path = nlcd_source(raw, 2025)
    c0, r1, width, height, transform = bounds_for_grid(source_path, bbox)
    with rasterio.open(source_path) as source:
        mask_layer = warp_band(source, (height, width), transform, Resampling.nearest)
    valid = np.isfinite(mask_layer) & (mask_layer >= 11) & (mask_layer <= 95)
    rr, cc = np.nonzero(valid)
    cols = c0 + cc.astype(np.int32)
    rows = r1 - rr.astype(np.int32) - 1
    xx = (cols.astype(np.float64) + 0.5) * CELL_M
    yy = (rows.astype(np.float64) + 0.5) * CELL_M
    lon, lat = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform(xx, yy)
    if bbox:
        within = ((lon >= bbox[0]) & (lon <= bbox[2])
                  & (lat >= bbox[1]) & (lat <= bbox[3]))
        valid[rr[~within], cc[~within]] = False
        cols, rows, xx, yy, lon, lat = (array[within] for array in (cols, rows, xx, yy, lon, lat))
    cell_id = pack_cell(cols, rows)
    frame = pd.DataFrame({
        "cell_id": cell_id, "grid_col": cols, "grid_row": rows,
        "center_x_m": xx, "center_y_m": yy,
        "center_lon": lon, "center_lat": lat,
    })
    return frame, valid, transform


def pack_cell(cols, rows):
    col = np.asarray(cols, dtype=np.int64)
    row = np.asarray(rows, dtype=np.int64)
    return (col << 32) | (row & np.int64(0xFFFFFFFF))


def sample_topography(raw: Path, shape, transform, valid) -> dict[str, np.ndarray]:
    result = {}
    base = raw / "3DEP"
    for name, folder, prefix in (
        ("elevation_m", base, "3dep_elevation_conus_*_approx1to3km.tif"),
        ("slope_deg", base / "derived", "3dep_slope_conus_*_approx1to3km.tif"),
        ("aspect_deg", base / "derived", "3dep_aspect_conus_*_approx1to3km.tif"),
        ("curvature_per_m", base / "derived", "3dep_curvature_conus_*_approx1to3km.tif"),
    ):
        paths = sorted(folder.glob(prefix))
        if len(paths) != 4:
            raise FileNotFoundError(f"Expected four CONUS {name} TIFFs in {folder}, found {len(paths)}")
        combined = np.full(shape, np.nan, dtype=np.float32)
        for path in paths:
            with rasterio.open(path) as source:
                method = Resampling.nearest if name == "aspect_deg" else Resampling.bilinear
                image = warp_band(source, shape, transform, method)
                if source.nodata is None:
                    image[np.isclose(image, -9999)] = np.nan
                fill = np.isfinite(image)
                combined[fill] = image[fill]
        result[name] = combined[valid]
    return result


def land_cover(raw: Path, year: int, shape, transform, valid) -> np.ndarray:
    with rasterio.open(nlcd_source(raw, year)) as source:
        values = warp_band(source, shape, transform, Resampling.mode)
    values = values[valid]
    values[~np.isfinite(values)] = 0
    return values.astype(np.uint8)


def vector_path(archive: Path) -> str:
    if not archive.is_file():
        raise FileNotFoundError(f"Missing complete vector ZIP: {archive}")
    with zipfile.ZipFile(archive) as package:
        names = package.namelist()
    gpkg = [name for name in names if name.lower().endswith(".gpkg")]
    if len(gpkg) == 1:
        return f"/vsizip/{archive.resolve().as_posix()}/{gpkg[0]}"
    gdb_roots = {name[:name.lower().index(".gdb") + 4] for name in names if ".gdb/" in name.lower()}
    if len(gdb_roots) == 1:
        return f"/vsizip/{archive.resolve().as_posix()}/{gdb_roots.pop()}"
    raise ValueError(f"Could not find one GPKG or GDB in {archive}")


def open_vector_source(path: Path) -> str:
    if path.suffix.lower() == ".zip":
        return vector_path(path)
    if path.exists():
        return str(path.resolve())
    raise FileNotFoundError(f"Vector source not found: {path}")


def select_layer(path: str, requested: str | None, candidates: tuple[str, ...]) -> str:
    layers = [str(row[0]) for row in pyogrio.list_layers(path)]
    if requested:
        if requested not in layers:
            raise ValueError(f"Layer {requested!r} not found. Available: {layers}")
        return requested
    for candidate in candidates:
        matches = [layer for layer in layers if layer.lower() == candidate.lower()]
        if len(matches) == 1:
            return matches[0]
    raise ValueError(f"Could not select a layer automatically. Available: {layers}. Use --road-layer or --stream-layer")


def source_bbox(bounds, source_crs: str):
    return transform_bounds(CRS, source_crs, *bounds, densify_pts=21)


def update_tile_distances(path: str, layer: str, source_crs: str,
                          ftype_field: str | None, bounds, indices: np.ndarray,
                          x: np.ndarray, y: np.ndarray, out: np.ndarray,
                          search_m: float) -> None:
    kwargs = {"layer": layer, "bbox": source_bbox(bounds, source_crs)}
    if ftype_field:
        kwargs["columns"] = [ftype_field]
    vector = pyogrio.read_dataframe(path, **kwargs)
    if ftype_field and not vector.empty:
        vector = vector[pd.to_numeric(vector[ftype_field], errors="coerce") == 460]
    if vector.empty:
        return
    vector = vector.to_crs(CRS)
    geometries = vector.geometry.to_numpy()
    geometries = geometries[~pd.isna(geometries)]
    if not len(geometries):
        return
    tree = STRtree(geometries)
    point_array = points(x[indices], y[indices])
    pairs, metres = tree.query_nearest(point_array, max_distance=search_m,
                                      return_distance=True)
    positions = indices[pairs[0]]
    candidates = metres.astype(np.float32)
    current = out[positions]
    out[positions] = np.where(np.isfinite(current), np.minimum(current, candidates),
                              candidates)


def distances_to_lines(path: str, layer: str, cells: pd.DataFrame,
                       search_m: float, tile_cells: int, stream: bool) -> np.ndarray:
    """Get exact center-to-line distances within a bounded search radius."""
    info = pyogrio.read_info(path, layer=layer)
    source_crs = info["crs"]
    ftype_field = None
    if stream:
        ftype_field = next((field for field in info["fields"] if field.lower() == "ftype"), None)
        if ftype_field is None:
            raise ValueError(f"{layer} has no FType field for stream selection")
    out = np.full(len(cells), np.nan, dtype=np.float32)
    col = cells["grid_col"].to_numpy()
    row = cells["grid_row"].to_numpy()
    x = cells["center_x_m"].to_numpy()
    y = cells["center_y_m"].to_numpy()
    tx = np.floor_divide(col, tile_cells)
    ty = np.floor_divide(row, tile_cells)
    tiles = pd.DataFrame({"tx": tx, "ty": ty}).groupby(["tx", "ty"], sort=True).indices
    near_m = min(search_m, 10 * 1609.344)
    for number, ((tile_x, tile_y), indices) in enumerate(tiles.items(), 1):
        indices = np.asarray(indices)
        left, right = x[indices].min() - near_m, x[indices].max() + near_m
        bottom, top = y[indices].min() - near_m, y[indices].max() + near_m
        update_tile_distances(path, layer, source_crs, ftype_field,
                              (left, bottom, right, top), indices, x, y, out,
                              near_m)
        unresolved = indices[~np.isfinite(out[indices])]
        fallback_groups = 0
        if near_m < search_m and len(unresolved):
            small = pd.DataFrame({"tx": np.floor_divide(col[unresolved], 16),
                                  "ty": np.floor_divide(row[unresolved], 16)})
            for local in small.groupby(["tx", "ty"], sort=True).indices.values():
                group = unresolved[np.asarray(local)]
                bounds = (x[group].min() - search_m, y[group].min() - search_m,
                          x[group].max() + search_m, y[group].max() + search_m)
                update_tile_distances(path, layer, source_crs, ftype_field,
                                      bounds, group, x, y, out, search_m)
                fallback_groups += 1
        print(f"{layer}: {number}/{len(tiles)} spatial tiles, "
              f"{len(unresolved)} distant cells in {fallback_groups} groups", flush=True)
    return out


def nhd_state_archives(raw: Path) -> list[Path]:
    folder = raw / "USGS_Hydrography"
    manifest = folder / "nhd_conus_state_archives.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing NHD state archive catalog: {manifest}")
    items = json.loads(manifest.read_text(encoding="utf-8"))["archives"]
    if len(items) != 49:
        raise ValueError(f"Expected 49 CONUS/DC NHD state archives, found {len(items)}")
    paths = [folder / Path(item["url"]).name for item in items]
    missing = [path.name for path, item in zip(paths, items)
               if not path.is_file() or path.stat().st_size != item["bytes"]]
    if missing:
        raise FileNotFoundError(f"Missing complete NHD state ZIPs: {', '.join(missing[:10])}"
                                + (f" ... and {len(missing) - 10} more" if len(missing) > 10 else ""))
    return paths


def nhd_state_fingerprint(raw: Path) -> str:
    digest = hashlib.sha256()
    for path in nhd_state_archives(raw):
        record_path = path.with_suffix(".zip.json")
        if not record_path.is_file():
            raise FileNotFoundError(f"Missing NHD source hash: {record_path}")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        digest.update(path.name.encode("utf-8"))
        digest.update(record["sha256"].encode("ascii"))
    return digest.hexdigest()


def distances_to_state_streams(raw: Path, cells: pd.DataFrame, search_m: float,
                               tile_cells: int, requested_layer: str | None) -> np.ndarray:
    sources = []
    for archive in nhd_state_archives(raw):
        path = vector_path(archive)
        layer = select_layer(path, requested_layer, ("NHDFlowline",))
        info = pyogrio.read_info(path, layer=layer)
        ftype_field = next((field for field in info["fields"]
                            if field.lower() == "ftype"), None)
        if ftype_field is None:
            raise ValueError(f"{archive.name}: {layer} has no FType field")
        extent = transform_bounds(info["crs"], CRS, *info["total_bounds"],
                                  densify_pts=21)
        sources.append((path, layer, info["crs"], ftype_field, extent))
    out = np.full(len(cells), np.nan, dtype=np.float32)
    col = cells["grid_col"].to_numpy()
    row = cells["grid_row"].to_numpy()
    x = cells["center_x_m"].to_numpy()
    y = cells["center_y_m"].to_numpy()
    tx = np.floor_divide(col, tile_cells)
    ty = np.floor_divide(row, tile_cells)
    tiles = pd.DataFrame({"tx": tx, "ty": ty}).groupby(["tx", "ty"], sort=True).indices
    near_m = min(search_m, 10 * 1609.344)

    def query_sources(indices: np.ndarray, radius: float) -> int:
        bounds = (x[indices].min() - radius, y[indices].min() - radius,
                  x[indices].max() + radius, y[indices].max() + radius)
        used = 0
        for path, layer, source_crs, ftype_field, extent in sources:
            if (bounds[2] < extent[0] or bounds[0] > extent[2]
                    or bounds[3] < extent[1] or bounds[1] > extent[3]):
                continue
            used += 1
            update_tile_distances(path, layer, source_crs, ftype_field, bounds,
                                  indices, x, y, out, radius)
        return used

    for number, (_, indices) in enumerate(tiles.items(), 1):
        indices = np.asarray(indices)
        used = query_sources(indices, near_m)
        unresolved = indices[~np.isfinite(out[indices])]
        fallback_groups = 0
        if near_m < search_m and len(unresolved):
            small = pd.DataFrame({"tx": np.floor_divide(col[unresolved], 16),
                                  "ty": np.floor_divide(row[unresolved], 16)})
            for local in small.groupby(["tx", "ty"], sort=True).indices.values():
                query_sources(unresolved[np.asarray(local)], search_m)
                fallback_groups += 1
        print(f"NHDFlowline: {number}/{len(tiles)} spatial tiles, "
              f"{used} state archives, {len(unresolved)} distant cells in "
              f"{fallback_groups} groups", flush=True)
    return out


def cached_national_road_distances(raw: Path, cells: pd.DataFrame,
                                   search_miles: float, road_source: Path | None):
    if road_source is not None or search_miles != 50:
        return None
    folder = raw / "USGS_Transportation"
    cache = folder / "road_distance_2mi_50mi.parquet"
    record_file = folder / "road_distance_2mi_50mi.json"
    source_file = folder / "Transportation_National_GPKG.gpkg.json"
    if not all(path.is_file() for path in (cache, record_file, source_file)):
        return None
    record = json.loads(record_file.read_text(encoding="utf-8"))
    source = json.loads(source_file.read_text(encoding="utf-8"))
    if (record.get("source_sha256") != source.get("sha256")
            or record.get("grid_crs") != CRS
            or record.get("cell_width_m") != CELL_M
            or record.get("search_miles") != search_miles):
        return None
    cached = pd.read_parquet(cache, columns=["cell_id", "road_distance_m"])
    if not np.array_equal(cached.cell_id.to_numpy(), cells.cell_id.to_numpy()):
        return None
    print(f"Using verified road-distance cache for {len(cells):,} cells", flush=True)
    return cached.road_distance_m.to_numpy(dtype=np.float32)


def cached_national_stream_distances(raw: Path, cells: pd.DataFrame,
                                     search_miles: float, stream_source: Path | None):
    if stream_source is not None or search_miles != 50:
        return None
    folder = raw / "USGS_Hydrography"
    cache = folder / "stream_distance_2mi_50mi.parquet"
    record_file = folder / "stream_distance_2mi_50mi.json"
    if not cache.is_file() or not record_file.is_file():
        return None
    record = json.loads(record_file.read_text(encoding="utf-8"))
    if (record.get("source_fingerprint") != nhd_state_fingerprint(raw)
            or record.get("grid_crs") != CRS
            or record.get("cell_width_m") != CELL_M
            or record.get("search_miles") != search_miles):
        return None
    cached = pd.read_parquet(cache, columns=["cell_id", "stream_distance_m"])
    if not np.array_equal(cached.cell_id.to_numpy(), cells.cell_id.to_numpy()):
        return None
    print(f"Using verified stream-distance cache for {len(cells):,} cells", flush=True)
    return cached.stream_distance_m.to_numpy(dtype=np.float32)


def static_cells(raw: Path, output: Path, bbox, search_miles: float, tile_cells: int,
                 road_layer: str | None, stream_layer: str | None, skip_proximity: bool,
                 road_source: Path | None, stream_source: Path | None):
    cells, valid, transform = make_grid(raw, bbox)
    shape = valid.shape
    if cells.empty:
        raise ValueError("No CONUS land-cover cells in the selected area")
    for name, values in sample_topography(raw, shape, transform, valid).items():
        cells[name] = values
    aspect_rad = np.deg2rad(cells["aspect_deg"].to_numpy())
    cells["aspect_sin"] = np.sin(aspect_rad).astype(np.float32)
    cells["aspect_cos"] = np.cos(aspect_rad).astype(np.float32)
    if skip_proximity:
        cells["road_distance_m"] = np.nan
        cells["stream_distance_m"] = np.nan
    else:
        road_zip = road_source or raw / "USGS_Transportation" / "Transportation_National_GPKG.gpkg"
        cached_roads = cached_national_road_distances(raw, cells, search_miles,
                                                       road_source) if bbox is None and road_layer is None else None
        if cached_roads is not None:
            cells["road_distance_m"] = cached_roads
        else:
            path = open_vector_source(road_zip)
            layer = select_layer(path, road_layer, ("Trans_RoadSegment", "RoadSegment", "Road"))
            cells["road_distance_m"] = distances_to_lines(path, layer, cells,
                                                           search_miles * 1609.344,
                                                           tile_cells, False)
        if stream_source:
            path = open_vector_source(stream_source)
            layer = select_layer(path, stream_layer, ("NHDFlowline",))
            cells["stream_distance_m"] = distances_to_lines(path, layer, cells,
                                                             search_miles * 1609.344,
                                                             tile_cells, True)
        else:
            cached_streams = cached_national_stream_distances(
                raw, cells, search_miles, None) if bbox is None and stream_layer is None else None
            if cached_streams is not None:
                cells["stream_distance_m"] = cached_streams
            else:
                cells["stream_distance_m"] = distances_to_state_streams(
                    raw, cells, search_miles * 1609.344, tile_cells, stream_layer)
    output.mkdir(parents=True, exist_ok=True)
    cells.to_parquet(output / "_cells.parquet", index=False, compression="zstd")
    (output / "_static_config.json").write_text(json.dumps({
        "input_root": str(raw.resolve()),
        "bbox": bbox, "search_miles": search_miles, "tile_cells": tile_cells,
        "road_layer": road_layer, "stream_layer": stream_layer,
        "skip_proximity": skip_proximity,
        "road_source": str(road_source.resolve()) if road_source else None,
        "stream_source": str(stream_source.resolve()) if stream_source else None,
    }, indent=2) + "\n", encoding="utf-8")
    return cells, valid, transform


def fire_counts(raw: Path, first: dt.date, last: dt.date):
    archive = raw / "FPA_FOD" / "RDS-2013-0009.7_Data_Format3_GPKG.zip"
    if not archive.is_file():
        raise FileNotFoundError(f"Missing USDA FPA-FOD ZIP: {archive}")
    with zipfile.ZipFile(archive) as package:
        members = [item for item in package.infolist() if item.filename.lower().endswith(".gpkg")]
        if len(members) != 1:
            raise ValueError("FPA-FOD ZIP must contain one GeoPackage")
        member = members[0]
        cached = raw / "FPA_FOD" / Path(member.filename).name
        if not cached.exists() or cached.stat().st_size != member.file_size:
            with package.open(member) as source, cached.open("wb") as target:
                shutil.copyfileobj(source, target, 4 * 1024 * 1024)
    counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    transformer = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    with closing(sqlite3.connect(cached)) as connection:
        tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        fire_table = next((name for name in tables if name.lower() == "fires"), None)
        if not fire_table:
            raise ValueError(f"FPA-FOD Fires table not found. Tables: {tables}")
        columns = {row[1].upper(): row[1] for row in connection.execute(f'PRAGMA table_info("{fire_table}")')}
        required = ("FIRE_YEAR", "DISCOVERY_DOY", "LONGITUDE", "LATITUDE")
        missing = [name for name in required if name not in columns]
        if missing:
            raise ValueError(f"FPA-FOD missing columns: {missing}; found {list(columns)}")
        selected = ", ".join('"' + columns[name] + '"' for name in required)
        sql = (f'SELECT {selected} FROM "{fire_table}" WHERE "{columns["FIRE_YEAR"]}" '
               'BETWEEN ? AND ?')
        chunks = pd.read_sql_query(sql, connection, params=(first.year, last.year), chunksize=200_000)
        for chunk in chunks:
            chunk.columns = required
            for name in required:
                chunk[name] = pd.to_numeric(chunk[name], errors="coerce")
            chunk = chunk.dropna(subset=list(required))
            chunk = chunk[(chunk.DISCOVERY_DOY >= 1) & (chunk.DISCOVERY_DOY <= 366)
                          & chunk.LATITUDE.between(24, 50) & chunk.LONGITUDE.between(-125, -66)]
            if chunk.empty:
                continue
            dates = pd.to_datetime(chunk.FIRE_YEAR.astype(int).astype(str) + "-01-01") + pd.to_timedelta(chunk.DISCOVERY_DOY.astype(int) - 1, unit="D")
            use = ((dates.dt.year.to_numpy() == chunk.FIRE_YEAR.to_numpy())
                   & (dates >= pd.Timestamp(first)) & (dates < pd.Timestamp(next_month(last))))
            chunk = chunk.loc[use]
            dates = dates.loc[use]
            if chunk.empty:
                continue
            xx, yy = transformer.transform(chunk.LONGITUDE.to_numpy(), chunk.LATITUDE.to_numpy())
            ids = pack_cell(np.floor(xx / CELL_M).astype(np.int32), np.floor(yy / CELL_M).astype(np.int32))
            grouped = pd.DataFrame({"month": dates.dt.strftime("%Y-%m").to_numpy(), "cell_id": ids}).value_counts()
            for (month, cell_id), count in grouped.items():
                counts[month][int(cell_id)] += int(count)
    return counts


def gridmet_month(raw: Path, variable: str, month: dt.date):
    path = raw / "gridMET" / f"{variable}_{month.year}.nc"
    if not path.is_file():
        raise FileNotFoundError(f"Missing complete gridMET file: {path}")
    with netCDF4.Dataset(path) as source:
        time_var = source.variables["day"] if "day" in source.variables else source.variables["time"]
        dates = netCDF4.num2date(time_var[:], time_var.units, only_use_cftime_datetimes=False)
        indices = [i for i, day in enumerate(dates) if day.year == month.year and day.month == month.month]
        if not indices:
            raise ValueError(f"No {month:%Y-%m} days in {path}")
        data_vars = [name for name, item in source.variables.items() if item.ndim == 3 and name not in ("crs",)]
        if len(data_vars) != 1:
            raise ValueError(f"Expected one 3-D climate variable in {path}; found {data_vars}")
        field = source.variables[data_vars[0]]
        values = np.ma.filled(field[indices, :, :], np.nan).astype(np.float32)
        units = getattr(field, "units", "")
        count = np.count_nonzero(np.isfinite(values), axis=0)
        total = np.nansum(values, axis=0)
        if variable == "pr":
            summary = total
        else:
            summary = np.full(total.shape, np.nan, dtype=np.float32)
            np.divide(total, count, out=summary, where=count > 0)
        summary[count == 0] = np.nan
        if variable in ("tmmx", "tmmn") and units.lower() in ("k", "kelvin"):
            summary -= 273.15
        lat = source.variables["lat"][:]
        lon = source.variables["lon"][:]
        # gridMET coordinates are cell centers. GDAL rasters start at the north.
        if lat[0] < lat[-1]:
            summary = summary[::-1, :]
        if lon[0] > lon[-1]:
            summary = summary[:, ::-1]
        lat_step = abs(float(lat[1] - lat[0]))
        lon_step = abs(float(lon[1] - lon[0]))
        source_transform = from_origin(float(min(lon)) - lon_step / 2,
                                       float(max(lat)) + lat_step / 2, lon_step, lat_step)
    return summary, source_transform


def climate_features(raw: Path, target: dt.date, shape, transform, valid):
    earlier = previous_month(target)
    result = {}
    for variable, name in CLIMATE.items():
        image, source_transform = gridmet_month(raw, variable, earlier)
        dest = np.full(shape, np.nan, dtype=np.float32)
        reproject(image, dest, src_transform=source_transform, src_crs="EPSG:4326",
                  src_nodata=np.nan, dst_transform=transform, dst_crs=CRS,
                  dst_nodata=np.nan, resampling=Resampling.nearest)
        result[name] = dest[valid]
    return result


def modis_files(raw: Path):
    catalog: dict[str, list[tuple[dt.date, Path]]] = collections.defaultdict(list)
    for path in (raw / "MOD13Q1").rglob("*_NDVI_quality.nc"):
        match = MODIS_NAME.search(path.name)
        if match:
            start = dt.date(int(match[1]), 1, 1) + dt.timedelta(days=int(match[2]) - 1)
            catalog[match[3]].append((start, path))
    return catalog


def modis_bounds(file: h5py.File):
    raw = file.attrs["StructMetadata_0"]
    text = raw.decode("ascii") if isinstance(raw, bytes) else str(raw)
    upper = re.search(r"UpperLeftPointMtrs=\(([-\d.]+),([-\d.]+)\)", text)
    lower = re.search(r"LowerRightMtrs=\(([-\d.]+),([-\d.]+)\)", text)
    if not upper or not lower:
        raise ValueError("MODIS tile has no sinusoidal bounds")
    return tuple(map(float, (*upper.groups(), *lower.groups())))


def ndvi_features(catalog, target: dt.date, shape, transform, valid):
    cutoff = target - dt.timedelta(days=16)  # composite must end before target month
    selected = []
    for variants in catalog.values():
        choices = [(start, path) for start, path in variants if start <= cutoff and (target - start).days <= 48]
        if choices:
            selected.append(max(choices, key=lambda pair: pair[0]))
    if not selected:
        raise FileNotFoundError(f"No complete pre-{target:%Y-%m} MOD13Q1 NDVI subsets found")
    output = np.full(shape, np.nan, dtype=np.float32)
    age_days = np.full(shape, np.nan, dtype=np.float32)
    tile_coverage = np.zeros(shape, dtype=np.uint8)
    full_bounds = rasterio.transform.array_bounds(*shape, transform)
    for start, path in selected:
        with h5py.File(path) as file:
            left, top, right, bottom = modis_bounds(file)
            tile_bounds = transform_bounds(MODIS_CRS, CRS, left, bottom, right, top, densify_pts=21)
            if not rasterio.coords.disjoint_bounds(full_bounds, tile_bounds):
                window = from_bounds(*tile_bounds, transform=transform)
                c0 = max(0, math.floor(window.col_off))
                r0 = max(0, math.floor(window.row_off))
                c1 = min(shape[1], math.ceil(window.col_off + window.width))
                r1 = min(shape[0], math.ceil(window.row_off + window.height))
                if c0 >= c1 or r0 >= r1:
                    continue
                group = file[GRID]
                ndvi = group["_250m_16_days_NDVI"][:]
                reliability = group["_250m_16_days_pixel_reliability"][:]
                source = np.where((ndvi >= -2000) & (ndvi <= 10000)
                                  & (reliability == 0), ndvi.astype(np.float32) / 10000,
                                  np.float32(-9999))
                tile_transform = from_origin(left, top, (right - left) / source.shape[1],
                                             (top - bottom) / source.shape[0])
                subwindow = Window(c0, r0, c1 - c0, r1 - r0)
                coverage_array = np.zeros((r1 - r0, c1 - c0), dtype=np.uint8)
                reproject(np.ones(source.shape, dtype=np.uint8), coverage_array,
                          src_transform=tile_transform, src_crs=MODIS_CRS, src_nodata=0,
                          dst_transform=window_transform(subwindow, transform),
                          dst_crs=CRS, dst_nodata=0, resampling=Resampling.nearest)
                tile_coverage[r0:r1, c0:c1] |= coverage_array
                target_array = np.full((r1 - r0, c1 - c0), np.nan, dtype=np.float32)
                reproject(source, target_array, src_transform=tile_transform,
                          src_crs=MODIS_CRS, src_nodata=-9999,
                          dst_transform=window_transform(subwindow, transform),
                          dst_crs=CRS, dst_nodata=np.nan, resampling=Resampling.average)
                section = output[r0:r1, c0:c1]
                age_section = age_days[r0:r1, c0:c1]
                fill = np.isfinite(target_array)
                section[fill] = target_array[fill]
                age_section[fill] = (target - (start + dt.timedelta(days=16))).days
    return (output[valid], age_days[valid], tile_coverage[valid],
            max(start for start, _ in selected), len(selected))


def check_inputs(raw: Path, first: dt.date, last: dt.date, skip_proximity: bool,
                 road_source: Path | None, stream_source: Path | None):
    missing = []
    for month in months(first, last):
        prior = previous_month(month)
        for variable in CLIMATE:
            path = raw / "gridMET" / f"{variable}_{prior.year}.nc"
            if not path.is_file():
                missing.append(str(path))
        cover = raw / "NLCD" / NLCD_NAME.format(year=month.year - 1)
        if not cover.is_file():
            missing.append(str(cover))
    for archive in (
        raw / "FPA_FOD" / "RDS-2013-0009.7_Data_Format3_GPKG.zip",
        raw / "NLCD" / NLCD_NAME.format(year=2025),
    ):
        if not archive.is_file():
            missing.append(str(archive))
    if not skip_proximity:
        for archive in (
            road_source or raw / "USGS_Transportation" / "Transportation_National_GPKG.gpkg",
            *([stream_source] if stream_source else []),
        ):
            if not archive.exists():
                missing.append(str(archive))
    if missing:
        unique = list(dict.fromkeys(missing))
        noun = "file" if len(unique) == 1 else "files"
        raise FileNotFoundError(f"Missing {len(unique)} complete input {noun}:\n" + "\n".join(unique[:30])
                                + (f"\n... and {len(unique) - 30} more" if len(unique) > 30 else ""))
    if not skip_proximity and not stream_source:
        nhd_state_archives(raw)


def combine_partitions(output: Path, single_file: Path) -> None:
    """Write completed monthly parts to one checked, atomic Parquet file."""
    parts = sorted(output.glob("year=*/month_num=*/data.parquet"))
    if not parts:
        raise FileNotFoundError(f"No monthly Parquet parts in {output}")
    metadata_path = output / "_dataset_metadata.json"
    cells_path = output / "_cells.parquet"
    if not metadata_path.is_file() or not cells_path.is_file():
        raise FileNotFoundError(f"Missing dataset metadata or cell grid in {output}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    part_months = [f"{part.parent.parent.name[5:]}-{part.parent.name[10:]}" for part in parts]
    if metadata.get("months_present") != part_months:
        raise ValueError("Monthly parts differ from the dataset metadata")
    if single_file.is_dir() or single_file in parts:
        raise ValueError("The single-file path must be outside the monthly parts")
    cells = pq.read_metadata(cells_path).num_rows
    single_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = single_file.with_name(single_file.name + ".part")
    source_schema = None
    file_schema = None
    total_rows = 0
    try:
        writer = None
        try:
            for month, part in zip(part_months, parts):
                source = pq.ParquetFile(part)
                if source.metadata.num_rows != cells:
                    raise ValueError(f"Incomplete monthly part: {part}")
                table = source.read()
                if writer is None:
                    source_schema = table.schema
                    file_metadata = dict(source_schema.metadata or {})
                    file_metadata[METADATA_KEY] = json.dumps(metadata).encode("utf-8")
                    file_schema = source_schema.with_metadata(file_metadata)
                    writer = pq.ParquetWriter(temporary, file_schema,
                                              compression="zstd", write_statistics=True)
                elif not table.schema.equals(source_schema, check_metadata=False):
                    raise ValueError(f"Monthly schema differs: {part}")
                writer.write_table(table.replace_schema_metadata(file_schema.metadata),
                                   row_group_size=131_072)
                total_rows += len(table)
                print(f"Added {month}: {len(table):,} rows", flush=True)
        finally:
            if writer is not None:
                writer.close()
        if pq.read_metadata(temporary).num_rows != total_rows:
            raise ValueError("The combined Parquet row count is incomplete")
        os.replace(temporary, single_file)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"Single dataset file: {single_file} ({total_rows:,} rows)", flush=True)


def run(args):
    raw = args.raw.resolve()
    output = args.output.resolve()
    if args.end < args.start:
        raise ValueError("--end must be on or after --start")
    if args.end > dt.date(2024, 12, 1):
        raise ValueError("The current FPA-FOD target ends in 2024; later months cannot have valid zero labels")
    if args.start < dt.date(2000, 4, 1):
        raise ValueError("MOD13Q1 started in 2000; use --start 2000-04 or later")
    check_inputs(raw, args.start, args.end, args.skip_proximity,
                 args.road_source, args.stream_source)
    modis = modis_files(raw)
    for month in months(args.start, args.end):
        cutoff = month - dt.timedelta(days=16)
        if not any(start <= cutoff and (month - start).days <= 48
                   for variants in modis.values() for start, _ in variants):
            raise FileNotFoundError(f"No complete pre-{month:%Y-%m} MOD13Q1 NDVI subset found")
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"Output is not empty: {output}; use --resume to keep completed partitions")
    static_file = output / "_cells.parquet"
    if args.resume and static_file.is_file():
        config_file = output / "_static_config.json"
        if not config_file.is_file():
            raise ValueError("Cached cells have no settings record; use a new output folder")
        previous = json.loads(config_file.read_text(encoding="utf-8"))
        expected = {"input_root": str(raw.resolve()),
                    "bbox": args.bbox, "search_miles": args.search_miles,
                    "tile_cells": args.tile_cells, "road_layer": args.road_layer,
                    "stream_layer": args.stream_layer, "skip_proximity": args.skip_proximity,
                    "road_source": str(args.road_source.resolve()) if args.road_source else None,
                    "stream_source": str(args.stream_source.resolve()) if args.stream_source else None}
        if previous != expected:
            raise ValueError("Static settings differ from the cached cells; use a new output folder")
        current_cells, valid, transform = make_grid(raw, args.bbox)
        cells = pd.read_parquet(static_file)
        if not np.array_equal(current_cells.cell_id.to_numpy(), cells.cell_id.to_numpy()):
            raise ValueError("Cached grid differs from this run's grid; use a new output folder")
    else:
        cells, valid, transform = static_cells(raw, output, args.bbox, args.search_miles,
                                               args.tile_cells, args.road_layer,
                                               args.stream_layer, args.skip_proximity,
                                               args.road_source, args.stream_source)
    counts = fire_counts(raw, args.start, args.end)
    shape = valid.shape
    all_ids = cells.cell_id.to_numpy()
    cover_cache = {}
    for month in months(args.start, args.end):
        partition = output / f"year={month.year}" / f"month_num={month.month:02d}"
        target = partition / "data.parquet"
        if target.is_file() and args.resume:
            print(f"Already complete: {month:%Y-%m}", flush=True)
            continue
        year = month.year - 1
        if year not in cover_cache:
            cover_cache[year] = land_cover(raw, year, shape, transform, valid)
        frame = cells.copy()
        frame["month"] = month.strftime("%Y-%m")
        frame["land_cover_previous_year"] = cover_cache[year]
        for name, values in climate_features(raw, month, shape, transform, valid).items():
            frame[name] = values
        ndvi, ndvi_age, tile_coverage, ndvi_start, tiles = ndvi_features(modis, month, shape, transform, valid)
        coverage_fraction = float(np.mean(tile_coverage))
        if coverage_fraction < args.min_ndvi_tile_coverage:
            raise ValueError(f"{month:%Y-%m} MODIS tiles cover {coverage_fraction:.1%} of cells; "
                             f"need {args.min_ndvi_tile_coverage:.1%}. Download missing tiles first.")
        frame["ndvi_prior"] = ndvi
        frame["ndvi_age_days"] = ndvi_age
        fire_map = counts.get(month.strftime("%Y-%m"), {})
        frame["fire_count"] = pd.Series(all_ids).map(fire_map).fillna(0).astype(np.int32).to_numpy()
        frame["wildfire_recorded"] = (frame.fire_count > 0).astype(np.uint8)
        partition.mkdir(parents=True, exist_ok=True)
        temporary = partition / "_data.parquet.part"
        frame.to_parquet(temporary, index=False, compression="zstd")
        temporary.replace(target)
        print(f"{month:%Y-%m}: {len(frame):,} cells, {frame.wildfire_recorded.sum():,} positive cells, "
              f"{np.isfinite(ndvi).mean():.1%} quality-screened NDVI, {tiles} MODIS tiles "
              f"(latest starts {ndvi_start})", flush=True)
    completed_months = sorted(
        f"{part.parent.parent.name[5:]}-{part.parent.name[10:]}"
        for part in output.glob("year=*/month_num=*/data.parquet")
    )
    metadata = {
        "grid_crs": CRS, "cell_width_m": CELL_M,
        "key": ["cell_id", "month"], "target": "wildfire_recorded",
        "zero_means": "No FPA-FOD fire recorded in this cell and month",
        "feature_timing": "Climate is previous calendar month; NDVI is latest completed composite before target month; NLCD is previous year.",
        "aggregation": {
            "grid_mask": "Current (2025) NLCD class at the cell center; classes 11-95 are included.",
            "climate": "Daily gridMET summarized over previous month, then nearest native approximately 4-km cell.",
            "land_cover": "Modal 30-m NLCD class within each 2-mile cell; zero means missing class.",
            "ndvi": "Mean of valid 250-m pixels within each cell; only pixel reliability 0 is used.",
            "terrain": "Elevation, slope, and curvature bilinearly sampled; aspect sampled by nearest pixel.",
        },
        "road_stream_distance": "Metres from cell center to nearest selected line within search radius; null if none found.",
        "min_modis_tile_coverage": args.min_ndvi_tile_coverage,
        "distance_search_miles": args.search_miles, "proximity_skipped": args.skip_proximity,
        "topography": "USGS 3DEP service extracts resampled to approximately 1-3 km; slope, aspect, curvature are derived.",
        "historical_vector_note": "Current road and stream geometry is used for historical months; road changes can affect retrospective results.",
        "fire_location_note": "Some FPA-FOD point locations are only accurate to a one-square-mile section.",
        "input_root": str(raw),
        "output_period": [completed_months[0] + "-01", completed_months[-1] + "-01"],
        "months_present": completed_months,
        "bbox_lonlat": args.bbox,
    }
    (output / "_dataset_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    combine_partitions(output, args.single_file)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=RAW_DEFAULT, help="Folder containing the source subfolders")
    parser.add_argument("--output", type=Path, default=HERE / "wildfire_dataset")
    parser.add_argument("--single-file", type=Path,
                        help="Final Parquet file; defaults to OUTPUT.parquet")
    parser.add_argument("--combine-only", action="store_true",
                        help="Make the single file from existing monthly parts without rereading raw data")
    parser.add_argument("--start", type=parse_month, default=dt.date(2000, 4, 1), help="First target month, YYYY-MM")
    parser.add_argument("--end", type=parse_month, default=dt.date(2024, 12, 1), help="Last target month, YYYY-MM")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH"),
                        help="Optional WGS84 bounds for a smaller run")
    parser.add_argument("--search-miles", type=float, default=50, help="Maximum road and stream search distance")
    parser.add_argument("--tile-cells", type=int, default=64, help="Width of vector query tiles in grid cells")
    parser.add_argument("--min-ndvi-tile-coverage", type=float, default=0.98,
                        help="Required fraction of grid cells covered by downloaded MODIS tiles")
    parser.add_argument("--road-layer", help="Override automatic road layer selection")
    parser.add_argument("--stream-layer", help="Override automatic stream layer selection")
    parser.add_argument("--road-source", type=Path, help="Optional extracted road GPKG/GDB or ZIP")
    parser.add_argument("--stream-source", type=Path, help="Optional extracted hydro GPKG/GDB or ZIP")
    parser.add_argument("--skip-proximity", action="store_true", help="Pilot only: leave road and stream distances null")
    parser.add_argument("--resume", action="store_true", help="Keep existing complete month partitions")
    args = parser.parse_args()
    args.single_file = (args.single_file or Path(str(args.output) + ".parquet")).resolve()
    if args.bbox and not (args.bbox[0] < args.bbox[2] and args.bbox[1] < args.bbox[3]):
        parser.error("--bbox must be WEST SOUTH EAST NORTH")
    if args.search_miles <= 0 or args.tile_cells <= 0:
        parser.error("--search-miles and --tile-cells must be positive")
    if not 0 < args.min_ndvi_tile_coverage <= 1:
        parser.error("--min-ndvi-tile-coverage must be between 0 and 1")
    try:
        if args.combine_only:
            combine_partitions(args.output.resolve(), args.single_file)
        else:
            run(args)
    except FileNotFoundError as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
