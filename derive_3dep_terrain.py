"""Make slope, aspect, and Laplacian curvature from 3DEP service extracts.

The exports use geographic coordinates. Horizontal distances are corrected by
latitude for each image row. Edges and cells near invalid source data are nodata.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, TiffImagePlugin
from scipy.ndimage import minimum_filter


ROOT = Path(__file__).resolve().parent / "raw_data" / "3DEP"
OUT = ROOT / "derived"
NODATA = np.float32(-9999)


def write_raster(path: Path, values: np.ndarray, source: Image.Image, units: str) -> None:
    tags = TiffImagePlugin.ImageFileDirectory_v2()
    for key, value in source.tag_v2.items():
        tags[key] = value
    tags[42113] = "-9999"
    Image.fromarray(values.astype(np.float32)).save(path, compression="tiff_lzw", tiffinfo=tags)
    check = Image.open(path)
    if check.mode != "F" or check.tag_v2.get(33550) != source.tag_v2.get(33550):
        raise IOError(f"Invalid derived GeoTIFF: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    record = {"file": path.name, "derived_from": source.filename,
              "units": units, "nodata": float(NODATA),
              "bytes": path.stat().st_size, "sha256": digest}
    path.with_suffix(".tif.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for source_path in sorted(ROOT.glob("3dep_elevation_*_approx1to3km.tif")):
        image = Image.open(source_path)
        elevation = np.asarray(image, dtype=np.float64)
        upper = (5000 if "conus" in source_path.name else
                 6500 if "alaska" in source_path.name else
                 4500 if "hawaii" in source_path.name else 3500)
        valid = np.isfinite(elevation) & (elevation > -500) & (elevation < upper)
        z = np.where(valid, elevation, 0.0)

        lon_step, lat_step = image.tag_v2[33550][:2]
        north_lat = image.tag_v2[33922][4]
        lat = np.deg2rad(north_lat - (np.arange(z.shape[0]) + 0.5) * lat_step)
        dx = lon_step * (111412.84 * np.cos(lat) - 93.5 * np.cos(3 * lat)
                         + 0.118 * np.cos(5 * lat))[:, None]
        dy = lat_step * (111132.92 - 559.82 * np.cos(2 * lat)
                         + 1.175 * np.cos(4 * lat))[:, None]

        dzdx = np.gradient(z, axis=1) / dx
        dzdy = -np.gradient(z, axis=0) / dy
        slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
        aspect = (np.degrees(np.arctan2(-dzdx, -dzdy)) + 360) % 360
        curvature = np.gradient(dzdx, axis=1) / dx - np.gradient(dzdy, axis=0) / dy

        mask3 = minimum_filter(valid.astype(np.uint8), size=3, mode="constant") == 1
        mask5 = minimum_filter(valid.astype(np.uint8), size=5, mode="constant") == 1
        aspect_mask = mask3 & (slope >= 1e-6)
        region = source_path.stem.removeprefix("3dep_elevation_")
        outputs = (
            ("slope", slope, mask3, "degrees"),
            ("aspect", aspect, aspect_mask, "degrees clockwise from north, downslope"),
            ("curvature", curvature, mask5, "1/metre; elevation Laplacian"),
        )
        for name, data, mask, units in outputs:
            target = OUT / f"3dep_{name}_{region}.tif"
            write_raster(target, np.where(mask, data, NODATA), image, units)
            print(f"{target.name}: {target.stat().st_size:,} bytes", flush=True)


if __name__ == "__main__":
    main()
