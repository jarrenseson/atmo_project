"""Download the source data for the 2023-2024 wildfire dataset.

Run ``python download_wildfire_sources.py --help`` for source groups and options.
The raw source data needs about 100 GiB of disk space. NASA MODIS subsets
require an Earthdata Login account in ~/.netrc. Downloads resume from .part
files when the server supports byte ranges. Source files stay outside Git.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import time
from pathlib import Path
from urllib.parse import urlencode

import requests


ROOT = Path(__file__).resolve().parent
CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"
MODIS_COLLECTION = "C1748066515-LPCLOUD"
MODIS_FIELDS = (
    "_250m_16_days_NDVI",
    "_250m_16_days_VI_Quality",
    "_250m_16_days_pixel_reliability",
)
MODIS_GRID = "/MODIS_Grid_16DAY_250m_500m_VI/Data_Fields/"
STATES = """Alabama Arizona Arkansas California Colorado Connecticut Delaware
District_of_Columbia Florida Georgia Idaho Illinois Indiana Iowa Kansas Kentucky
Louisiana Maine Maryland Massachusetts Michigan Minnesota Mississippi Missouri
Montana Nebraska Nevada New_Hampshire New_Jersey New_Mexico New_York
North_Carolina North_Dakota Ohio Oklahoma Oregon Pennsylvania Rhode_Island
South_Carolina South_Dakota Tennessee Texas Utah Vermont Virginia Washington
West_Virginia Wisconsin Wyoming""".split()
GROUPS = ("gridmet", "nlcd", "3dep", "transportation", "hydrography", "fpa-fod", "modis")


def record_path(path: Path) -> Path:
    return path.with_name(path.name + ".json")


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_record(raw: Path, path: Path, url: str, extra: dict | None = None) -> dict:
    record = {"file": str(path.relative_to(raw)), "source_url": url,
              "bytes": path.stat().st_size, "sha256": sha256(path)}
    if extra:
        record.update(extra)
    save_json(record_path(path), record)
    return record


def existing_file(path: Path, verify: bool) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    sidecar = record_path(path)
    if not sidecar.is_file():
        return False
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    if path.stat().st_size != record.get("bytes"):
        return False
    return not verify or sha256(path) == record.get("sha256")


def download(session: requests.Session, raw: Path, path: Path, url: str,
             *, verify: bool, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if existing_file(path, verify):
        print(f"Have {path.relative_to(raw)}", flush=True)
        return
    if path.is_file() and path.stat().st_size > 0 and not record_path(path).exists():
        source_record(raw, path, url, extra)
        print(f"Recorded {path.relative_to(raw)}", flush=True)
        return

    part = path.with_name(path.name + ".part")
    for attempt in range(1, 5):
        offset = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with session.get(url, headers=headers, stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "").lower()
                if ("text/html" in content_type or "text/xml" in content_type
                        or "json" in content_type):
                    raise ValueError(f"Received a web page instead of data from {response.url}")
                if offset and response.status_code == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise IOError(f"Unexpected Content-Range: {content_range}")
                    mode = "ab"
                elif offset and response.status_code == 200:
                    mode = "wb"  # The server did not accept the range request.
                else:
                    mode = "wb"
                print(f"Get {path.relative_to(raw)}", flush=True)
                with part.open(mode) as target:
                    for block in response.iter_content(chunk_size=4 * 1024 * 1024):
                        if block:
                            target.write(block)
                expected = response.headers.get("Content-Length")
                if expected and part.stat().st_size != int(expected) + (offset if mode == "ab" else 0):
                    raise IOError(f"Incomplete transfer: {path.name}")
            if part.stat().st_size == 0:
                raise IOError(f"Empty transfer: {path.name}")
            os.replace(part, path)
            source_record(raw, path, url, extra)
            return
        except (requests.RequestException, IOError) as exc:
            if attempt == 4:
                raise RuntimeError(f"Could not download {path.name}: {exc}") from exc
            print(f"Retry {attempt}/4 for {path.name}: {exc}", flush=True)
            time.sleep(min(2 ** attempt, 10))


def static_sources(group: str, raw: Path):
    if group == "gridmet":
        for variable in ("tmmx", "tmmn", "pr", "vs"):
            for year in (2022, 2023, 2024):
                name = f"{variable}_{year}.nc"
                yield raw / "gridMET" / name, f"https://www.northwestknowledge.net/metdata/data/{name}", None
    elif group == "nlcd":
        for year in (2022, 2023, 2025):
            name = f"Annual_NLCD_LndCov_{year}_CU_C1V2.zip"
            yield raw / "NLCD" / name, f"https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/data-bundles/{name}", None
    elif group == "3dep":
        for region, bbox, size in (
            ("ne", "-95,37,-66,50", "2320,1040"),
            ("nw", "-125,37,-95,50", "2400,1040"),
            ("se", "-95,24,-66,37", "2320,1040"),
            ("sw", "-125,24,-95,37", "2400,1040"),
        ):
            name = f"3dep_elevation_conus_{region}_approx1to3km.tif"
            service = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
            parameters = {"bbox": bbox, "bboxSR": 4326, "imageSR": 4326,
                          "size": size, "format": "tiff", "pixelType": "F32",
                          "interpolation": "RSP_BilinearInterpolation", "f": "image"}
            yield raw / "3DEP" / name, f"{service}?{urlencode(parameters)}", {"parameters": parameters}
    elif group == "transportation":
        name = "Transportation_National_GPKG.zip"
        yield raw / "USGS_Transportation" / name, f"https://prd-tnm.s3.amazonaws.com/StagedProducts/Tran/National/GPKG/{name}", None
    elif group == "hydrography":
        for state in STATES:
            name = f"NHD_H_{state}_State_GDB.zip"
            yield raw / "USGS_Hydrography" / name, f"https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHD/State/GDB/{name}", None
    elif group == "fpa-fod":
        name = "RDS-2013-0009.7_Data_Format3_GPKG.zip"
        yield raw / "FPA_FOD" / name, f"https://www.fs.usda.gov/rds/archive/products/RDS-2013-0009.7/{name}", None


def modis_granules(session: requests.Session):
    page = 1
    while True:
        response = session.get(CMR, params={
            "collection_concept_id": MODIS_COLLECTION,
            "bounding_box": "-125,24,-66,50",
            "temporal": "2022-12-03T00:00:00Z,2024-12-31T23:59:59Z",
            "page_size": 2000, "page_num": page,
        }, timeout=(30, 180))
        response.raise_for_status()
        entries = response.json()["feed"]["entry"]
        if not entries:
            break
        for item in entries:
            granule_id = item.get("producer_granule_id") or item.get("title", "").removesuffix(".hdf")
            match = re.fullmatch(r"MOD13Q1\.A(\d{4})(\d{3})\.h\d{2}v\d{2}\.061\.\d+", granule_id)
            if not match:
                continue
            start = dt.date(int(match[1]), 1, 1) + dt.timedelta(days=int(match[2]) - 1)
            if start != dt.date(2022, 12, 3) and start.year not in (2023, 2024):
                continue
            yield granule_id, start, item
        if len(entries) < 2000:
            break
        page += 1


def download_modis(session: requests.Session, raw: Path, verify: bool) -> None:
    fields = ";".join(MODIS_GRID + field for field in MODIS_FIELDS)
    query = urlencode({"dap4.ce": fields})
    found = 0
    for granule_id, start, item in modis_granules(session):
        url = (f"https://opendap.earthdata.nasa.gov/collections/{MODIS_COLLECTION}/"
               f"granules/{granule_id}.dap.nc4?{query}")
        name = f"{granule_id}_NDVI_quality.nc"
        path = raw / "MOD13Q1" / f"{start.year}_NDVI_quality" / start.isoformat() / name
        extra = {"granule_id": granule_id, "time_start": item.get("time_start"),
                 "variables": list(MODIS_FIELDS)}
        download(session, raw, path, url, verify=verify, extra=extra)
        found += 1
    if not found:
        raise RuntimeError("CMR found no matching MOD13Q1 granules")
    print(f"MODIS granules: {found}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "raw_data")
    parser.add_argument("--only", choices=GROUPS, nargs="+", default=GROUPS,
                        help="Download only these source groups")
    parser.add_argument("--verify", action="store_true",
                        help="Recheck SHA-256 hashes of existing files")
    args = parser.parse_args()
    raw = args.raw_dir.resolve()
    session = requests.Session()  # Uses ~/.netrc for Earthdata Login redirects.
    for group in args.only:
        if group == "modis":
            download_modis(session, raw, args.verify)
            continue
        for path, url, extra in static_sources(group, raw):
            download(session, raw, path, url, verify=args.verify, extra=extra)
        if group == "hydrography":
            archives = [{"state": state,
                         "url": f"https://prd-tnm.s3.amazonaws.com/StagedProducts/Hydrography/NHD/State/GDB/NHD_H_{state}_State_GDB.zip",
                         "bytes": (raw / "USGS_Hydrography" / f"NHD_H_{state}_State_GDB.zip").stat().st_size}
                        for state in STATES]
            save_json(raw / "USGS_Hydrography" / "nhd_conus_state_archives.json",
                      {"source_listing": "https://prd-tnm.s3.amazonaws.com/?prefix=StagedProducts/Hydrography/NHD/State/GDB/",
                       "archives": archives})


if __name__ == "__main__":
    main()
