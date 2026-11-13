"""Extract the national USGS road GeoPackage for indexed spatial queries.

The ZIP remains in raw_data as the source download. The extracted GeoPackage
is also kept there because random queries inside a compressed SQLite file are
too slow for a national grid.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path


FOLDER = Path(__file__).resolve().parent / "raw_data" / "USGS_Transportation"
ARCHIVE = FOLDER / "Transportation_National_GPKG.zip"
OUTPUT = FOLDER / "Transportation_National_GPKG.gpkg"
PART = FOLDER / "Transportation_National_GPKG.gpkg.part"


def main() -> None:
    with zipfile.ZipFile(ARCHIVE) as package:
        members = [item for item in package.infolist() if item.filename.lower().endswith(".gpkg")]
        if len(members) != 1 or Path(members[0].filename).name != OUTPUT.name:
            raise ValueError(f"Expected one {OUTPUT.name} in {ARCHIVE}")
        member = members[0]
        if OUTPUT.is_file() and OUTPUT.stat().st_size != member.file_size:
            raise ValueError(f"Existing GeoPackage has the wrong size: {OUTPUT}")
        if not OUTPUT.is_file():
            if PART.exists():
                PART.unlink()
            available = shutil.disk_usage(FOLDER).free
            if available < member.file_size + 4 * 1024**3:
                raise OSError("Need GeoPackage size plus 4 GiB free before extraction")
            copied = 0
            with package.open(member) as source, PART.open("wb") as target:
                while block := source.read(4 * 1024 * 1024):
                    target.write(block)
                    copied += len(block)
                    if copied // (1024**3) != (copied - len(block)) // (1024**3):
                        print(f"Extracted {copied / 1024**3:.1f} / {member.file_size / 1024**3:.1f} GiB", flush=True)
            if PART.stat().st_size != member.file_size:
                raise IOError("Incomplete GeoPackage extraction")
            os.replace(PART, OUTPUT)
    digest = hashlib.sha256()
    with OUTPUT.open("rb") as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(block)
    record = {
        "file": str(OUTPUT.relative_to(FOLDER.parent)),
        "source_archive": str(ARCHIVE.relative_to(FOLDER.parent)),
        "bytes": OUTPUT.stat().st_size,
        "sha256": digest.hexdigest(),
        "zip_crc32": f"{member.CRC:08x}",
    }
    OUTPUT.with_suffix(".gpkg.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
