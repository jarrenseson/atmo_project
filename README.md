# US wildfire occurrence by 2-mile grid cell

This project combines climate, land cover, vegetation, terrain, roads, streams,
and recorded wildfire locations. Each output row describes one 2 × 2 mile cell
in the contiguous United States during one month. The key is `(cell_id, month)`.
The target, `wildfire_recorded`, is 1 when FPA-FOD records a fire in that cell
and month. A value of 0 means no fire was recorded.

The local `wildfire_dataset.parquet` file covers January 2023 through December
2024. Its audit passed for 779,958 cells and 18,718,992 cell-month rows. See
[WILDFIRE_PIPELINE.md](WILDFIRE_PIPELINE.md) for feature definitions and limits.

## What is in Git

Git contains the Python code, requirements, and project instructions. It does
not contain the large `wildfire_dataset.parquet` file, the `raw_data/` source
folder, or the monthly processing folder. Use the downloader and builder to
make the dataset after cloning.

## Run the pipeline

Use Python 3.11 or newer. Install the packages and download the source files.
NASA MODIS downloads need an Earthdata Login account in `~/.netrc`; see
[DATA_SOURCES.md](DATA_SOURCES.md). The full source set needs about 100 GiB of
disk space. Then run:

```powershell
python -m pip install -r requirements_wildfire_pipeline.txt
python download_wildfire_sources.py
python derive_3dep_terrain.py
python extract_transportation_gpkg.py
python precompute_national_roads.py
python precompute_national_streams.py
python build_wildfire_dataset.py --start 2023-01 --end 2024-12 --output wildfire_dataset --resume
python audit_wildfire_dataset.py wildfire_dataset.parquet
```

The builder writes all rows to one file, `wildfire_dataset.parquet`. It keeps
monthly files in `wildfire_dataset/` so an interrupted build can resume. To
make the single file again from completed monthly files, run
`python build_wildfire_dataset.py --combine-only`. Read only the rows or
columns you need; loading every row at once can use too much memory.
