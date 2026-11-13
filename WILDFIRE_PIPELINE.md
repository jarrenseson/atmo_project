# Wildfire aggregation pipeline

`build_wildfire_dataset.py` combines the files in `raw_data` into one Parquet
file, `wildfire_dataset.parquet`. Each row represents one 2-mile by 2-mile cell in the
contiguous United States and one calendar month. The key is `(cell_id, month)`.
`wildfire_recorded` is `1` if USDA FPA-FOD lists at least one ignition in the
cell during that month. `0` means no fire was recorded. `fire_count` keeps the
number of records.

## Install and run

The required packages are in `requirements_wildfire_pipeline.txt`. This
workspace also has a local `.deps` folder that the script uses automatically.
It is excluded from Git. To install into another Python environment:

```powershell
python -m pip install -r requirements_wildfire_pipeline.txt
```

The national run uses 49 CONUS/DC NHD state ZIPs. The national road ZIP
contains a compressed GeoPackage. Run
`python extract_transportation_gpkg.py` before the national build so spatial
queries use its extracted, indexed copy. The optional
`python precompute_national_roads.py` step caches road distances for the
same 2-mile grid and 50-mile search radius. The optional
`python precompute_national_streams.py` step does the same for streams. The
builder checks cached cell IDs and source hashes before use. See
`DATA_SOURCES.md` for the required source files and provider links.

The local source files and distance caches are complete. They are excluded
from Git. A new clone can run `python download_wildfire_sources.py` and then
the preparation scripts in `README.md`. The available common period is January
2023 through December 2024.
Build or resume it with:

```powershell
python build_wildfire_dataset.py --start 2023-01 --end 2024-12 --output wildfire_dataset --resume
```

The finished local dataset is `wildfire_dataset.parquet`. Its audit passed for
24 months, 779,958 cells, and 18,718,992 unique cell-month rows. It contains
147,295 fire-positive rows. There are 15,303,339 rows with every source
feature present.

| Source | Output fields |
| --- | --- |
| gridMET | `tmax_c`, `tmin_c`, `precip_mm`, `wind_m_s` |
| Annual NLCD | `land_cover_previous_year`; 2025 NLCD also defines the CONUS grid mask |
| MOD13Q1 | `ndvi_prior`, `ndvi_age_days` |
| USGS 3DEP | `elevation_m`, `slope_deg`, `aspect_deg`, `curvature_per_m`, `aspect_sin`, `aspect_cos` |
| USGS Transportation | `road_distance_m` |
| USGS Hydrography | `stream_distance_m` |
| USDA FPA-FOD | `fire_count`, `wildfire_recorded` |

For the full common time range, after all matching historical NLCD, MOD13Q1,
gridMET, and vector files are available:

```powershell
python build_wildfire_dataset.py --start 2000-04 --end 2024-12 --output wildfire_dataset
```

The script checks for complete files before processing. It ignores `.part`
files and reports missing inputs. Earlier years need the matching previous-year
NLCD, gridMET, and MOD13Q1 files before each target month. The current 2025
MOD13Q1 collection cannot supply NDVI for 2000-2024 fire labels.

## Output

The final file is `wildfire_dataset.parquet`. It includes the cell ID, cell
center, month, target, climate, land cover, NDVI, terrain, road distance, and
stream distance. It stores aggregation metadata in the Parquet file. Monthly
files under `wildfire_dataset/year=YYYY/month_num=MM/` are a resume cache.
The final file, raw sources, and monthly cache remain outside Git.
`wildfire_dataset/_cells.parquet` stores the static grid. Recreate only the
single file from completed monthly files with:

```powershell
python build_wildfire_dataset.py --combine-only
```

Read the finished file with:

```python
import pandas as pd
data = pd.read_parquet("wildfire_dataset.parquet")
```

For national runs, read only the years or columns you need. Loading all months
into memory at once is likely to use too much RAM.
Check every month and source field with:

```powershell
python audit_wildfire_dataset.py wildfire_dataset.parquet
```

The audit writes `wildfire_dataset.audit.json` with row counts, fire labels,
complete feature rows, and nonmissing values for each source field.

## Feature timing and limits

- Temperature and wind are means from the **previous calendar month**.
  Precipitation is the previous month's total.
- NDVI is the mean of quality-screened pixels from the latest 16-day composite
  that ended before the target month. `ndvi_age_days` records its age.
- Land cover is the most common class in the cell from the **previous year**.
- Elevation, slope, aspect, and curvature come from the available 3DEP service
  extracts. These extracts are about 1-3 km resolution, so slope and curvature
  do not represent fine terrain.
- Road and stream distances are measured from each cell center to the nearest
  selected line within 50 miles by default. A missing value means no selected
  line was found within that search distance. Stream lines are NHD `FType=460`
  (Stream/River).
- Present-day road and stream geometry is used for historical months. Old road
  locations may differ from the present network.
- The grid uses EPSG:5070 and exact 3,218.688 m sides. The current 2025 NLCD
  footprint selects cells; Alaska and Hawaii are excluded.
- FPA-FOD point locations can be less precise than one 2-mile cell. Fire labels
  near cell edges can be uncertain.

Sources: [gridMET](https://www.climatologylab.org/gridmet.html),
[Annual NLCD](https://www.mrlc.gov/data),
[MOD13Q1](https://doi.org/10.5067/MODIS/MOD13Q1.061),
[3DEP](https://www.usgs.gov/3d-elevation-program),
[USGS Transportation](https://www.usgs.gov/the-national-map-data-delivery/gis-data-download),
[USGS Hydrography](https://www.usgs.gov/national-hydrography/access-national-hydrography-products),
[FPA-FOD](https://www.fs.usda.gov/rds/archive/catalog/RDS-2013-0009.7).
