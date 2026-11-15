# Local source files

`raw_data/` is kept on the local computer and excluded from Git. The current
folder is about 96 GiB, and the final single Parquet file is about 1.35 GiB.
Allow enough disk space before you reproduce the national build. Run
`python download_wildfire_sources.py` to get the source files. The script
downloads into `raw_data/`, resumes partial transfers, and writes source URL
and SHA-256 records. Use `--only gridmet` or another source group to download
one group. Use `--verify` to check hashes of files already present.

NASA MODIS subsets require an Earthdata Login account. Put the account in a
`~/.netrc` file as described in the [Earthdata Login instructions](https://urs.earthdata.nasa.gov/documentation/for_users/data_access/curl_and_wget).
Do not put credentials in this repository. The MODIS service can change; if
the downloader reports an access error, check the current NASA data service.

The pipeline uses the following files for January 2023 through December 2024.
Source addresses and product versions can change; check the provider pages.

| Source | Local files needed | Source page |
| --- | --- | --- |
| gridMET | `gridMET/{tmmx,tmmn,pr,vs}_{2022,2023,2024}.nc` | [gridMET](https://www.climatologylab.org/gridmet.html) |
| Annual NLCD | `NLCD/Annual_NLCD_LndCov_{2022,2023,2025}_CU_C1V2.zip` | [MRLC](https://www.mrlc.gov/data) |
| MOD13Q1.061 | `MOD13Q1/2022_NDVI_quality/` for December 3; full `2023_NDVI_quality/` and `2024_NDVI_quality/` subsets, with quality fields and records | [NASA](https://doi.org/10.5067/MODIS/MOD13Q1.061) |
| 3DEP | Four CONUS `3dep_elevation_*_approx1to3km.tif` service extracts in `3DEP/` | [USGS 3DEP](https://www.usgs.gov/3d-elevation-program) |
| Transportation | `USGS_Transportation/Transportation_National_GPKG.zip` and its extracted GeoPackage | [USGS Transportation](https://www.usgs.gov/the-national-map-data-delivery/gis-data-download) |
| Hydrography | 49 CONUS/DC `NHD_H_*_State_GDB.zip` files, the state archive manifest, and source hash records in `USGS_Hydrography/` | [USGS Hydrography](https://www.usgs.gov/national-hydrography/access-national-hydrography-products) |
| FPA-FOD | `FPA_FOD/RDS-2013-0009.7_Data_Format3_GPKG.zip` | [USDA Forest Service](https://research.fs.usda.gov/firelab/products/dataandtools/us-wildfires) |

The 2022 MODIS composite supplies January 2023 NDVI. Annual NLCD 2022 and
2023 supply previous-year land cover; NLCD 2025 defines the current CONUS
cell mask. The local MODIS files are NASA NetCDF subsets with NDVI and quality
fields. An Earthdata account is needed to obtain them. The 3DEP TIFFs are
regional image-service exports at about 1 to 3 km spacing.

After the files are present, the remaining processing scripts extract roads
and make reusable distance caches:

```powershell
python extract_transportation_gpkg.py
python precompute_national_roads.py
python precompute_national_streams.py
```

The local `raw_data/` folder also contains the source catalogs and SHA-256
records used to check the caches. A clone needs those records with the source
files to rebuild the same dataset. The source data and caches stay out of Git.
