# Data Inventory

| Folder | Content | Storage |
| --- | --- | --- |
| `data/raw/` | Original weather dataset | DVC with DagsHub remote |
| `data/preprocessed/` | Modeling tables, selected feature lists, feature rankings, and location metadata | DVC with DagsHub remote for large datasets, normal Git for small metadata |
| `references/` | Station and climate reference files | Normal Git |

| File | Size | Role |
| --- | ---: | --- |
| `data/raw/weatherAUS.csv` | 13.44 MB | Original weather dataset |
| `data/preprocessed/rain_model_dataset.csv` | 90.69 MB | Main processed modeling table |
| `data/preprocessed/rain_model_dataset_aligned.csv` | 99.25 MB | Aligned processed table used by the winner feature workflow |
| `data/preprocessed/rain_model_dataset_feature_experiments.csv` | 121.51 MB | Extended processed table used for feature experiments |
| `data/preprocessed/ines_selected_top25_features.txt` | < 1 MB | Selected top-25 feature list |
| `data/preprocessed/ines_selected_top25_features_aligned.txt` | < 1 MB | Aligned top-25 feature list retained as modeling evidence |
| `data/preprocessed/ines_selected_top25_features_feature_experiments.txt` | < 1 MB | Feature-experiment selected feature list |
| `data/preprocessed/ines_feature_ranking.csv` | < 1 MB | Feature ranking output |
| `data/preprocessed/ines_feature_ranking_aligned.csv` | < 1 MB | Aligned feature ranking output |
| `data/preprocessed/ines_feature_ranking_feature_experiments.csv` | < 1 MB | Feature-experiment ranking output |
| `data/preprocessed/locations_metadata.csv` | < 1 MB | Station metadata used by downstream interfaces |
| `data/preprocessed/daily_zonal_locations_metadata.csv` | < 1 MB | Derived station metadata used by the winner feature builder |
| `references/` | 2.69 MB | Climate and station reference files |

## New Local Data

Use the Airflow `data_model_versioning` DAG with `WEATHER_AUS_SOURCE` pointing
to a local WeatherAUS-format CSV, ZIP, or directory. The extractor validates the
schema, merges by `Date` + `Location`, writes an ingest manifest, and updates the
local DVC pointer for `data/raw/weatherAUS.csv`.

The default merge mode is `upsert`, which inserts new rows and applies corrected
values for matching `Date` + `Location` keys. Use `append` for strict new-row
loads or `replace` for a full raw dataset refresh.

For online sources, set `WEATHER_AUS_SOURCE_URL` to a direct CSV/ZIP URL or set
`WEATHER_AUS_KAGGLE_DATASET` with `KAGGLE_USERNAME` and `KAGGLE_KEY`. Online
downloads are still written and versioned locally; the workflow does not push to
GitHub or DagsHub.

For automated daily non-Kaggle data, use `WEATHER_AUS_DAILY_PROVIDER=open-meteo`.
This fetches recent weather for the known station coordinates, writes
`data/incoming/open_meteo_weatherAUS_daily.csv`, and upserts it into
`data/raw/weatherAUS.csv`.
