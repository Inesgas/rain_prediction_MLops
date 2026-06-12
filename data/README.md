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
