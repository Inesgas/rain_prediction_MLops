# Local Airflow Data and Model Versioning

The project uses DVC for artifact identity and Airflow for orchestration. This
workflow is local-only: it does not push to GitHub or DagsHub.

## DAG

`airflow/dags/data_model_versioning_dag.py` defines `data_model_versioning`.

The DAG performs these steps:

1. Extract or validate `data/raw/weatherAUS.csv`.
2. Run `dvc add data/raw/weatherAUS.csv` locally.
3. Check that local DVC-tracked input datasets are present.
4. Snapshot Git, DVC pointer, data, and model metadata before training.
5. Run `python -m src.models.train_winner`.
6. Run `dvc add models/final_winner/winner_model.joblib` locally.
7. Snapshot output model metadata and DVC status.
8. End with a local-only notice.

## Running Locally

```powershell
docker compose -f src/docker/docker-compose-dev.yml up --build airflow
```

Open `http://localhost:8080`, sign in with `airflow` / `airflow`, and trigger
the `data_model_versioning` DAG.

`daily_weather_ingestion` is scheduled at `03:00` UTC each day. It performs only
daily raw ingestion, local DVC metadata updates, and local manifests. It does not
train or promote a model.

## New Data

Set `WEATHER_AUS_SOURCE` to a local WeatherAUS-format CSV, ZIP, or directory.
The extractor validates the required columns before updating `data/raw/weatherAUS.csv`.

Modes are controlled with `WEATHER_AUS_EXTRACT_MODE`:

| Mode | Behavior |
| --- | --- |
| `upsert` | Default. Insert new `Date` + `Location` rows and replace changed matching rows. |
| `append` | Insert only new keys and fail if matching rows contain changed values. |
| `replace` | Treat the source as a complete refreshed raw dataset. |

Example:

```powershell
$env:WEATHER_AUS_SOURCE = "data/incoming/weatherAUS_new.csv"
$env:WEATHER_AUS_EXTRACT_MODE = "upsert"
docker compose -f src/docker/docker-compose-dev.yml up --build airflow
```

Ingest manifests record inserted, updated, unchanged, and duplicate incoming
rows under `reports/versioning/`.

## Online Extraction

The extractor can download a direct online CSV or ZIP before validation:

```powershell
$env:WEATHER_AUS_SOURCE_URL = "https://example.com/weatherAUS.csv"
$env:WEATHER_AUS_SOURCE_SHA256 = "<optional expected sha256>"
$env:WEATHER_AUS_EXTRACT_MODE = "upsert"
docker compose -f src/docker/docker-compose-dev.yml up --build airflow
```

It can also use Kaggle when credentials are available:

```powershell
$env:KAGGLE_USERNAME = "<your username>"
$env:KAGGLE_KEY = "<your api key>"
$env:WEATHER_AUS_KAGGLE_DATASET = "jsphyg/weather-dataset-rattle-package"
$env:WEATHER_AUS_KAGGLE_FILE = "weatherAUS.csv"
$env:WEATHER_AUS_EXTRACT_MODE = "replace"
docker compose -f src/docker/docker-compose-dev.yml up --build airflow
```

Use `replace` for full online dataset refreshes and `upsert` for incremental
online feeds with `Date` + `Location` keys.

## Daily Open-Meteo Ingestion

For automated non-Kaggle daily ingestion, use the Open-Meteo provider:

```powershell
$env:WEATHER_AUS_DAILY_PROVIDER = "open-meteo"
$env:WEATHER_AUS_EXTRACT_MODE = "upsert"
$env:WEATHER_AUS_DAILY_DAYS_BACK = "7"
$env:WEATHER_AUS_DAILY_END_LAG_DAYS = "1"
docker compose -f src/docker/docker-compose-dev.yml up --build airflow
```

The provider fetches recent daily and hourly variables for the project locations
from `data/preprocessed/locations_metadata.csv`, converts wind directions to the
WeatherAUS compass labels, converts cloud cover percent to oktas, derives
`RainToday`, and fills `RainTomorrow` from the next day's rainfall when present.

The last fetched date usually has a blank `RainTomorrow`, because its next day
has not been observed yet. The next scheduled run backfills that label.

## Outputs

Runtime manifests are written to `reports/versioning/`. They are ignored by Git
except for `.gitkeep`, so they can be used as local evidence or logged later by
the separate MLflow tracking work.

The local DVC pointer for `models/final_winner/winner_model.joblib` may change
after retraining. Review that pointer before deciding whether to commit it.

When raw data changes, rebuild and version the processed feature datasets before
using the retrained model for evaluation or promotion. This DAG keeps the raw
ingest auditable, but processed feature rebuilds remain a separate project step.
