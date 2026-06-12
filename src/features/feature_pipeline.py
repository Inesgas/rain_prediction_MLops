import pandas as pd
import numpy as np
from pathlib import Path

try:
    import xarray as xr
except ImportError:  # optional dependency for Köppen enrichment
    xr = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA_PATH = PROJECT_ROOT / 'data' / 'preprocessed' / 'locations_metadata.csv'
DEFAULT_STATIONS_PATH = PROJECT_ROOT / 'references' / 'stations.txt'
DEFAULT_KOPPEN_PATH = PROJECT_ROOT / 'references' / 'koppen_zones_grid.nc'
DEFAULT_SEASONAL_RAINFALL_PATH = PROJECT_ROOT / 'references' / 'seasonal_rainfall_zones_grid.nc'
DEFAULT_TEMPERATURE_HUMIDITY_PATH = PROJECT_ROOT / 'references' / 'temperature_humidity_zones_grid.nc'


BASE_MISSINGNESS_COLUMNS = ['sunshine', 'evaporation', 'cloud_9am', 'cloud_3pm']
ALIGNED_MISSINGNESS_COLUMNS = [
    'rainfall',
    'sunshine',
    'evaporation',
    'wind_gust_speed',
    'humidity_9am',
    'humidity_3pm',
    'pressure_9am',
    'pressure_3pm',
    'cloud_9am',
    'cloud_3pm',
]


def _resolve_project_path(path_like):
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path



# ============================================================
# 1. COLUMN NAMING
# ============================================================
def rename_cols(df):
    """
    Rename original weather dataset columns to snake_case.
    This preserves the original column mapping exactly.
    """
    df = df.copy()
    dictionary = {
        'Date': 'date',
        'Location': 'location',
        'MinTemp': 'min_temp',
        'MaxTemp': 'max_temp',
        'Rainfall': 'rainfall',
        'Evaporation': 'evaporation',
        'Sunshine': 'sunshine',
        'WindGustDir': 'wind_gust_dir',
        'WindGustSpeed': 'wind_gust_speed',
        'WindDir9am': 'wind_dir_9am',
        'WindDir3pm': 'wind_dir_3pm',
        'WindSpeed9am': 'wind_speed_9am',
        'WindSpeed3pm': 'wind_speed_3pm',
        'Humidity9am': 'humidity_9am',
        'Humidity3pm': 'humidity_3pm',
        'Pressure9am': 'pressure_9am',
        'Pressure3pm': 'pressure_3pm',
        'Cloud9am': 'cloud_9am',
        'Cloud3pm': 'cloud_3pm',
        'Temp9am': 'temp_9am',
        'Temp3pm': 'temp_3pm',
        'RainToday': 'rain_today',
        'RainTomorrow': 'rain_tomorrow'
    }
    return df.rename(columns=dictionary)


# ============================================================
# 2. SPATIAL FEATURES
# ============================================================
def build_location_metadata(
    stations_path=DEFAULT_STATIONS_PATH,
    output_path=DEFAULT_METADATA_PATH,
):
    """Build station metadata from the BoM stations table when no cached CSV exists."""
    stations_path = _resolve_project_path(stations_path)
    output_path = _resolve_project_path(output_path)
    if not stations_path.exists():
        return None

    df_bom = pd.read_fwf(stations_path, skiprows=2, header=0, na_values=['..', '.....'])
    for col in ['Start', 'End', 'WMO']:
        if col in df_bom.columns:
            df_bom[col] = pd.to_numeric(df_bom[col], errors='coerce')

    mask = (
        df_bom['WMO'].notna()
        & (df_bom['WMO'] != 0)
        & ((df_bom['End'] >= 2015) | (df_bom['End'].isna()))
    )
    df_active = df_bom[mask].copy()

    locations = [
        'Adelaide', 'Albany', 'Albury', 'AliceSprings', 'BadgerysCreek',
        'Ballarat', 'Bendigo', 'Brisbane', 'Cairns', 'Canberra',
        'Cobar', 'CoffsHarbour', 'Dartmoor', 'Darwin', 'GoldCoast',
        'Hobart', 'Katherine', 'Launceston', 'Melbourne', 'MelbourneAirport',
        'Mildura', 'Moree', 'MountGambier', 'MountGinini', 'Newcastle',
        'Nhil', 'NorahHead', 'NorfolkIsland', 'Nuriootpa', 'PearceRAAF',
        'Penrith', 'Perth', 'PerthAirport', 'Portland', 'Richmond',
        'Sale', 'SalmonGums', 'Sydney', 'SydneyAirport', 'Townsville',
        'Tuggeranong', 'Uluru', 'WaggaWagga', 'Walpole', 'Watsonia',
        'Williamtown', 'Witchcliffe', 'Wollongong', 'Woomera',
    ]
    name_fixes = {
        'Adelaide': 'ADELAIDEAIRPORTM.O.',
        'Cobar': 'COBARAIRPORTAWS',
        'Hobart': 'HOBARTAIRPORTWEST',
        'Launceston': 'LAUNCESTONAIRPORT',
        'Melbourne': 'MELBOURNE(OLYMPICPARK)',
        'Nhil': 'NHILLAERODROME',
        'Watsonia': 'VIEWBANK',
        'Katherine': 'TINDAL',
        'Richmond': 'RICHMONDRAAF',
        'Uluru': 'YULARAAIRPORT',
        'GoldCoast': 'GOLDCOASTSEAWAY',
        'Perth': 'PERTHMETRO',
        'Sale': 'EASTSALE',
        'Walpole': 'NORTHWALPOLE',
        'Sydney': 'SYDNEY(OBSERVATORYHILL)',
    }

    site_name_norm = (
        df_active['Site name']
        .fillna('')
        .astype(str)
        .str.replace(' ', '', regex=False)
        .str.lower()
    )

    metadata_rows = []
    for loc in locations:
        search_term = name_fixes.get(loc, loc).replace(' ', '').lower()
        matches = df_active[site_name_norm.str.startswith(search_term)]
        if matches.empty:
            continue

        best_match = matches.sort_values(by='Start', ascending=True).iloc[0]
        metadata_rows.append({
            'location': loc,
            'site_id': best_match['Site'],
            'elevation': best_match['Height (m)'],
            'lat': best_match['Lat'],
            'lon': best_match['Lon'],
        })

    if not metadata_rows:
        return None

    meta_df = pd.DataFrame(metadata_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_df.to_csv(output_path, index=False)
    return meta_df


def add_coordinates(
    df,
    metadata_path=DEFAULT_METADATA_PATH,
    stations_path=DEFAULT_STATIONS_PATH,
):
    """Merge station latitude/longitude/elevation using cached or raw station metadata."""
    df = df.copy()
    metadata_path = _resolve_project_path(metadata_path)
    stations_path = _resolve_project_path(stations_path)
    meta_df = pd.read_csv(metadata_path) if metadata_path.exists() else build_location_metadata(
        stations_path=stations_path,
        output_path=metadata_path,
    )
    if meta_df is None:
        return df
    keep_cols = [col for col in ['location', 'lat', 'lon', 'elevation'] if col in meta_df.columns]
    if 'location' not in keep_cols:
        return df
    meta_subset = meta_df[keep_cols].drop_duplicates(subset=['location'])
    return df.merge(meta_subset, on='location', how='left')


def add_ncc_zones(df):
    """Add NCC climate zone. This is the zonation kept for the notebook."""
    df = df.copy()
    zones = {
        "Albury": 4, "BadgerysCreek": 6, "Cobar": 4, "CoffsHarbour": 2, "Moree": 4,
        "Newcastle": 5, "NorahHead": 5, "NorfolkIsland": 2, "Penrith": 6, "Richmond": 6,
        "Sydney": 5, "SydneyAirport": 5, "WaggaWagga": 4, "Williamtown": 5, "Wollongong": 5,
        "Canberra": 7, "Tuggeranong": 7, "MountGinini": 8, "Ballarat": 7, "Bendigo": 6,
        "Sale": 6, "MelbourneAirport": 6, "Melbourne": 6, "Mildura": 4, "Nhil": 4,
        "Portland": 6, "Watsonia": 6, "Dartmoor": 6, "Brisbane": 2, "Cairns": 1,
        "GoldCoast": 2, "Townsville": 1, "Adelaide": 5, "MountGambier": 6, "Nuriootpa": 6,
        "Woomera": 4, "Albany": 6, "Witchcliffe": 5, "PearceRAAF": 5, "PerthAirport": 5,
        "Perth": 5, "SalmonGums": 4, "Walpole": 6, "Hobart": 7, "Launceston": 7,
        "AliceSprings": 3, "Darwin": 1, "Katherine": 1, "Uluru": 3
    }
    df['ncc_zone'] = df['location'].map(zones)
    return df


def _pick_zone_grid_var(ds):
    for name, da in ds.data_vars.items():
        dims = set(da.dims)
        if {'lat', 'lon'}.issubset(dims):
            return name
    return None


def _pick_zone_legend(ds):
    class_ids = None
    class_names = None

    for candidate in ['class_code', 'zone_code', 'zone_id']:
        if candidate in ds:
            class_ids = ds[candidate].values
            break

    for candidate in ds.data_vars:
        lower = candidate.lower()
        if any(token in lower for token in ['description', 'label', 'name']):
            values = ds[candidate].astype(str).values
            if values.ndim == 1:
                class_names = values
                break

    if class_ids is None or class_names is None:
        return None
    return dict(zip(class_ids, class_names))


def _add_zone_from_grid(df, zone_col, nc_path, fallback_override=None):
    df = df.copy()
    required = {'lat', 'lon'}
    nc_path = _resolve_project_path(nc_path)
    if xr is None or not required.issubset(df.columns) or not nc_path.exists():
        return df

    geo_df = df[['location', 'lat', 'lon']].drop_duplicates().copy()
    try:
        ds = xr.open_dataset(nc_path)
    except Exception:
        return df
    grid_var = _pick_zone_grid_var(ds)
    if grid_var is None:
        ds.close()
        return df

    legend_dict = _pick_zone_legend(ds)

    def coerce_numeric(val):
        try:
            return float(val)
        except (TypeError, ValueError):
            return np.nan

    def map_value(val):
        if pd.isna(val):
            return 'Unknown'
        if legend_dict is None:
            numeric_val = coerce_numeric(val)
            if pd.isna(numeric_val):
                return str(val)
            return str(int(numeric_val)) if numeric_val.is_integer() else str(numeric_val)
        numeric_val = coerce_numeric(val)
        if pd.isna(numeric_val):
            return 'Unknown'
        return legend_dict.get(numeric_val, 'Unknown')

    def get_zone_safe(row):
        try:
            val = ds[grid_var].sel(lat=row['lat'], lon=row['lon'], method='nearest').item()
        except Exception:
            return 'Unknown'
        numeric_val = coerce_numeric(val)

        if pd.isna(numeric_val) or numeric_val < 0:
            try:
                subset = ds[grid_var].sel(
                    lat=slice(row['lat'] - 0.3, row['lat'] + 0.3),
                    lon=slice(row['lon'] - 0.3, row['lon'] + 0.3),
                )
                if subset.values.size > 0:
                    non_null = pd.Series(subset.values.flatten()).dropna()
                    val = non_null.mode().iloc[0] if not non_null.empty else np.nan
                else:
                    val = np.nan
            except Exception:
                val = np.nan

        return map_value(val)

    geo_df[zone_col] = geo_df.apply(get_zone_safe, axis=1)
    ds.close()
    if fallback_override:
        for location, label in fallback_override.items():
            geo_df.loc[geo_df['location'] == location, zone_col] = label
    return df.merge(geo_df[['location', zone_col]], on='location', how='left')


def add_koppen_zone(df, nc_path=DEFAULT_KOPPEN_PATH):
    """
    Add a Köppen-style climate label from the BoM grid when the file is available.

    The lookup is optional so notebook copies stay runnable even when the external
    NetCDF file is missing locally.
    """
    return _add_zone_from_grid(
        df,
        zone_col='koppen_zone',
        nc_path=nc_path,
        fallback_override={'NorfolkIsland': 'Subtropical'},
    )


def add_seasonal_rainfall_zone(df, nc_path=DEFAULT_SEASONAL_RAINFALL_PATH):
    """Add a seasonal rainfall regime label from the BoM grid when available."""
    return _add_zone_from_grid(df, zone_col='seasonal_rainfall_zone', nc_path=nc_path)


def add_temperature_humidity_zone(df, nc_path=DEFAULT_TEMPERATURE_HUMIDITY_PATH):
    """Add a temperature-humidity regime label from the BoM grid when available."""
    return _add_zone_from_grid(df, zone_col='temperature_humidity_zone', nc_path=nc_path)


def add_longitude_strips(df):
    """Add a coarse east-west climate strip derived from station longitude."""
    df = df.copy()
    if 'lon' not in df.columns:
        return df

    conditions = [
        (df['lon'] < 125),
        (df['lon'] >= 125) & (df['lon'] < 140),
        (df['lon'] >= 140) & (df['lon'] < 148),
        (df['lon'] >= 148),
    ]
    values = ['west_coast', 'central_arid', 'east_inland', 'east_coast']
    df['land_strip'] = np.select(conditions, values, default='unknown')
    return df


# ============================================================
# 3. MISSING-VALUE HELPERS
# ============================================================
def add_missing_indicators(df, columns):
    """Add binary indicators for informative missingness."""
    df = df.copy()
    if isinstance(columns, str):
        columns = [columns]
    for col in columns:
        if col in df.columns:
            df[f'{col}_missing'] = df[col].isna().astype(int)
    return df


def fill_with_zonal_stats(df, reference_df, columns, strategy='median', threshold=0.2):
    """
    Fill missing values using ncc_zone statistics and optionally add missing indicators.
    """
    df = df.copy()
    if isinstance(columns, str):
        columns = [columns]

    for col in columns:
        if col not in df.columns:
            continue

        nan_rate = reference_df[col].isnull().mean()

        if nan_rate > threshold:
            df[f"{col}_missing"] = df[col].isnull().astype(int)

        if strategy == 'median':
            zonal_stats = reference_df.groupby('ncc_zone')[col].median()
        elif strategy == 'mean':
            zonal_stats = reference_df.groupby('ncc_zone')[col].mean()
        elif strategy == 'mode':
            zonal_stats = reference_df.groupby('ncc_zone')[col].apply(
                lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
            )
        else:
            raise ValueError("strategy must be 'median', 'mean', or 'mode'")

        if 8 not in zonal_stats.index or pd.isna(zonal_stats.get(8)):
            if 7 in zonal_stats.index:
                zonal_stats[8] = zonal_stats[7]

        df[col] = df[col].fillna(df['ncc_zone'].map(zonal_stats))

    return df


# ============================================================
# 4. ENGINEERED WEATHER FEATURES
# ============================================================
def add_daily_diffs(df, prefixes):
    """3pm - 9am difference for selected prefixes."""
    df = df.copy()
    if isinstance(prefixes, str):
        prefixes = [prefixes]

    for p in prefixes:
        col_3pm = f"{p}_3pm"
        col_9am = f"{p}_9am"
        if col_3pm in df.columns and col_9am in df.columns:
            df[f"{p}_day_diff"] = df[col_3pm] - df[col_9am]
    return df


def add_overnight_diffs(df, prefixes):
    """Today 9am - yesterday 3pm."""
    df = df.copy()
    if isinstance(prefixes, str):
        prefixes = [prefixes]

    for p in prefixes:
        col_9am = f"{p}_9am"
        col_3pm = f"{p}_3pm"

        if col_9am in df.columns and col_3pm in df.columns:
            temp = df[['date', 'location', col_3pm]].copy()
            temp['date'] = pd.to_datetime(temp['date']) + pd.Timedelta(days=1)
            temp = temp.rename(columns={col_3pm: f"{p}_3pm_yesterday"})

            df = df.merge(temp, on=['date', 'location'], how='left')
            df[f"{p}_overnight_change"] = df[col_9am] - df[f"{p}_3pm_yesterday"]
            df = df.drop(columns=[f"{p}_3pm_yesterday"])

    return df


def add_yesterday_lag(df, columns):
    """Lag selected columns by one day within each location."""
    df = df.copy()
    if isinstance(columns, str):
        columns = [columns]

    for col in columns:
        temp = df[['date', 'location', col]].copy()
        temp['date'] = pd.to_datetime(temp['date']) + pd.Timedelta(days=1)
        temp = temp.rename(columns={col: f"{col}_yesterday"})
        df = df.merge(temp, on=['date', 'location'], how='left')

    return df


def add_dewpoint_features(df):
    """Dewpoint and dew point spread using Magnus formula."""
    df = df.copy()
    a = 17.27
    b = 237.7

    for time in ['9am', '3pm']:
        t_col = f'temp_{time}'
        h_col = f'humidity_{time}'
        dp_col = f'dewpoint_{time}'
        s_col = f'dew_point_spread_{time}'

        if t_col in df.columns and h_col in df.columns:
            safe_h = df[h_col].clip(lower=0.01, upper=100.0)
            alpha = ((a * df[t_col]) / (b + df[t_col])) + np.log(safe_h / 100.0)
            df[dp_col] = (b * alpha) / (a - alpha)
            df[s_col] = df[t_col] - df[dp_col]

    return df


def _direction_to_degrees():
    return {
        "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
        "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
        "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
        "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5
    }


def encode_wind_direction(df, columns=None, drop_original=False):
    """Encode wind direction into x/y components."""
    df = df.copy()
    if columns is None:
        columns = ['wind_dir_9am', 'wind_dir_3pm', 'wind_gust_dir']

    angle_map = _direction_to_degrees()

    for col in columns:
        if col in df.columns:
            degrees = df[col].map(angle_map)
            radians = np.deg2rad(degrees)
            df[f'{col}_x'] = np.cos(radians)
            df[f'{col}_y'] = np.sin(radians)

            if drop_original:
                df = df.drop(columns=[col])

    return df


def add_wind_shift(df):
    """Dot product between 9am and 3pm wind vectors."""
    df = df.copy()
    cols = ['wind_dir_9am_x', 'wind_dir_3pm_x', 'wind_dir_9am_y', 'wind_dir_3pm_y']
    if all(c in df.columns for c in cols):
        df['wind_shift_score'] = (
            df['wind_dir_9am_x'] * df['wind_dir_3pm_x'] +
            df['wind_dir_9am_y'] * df['wind_dir_3pm_y']
        )
    return df


def add_trend_indicators(df):
    """Binary weather trend flags."""
    df = df.copy()

    if 'pressure_3pm' in df.columns and 'pressure_9am' in df.columns:
        df['pressure_fall'] = (df['pressure_3pm'] < df['pressure_9am']).astype(int)

    if 'humidity_3pm' in df.columns and 'humidity_9am' in df.columns:
        df['humidity_rising_fast'] = (df['humidity_3pm'] > (df['humidity_9am'] * 1.1)).astype(int)

    if 'temp_3pm' in df.columns and 'temp_9am' in df.columns:
        df['warming_day'] = (df['temp_3pm'] > df['temp_9am']).astype(int)

    return df


def add_cyclical_time_features(df):
    """Seasonality encoded as sin/cos of day of year."""
    df = df.copy()
    if 'date' in df.columns:
        day_of_year = pd.to_datetime(df['date']).dt.dayofyear
        df['day_of_year_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
        df['day_of_year_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)
    return df


def add_temperature_structure_features(df):
    """Add temperature range and simple moisture-temperature interactions."""
    df = df.copy()

    if {'max_temp', 'min_temp'}.issubset(df.columns):
        df['temp_range'] = df['max_temp'] - df['min_temp']

    if {'temp_3pm', 'humidity_3pm'}.issubset(df.columns):
        df['humidity_temp_3pm_interaction'] = df['temp_3pm'] * (df['humidity_3pm'] / 100.0)

    if {'temp_9am', 'humidity_9am'}.issubset(df.columns):
        df['humidity_temp_9am_interaction'] = df['temp_9am'] * (df['humidity_9am'] / 100.0)

    if {'temp_3pm', 'max_temp'}.issubset(df.columns):
        df['temp_3pm_vs_max_gap'] = df['max_temp'] - df['temp_3pm']

    return df


def add_pressure_moisture_features(df):
    """Combine pressure, humidity, cloud and dew-point information."""
    df = df.copy()

    if {'pressure_9am', 'humidity_9am'}.issubset(df.columns):
        df['pressure_humidity_9am_ratio'] = df['pressure_9am'] / (df['humidity_9am'] + 1.0)

    if {'pressure_3pm', 'humidity_3pm'}.issubset(df.columns):
        df['pressure_humidity_3pm_ratio'] = df['pressure_3pm'] / (df['humidity_3pm'] + 1.0)

    if {'cloud_3pm', 'humidity_3pm'}.issubset(df.columns):
        df['cloud_humidity_3pm_interaction'] = df['cloud_3pm'] * df['humidity_3pm']

    if {'dew_point_spread_3pm', 'humidity_3pm'}.issubset(df.columns):
        df['moisture_stability_3pm'] = df['dew_point_spread_3pm'] * (100.0 - df['humidity_3pm'])

    return df


def _days_since_last_rain_event(series, threshold=1.0):
    shifted = series.shift(1)
    days_since = []
    counter = np.nan

    for value in shifted:
        if pd.notna(value) and value >= threshold:
            counter = 0.0
        elif pd.isna(counter):
            counter = np.nan
        else:
            counter += 1.0
        days_since.append(counter)

    return pd.Series(days_since, index=series.index)


def add_location_lag_rollups(df):
    """
    Add leak-safe location-wise persistence and rolling features using only past days.
    """
    df = df.copy()
    grouped = df.groupby('location', group_keys=False)

    if 'rainfall' in df.columns:
        rainfall_shift = grouped['rainfall'].shift(1)
        df['rainfall_prev_1d'] = rainfall_shift
        df['rainfall_roll3_mean'] = (
            grouped['rainfall']
            .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        )
        df['rainfall_roll7_sum'] = (
            grouped['rainfall']
            .transform(lambda s: s.shift(1).rolling(7, min_periods=1).sum())
        )
        df['rainfall_roll7_max'] = (
            grouped['rainfall']
            .transform(lambda s: s.shift(1).rolling(7, min_periods=1).max())
        )
        df['days_since_rain_1mm'] = grouped['rainfall'].transform(_days_since_last_rain_event)

    if 'rain_today' in df.columns:
        rain_today_num = df['rain_today'].map({'No': 0, 'Yes': 1}).fillna(0)
        df['rain_today_streak_3'] = (
            rain_today_num.groupby(df['location'])
            .transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum())
        )
        df['rain_today_streak_7'] = (
            rain_today_num.groupby(df['location'])
            .transform(lambda s: s.shift(1).rolling(7, min_periods=1).sum())
        )

    for col, short_window, long_window in [
        ('humidity_3pm', 3, 7),
        ('pressure_3pm', 3, 7),
        ('temp_3pm', 3, 7),
        ('wind_gust_speed', 3, 7),
    ]:
        if col in df.columns:
            df[f'{col}_roll{short_window}_mean'] = (
                grouped[col].transform(lambda s: s.shift(1).rolling(short_window, min_periods=1).mean())
            )
            df[f'{col}_roll{long_window}_mean'] = (
                grouped[col].transform(lambda s: s.shift(1).rolling(long_window, min_periods=1).mean())
            )

    return df


def add_feature_experiment_rollups(df):
    """
    Add longer-memory persistence features for rainfall, pressure, humidity, and cloud.

    These are experimental extensions beyond the aligned notebook. The goal is to
    capture slower wet/dry regimes and short-vs-long atmospheric tendency signals
    without leaking future information.
    """
    df = df.copy()
    grouped = df.groupby('location', group_keys=False)

    if 'rainfall' in df.columns:
        df['rainfall_roll14_sum'] = (
            grouped['rainfall']
            .transform(lambda s: s.shift(1).rolling(14, min_periods=1).sum())
        )
        df['rainfall_roll30_sum'] = (
            grouped['rainfall']
            .transform(lambda s: s.shift(1).rolling(30, min_periods=1).sum())
        )
        df['rainfall_roll14_max'] = (
            grouped['rainfall']
            .transform(lambda s: s.shift(1).rolling(14, min_periods=1).max())
        )
        df['days_since_rain_5mm'] = (
            grouped['rainfall']
            .transform(lambda s: s.shift(1).ge(5.0).astype(float))
            .groupby(df['location'])
            .transform(lambda s: s.groupby(s.eq(1).cumsum()).cumcount())
        )

    if 'rain_today' in df.columns:
        rain_today_num = df['rain_today'].map({'No': 0, 'Yes': 1}).fillna(0)
        rain_grouped = rain_today_num.groupby(df['location'])
        df['rain_days_last_7'] = (
            rain_grouped.transform(lambda s: s.shift(1).rolling(7, min_periods=1).sum())
        )
        df['rain_days_last_14'] = (
            rain_grouped.transform(lambda s: s.shift(1).rolling(14, min_periods=1).sum())
        )
        df['dry_days_last_7'] = (
            rain_grouped.transform(lambda s: (1.0 - s.shift(1).fillna(0)).rolling(7, min_periods=1).sum())
        )
        df['recent_wet_ratio_14'] = df['rain_days_last_14'] / 14.0

    if 'pressure_3pm' in df.columns:
        df['pressure_3pm_roll14_mean'] = (
            grouped['pressure_3pm']
            .transform(lambda s: s.shift(1).rolling(14, min_periods=1).mean())
        )
        df['pressure_3pm_prev_2d_change'] = grouped['pressure_3pm'].transform(
            lambda s: s.shift(1) - s.shift(2)
        )
        df['pressure_3pm_trend_3v7'] = (
            grouped['pressure_3pm']
            .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
            - grouped['pressure_3pm']
            .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
        )

    if 'humidity_3pm' in df.columns:
        df['humidity_3pm_roll14_mean'] = (
            grouped['humidity_3pm']
            .transform(lambda s: s.shift(1).rolling(14, min_periods=1).mean())
        )
        df['humidity_3pm_prev_2d_mean'] = (
            grouped['humidity_3pm']
            .transform(lambda s: s.shift(1).rolling(2, min_periods=1).mean())
        )
        df['humidity_3pm_trend_3v7'] = (
            grouped['humidity_3pm']
            .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
            - grouped['humidity_3pm']
            .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
        )

    if 'cloud_3pm' in df.columns:
        df['cloud_3pm_roll7_mean'] = (
            grouped['cloud_3pm']
            .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
        )
        df['cloud_3pm_roll14_mean'] = (
            grouped['cloud_3pm']
            .transform(lambda s: s.shift(1).rolling(14, min_periods=1).mean())
        )
        df['cloud_3pm_prev_2d_mean'] = (
            grouped['cloud_3pm']
            .transform(lambda s: s.shift(1).rolling(2, min_periods=1).mean())
        )

    return df


def add_season_zone_interactions(df):
    """
    Add simple seasonal interaction labels with regional climate context.

    These categorical features test whether the same station/climate region behaves
    differently across broader seasonal regimes.
    """
    df = df.copy()
    if 'date' not in df.columns:
        return df

    month = pd.to_datetime(df['date']).dt.month
    season_map = {
        12: 'summer', 1: 'summer', 2: 'summer',
        3: 'autumn', 4: 'autumn', 5: 'autumn',
        6: 'winter', 7: 'winter', 8: 'winter',
        9: 'spring', 10: 'spring', 11: 'spring',
    }
    season_label = month.map(season_map)
    df['season_label'] = season_label

    if 'ncc_zone' in df.columns:
        df['season_ncc_zone'] = season_label.astype(str) + '_zone_' + df['ncc_zone'].fillna(-1).astype(int).astype(str)

    if 'koppen_zone' in df.columns:
        df['season_koppen_zone'] = season_label.astype(str) + '_' + df['koppen_zone'].fillna('Unknown').astype(str)

    return df


# ============================================================
# 5. MAIN PIPELINE FOR THE NOTEBOOK
# ============================================================
def build_features_pipeline(
    raw_df,
    profile='base',
    use_ncc=True,
    keep_direction_labels=True,
    use_geo=None,
    use_koppen=False,
    use_seasonal_rainfall=False,
    use_temperature_humidity=False,
    add_experimental_rollups=None,
    add_season_zone_features=None,
    missing_indicator_columns=None,
    metadata_path=DEFAULT_METADATA_PATH,
    stations_path=DEFAULT_STATIONS_PATH,
    nc_path=DEFAULT_KOPPEN_PATH,
    seasonal_rainfall_nc_path=DEFAULT_SEASONAL_RAINFALL_PATH,
    temperature_humidity_nc_path=DEFAULT_TEMPERATURE_HUMIDITY_PATH,
):
    """
    Shared feature-building pipeline for the project notebook track.

    Profiles:
    - base: the original notebook 02 feature set
    - aligned: the aligned geographic/climate extension from notebook 05
    - experimental: the aligned pipeline plus longer-memory and season-zone extras
    """
    if profile not in {'base', 'aligned', 'experimental'}:
        raise ValueError("profile must be 'base', 'aligned', or 'experimental'")

    if use_geo is None:
        use_geo = profile in {'aligned', 'experimental'}

    if add_experimental_rollups is None:
        add_experimental_rollups = profile == 'experimental'

    if add_season_zone_features is None:
        add_season_zone_features = profile == 'experimental'

    if missing_indicator_columns is None:
        missing_indicator_columns = (
            ALIGNED_MISSINGNESS_COLUMNS if profile == 'aligned' else BASE_MISSINGNESS_COLUMNS
        )

    df = rename_cols(raw_df)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['location', 'date']).reset_index(drop=True)

    if use_geo:
        df = add_coordinates(df, metadata_path=metadata_path, stations_path=stations_path)

    if use_ncc:
        df = add_ncc_zones(df)

    if use_geo:
        df = add_longitude_strips(df)

    if use_koppen:
        df = add_koppen_zone(df, nc_path=nc_path)

    if use_seasonal_rainfall:
        df = add_seasonal_rainfall_zone(df, nc_path=seasonal_rainfall_nc_path)

    if use_temperature_humidity:
        df = add_temperature_humidity_zone(df, nc_path=temperature_humidity_nc_path)

    df = add_missing_indicators(df, missing_indicator_columns)

    # main engineered features
    df = add_daily_diffs(df, ['humidity', 'pressure', 'temp', 'wind_speed', 'cloud'])
    df = add_overnight_diffs(df, ['humidity', 'pressure', 'temp'])
    df = add_yesterday_lag(df, ['rainfall', 'pressure_3pm', 'cloud_3pm'])
    df = add_dewpoint_features(df)
    df = encode_wind_direction(df, drop_original=not keep_direction_labels)
    df = add_wind_shift(df)
    df = add_trend_indicators(df)
    df = add_cyclical_time_features(df)
    df = add_temperature_structure_features(df)
    df = add_pressure_moisture_features(df)
    df = add_location_lag_rollups(df)

    if add_experimental_rollups:
        df = add_feature_experiment_rollups(df)

    if add_season_zone_features:
        df = add_season_zone_interactions(df)

    return df


# backward-compatible alias
def make_model_ready_dataset(raw_df, add_zone='ncc', drop_original_direction=False, profile='base'):
    use_ncc = (add_zone == 'ncc')
    keep_direction_labels = not drop_original_direction
    return build_features_pipeline(
        raw_df,
        profile=profile,
        use_ncc=use_ncc,
        keep_direction_labels=keep_direction_labels
    )
