import pandas as pd
import numpy as np
import os
from pathlib import Path

try:
    import xarray as xr
except ImportError:  # pragma: no cover
    xr = None

def rename_cols(df):
    df=df.copy()
    #rename variables
    dictionary={
        'Date':'date',
        'Location':'location',
        'MinTemp':'min_temp',
        'MaxTemp':'max_temp',
        'Rainfall':'rainfall',
        'Evaporation':'evaporation',
        'Sunshine':'sunshine',
        'WindGustDir':'wind_gust_dir',
        'WindGustSpeed':'wind_gust_speed',
        'WindDir9am':'wind_dir_9am',
        'WindDir3pm':'wind_dir_3pm',
        'WindSpeed9am':'wind_speed_9am',
        'WindSpeed3pm':'wind_speed_3pm',
        'Humidity9am':'humidity_9am',
        'Humidity3pm':'humidity_3pm',
        'Pressure9am':'pressure_9am',
        'Pressure3pm':'pressure_3pm',
        'Cloud9am':'cloud_9am',
        'Cloud3pm':'cloud_3pm',
        'Temp9am':'temp_9am',
        'Temp3pm':'temp_3pm',
        'RainToday':'rain_today',
        'RainTomorrow':'rain_tomorrow'
    }
    df=df.rename(columns=dictionary)
    return df

# !!!! all functions assume the columns were renamed to snake_case !!!!

def get_metadata(src_path='references/stations.txt', target_path='data/preprocessed/locations_metadata.csv'):
    """
    Gets the metadata for the locations of the Australian weather dataframe
    and saves them in a csv file.
    Metadata: Location, Site ID, Elevation, Latitude, Longitude, 
    Koppen Zone, Seasonal Rainfall Zone, Temperature-Humidity Zone
    Arguments: src_path: The stations file
               target_path: The file to save the metadata to
    The stations file has to be downloaded manually from the BoM website, 
    since automated access is blocked:
    https://www.bom.gov.au/climate/data/lists_by_element/stations.txt
    Same for the climate grid files from: 
    https://www.bom.gov.au/climate/maps/averages/climate-classification
    """
    if not Path(src_path).exists():
        raise FileNotFoundError(f"File not found: {src_path}")
    df_bom = pd.read_fwf(src_path, skiprows=2, header=0, na_values=['..','.....'])

    #only keep stations that were still active after 2017 
    #or whose 'End' column is empty (NaN) (= still open).
    df_bom['End'] = pd.to_numeric(df_bom['End'], errors='coerce')
    mask = (df_bom['WMO'].notna()) & (df_bom['WMO'] != 0) & \
        ((df_bom['End'] >= 2017) | (df_bom['End'].isna()))
    df_active = df_bom[mask].copy()

    df_bom['Start'] = pd.to_numeric(df_bom['Start'], errors='coerce')
    df_bom['End'] = pd.to_numeric(df_bom['End'], errors='coerce')
    df_bom['WMO'] = pd.to_numeric(df_bom['WMO'], errors='coerce')

    #filter for high-quality, active stations
    #must have a WMO ID AND (be active until 2015 OR currently open)
    mask = (df_bom['WMO'].notna()) & (df_bom['WMO'] != 0) &\
        ((df_bom['End'] >= 2015) | (df_bom['End'].isna()))
    df_active = df_bom[mask].copy()

    #list of 49 locations
    locations = [
        "Adelaide", "Albany", "Albury", "AliceSprings", "BadgerysCreek", 
        "Ballarat", "Bendigo", "Brisbane", "Cairns", "Canberra", 
        "Cobar", "CoffsHarbour", "Dartmoor", "Darwin", "GoldCoast", 
        "Hobart", "Katherine", "Launceston", "Melbourne", "MelbourneAirport", 
        "Mildura", "Moree", "MountGambier", "MountGinini", "Newcastle", 
        "Nhil", "NorahHead", "NorfolkIsland", "Nuriootpa", "PearceRAAF", 
        "Penrith", "Perth", "PerthAirport", "Portland", "Richmond", 
        "Sale", "SalmonGums", "Sydney", "SydneyAirport", "Townsville", 
        "Tuggeranong", "Uluru", "WaggaWagga", "Walpole", "Watsonia", 
        "Williamtown", "Witchcliffe", "Wollongong", "Woomera"
    ]

    #station names found manually on bom.gov.au
    name_fixes = {
        "Adelaide": "ADELAIDEAIRPORTM.O.",
        "Cobar": "COBARAIRPORTAWS",
        "Hobart": "HOBARTAIRPORTWEST",
        "Launceston": "LAUNCESTONAIRPORT",
        "Melbourne": "MELBOURNE(OLYMPICPARK)",
        "Nhil": "NHILLAERODROME",
        "Watsonia": "VIEWBANK",
        "Katherine": "TINDAL",
        "Richmond": "RICHMONDRAAF",
        "Uluru": "YULARAAIRPORT",
        "GoldCoast": "GOLDCOASTSEAWAY",
        "Perth": "PERTHMETRO",
        "Sale": "EASTSALE",
        "Walpole": "NORTHWALPOLE",
        "Sydney": "SYDNEY(OBSERVATORYHILL)"
    }

    #find matching entries
    metadata_list = []
    for loc in locations:
        #search the location in name_fixes otherwise use loc
        search_term = name_fixes.get(loc, loc).upper() #remove spaces and convert to uppercase

        #search for the name in the stations dataframe (without spaces)
        match = df_active[df_active['Site name'].str.replace(' ', '').str.startswith(search_term)]

        if not match.empty:
            #extract first row of the series (oldest station)
            best_match = match.sort_values(by='Start', ascending=True).iloc[0]
            metadata_list.append({
                'location': loc,
                'site_id': best_match['Site'],
                'elevation': best_match['Height (m)'],
                'lat': best_match['Lat'],
                'lon': best_match['Lon']
            })
        else:
            #warning if a location is not found in the filtered BoM list
            print(f"Warning: Location '{loc}' not found in active station list!")

    #convert into df
    geo_df = pd.DataFrame(metadata_list)
    #convert into numeric values
    geo_df[['lat', 'lon']] = geo_df[['lat', 'lon']].apply(pd.to_numeric, errors='coerce')

    #get climate zones
    configs = {
        'koppen': {
            'nc_path': 'references/koppen_zones_grid.nc',
            'grid_var': 'stern_dehoedt_2000_major',
            'legend_var': 'class long description',
            'legend_dim': 'class_code',
            'zone_col_name': 'koppen_zone'
        },
        'seasonal': {
            'nc_path': 'references/seasonal_rainfall_zones_grid.nc',
            'grid_var': 'rainfall_zone_numeric',
            'legend_var': 'rainfall_zone_description',
            'legend_dim': 'rz',
            'zone_col_name': 'rainfall_zone'
        },
        'thermal': {
            'nc_path': 'references/temperature_humidity_zones_grid.nc',
            'grid_var': 'temperature-humidity-zone-code',
            'legend_var': 'temp_humidity_zone_description',
            'legend_dim': 'thz',
            'zone_col_name': 'thermal_zone'
        }
    }
    for z in configs:
        geo_df = get_zone_from_grid(geo_df, **configs[z])

    #save as csv
    Path(target_path).parent.mkdir(parents=True, exist_ok=True)
    geo_df.to_csv(target_path, index=False)

    print(f"\nMetadata for {len(geo_df)} locations saved to {target_path}.")
    return target_path


def add_coordinates(df, metadata_path='data/preprocessed/locations_metadata.csv'):
    """
    Merges latitude and longitude from a metadata CSV into the main DataFrame.
    """
    df=df.copy()
    required = ['lat', 'lon', 'elevation']
    # Check which of the required columns are actually missing
    missing = [col for col in required if col not in df.columns]

    if not missing:
        return df # Nothing to do, everything is already there
    
    if not Path(metadata_path).exists():
        print("Locations metadata file not found. Generating it now (this may take a moment)...")
        metadata_path = get_metadata()
    meta_df = pd.read_csv(metadata_path)
    
    #left merge: keep all rows from df, add coords where location matches
    df = df.merge(meta_df[['location'] + missing], on='location', how='left')
    
    return df


def add_ncc_zones(df):
    '''Assigns climate zones to all locations in a new column "ncc_zone"'''
    df=df.copy()
    #assign climate zones
    #see https://www.abcb.gov.au/abcb-climate-map
    zones={ "Albury":4, "BadgerysCreek":6, "Cobar":4, "CoffsHarbour":2, "Moree":4, "Newcastle":5, "NorahHead":5,
            "NorfolkIsland":2, "Penrith":6, "Richmond":6, "Sydney":5, "SydneyAirport":5, "WaggaWagga":4,
            "Williamtown":5, "Wollongong":5, "Canberra":7, "Tuggeranong":7, "MountGinini":8,
            "Ballarat":7, "Bendigo":6, "Sale":6, "MelbourneAirport":6, "Melbourne":6, "Mildura":4,
            "Nhil":4, "Portland":6, "Watsonia":6, "Dartmoor":6, "Brisbane":2, "Cairns":1, "GoldCoast":2,
            "Townsville":1, "Adelaide":5, "MountGambier":6, "Nuriootpa":6, "Woomera":4, "Albany":6,
            "Witchcliffe":5, "PearceRAAF":5, "PerthAirport":5, "Perth":5, "SalmonGums":4, "Walpole":6,
            "Hobart":7, "Launceston":7, "AliceSprings":3, "Darwin":1, "Katherine":1, "Uluru":3
    } #Norfork Island is in the pacific and not on mainland of australia
    df['ncc_zone']=df['location'].map(zones)
    return df


def get_zone_from_grid(df, nc_path, grid_var, legend_var, legend_dim, zone_col_name):
    """
    Enriches the DataFrame with climate zones by performing a spatial 
    lookup in a NetCDF grid. Includes a proximity search for coastal locations 
    to prevent 'Unknown' values caused by land-sea masks.

    The function dynamically handles different BoM (Bureau of Meteorology) 
    grid structures and applies internal cleaning for inconsistent IDs 
    and verbose descriptions.

    Args:
        df (pd.DataFrame): DataFrame containing 'location', 'lat', and 'lon' columns.
        nc_path (str): Path to the official BoM NetCDF file (.nc).
        grid_var (str): The specific variable name within the NetCDF that contains 
                        the spatial data (e.g., 'rainfall_zone_numeric').
        legend_var (str): The variable name containing the descriptive text for 
                          the zones (e.g., 'rainfall_zone_description').
        legend_dim (str): The dimension name used to index the legend 
                          (e.g., 'rz', 'thz', or 'class_code').
        zone_col_name (str): The name of the new column to be added to the 
                             resulting DataFrame (e.g., 'koppen_zone').

    Returns:
        pd.DataFrame: A copy of the input DataFrame with the new zone column 
                      merged based on the location.
    """

    if xr is None:
        raise ImportError(
            "xarray is required to read the BoM NetCDF climate grids. "
            "Install xarray before running get_zone_from_grid or get_metadata."
        )

    geo_df = df[['location', 'lat', 'lon']].drop_duplicates().copy()
    ds = xr.open_dataset(nc_path)

    # Mapping for Seasonal Rainfall
    rain_map = {
        100: "Summer dominant", 101: "Summer", 102: "Uniform", 
        103: "Winter", 104: "Winter dominant", 105: "Arid"
    }

    # Mapping for Temp/Hum
    temp_hum_map = {'hd_cW': 'hdScW', 'hd_wW': 'hdSwW'}

    # Create the Legend Dictionary
    ids = ds[legend_dim].values
    raw_names = [n.decode('utf-8') if isinstance(n, bytes) else str(n) for n in ds[legend_var].values]
    legend_dict = dict(zip(ids, raw_names))

    if "seasonal_rainfall" in nc_path:
        legend_dict.update(rain_map)

    def get_zone_safe(row):
        # Initial lookup
        val = ds[grid_var].sel(lat=row['lat'], lon=row['lon'], method='nearest').item()
    
        if np.isnan(val) or (isinstance(val, (int, float)) and val < 0):
            # take a small window around the coordinate (approx. 0.3 degrees)
            # and look for the first value there that is NOT NaN
            subset = ds[grid_var].sel(
                lat=slice(row['lat'] - 0.3, row['lat'] + 0.3), 
                lon=slice(row['lon'] - 0.3, row['lon'] + 0.3)
            )
            #take the most common value from this map section
            valid_vals = pd.Series(subset.values.flatten()).dropna()
            val = valid_vals.mode().iloc[0] if not valid_vals.empty else np.nan
        
        if np.isnan(val):
            return "Unknown"
        
    # Handle string vs numeric IDs for dictionary lookup
        lookup_id = int(val)

        if "temperature_humidity" in nc_path:
            raw_label = ids[lookup_id]
            return temp_hum_map.get(raw_label, raw_label)
        return legend_dict.get(lookup_id, "Unknown")

    # Apply to the unique locations
    geo_df[zone_col_name] = geo_df.apply(get_zone_safe, axis=1)
    ds.close()

    # Manual Fixes for offshore locations
    fixes = {'koppen_zone': 'Subtropical', 'rainfall_zone': 'Summer', 'thermal_zone': 'whSmW'}
    if zone_col_name in fixes:
        geo_df.loc[geo_df['location'] == 'NorfolkIsland', zone_col_name] = fixes[zone_col_name]

    # Merge results back to main dataframe
    return df.merge(geo_df[['location', zone_col_name]], on='location', how='left')


def add_bom_zones(df, zones=['koppen', 'rainfall', 'thermal'], metadata_path='data/preprocessed/locations_metadata.csv'):
    """
    Enriches the DataFrame with pre-calculated climate zones from the metadata master file.
    This function avoids expensive NetCDF grid lookups by reading from a processed CSV.

    Args:
        df (pd.DataFrame): The main weather DataFrame containing a 'location' column.
        zones (list): The classifications to add. Options:
                      - 'koppen': Köppen-Geiger climate zones.
                      - 'rainfall': Seasonal rainfall regimes (e.g., Summer dominant, Arid).
                      - 'thermal': Temperature & humidity short codes (e.g., hdSwW).
        metadata_path (str): Path to the CSV containing pre-calculated location features.

    Returns:
        pd.DataFrame: The DataFrame with the requested zone columns added.
    """
    if not Path(metadata_path).exists():
        print("Metadata file not found. Generating it now (this may take a moment)...")
        get_metadata(target_path=metadata_path)

    meta_df = pd.read_csv(metadata_path)
    mapping = {
        'koppen': 'koppen_zone',
        'rainfall': 'rainfall_zone',
        'thermal': 'thermal_zone'
    }
    #get only zones that do not exist yet
    zones_to_add = [z for z in zones if mapping[z] not in df.columns]
    if not zones_to_add:
        return df
    cols_to_keep = ['location'] + [mapping[z] for z in zones_to_add]
    return df.merge(meta_df[cols_to_keep], on='location', how='left')


def add_climatic_strips(df):
    """
    Categorizes locations into longitudinal strips to better represent 
    coastal vs. inland climate differences.
    """
    df = df.copy()
    #define strips based on longitude
    conditions = [
        (df['lon'] < 125),
        (df['lon'] >= 125) & (df['lon'] < 140),
        (df['lon'] >= 140) & (df['lon'] < 148),
        (df['lon'] >= 148)
    ]
    values = ['West_Coast', 'Central_Arid', 'East_Inland', 'East_Coast']
    
    df['land_strip'] = np.select(conditions, values, default='Unknown')
    
    return df


def fill_with_zonal_stats(df, reference_df, columns, strategy='median', threshold=0.2):
    """
    Fills NaNs in specific columns based on zonal statistics from a reference dataframe.
    Automatically adds a Missing Indicator if the NaN-rate is above the threshold.
    Reports locations that remain empty.
    Handles Zone 8 (Alpine) by using Zone 7 as a proxy.
    Strategy can be 'median', 'mean', or 'mode'.
    """
    df = df.copy()
    if isinstance(columns, str):
        columns = [columns]

    for col in columns:
        if col not in df.columns:
            print(f"Skipping '{col}': Column not found.")
            continue

        #calculate the missing rate (NaN-rate)
        nan_rate = reference_df[col].isnull().mean()

        #ddd indicator if missing rate > threshold
        if nan_rate > threshold:
            print(f"Adding indicator for '{col}' (Missing Rate: {nan_rate:.1%})")
            df[f"{col}_missing"] = df[col].isnull().astype(int)
        else:
            print(f"No indicator for '{col}' (Missing Rate: {nan_rate:.1%} < {threshold:.1%})")

        #define the strategy for calculating zonal stats
        if strategy == 'median':
            zonal_stats = reference_df.groupby('ncc_zone')[col].median()
        elif strategy == 'mean':
            zonal_stats = reference_df.groupby('ncc_zone')[col].mean()
        elif strategy == 'mode':
            # Mode can return multiple values, so we take the first one [.iloc[0]]
            zonal_stats = reference_df.groupby('ncc_zone')[col].apply(
                lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
            )
        # Manually link Zone 8 to Zone 7 if Zone 8 is missing
        if 8 not in zonal_stats.index or pd.isna(zonal_stats.get(8)):
            if 7 in zonal_stats.index:
                zonal_stats[8] = zonal_stats[7]
                print(f"Note: Zone 8 inherited stats from Zone 7 for '{col}'.")

        #map and fill based on zone
        df[col] = df[col].fillna(df['ncc_zone'].map(zonal_stats))
            
        #report locations that are still NaN for this column
        remaining = df[df[col].isnull()]['location'].unique()
        if len(remaining) > 0:
            print(f"INFO: '{col}' still missing in: {remaining}")
    return df

def fill_with_daily_zonal_stats(df, reference_df, columns, zone_col='rainfall_zone', strategy='median', threshold=0.1):
    """
    Fills NaNs in specific columns based on DAILY zonal statistics from a reference dataframe.
    Adds missing indicator if missing rate > threshold
    Parameters:
    - zone_col: The grouping column to use (e.g., 'ncc_zone', 'koppen_zone', 'location').
    - strategy: 'median', 'mean', or 'mode'.
    """

    def get_stats(group_obj, strat, min_samples=10):
        if strat == 'mean': 
            res = group_obj.mean()
        elif strat == 'mode': 
            res = group_obj.apply(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        else: 
            res = group_obj.median()
        
        count = group_obj.count()
        res[count < min_samples] = np.nan
        return res
    
    df = df.copy()
    reference_df = reference_df.copy()

    if isinstance(columns, str):
        columns = [columns]

    # Ensure date is datetime for proper mapping
    df['date'] = pd.to_datetime(df['date'])
    reference_df['date'] = pd.to_datetime(reference_df['date'])

    for col in columns:
        if col not in df.columns:
            print(f"Skipping '{col}': Column not found.")
            continue

        nan_mask = df[col].isnull()
        # Calculate the missing rate (NaN-rate)
        nan_rate = reference_df[col].isnull().mean()
        # Add indicator if missing rate > threshold
        if nan_rate > threshold:
            df[f"{col}_missing"] = df[col].isnull().astype(int)

        # Define the strategy for calculating DAILY zonal stats
        # We group by both: the Zone AND the specific Date
        daily_stats = get_stats(reference_df.groupby([zone_col, 'date'])[col], strategy)

        if zone_col == 'ncc_zone':
            try:
                # all rows of zone 7
                s7 = daily_stats.xs(7, level=zone_col)
                # new series for zone 8 with the same data
                s8 = pd.Series(s7.values, index=pd.MultiIndex.from_product([[8], s7.index], names=[zone_col, 'date']))
                
                daily_stats = daily_stats.combine_first(s8)
            except KeyError:
                pass

        # fill daily zone stats
        df = df.set_index([zone_col, 'date'])
        df[col] = df[col].fillna(daily_stats)
        df = df.reset_index()

        # SEASONAL SAFETY NET (If daily failed)
        if df[col].isnull().any():
            df['month'] = df['date'].dt.month
            reference_df['month'] = reference_df['date'].dt.month

            seasonal_stats = get_stats(reference_df.groupby([zone_col, 'month'])[col], strategy)

            if zone_col == 'ncc_zone' and 7 in seasonal_stats.index:
                if 8 not in seasonal_stats.index or pd.isna(seasonal_stats.loc[8]).any():
                    # Create entry for Zone 8 using Zone 7 values
                    seasonal_stats.loc[8] = seasonal_stats.loc[7]

            df = df.set_index([zone_col, 'month'])
            df[col] = df[col].fillna(seasonal_stats)
            df = df.reset_index()
            df = df.drop(columns=['month']) # Clean up

            # Bound certain values physically (e.g., humidity can't be > 100 or < 0)
            if 'humidity' in col:
                df[col] = df[col].clip(0, 100)
            elif 'sunshine' in col:
                df[col] = df[col].clip(0, 16) # Max sunshine hours approx 16
            elif 'cloud' in col:
                df[col] = df[col].clip(0, 8)

        # Report locations that are still NaN
        remaining = df[df[col].isnull()]['location'].unique()
        if len(remaining) > 0:
            print(f"INFO: '{col}' still missing in: {remaining}")

    return df


def add_daily_diffs(df, prefixes=['humidity', 'temp', 'wind_speed', 'pressure', 'cloud']):
    """
    Calculates the change during the day (3pm - 9am) for a given list of features.
    df: The DataFrame containing the features.
    prefixes: The features to calculate the difference for.
              Accepts a single string or a list of strings.
              Possible values:
              ['humidity', 'temp', 'wind_speed', 'pressure', 'cloud']
    """
    df=df.copy()
    if isinstance(prefixes,str):
        prefixes = [prefixes]
    for p in prefixes:
        col_3pm = f"{p}_3pm"
        col_9am = f"{p}_9am"
        if col_3pm in df.columns and col_9am in df.columns:
            df[f"{p}_day_diff"] = df[col_3pm] - df[col_9am]
        else:
            print(f"Note: Could not create diff for '{p}' (Columns not found).")
    return df


def add_overnight_diffs(df, prefixes=['humidity', 'temp', 'wind_speed', 'pressure', 'cloud']):
    """
    Calculates the change from yesterday 3pm to today 9am for a given list of features.
    Uses a time-safe merge to ensure 'yesterday' is actually the previous day.
    df: The DataFrame containing the features.
    prefixes: The features to calculate the difference for.
              Accepts a single string or a list of strings.
              Possible values:
              ['humidity', 'temp', 'wind_speed', 'pressure', 'cloud']
    """
    df = df.copy() 
    df = datesplit(df)
    df = basic_clean(df)
    if isinstance(prefixes, str):
        prefixes = [prefixes]
    for p in prefixes:
        col_9am = f"{p}_9am"
        col_3pm = f"{p}_3pm"
        if col_9am in df.columns and col_3pm in df.columns:
            # Build mapping for yesterday's 3pm values.
            temp = df[['date', 'location', col_3pm]].copy()
            #shift the date forward by 1 day so it matches 'today'
            temp['date'] = temp['date'] + pd.Timedelta(days=1)
            temp = temp.rename(columns={col_3pm: f"{p}_3pm_yest"})
            
            #merge the yesterday values onto the current dataframe
            df = pd.merge(df, temp, on=['date', 'location'], how='left')
            
            #calculate the difference
            df[f"{p}_overnight_diff"] = df[col_9am] - df[f"{p}_3pm_yest"]
            
            # Remove the helper column after the difference is computed.
            df = df.drop(columns=[f"{p}_3pm_yest"])
        else:
            print(f"Note: Could not create overnight diff for '{p}'(Columns not found).")
    return df


def add_yesterday_lag(df, columns=['humidity_3pm','cloud_3pm','max_temp','min_temp','pressure_3pm','rain_today','rainfall']):
    """
    Creates a lag of 1 day for a given list of columns.
    Uses a time-safe merge to ensure 'yesterday' is actually the previous day.
    df: The DataFrame containing the features.
    columns: The features to calculate the difference for.
              Accepts a single string or a list of strings.
    """
    df = df.copy()

    if isinstance(columns, str):
        columns = [columns]
        
    for col in columns:
        # Build a shifted-date helper frame for yesterday's values.
        temp = df[['date', 'location', col]].copy()
        #shift 'date' forward by 1 day so it matches 'today'
        temp['date'] = temp['date'] + pd.Timedelta(days=1)
        
        #rename the column to indicate it is the value from yesterday
        new_col_name = f"{col}_yest"
        temp = temp.rename(columns={col: new_col_name})
        
        #merge the yesterday value onto the original dataframe
        df = pd.merge(df, temp, on=['date', 'location'], how='left')
        
        #print status for the user
        print(f"Created feature: {new_col_name}")
    return df


def add_dewpoint_features(df):
    """
    Calculates Dewpoint and Dewpoint Spread for 9am and 3pm using the Magnus formula.
    Spread = Temperature - Dewpoint.
    """
    df = df.copy()
    #magnus formula constants
    a = 17.27
    b = 237.7
    
    for time in ['9am', '3pm']:
        t_col = f'temp_{time}'
        h_col = f'humidity_{time}'
        dp_col = f'dewpoint_{time}'
        s_col = f'dewpoint_spread_{time}'
        
        #check if both required columns exist
        if t_col in df.columns and h_col in df.columns:
            #prepare safe humidity (avoid log(0) or negative values)
            safe_h = df[h_col].clip(lower=0.01, upper=100.0)
            #compute the alpha term
            alpha = ((a * df[t_col]) / (b + df[t_col])) + np.log(safe_h / 100.0)
            #compute Dewpoint
            df[dp_col] = (b * alpha) / (a - alpha)
            #compute Spread (How close is the air to saturation?)
            df[s_col] = df[t_col] - df[dp_col]
    
            print(f"Features created: {dp_col} and {s_col}")
        else:
            print(f"Skipping {time}: Required columns {t_col} or {h_col} not found.")
            
    return df

def add_24h_diff(df, columns=['evaporation','min_temp','max_temp','pressure_3pm','humidity_3pm',
                              'wind_gust_speed','wind_dir_3pm','cloud_3pm','temp_3pm','dewpoint_spread_3pm','dewpoint_3pm']):
    df = df.copy() 
    df['date'] = pd.to_datetime(df['date'])
    for col in columns: 
        if col in df.columns:
            # Build mapping for yesterday's values.
            temp = df[['date', 'location', col]].copy()
            #shift the date forward by 1 day so it matches 'today'
            temp['date'] = temp['date'] + pd.Timedelta(days=1)
            temp = temp.rename(columns={col: f"{col}_yest"})

            #merge the yesterday values onto the current dataframe
            df = pd.merge(df, temp, on=['date', 'location'], how='left')
            #calculate the difference
            df[f"{col}_24h_diff"] = df[col] - df[f"{col}_yest"]

            # Remove the helper column after the difference is computed.
            df = df.drop(columns=[f"{col}_yest"])
        else:
            print(f"Note: Could not create 24h diff for '{col}'(Column not found).")
    return df

def add_wind_shift(df):
    """ Calculates wind shift score (dot product of x/y components).
        Assumes x/y components exist. 
        1.0= same direction, -1.0= opposite direction, 0= 90 degree shift
    """
    df = df.copy()
    cols = ['wind_dir_9am_x', 'wind_dir_3pm_x', 'wind_dir_9am_y', 'wind_dir_3pm_y']
    if all(c in df.columns for c in cols):
        df['wind_shift_score'] = (df['wind_dir_9am_x'] * df['wind_dir_3pm_x'] + 
                                  df['wind_dir_9am_y'] * df['wind_dir_3pm_y'])
    return df

def basic_clean(df):
    #
    df=df.copy()
    df=rename_cols(df)
    df['date']= pd.to_datetime(df['date'])
    #replace yes/no with 1 and 0
    df[['rain_today','rain_tomorrow']]=df[['rain_today','rain_tomorrow']].replace(to_replace=['Yes','No'], value=[1,0]).astype('float')
    #drop rows with missing target(rain) values 
    #df=df.dropna(subset=['rain_tomorrow'])
    #df['rain_tomorrow']=df['rain_tomorrow'].astype(int)
    return df

def datesplit(df):
    """Splits the date into the columns year, month, day."""
    df=df.copy()
    df=rename_cols(df)
    df['date'] = pd.to_datetime(df['date'])
    df['year']=df['date'].dt.year
    df['month']=df['date'].dt.month 
    df['day']=df['date'].dt.day
    return df  

def add_circular_features(df, features=['month', 'day', 'wind', 'year_cycle']):
    df = df.copy()
    if "month" in features:
        #map months to sin/cos to preserve the circular nature of time
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    if "day" in features:
        #map days to sin/cos to preserve the circular nature of time
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)

    if "wind" in features:
        #adding wind directions
        wd_map = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 
            'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
            'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
            'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
        }
        wind_cols = ['wind_dir_9am', 'wind_dir_3pm', 'wind_gust_dir']

        for col in wind_cols:
            #map cardinal directions to degrees
            df[f'{col}_deg'] = df[col].map(wd_map)
            #convert to Radians and then to X/Y unit vectors
            rads = np.radians(df[f'{col}_deg'])
            df[f'{col}_y'] = np.cos(rads)
            df[f'{col}_x'] = np.sin(rads)
            #drop string columns and degree columns to avoid circular data issues in the model
            df = df.drop(columns=[col, f'{col}_deg'])

    # Seasonal Cycle (Replaces Month & Day)
    if "year_cycle" in features:
        # Get day of year (1-365/366)
        day_of_year = df['date'].dt.dayofyear
        
        # Use 365.25 to account for leap years
        df['year_cycle_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
        df['year_cycle_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)
        
        # Drop month/day if they exist
        #cols_to_drop = [c for c in ['month', 'day'] if c in df.columns]
        #df = df.drop(columns=cols_to_drop)
    return df

def new_raw_df(path='../data/raw/weatherAUS.csv'):
    df=pd.read_csv(path)
    return df

def selected_features(df):
    df = df.copy()
    df = basic_clean(df)
    df = add_circular_features(df, features=['wind', 'year_cycle'])
    df = add_coordinates(df)
    df = add_bom_zones(df, zones=['koppen'])

    num_cols = [
    'min_temp', 'max_temp', 'temp_9am', 'temp_3pm','sunshine', 'wind_gust_speed', 'wind_speed_3pm', 'humidity_9am',
    'humidity_3pm', 'pressure_9am', 'pressure_3pm', 'wind_dir_9am_y', 'wind_dir_3pm_y',
    'wind_dir_3pm_x', 'wind_gust_dir_y', 'wind_gust_dir_x'
    ]
    cat_cols = [ 'cloud_9am', 'cloud_3pm']

    train, test = split_chronological(df)
    train = fill_with_daily_zonal_stats(train, train, columns=num_cols, zone_col='koppen_zone', threshold=1)
    test = fill_with_daily_zonal_stats(test, train, columns=num_cols, zone_col='koppen_zone', threshold=1)
    train = fill_with_daily_zonal_stats(train, train, columns=cat_cols, zone_col='koppen_zone', threshold=1, strategy='mode')
    test = fill_with_daily_zonal_stats(test, train, columns=cat_cols, zone_col='koppen_zone', threshold=1, strategy='mode')

    processed_sets = []
    for d in [train, test]:
        d = add_daily_diffs(d, prefixes=['pressure'])
        d = add_dewpoint_features(d)
        d['temp_range'] = d['max_temp'] - d['min_temp']
        d = add_24h_diff(d, columns=['pressure_3pm','humidity_3pm','wind_gust_speed'])
        d = d.drop(['humidity_9am','dewpoint_3pm','dewpoint_9am','evaporation',
                    'temp_9am','wind_speed_9am','temp_3pm','max_temp'],axis=1)
        d=d.dropna(subset=['rain_tomorrow','rain_today','rainfall'])
        d['rain_tomorrow']=d['rain_tomorrow'].astype(int)
        d = pd.get_dummies(d,columns=['koppen_zone'], drop_first=True)
        d = d.drop(['location','date'],axis=1)
        d = d.dropna()
        processed_sets.append(d)
    return processed_sets[0], processed_sets[1] # Return train, test

def split_chronological(df, test_size=0.2):
    df = df.copy()
    #sorting chronologically
    df = df.sort_values('date')

    #save lists of location dfs here:
    train_list = []
    test_list = []

    for location in df['location'].unique():
        location_df = df[df['location'] == location]
        
        # calculate split point 80/20
        split_idx = int(len(location_df) * (1-test_size))

        #append to lists:
        #all rows before splitting point (80%) = train base
        train_list.append(location_df.iloc[:split_idx])
        #all rows from splitting point (20%) = test base
        test_list.append(location_df.iloc[split_idx:])

    #combine everything back together
    train = pd.concat(train_list)
    test = pd.concat(test_list)

    return train,test

def build_all_features(df):
    """
    Builds all features from the raw weather DataFrame.

    Pipeline order:
    1. Basic cleaning (rename cols, parse dates, encode rain_today/tomorrow)
    2. Add static location features (coordinates, zones)
    3. Add circular/temporal features (wind direction, year cycle)
    4. Impute NaNs on raw columns FIRST (daily zone mean → monthly zone mean)
    5. Create derived features AFTER imputation (diffs, dewpoint, lag, wind shift)
    6. Encode categorical columns
    7. Chronological train/test split

    Returns:
        train (pd.DataFrame), test (pd.DataFrame)
    """
    df = df.copy()

    # -------------------------------------------------------------------------
    # STEP 1: Basic cleaning
    # -------------------------------------------------------------------------
    df = basic_clean(df)          # rename cols, parse date, encode rain yes/no
    df = datesplit(df)            # adds year, month, day columns (needed for imputer)

    # -------------------------------------------------------------------------
    # STEP 2: Static location features (no NaNs introduced here)
    # -------------------------------------------------------------------------
    df = add_coordinates(df)      # lat, lon, elevation
    df = add_ncc_zones(df)        # ncc_zone (int)
    df = add_bom_zones(df, zones=['koppen', 'rainfall', 'thermal'])
    df = add_climatic_strips(df)  # land_strip

    # -------------------------------------------------------------------------
    # STEP 3: Circular / temporal features on raw columns
    # wind dir encoding is safe before imputation (NaN stays NaN → imputed later)
    # year_cycle only needs 'date', no NaN risk
    # -------------------------------------------------------------------------
    df = add_circular_features(df, features=['wind', 'year_cycle'])

    wind_xy_cols = [
    'wind_dir_9am_x', 'wind_dir_9am_y',
    'wind_dir_3pm_x', 'wind_dir_3pm_y',
    'wind_gust_dir_x', 'wind_gust_dir_y'
    ]

    
    # -------------------------------------------------------------------------
    # STEP 4: Chronological split BEFORE imputation
    # Critical: fit imputation stats only on train, then apply to test
    # -------------------------------------------------------------------------
    train, test = split_chronological(df)

    # Fill the wind x/y columns with 0 (= ‘no wind / unknown direction’)
    wind_xy_cols = [
    'wind_dir_9am_x', 'wind_dir_9am_y',
    'wind_dir_3pm_x', 'wind_dir_3pm_y',
    'wind_gust_dir_x', 'wind_gust_dir_y'
    ]

    for split in [train, test]:
        wind_present = [c for c in wind_xy_cols if c in split.columns]
        split[wind_present] = split[wind_present].fillna(0)

    # Columns to impute — all raw numerical/categorical columns that may have NaNs
    # These must exist BEFORE derived features are created
    num_cols = [
        'min_temp', 'max_temp',
        'temp_9am', 'temp_3pm',
        'humidity_9am', 'humidity_3pm',
        'pressure_9am', 'pressure_3pm',
        'wind_gust_speed', 'wind_speed_9am', 'wind_speed_3pm',
        'sunshine', 'evaporation', 'rainfall',
        'dewpoint_9am', 'dewpoint_3pm',       # if already present from a prior run
    ]
    cat_cols = ['cloud_9am', 'cloud_3pm']

    # Only keep columns that actually exist in the dataframe
    num_cols = [c for c in num_cols if c in train.columns]
    cat_cols = [c for c in cat_cols if c in train.columns]

    # Impute: daily zone mean → monthly zone mean (fit on train, apply to both)
    for zone_col in ['ncc_zone']:
        train = fill_with_daily_zonal_stats(
            train, train,
            columns=num_cols,
            zone_col=zone_col,
            threshold=0,
            strategy='median'
        )
        test = fill_with_daily_zonal_stats(
            test, train,           # <-- reference always = train
            columns=num_cols,
            zone_col=zone_col,
            threshold=0,
            strategy='median'
        )
        train = fill_with_daily_zonal_stats(
            train, train,
            columns=cat_cols,
            zone_col=zone_col,
            threshold=0,
            strategy='mode'
        )
        test = fill_with_daily_zonal_stats(
            test, train,
            columns=cat_cols,
            zone_col=zone_col,
            threshold=0,
            strategy='mode'
        )

    # -------------------------------------------------------------------------
    # STEP 5: Derived features (AFTER imputation — no NaN propagation)
    # -------------------------------------------------------------------------
    processed = []
    for split in [train, test]:

        # Day diffs: 3pm - 9am  (needs temp, humidity, pressure, wind_speed, cloud)
        split = add_daily_diffs(split, prefixes=['humidity', 'temp', 'wind_speed', 'pressure', 'cloud'])

        # Overnight diffs: today 9am - yesterday 3pm
        split = add_overnight_diffs(split, prefixes=['humidity', 'temp', 'wind_speed', 'pressure', 'cloud'])

        # Dewpoint & spread (needs temp + humidity — now imputed)
        split = add_dewpoint_features(split)

        # 24h diffs (needs dewpoint — now created above)
        split = add_24h_diff(split, columns=[
            'evaporation', 'min_temp', 'max_temp',
            'pressure_3pm', 'humidity_3pm',
            'wind_gust_speed',
            'cloud_3pm', 'temp_3pm',
            'dewpoint_spread_3pm', 'dewpoint_3pm'
        ])

        # Yesterday lag features
        split = add_yesterday_lag(split, columns=[
            'sunshine', 'wind_dir_3pm_x', 'wind_dir_3pm_y',
            'wind_gust_speed', 'humidity_3pm', 'cloud_3pm',
            'max_temp', 'min_temp', 'pressure_3pm',
            'rain_today', 'rainfall'
        ])

        # Wind shift score (needs wind x/y components)
        split = add_wind_shift(split)

        processed.append(split)

    train, test = processed

    # -------------------------------------------------------------------------
    # STEP 6: Encode categoricals + drop unused columns
    # -------------------------------------------------------------------------
    cat_encode_cols = ['ncc_zone', 'koppen_zone', 'rainfall_zone', 'thermal_zone', 'land_strip']

    for split_name, split in [('train', train), ('test', test)]:
        # Drop target NaNs (must not be imputed)
        split = split.dropna(subset=['rain_tomorrow', 'rain_today', 'rainfall'])
        split['rain_tomorrow'] = split['rain_tomorrow'].astype(int)

        # One-hot encode climate zones and location
        encode_present = [c for c in cat_encode_cols if c in split.columns]
        split = pd.get_dummies(split, columns=encode_present + ['location'], drop_first=False)

        # Bool → int (for TensorFlow compatibility)
        bool_cols = split.select_dtypes(include='bool').columns
        split[bool_cols] = split[bool_cols].astype(int)

        # Drop columns not useful for modeling
        drop_cols = [c for c in ['date', 'year', 'month', 'day'] if c in split.columns]
        split = split.drop(columns=drop_cols)

        # Final NaN drop (should be minimal — only first/last day lag edges)
        n_before = len(split)
        split = split.dropna()
        n_dropped = n_before - len(split)
        if n_dropped > 0:
            print(f"INFO ({split_name}): Dropped {n_dropped} rows with remaining NaNs (lag edges).")

        if split_name == 'train':
            train = split
        else:
            test = split

    # -------------------------------------------------------------------------
    # STEP 7: Align columns (train and test must have identical columns)
    # -------------------------------------------------------------------------
    train, test = train.align(test, join='left', axis=1, fill_value=0)

    print(f"\nDone. Train: {train.shape}, Test: {test.shape}")
    print(f"Features: {train.shape[1] - 1} (excl. target)")

    return train, test
