import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.features import build_features as ft

def zone_trends(df, group_col='koppen_zone', target_var='rainfall', time_scale='month', monthly_op='sum', fiscal_year=True, colors=None):
    """
    Universal plotting function for climate zone trends (monthly totals/averages or daily resolution).
    - df: the dataframe to plot
    - group_col: the zone column to group by
    - target_var: the variable to plot
    - time_scale: 'month' for aggregated monthly view, 'day' for 365-day year view.
    - monthly_op: 'sum' (useful for rainfall/rain_today) or 'mean' (standard for others).
    - fiscal_year: If True, starts plot in July to show the full Australian summer continuously.
    - colors: color palette
    """

    # --- SMART COLOR HANDLING ---
    if colors is None:
        if group_col == 'koppen_zone':
            #Koppen Zones colors from https://www.bom.gov.au/climate/maps/averages/climate-classification
            colors = {
                'Equatorial': "#a68936", 'Tropical': "#8ca13e", 
                'Subtropical': "#dae6b1", 'Desert': "#f69d64", 
                'Grassland': "#fbe396", 'Temperate': "#64c4ec"
            }
        
        elif group_col == 'ncc_zone':
            # NCC Zones digitized from https://ncc.abcb.gov.au/abcb-climate-map
            colors = {
                1: '#b03a2e',
                2: "#fff01f",
                3: "#ff9900",
                4: "#d6ac58",
                5: '#b2ff8d',
                6: '#5de1ff',
                7: '#4d8bb8',
                8: '#7d7d7d'
            }
            
        elif group_col == 'rainfall_zone':
            #Seasonal Rainfall Zones colors digitized from https://www.bom.gov.au/climate/maps/averages/climate-classification
            colors = {
                'Summer dominant': "#c62828", # Deep Red
                'Summer':          "#fdd835", # Deep Yellow
                'Uniform':         "#43a047", # Green
                'Arid':            "#dfce99", # Cream/Off-white
                'Winter':          "#78909c", # Blue-Gray
                'Winter dominant': "#1565c0"  # Strong Blue
            }
            
        elif group_col == 'thermal_zone':
            #Temperature & Humidity Zones colors digitized from https://www.bom.gov.au/climate/maps/averages/climate-classification
            colors = {
                'hhSwW':"#f4511e", # Red-Orange
                'whSmW':"#ffb74d", # Light Orange
                'hdSwW':"#e0e0e0", # Gray
                'hdScW':"#a1887f", # Brown
                'wScW': "#4db6ac", # Teal
                'mScW': "#4fc3f7"  # Sky Blue
            }
        
        else:
            #Default Fallback
            colors = "rocket"

    # Summation only makes sense for precipitation-related variables.
    if target_var not in ['rainfall', 'rain_today'] and monthly_op == 'sum': monthly_op = 'mean'

    df_plot = df.copy()
    df_plot= ft.datesplit(df_plot)
    df_plot['dayofyear'] = df_plot['date'].dt.dayofyear
    
    # get representative daily mean per zone (averaging all stations)
    daily_res = df_plot.groupby([group_col, 'location', 'year', 'month', 'day', 'dayofyear'])[target_var].mean().reset_index()
    zone_daily = daily_res.groupby([group_col, 'year', 'month', 'day', 'dayofyear'])[target_var].mean().reset_index()

    if time_scale == 'month':
        #MONTHLY VIEW: Aggregate daily values per month and year
        monthly = zone_daily.groupby([group_col, 'year', 'month'])[target_var].agg(monthly_op).reset_index()
        #average those monthly values across all available years
        plot_data = monthly.groupby([group_col, 'month'])[target_var].mean().unstack(0)
        
        xlabel = 'Month'
        if fiscal_year:
            # Reorder indices to start from July (7) to June (6)
            new_order = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8]
            plot_data = plot_data.reindex(new_order)
            xtick_labels = ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul*', 'Aug*']
        else:
            new_order = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2]
            plot_data = plot_data.reindex(new_order)
            xtick_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan*', 'Feb*']
        xticks = range(len(plot_data))
        
    else:
        #DAILY VIEW: Average each of the 365 days across all years
        plot_data = zone_daily.groupby([group_col, 'dayofyear'])[target_var].mean().unstack(0)
        # Apply 7-day rolling mean to smooth out daily noise (centered window)
        plot_data = plot_data.rolling(window=7, center=True).mean()
        
        xlabel = 'Day of the Year (7-day Smoothed)'
        if fiscal_year:
            # Reorder indices to start from July (7) to June (6)
            part1 = plot_data.loc[182:] # July to Dec
            part2 = plot_data.loc[:181] # Jan to June
            plot_data = pd.concat([part1, part2])
            xticks = [0, 31, 62, 92, 123, 153, 184, 215, 243, 274, 304, 335]
            xtick_labels = ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        else:
            # Set ticks to the start of each month
            xticks = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
            xtick_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    #--- PLOTTING ---
    plt.figure(figsize=(15, 7))
    sns.lineplot(data=plot_data.reset_index(drop=True), palette=colors, linewidth=2.5, marker='o', dashes=False)
    
    #Dynamic title and labels
    if group_col == 'koppen_zone':
        zone_name = 'Köppen Climate Zone'
    elif group_col == 'ncc_zone':
        zone_name = 'NCC Climate Zone'
    elif group_col == 'rainfall_zone':
        zone_name = 'Seasonal Rainfall Zone'
    elif group_col == 'thermal_zone':
        zone_name = 'Temperature & Humidity Zone'
    else:
        zone_name = group_col.capitalize()
    title_type = "Total" if monthly_op == 'sum' and time_scale == 'month' else "Average"
    plt.title(f'{title_type} {target_var.capitalize()} by {zone_name}', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel(target_var.capitalize(), fontsize=12)
    plt.xlabel(xlabel, fontsize=12)
    plt.xticks(xticks, xtick_labels)
    plt.xlim(0, 13 if time_scale == 'month' else 364)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(title=zone_name, title_fontsize='13', fontsize='11', 
           bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    plt.tight_layout()
    plt.show()


def location_loss(df, threshold, features=None):
    """
    Plots the percentage of data loss per location based on a NaN threshold.
    
    Args:
        df (pd.DataFrame): The weather dataset.
        threshold (int): Max allowed NaNs (rows with at least this amount are marked as loss).
        features (list, optional): Specific columns to check. If None, all columns are checked.
    """
    # Determine which columns to audit
    audit_df = df[features] if features is not None else df
    feature_type = "Core Features" if features is not None else "All Features"
    
    # Identify 'garbage' rows (True/False)
    # Check if the sum of NaNs in the selected features is GREATER than the threshold
    is_garbage = audit_df.isnull().sum(axis=1) >= threshold
    
    # Calculate percentage loss per location
    location_loss = (is_garbage.groupby(df['location']).mean() * 100).sort_values(ascending=False)
    
    # Plotting
    plt.figure(figsize=(14, 6))
    color = '#3498db' if features is not None else '#e74c3c' # Blue for Core, Red for All
    
    location_loss.head(20).plot(kind='bar', color=color, label=f'{threshold}+ missing values')
    
    # Add visual guide for the 10% threshold
    plt.axhline(y=10, color='black', linestyle='--', alpha=0.6, label='10% Loss Threshold')
    
    # Labels and Titles
    plt.title(f'Data Loss per Location ({threshold}+ NaNs in {feature_type})', fontsize=15)
    plt.ylabel('Percentage of Rows Lost (%)', fontsize=12)
    plt.xlabel('Location', fontsize=12)
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print the top 10 for quick reference
    print(f"Top 10 Data Loss Locations using {feature_type} (Threshold > {threshold}):")
    print(location_loss.head(10))
    print("-" * 30)
