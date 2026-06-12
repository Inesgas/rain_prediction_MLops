import pandas as pd
from pycaret.classification import setup, compare_models, save_model, finalize_model

# 1. Load data
train = pd.read_csv("../data/preprocessed/train_all_features.csv")
test = pd.read_csv("../data/preprocessed/test_all_features.csv")

# 2. Setup - The core of the AutoML pipeline
# This replaces the manual preprocessing, scaling, and balancing
s = setup(
    data=train, 
    test_data=test, 
    target='rain_tomorrow',
    fold_strategy='timeseries',      # Crucial: Respects chronological order
    fold=5,                          # Standard for stable validation
    session_id=42,                   # Ensures reproducibility
    
    # --- PREPROCESSING & CLEANING ---
    normalize=True,                  # Scales features
    normalize_method='robust',       # Best for weather data (outlier resistant)
    imputation_type='simple',        # Handles missing values
    
    # --- IMBALANCE HANDLING ---
    fix_imbalance=True,              # Uses SMOTE to balance 'Rain' vs 'No Rain'
    
    # --- FEATURE ENGINEERING ---
    polynomial_features=True,        # Creates interactions like Temp * Humidity
    polynomial_degree=2,             # Complexity of interactions
    feature_interaction=True,        # Mathematically combines meaningful features
    
    # --- FEATURE SELECTION ---
    feature_selection=True,          # Removes noise
    n_features_to_select=0.2,        # Keeps top 20% of all generated features
    remove_multicollinearity=True,   # Removes redundant, highly correlated features
    
    # --- SYSTEM SETTINGS ---
    n_jobs=-1,                       # Uses all CPU cores
    use_gpu=False,                   # Set to True if you have a compatible NVIDIA GPU
    verbose=True
)

# 3. Model Comparison
# We optimize for F1-Score because of the class imbalance (Rain is rare)
# This will test XGBoost, CatBoost, LightGBM, Random Forest, etc.
print("Starting Model Comparison (this will take several hours)...")
best_model = compare_models(sort='F1')

# 4. Finalize & Save
# Finalize trains the model on the entire training dataset
final_model = finalize_model(best_model)
save_model(final_model, 'best_automl_weather_model_v1')

print("Run completed successfully!")
