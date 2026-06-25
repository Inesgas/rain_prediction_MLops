# Variable-Specific Missingness Design Notes

## Goal

Test whether a more scientific missing-data strategy improves predictive quality more than adding more model complexity.

## Core Idea

Stop treating all missing variables the same.

Instead, split them into logical groups:

- stable physical numeric variables: temperature, pressure, humidity, rainfall, wind speed
- observational numeric variables: sunshine, evaporation, cloud
- directional categorical variables: wind direction columns

## Experiments

1. Current hybrid imputer baseline
2. Current hybrid imputer plus expanded missingness features
3. Variable-specific missing-data design by group

## Important Constraint

The current CatBoost wrapper still median-fills numeric NaNs before fitting, so this experiment uses conservative fallbacks rather than a raw leave-missing numeric path. That limitation is documented explicitly so the result is still interpretable.
