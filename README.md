# Data-Driven Imputation of Missing Economic Indicators for EU Regions

Code, data and results for the study **"Data-Driven Imputation of Missing Economic Indicators for EU Regions: A Multi-Metric Accuracy Assessment"**, presented at DAMSS 2026 (17th International Conference Data Analysis Methods for Software Systems, Druskininkai, Lithuania).

**Authors:** Irmantas Pilypas, Irma Šileikienė, Asta Slotkienė (Vilnius Gediminas Technical University)

## About

The study compares Random Forest and XGBoost for imputing missing regional economic indicators at the NUTS 2 level. The dataset was compiled from the Eurostat regional database: 260 regions, eight indicators (ROD1-ROD8), 1990-2023, 6,984 observations, with 10.01-53.81% missing values per indicator.

A separate model is trained for each indicator, using the remaining indicators, region and year as predictors. For each indicator, 20% of the observed values are withheld as a test set. Accuracy is assessed with nRMSE, nMAE, R² and sMAPE using five-fold cross-validation, for a baseline configuration and a hyperparameter-optimized one (randomized search over 30 configurations).

| ID | Indicator |
|------|-------------------------------|
| ROD1 | Employment rate |
| ROD2 | R&D expenditure (% of GDP) |
| ROD3 | Unemployment rate |
| ROD4 | Tertiary education attainment |
| ROD5 | Youth unemployment rate |
| ROD6 | Population density |
| ROD7 | Long-term unemployment share |
| ROD8 | Female employment rate |

## Repository structure

```
Baseline_Economic_Indicator_Imputation.ipynb    Notebook 1: exploratory analysis and baseline models
Optimized_Economic_Indicator_Imputation.ipynb   Notebook 2: hyperparameter-optimized models
imputation_models/                              RandomForestImputer and XGBoostImputer classes
uploads/                                        Input dataset (Eurostat NUTS 2 indicators)
results/                                        Cross-validation metrics, imputed datasets and figures
requirements.txt                                Package versions used to obtain the reported results
```

## How to run

Python 3.11 was used.

```bash
pip install -r requirements.txt
jupyter notebook
```

Run Notebook 1 first, then Notebook 2: the second notebook reads `results/cv_metrics_baseline.csv` produced by the first one to compare the baseline and optimized models.

xgboost >= 3.0 is required to obtain exactly the reported XGBoost metrics.

## Data source

Eurostat, Regional statistics by NUTS classification (database): https://ec.europa.eu/eurostat/web/regions/database

## License

The code is released under the [MIT License](LICENSE). The input data originate from Eurostat and are subject to the Eurostat reuse policy.
