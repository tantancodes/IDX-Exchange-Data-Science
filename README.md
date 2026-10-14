# IDX Exchange — California Property Price Prediction

Machine learning project for predicting the closing price (`ClosePrice`) of single-family residential properties in California using historical CRMLS transaction data.

## Overview

The objective of this project is to develop a reproducible machine learning pipeline capable of estimating the closing price of a California single-family property using characteristics available at prediction time.

The model is intended to support predictions for both listed and off-market properties. Accordingly, model inputs are restricted to property characteristics that can reasonably be known independently of an active listing.

The current dataset contains **27 months of CRMLS sold-property data from January 2024 through March 2026**.

## Data

Monthly CRMLS datasets are combined and restricted to:

- `PropertyType == "Residential"`
- `PropertySubType == "SingleFamilyResidence"`

This produces **297,245 single-family residential transactions** prior to preprocessing.

Raw CRMLS data is proprietary and is **not included in this repository**.

### Initial Modeling Features

The current baseline feature set includes:

**Numeric**
- `LivingArea`
- `BedroomsTotal`
- `BathroomsTotalInteger`
- `LotSizeSquareFeet`

**Geographic**
- `City`
- `PostalCode`

This feature set is intentionally limited for the initial modeling pipeline. Additional property, temporal, and geographic features will be evaluated during feature engineering.

### Leakage Prevention

Because the model must also value properties that are not currently listed, listing-dependent variables are excluded from model inputs, including:

- `ListPrice`
- `OriginalListPrice`
- `DaysOnMarket` and cumulative DOM fields
- `PurchaseContractDate`
- `ListingContractDate`
- Features derived from listing price or contract timing

`ClosePrice` is used exclusively as the prediction target.

## Exploratory Data Analysis

Exploratory analysis was performed before applying cleaning or imputation rules.

Key findings include:

- `ClosePrice` is strongly right-skewed and contains a small number of extreme observations.
- Missingness is relatively low among the initial numeric features, with `LotSizeSquareFeet` having the highest missingness at approximately 1.7%.
- `LivingArea` has a clear positive relationship with sale price, although substantial variation remains among similarly sized properties.
- Lot size has a weaker standalone relationship with sale price.
- Geographic location is strongly associated with property value.
- Among the 15 highest-volume cities, median sale prices range from approximately **$440,000 in Victorville to $1.67 million in San Jose**.
- Monthly median prices vary over time, supporting chronological rather than random model evaluation.
- Initial data-quality issues included zero values, extreme observations, exact duplicates, and repeated `ListingKey` values.

See [`notebooks/01_exploration.ipynb`](notebooks/01_exploration.ipynb) for the complete analysis.

## Preprocessing

Preprocessing is designed to create a reproducible modeling dataset while minimizing data leakage.

### Duplicate Handling

The initial dataset contained **29 exact duplicate rows**.

Repeated `ListingKey` values were investigated separately rather than automatically removed. Some repeated keys correspond to records with different close dates or close prices and therefore may represent distinct transactions, relistings, or MLS updates.

The preprocessing procedure therefore:

- Removes exact duplicate rows.
- Does not remove observations solely because `ListingKey` repeats.
- Verifies transaction-level duplication using `ListingKey`, `CloseDate`, and `ClosePrice`.

After exact duplicate removal, no duplicate combinations of these three fields remained.

### Target Cleaning

Because `ClosePrice` is the prediction target, missing target values are not imputed.

Records are removed when:

- `ClosePrice` is missing.
- `ClosePrice <= 0`.
- `ClosePrice > $100,000,000`.

The upper bound was selected conservatively after inspecting the extreme tail of the distribution. It removes a very small number of highly implausible observations while retaining legitimate luxury-market transactions.

After duplicate and target-quality cleaning, **297,193 observations remain**.

### Predictor Cleaning

Zero values in key physical-property fields are treated as missing where they do not represent meaningful measurements:

- `LivingArea`
- `BedroomsTotal`
- `BathroomsTotalInteger`
- `LotSizeSquareFeet`

Rather than dropping an entire transaction because a predictor is unavailable, missing predictor values are handled during the preprocessing pipeline.

### Temporal Train/Test Split

Model evaluation uses a chronological split to better approximate prediction on future transactions.

The project specification reserves the most recent available month as the test set:

| Split | Period | Observations |
| --- | --- | ---: |
| Training | Jan 2024 – Feb 2026 | 285,632 |
| Test | Mar 2026 | 11,561 |

March 2026 is kept completely outside the training dataset.

The number of preceding months used for model training will be treated as a tunable modeling decision rather than assuming that the full historical window is necessarily optimal.

### Training-Only Transformations

Data-dependent transformations are fit using the training set only to prevent information from the held-out test period from leaking into model development.

The initial preprocessing pipeline includes:

- Median imputation for numeric variables
- Standardization of numeric variables
- Categorical missing-value imputation
- One-hot encoding of categorical variables
- Handling of previously unseen test-set categories

The pipeline is implemented using scikit-learn preprocessing components so the same fitted transformations can be consistently applied during training and inference.

See [`notebooks/02_preprocessing.ipynb`](notebooks/02_preprocessing.ipynb) for the preprocessing workflow.

## Baseline Model (Week 4)

[`notebooks/03_baseline_model.ipynb`](notebooks/03_baseline_model.ipynb) cleans the raw files from scratch and fits a Linear Regression on the six features above.

### Evaluation Design

Following the IDX AVM best-practices standard:

- Each sale is assigned to its month by `CloseDate`. Models train on all prior months, are validated on **February 2026**, and are tested once on **March 2026**.
- The same pipeline is re-run at three earlier cutoffs (November 2025 – January 2026) to check that results are stable over time.
- The 0.5th / 99.5th percentile `ClosePrice` bounds are computed from each training period only and then applied unchanged to the evaluation month. Metrics are reported both within those bounds and for all valid sales.
- Imputation, scaling, and encoding are fit inside a scikit-learn `Pipeline` on training data only.

Cleaning in notebook 03 adds a few rules on top of notebook 02:

- Records with `CloseDate` before `ListingContractDate` are removed (41 rows). The listing date is used only for this check, never as a model input.
- Physical values outside plausible bounds (living area 300–30,000 sq ft, 1–20 bedrooms or bathrooms) are treated as missing.
- Missing-value indicator columns are added for the numeric features.

### Results

A log-log variant (log price regressed on log living area and log lot size) is reported alongside the required baseline as a stronger linear reference point.

March 2026 test results, for sales within the training-derived price bounds (98.8% of March sales):

| Model | R² | MdAPE | MAPE | MAE |
| --- | ---: | ---: | ---: | ---: |
| Linear Regression (baseline) | **0.812** | 16.6% | 22.6% | $238,462 |
| Log-log Linear Regression | 0.826 | 10.0% | 14.1% | $186,273 |

- Baseline R² stayed between 0.80 and 0.81 at every historical cutoff.
- Training on all available months outperformed 3-, 6-, 12-, and 18-month windows.
- Error is largest for homes under about $560K (29.6% MdAPE) and in Inland Empire and high-desert cities.
- Including the ~1.2% of sales outside the price bounds lowers baseline R² to 0.63, while MdAPE barely changes.

## Repository Structure

```text
IDX-Exchange-Data-Science/
├── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_baseline_model.ipynb
├── scripts/
│   └── crmls_sold.py
├── results/              # metrics and data snapshot (local; CSVs are git-ignored)
├── requirements.txt
├── .gitignore
└── README.md
```

Local raw and processed datasets are excluded from version control.

## Project Status

| Stage | Status |
| --- | --- |
| Data access and integration | Complete |
| Exploratory data analysis | Complete |
| Data-quality investigation | Complete |
| Duplicate and target cleaning | Complete |
| Chronological train/test split | Complete |
| Preprocessing pipeline | Complete |
| Linear Regression baseline | Complete |
| Decision Tree / Random Forest comparison | Not started |
| Feature engineering | Not started |
| Advanced models | Not started |
| Expanded evaluation | Not started |

## Roadmap

The project follows a 12-week development plan:

1. **Setup and data access** — Complete
2. **Exploratory data analysis** — Complete
3. **Data preprocessing** — Complete
4. **Linear Regression baseline** — Complete
5. **Decision Tree and Random Forest comparison**
6. **Feature engineering and school-district geographic features**
7. **Gradient boosting and hyperparameter tuning**
8. **Expanded evaluation using R², MAPE, and MdAPE**
9. **Optional prediction application**
10. **Documentation**
11. **Presentation preparation**
12. **Final presentation and project handoff**

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Place the `CRMLSSold*.csv` files in `data/raw/` and run the notebooks in order. Notebook 03 prints a fingerprint of the source files (`2ff3877a04bd` for the January 2024 – March 2026 pull) and saves per-file hashes to `results/data_snapshot.csv` and all metrics to `results/baseline_metrics.csv`.

## Next Steps

Week 5 compares **Decision Tree and Random Forest** regressors against both linear baselines. Model selection will use February 2026 and the earlier cutoffs; the March 2026 test month is not used for those decisions.
