# IDX Exchange Data Science

Data science internship project focused on analyzing California residential real estate data from CRMLS and developing machine learning models to predict property closing prices.

## Project Overview

The goal of this project is to develop a machine learning model that predicts `ClosePrice` for a California single-family property using characteristics that would be available at the time of prediction.

Because the model should be applicable to properties that are either listed or off-market, model features are restricted to information that could reasonably be known for any property. Listing-dependent variables such as `ListPrice`, `OriginalListPrice`, `DaysOnMarket`, `PurchaseContractDate`, and `ListingContractDate` are therefore excluded from model inputs.

The current dataset covers **27 months from January 2024 through March 2026**. After combining the monthly CRMLS datasets, observations are restricted to:

- `PropertyType == Residential`
- `PropertySubType == SingleFamilyResidence`

This produces **297,245 single-family residential transactions** before preprocessing.

---

## Week 2 — Exploratory Data Analysis

The exploratory analysis investigates price distributions, property characteristics, missing values, anomalies, geographic differences, and changes in sale prices over time.

### Key Findings

- **Sale prices are strongly right-skewed.** Extreme observations substantially affect the raw `ClosePrice` distribution.
- **Missingness is relatively low** among the primary numeric features. `LotSizeSquareFeet` has the highest missingness among the initial core variables at approximately 1.7%.
- **Living area is strongly associated with sale price.** Larger homes generally sell for more, although substantial price variation remains among similarly sized properties.
- **Lot size has a weaker relationship with price.** Large lots occur across a wide range of sale prices.
- **Location is an important predictor.** Among the 15 cities with the most transactions, median sale prices range from approximately $440,000 in Victorville to $1.67 million in San Jose.
- **Sale prices vary over time.** Monthly median prices change meaningfully across the dataset, supporting chronological rather than random model evaluation.
- Data-quality issues identified during EDA include zero values, extreme observations, **29 exact duplicate rows**, and repeated `ListingKey` values.

No observations were removed during EDA. The purpose of the exploratory notebook was to identify issues requiring investigation during preprocessing.

---

## Week 3 — Data Preprocessing

Preprocessing is designed to create a reproducible, leakage-safe modeling dataset while preserving legitimate California housing-market variation.

### Duplicate Investigation

EDA identified both exact duplicate records and repeated `ListingKey` values.

Repeated listing keys were investigated before removal. Several repeated `ListingKey` values corresponded to observations with different close dates or close prices, meaning that automatically keeping only one record per `ListingKey` could remove legitimate transaction information.

Therefore:

- **29 exact duplicate records were removed.**
- Records were **not removed solely because they shared a `ListingKey`.**
- After exact duplicates were removed, no duplicate combinations of `ListingKey`, `CloseDate`, and `ClosePrice` remained.

### Target Cleaning

`ClosePrice` is the prediction target and therefore is not imputed.

Records with missing or non-positive close prices were removed. Inspection of the extreme upper tail also revealed a small number of implausible values, including several hundreds-of-millions-of-dollars prices associated with otherwise ordinary-sized single-family homes.

A conservative **$100 million upper data-quality bound** was used to remove only the most extreme likely errors while preserving legitimate luxury-market transactions.

Target cleaning removed **23 additional observations**.

After duplicate and target-quality cleaning, **297,193 observations remain**.

### Invalid Predictor Values

Zero values were identified in several physical property characteristics, including:

- `LivingArea`
- `BedroomsTotal`
- `BathroomsTotalInteger`
- `LotSizeSquareFeet`

Rather than dropping an entire property because one predictor is unavailable or invalid, these zero values are treated as missing values where appropriate.

Missing predictor values are handled using preprocessing statistics learned from the training data rather than the test data.

### Leakage-Safe Feature Selection

The model is intended to estimate the value of **any California single-family property**, including properties that are not currently listed.

The initial modeling feature set therefore uses property characteristics available independently of the sales process:

**Numeric features**
- `LivingArea`
- `BedroomsTotal`
- `BathroomsTotalInteger`
- `LotSizeSquareFeet`

**Geographic features**
- `City`
- `PostalCode`

Listing-dependent variables are excluded from model inputs, including:

- `ListPrice`
- `OriginalListPrice`
- `DaysOnMarket`
- cumulative days-on-market fields
- `PurchaseContractDate`
- `ListingContractDate`
- features derived from listing price or contract timing

Additional property and geographic features will be evaluated during the feature-engineering stage.

### Chronological Train/Test Split

A chronological split is used instead of a random train/test split because the intended use case is prediction on future property transactions.

Per the project specification, the **most recent month, March 2026, is reserved as the held-out test set**.

Current split:

| Dataset | Period | Observations |
| --- | --- | ---: |
| Training | Jan. 2024 – Feb. 2026 | 285,632 |
| Test | Mar. 2026 | 11,561 |

There is **no March 2026 overlap in the training dataset**.

The number of preceding months used for training will later be evaluated as a modeling choice rather than assuming that the longest possible training window is necessarily optimal.

### Training-Only Preprocessing

The preprocessing workflow is designed so that transformations that learn information from the data are fitted using **training observations only**.

The initial pipeline includes:

- Median imputation for missing numeric features
- Standardization of numeric features
- Imputation of missing categorical features
- One-hot encoding of categorical features
- Safe handling of categories that appear in the future test month but not in training

This prevents information from the March 2026 holdout period from influencing model preparation.

---

## Project Structure

```text
IDX-Exchange-Data-Science/
├── notebooks/
│   ├── 01_exploration.ipynb
│   └── 02_preprocessing.ipynb
├── scripts/
│   └── crmls_sold.py
├── .gitignore
└── README.md
```

Raw and processed CRMLS datasets are stored locally and are **not included in this public repository**.

---

## Project Progress

- [x] Data access and multi-month integration
- [x] Residential single-family filtering
- [x] Exploratory data analysis
- [x] Missing-value and anomaly investigation
- [x] Initial feature relationship analysis
- [x] Geographic and temporal analysis
- [x] Duplicate investigation and exact-duplicate removal
- [x] Target-quality investigation and cleaning
- [x] Leakage-safe initial feature selection
- [x] Chronological train/test split
- [ ] Finalize and validate preprocessing pipeline
- [ ] Baseline Linear Regression
- [ ] Decision Tree and Random Forest comparison
- [ ] Feature engineering
- [ ] School-district geographic features
- [ ] Gradient boosting / advanced models
- [ ] Expanded model evaluation
- [ ] Final documentation and presentation

---

## Current Status — Week 3

The project has progressed from exploratory analysis into data preprocessing.

The working dataset combines **27 months of CRMLS data from January 2024 through March 2026** and contains **297,193 observations after current duplicate and target-quality cleaning**.

The March 2026 data is isolated as a future holdout set with **11,561 transactions**, while **285,632 earlier transactions** are currently available for training.

The current focus is completing and validating the training-only preprocessing pipeline so that missing-value handling, scaling, and categorical encoding do not introduce information from the test period.

---

## Meeting Update

**This week's progress:**

I completed the exploratory analysis and moved into Week 3 preprocessing. I am working with all 27 available months of CRMLS data from January 2024 through March 2026, giving approximately 297,000 single-family residential transactions.

During preprocessing, I investigated duplicate `ListingKey` values rather than automatically removing them. Some repeated keys had different close dates or prices, so I retained those records and removed only confirmed exact duplicates.

I also investigated invalid and extreme target values. Missing and non-positive close prices were removed, along with a very small number of implausible prices above $100 million. After current cleaning, 297,193 observations remain.

Based on the updated project guidance, I am restricting model inputs to characteristics that would be available for any property, including an off-market home. This means listing-dependent variables such as list price and days on market are excluded.

For model evaluation, I reserved March 2026 as the held-out test month. This gives me **285,632 training observations and 11,561 test observations**. The preprocessing pipeline is being structured so that imputation, scaling, and categorical encoding learn only from the training data, preventing test-set leakage.

**Next step:** finish validating the preprocessing pipeline and then begin the Week 4 Linear Regression baseline, evaluated on the held-out March 2026 data using R².

---

## Roadmap

**Week 1:** Environment setup and CRMLS data access  
**Week 2:** Exploratory data analysis — Complete  
**Week 3:** Data preprocessing — In progress  
**Week 4:** Linear Regression baseline  
**Week 5:** Decision Tree and Random Forest comparison  
**Week 6:** Feature engineering and school-district geography  
**Week 7:** Gradient boosting and advanced models  
**Week 8:** Expanded evaluation using R², MAPE, and MdAPE  
**Weeks 9–12:** Optional application, documentation, presentation, and final handoff
