# IDX Exchange Data Science

Data science internship project focused on analyzing California residential real estate data from CRMLS and developing machine learning models to predict property closing prices.

## Project Overview

This project uses monthly CRMLS sold-property records to explore the factors associated with California single-family home prices and build models for predicting `ClosePrice`.

The current analysis covers **27 months of data from January 2024 through March 2026**. After combining the monthly datasets, the analysis is restricted to:

- `PropertyType == Residential`
- `PropertySubType == SingleFamilyResidence`

This results in approximately **297,000 single-family home sales** for analysis.

## Exploratory Data Analysis

The initial exploratory analysis investigates price distributions, property characteristics, missing values, potential anomalies, geographic differences, and changes in sale prices over time.

### Key Findings

- **Sale prices are strongly right-skewed.** Extreme observations substantially affect the raw `ClosePrice` distribution and will require further investigation during preprocessing.
- **Missingness is relatively low** among the primary numeric features. `LotSizeSquareFeet` has the highest missingness among the initial core variables at approximately 1.7%.
- **Living area is strongly associated with sale price.** Larger homes generally sell for more, although substantial price variation remains among homes of similar size.
- **Lot size has a weaker relationship with price.** Large lots occur across a wide range of sale prices, suggesting that lot size alone provides limited information without additional context.
- **Location appears to be an important predictor.** Among the 15 cities with the most transactions, median sale prices range from approximately $440,000 in Victorville to $1.67 million in San Jose.
- **Sale prices vary over time.** Monthly median prices change meaningfully across the dataset, supporting the use of chronological validation and testing for future modeling.
- Potential data-quality issues identified during EDA include zero values, extreme observations, 29 exact duplicate rows, and repeated `ListingKey` values.

No outlier removal or missing-value imputation has been applied during EDA. These issues are identified here for investigation during preprocessing.

## Project Structure

```text
IDX-Exchange-Data-Science/
├── notebooks/
│   └── 01_exploration.ipynb
├── scripts/
├── .gitignore
└── README.md
```

Raw CRMLS datasets are stored locally and are **not included in this repository**.

## Project Progress

- [x] Data loading and multi-month integration
- [x] Exploratory data analysis
- [x] Missing-value and anomaly investigation
- [x] Initial feature relationship analysis
- [x] Geographic and temporal analysis
- [ ] Data preprocessing and cleaning
- [ ] Chronological train/validation/test split
- [ ] Baseline regression model
- [ ] Model comparison
- [ ] Feature engineering
- [ ] Advanced modeling and tuning
- [ ] Final evaluation

## Next Steps

The next stage of the project will focus on preprocessing the CRMLS data. This includes investigating duplicate records, handling invalid and missing values, establishing defensible outlier rules, selecting modeling features, and preparing chronological training, validation, and test datasets.
