# Forecasting German Electricity Demand

Time-series forecasting project for German electricity demand using statistical, machine learning, and deep learning models.

This project was completed as part of an MSc Data Science forecasting assignment and investigates whether advanced forecasting methods provide meaningful improvements over traditional benchmark models.

---

## Project Objectives

The main objectives of this project are:

- Forecast German electricity demand over a 2-year horizon.
- Identify trend and seasonal components in electricity demand.
- Evaluate stationarity using formal statistical tests.
- Compare benchmark forecasting methods with advanced models.
- Assess the value of weather variables as exogenous regressors.
- Evaluate whether machine learning and deep learning models justify their additional complexity.
- Recommend an operational forecasting approach.

---

## Dataset

### Electricity Demand Data

Source: Open Power System Data (OPSD)

https://data.open-power-system-data.org/time_series/

Dataset used:

```text
time_series_60min_singleindex.csv
```

Country:

```text
Germany (DE)
```

Period analysed:

```text
01 January 2015 – 04 October 2020
```

Original frequency:

```text
Hourly
```

Aggregated to:

```text
Daily
Weekly
```

Target variable:

```text
load_gw
```

---

### Temperature Data

Source: Open-Meteo Historical Weather API

https://open-meteo.com/en/docs/historical-weather-api

Location:

```text
Berlin, Germany
```

Temperature features:

```text
temp_mean
temp_min
temp_max
heating_degree_days
cooling_degree_days
```

Observed temperatures were used during the test period. Therefore, SARIMAX results should be interpreted as conditional or explanatory forecasts rather than fully operational forecasts.

---

## Methodology

### Part 1 – Exploratory Data Analysis

The following analyses were performed:

- Missing value analysis
- Time-series visualisation
- STL decomposition
- Trend and seasonal strength analysis
- ACF and PACF analysis
- Augmented Dickey-Fuller test
- KPSS test
- First differencing
- Seasonal differencing

---

### Part 2 – Benchmark Forecasting Models

The following benchmark models were implemented:

- Mean Forecast
- Naive Forecast
- Seasonal Naive Forecast
- Drift Forecast

Forecast horizon:

```text
104 weeks (2 years)
```

---

### Part 3 – SARIMA

A grid search was performed over:

```text
p = 0–6
d = 0–2
q = 0–6
```

Model selection was based on:

```text
AIC minimisation
```

Residual diagnostics included:

- Residual plots
- Histogram
- Q-Q plot
- Residual ACF
- Ljung-Box test

---

### Part 4 – SARIMAX

Temperature variables were incorporated as exogenous regressors.

These included:

- Mean temperature
- Minimum temperature
- Maximum temperature
- Heating degree days
- Cooling degree days

Forecast intervals included:

- 80% confidence interval
- 95% confidence interval

---

### Part 5 – Machine Learning Models

Feature engineering included:

- Load lags
- Seasonal lags
- Rolling statistics
- Temperature features
- Fourier seasonal terms
- Calendar variables

Models evaluated:

- Random Forest Regressor
- Gradient Boosting Regressor

---

### Part 6 – Deep Learning

The deep learning model used was:

```text
Long Short-Term Memory (LSTM)
```

Key design choices included:

- Hourly demand data
- 168-hour input window
- Training-only scaling
- Early stopping
- Hyperparameter tuning
- Recursive multi-step forecasting

The final hourly forecasts were aggregated to weekly values for comparison with the other models.

---

## Evaluation Metrics

All models were evaluated using:

```text
MAE
RMSE
MASE
Bias
```

The Seasonal Naive Forecast was used as the primary benchmark.

---

## Results Summary

| Model | Relative Performance |
|---|---|
| Mean Forecast | Poor |
| Naive Forecast | Poor |
| Drift Forecast | Poor |
| Seasonal Naive | Strong benchmark |
| SARIMA | Moderate |
| SARIMAX | Slight improvement over SARIMA |
| Gradient Boosting | Strong |
| Random Forest | Best overall |
| LSTM | Underperformed on long-horizon forecasting |

### Key Findings

- German electricity demand exhibits strong annual seasonality.
- Seasonal Naive is a difficult benchmark to outperform.
- Temperature provides a modest improvement in forecasting accuracy.
- Random Forest achieved the best overall forecasting performance.
- LSTM performance deteriorated over a two-year recursive forecasting horizon.
- Random Forest provides the best balance of accuracy, interpretability, and maintainability.

---

## Repository Structure

```text
electricity-demand-forecasting/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── outputs/
│   ├── figures/
│   ├── forecasts/
│   ├── metrics/
│   └── model_objects/
│
├── reports/
│
├── scripts/
│   ├── download_data.py
│   ├── make_features.py
│   ├── search_sarima.py
│   └── run_pipeline.py
│
├── src/
│   └── electricity_demand/
│       ├── config.py
│       ├── data.py
│       ├── evaluation.py
│       ├── features.py
│       ├── pipeline.py
│       ├── plotting.py
│       └── models/
│           ├── benchmarks.py
│           ├── feature_models.py
│           ├── neural.py
│           ├── sarima.py
│           └── sarimax.py
│
├── tests/
│   ├── test_benchmarks.py
│   ├── test_evaluation.py
│   └── test_features.py
│
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Ram-sai7/electricity-demand-forecasting.git
cd electricity-demand-forecasting
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Install the local package:

```bash
pip install -e .
```

---

## Running the Project

### Download the Data

```bash
python scripts/download_data.py
```

### Generate Features

```bash
python scripts/make_features.py
```

### Run the SARIMA Grid Search

```bash
python scripts/search_sarima.py
```

This step may take several hours because it evaluates many SARIMA parameter combinations.

### Run the Forecasting Pipeline

```bash
python scripts/run_pipeline.py
```

---

## Generated Outputs

### Forecast Files

```text
outputs/forecasts/all_forecasts.csv
outputs/forecasts/sarimax_confidence_intervals.csv
```

### Metrics

```text
outputs/metrics/model_comparison.csv
```

### Figures

```text
outputs/figures/forecast_comparison.png
outputs/figures/LSTM.png
outputs/figures/2yrs.png
outputs/figures/temp.png
```

---

## Testing

Run the project tests using:

```bash
pytest
```

The tests check:

- Benchmark forecast lengths
- Evaluation metrics
- Lag feature construction
- Protection against future-value leakage

---

## Report

The project report is stored in:

```text
reports/
```

The report includes:

- Introduction
- Data and preprocessing
- Exploratory analysis
- Stationarity testing
- Forecasting methodology
- Model diagnostics
- Forecast evaluation
- Critical discussion
- Limitations
- Conclusion

---

## Main Limitations

- The SARIMA grid search is computationally expensive.
- The selected minimum-AIC SARIMA model showed convergence concerns.
- Observed future temperature was used, so SARIMAX is a conditional forecast.
- Holiday covariates were not included in the final fitted models.
- Feature-based models were evaluated in a rolling one-week-ahead framework.
- Recursive LSTM forecasting accumulated error over the long forecast horizon.
- Random Forest does not produce prediction intervals directly.

---

## Recommended Model

Random Forest is recommended for this forecasting task because it achieved the lowest forecasting error and offered a good balance of:

- Accuracy
- Interpretability
- Computational efficiency
- Ease of maintenance

Gradient Boosting also performed strongly and produced similar results.

---

## Author

Ram Sai Sayani

MSc Data Science

Advanced Research Topics – Assignment 1

German Electricity Demand Forecasting
