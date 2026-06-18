# Explainable Financial Anomaly Detection Dashboard

This project is part of an M.Sc. Data Science dissertation titled:

**Evaluating Explainable AI-Based Anomaly Detection Models for Financial Trading Surveillance**

The project develops an Explainable AI-based anomaly detection framework to identify unusual trading behaviour in financial market data around quarterly earnings announcements.

## Project Overview

The dashboard focuses on detecting and interpreting unusual market activity using unsupervised machine learning models. The analysis is based on public financial market data and focuses on a 10-trading-day pre-earnings event window.

The project does not claim to detect fraud, insider trading, or market manipulation. Instead, it identifies model-flagged anomalous trading behaviour for academic research and surveillance-oriented analysis.

## Companies Analysed

The analysis focuses on five major US-listed technology companies:

* Apple Inc. (AAPL)
* Microsoft Corporation (MSFT)
* NVIDIA Corporation (NVDA)
* Amazon.com Inc. (AMZN)
* Tesla Inc. (TSLA)

## Time Period

The study covers market data from:

**January 2021 to December 2025**

## Methodology

The project follows a structured data science workflow:

1. Data collection using public financial market sources
2. Earnings event collection and preparation
3. Data preprocessing
4. Feature engineering
5. Event-window labelling
6. Unsupervised anomaly detection
7. Event-based model evaluation
8. Explainable AI analysis using SHAP
9. Dashboard-based visualisation and interpretation

## Features Used

The feature set includes:

* Daily Return
* Relative Volume
* Rolling Volatility 10-Day
* 10-Day Moving Average
* Volume Change %
* Price Change %
* Distance From Earnings Event
* Event Window Indicator

## Models Used

The following unsupervised anomaly detection models are used:

* Isolation Forest
* One-Class Support Vector Machine
* Local Outlier Factor

## Evaluation Approach

The evaluation is framed as unsupervised event-based evaluation. The project avoids supervised classification metrics unless labelled ground truth is introduced.

The evaluation focuses on:

* Anomaly detection rate within event windows
* Relative anomaly concentration
* Model agreement and consistency
* Anomaly score stability
* False-positive analysis outside event windows
* Explainability of anomaly-driving features

## Explainable AI

SHAP is used to interpret the contribution of financial features to model-flagged anomalies, particularly for the most suitable anomaly detection model.

## Dashboard

The dashboard is built using Streamlit and provides a finance-surveillance-style interface for viewing:

* Market data overview
* Earnings event windows
* Model-detected anomalies
* Model comparison results
* SHAP-based feature importance
* Research findings and limitations

## Ethical Scope

This project uses only public secondary financial market data. It does not involve human participants, personal data, confidential data, or sensitive identifiable information.

The outputs should not be interpreted as investment advice, legal evidence, or confirmation of insider trading, fraud, or market manipulation.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* SHAP
* Plotly
* Matplotlib
* Streamlit
* Yahoo Finance data via yfinance

## How to Run the Project

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit dashboard:

```bash
streamlit run app.py
```

## Project Structure

`## Project Structure

```text
financial-anomaly-dissertation/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── .streamlit/
│   └── config.toml
│
├── assets/
│   └── style.css
│
├── data/
│   ├── raw/
│   │   ├── earnings_dates/
│   │   └── stock_prices/
│   │
│   └── processed/
│       ├── all_earnings_dates.csv
│       ├── earnings_dates_summary.csv
│       ├── event_window_dataset.csv
│       ├── event_window_detail.csv
│       └── final_feature_dataset.csv
│
├── notebooks/
│   ├── 01_data_collection_check.ipynb
│   ├── 02_feature_engineering_check.ipynb
│   ├── 03_model_testing.ipynb
│   ├── 04_evaluation_results.ipynb
│   └── 05_shap_explainability.ipynb
│
├── outputs/
│   ├── charts/
│   │   ├── 01_adjusted_closing_price_trends.html
│   │   ├── 02_trading_volume_trends.html
│   │   ├── 03_event_window_timeline.html
│   │   ├── 04_total_anomaly_alerts_by_model.html
│   │   ├── 05_event_vs_non_event_anomalies.html
│   │   ├── 06_event_window_detection_rate.html
│   │   ├── 07_relative_anomaly_concentration.html
│   │   ├── 08_model_agreement_summary.html
│   │   ├── 09_event_based_false_positive_rate.html
│   │   ├── 10_anomaly_score_stability.html
│   │   ├── 11_overall_model_evaluation_summary.html
│   │   ├── 12_best_model_selection_rank.html
│   │   ├── 13_isolation_forest_anomaly_timeline.html
│   │   ├── 14_shap_feature_importance.html
│   │   ├── 15_shap_feature_importance_by_company.html
│   │   ├── shap_beeswarm_plot.html
│   │   ├── shap_feature_importance_bar.html
│   │   └── shap_feature_importance_by_ticker.html
│   │
│   ├── model_results/
│   └── tables/
│
├── reports/
│
└── src/
    ├── __init__.py
    ├── config.py
    ├── data_collection.py
    ├── earnings_loader.py
    ├── event_windows.py
    ├── feature_engineering.py
    ├── models.py
    ├── evaluation.py
    ├── shap_analysis.py
    └── visualizations.py
```

### Key Folders

* `app.py` contains the Streamlit dashboard interface.
* `.streamlit/config.toml` stores the Streamlit dark theme configuration.
* `assets/style.css` contains custom dashboard styling.
* `data/raw/` contains original market and earnings data.
* `data/processed/` contains cleaned datasets used for modelling and dashboard display.
* `notebooks/` contains experimental and validation notebooks.
* `outputs/charts/` contains generated HTML visualisations used in the dashboard.
* `outputs/model_results/` stores anomaly detection model outputs.
* `outputs/tables/` stores evaluation and summary tables.
* `src/` contains the core Python modules for data collection, preprocessing, feature engineering, modelling, evaluation, SHAP analysis, and visualisation.

```
```


## Disclaimer

This dashboard is developed for academic research purposes only. The results identify unusual trading behaviour using unsupervised anomaly detection models and do not represent financial advice, trading recommendations, fraud detection, insider trading confirmation, or legal conclusions.
