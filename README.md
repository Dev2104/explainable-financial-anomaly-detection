# Explainable Financial Anomaly Detection

## MSc Data Science Dissertation Project

**Project Title:**
**Evaluating Explainable AI-Based Anomaly Detection Models for Financial Trading Surveillance**

**Author:** Dev Tailor
**Programme:** M.Sc. Data Science
**University:** Arden University Berlin

---

## Live Project Links

* **GitHub Repository:**
  https://github.com/Dev2104/explainable-financial-anomaly-detection

* **Live Streamlit Dashboard:**
  https://explainable-financial-anomaly-detection-devt2104.streamlit.app/

---

## Project Overview

This project develops an Explainable AI-based anomaly detection framework to identify unusual trading behaviour in financial market data around quarterly earnings announcements.

The project focuses on five major technology companies:

* Apple Inc. (AAPL)
* Microsoft Corporation (MSFT)
* NVIDIA Corporation (NVDA)
* Amazon.com Inc. (AMZN)
* Tesla Inc. (TSLA)

The analysis covers the period from **January 2021 to December 2025** and focuses on a **10-trading-day pre-earnings event window**.

The project uses unsupervised anomaly detection models to detect unusual price, volume, and volatility patterns. Explainable AI is then applied to improve interpretability of the selected model.

This project does **not** make claims of fraud, insider trading, or market manipulation. The analysis is limited to identifying unusual trading behaviour and event-driven financial anomalies using public secondary financial data.

---

## Research Scope

| Component       | Description                                             |
| --------------- | ------------------------------------------------------- |
| Companies       | AAPL, MSFT, NVDA, AMZN, TSLA                            |
| Time Period     | January 2021 to December 2025                           |
| Event Focus     | 10 trading days before quarterly earnings announcements |
| Data Type       | Public secondary financial market data                  |
| Models          | Isolation Forest, One-Class SVM, Local Outlier Factor   |
| Evaluation Type | Unsupervised event-based evaluation                     |
| Explainability  | SHAP-based interpretation for the selected model        |

---

## Methodology Summary

The project follows a structured data science workflow:

1. Data collection
2. Data preprocessing
3. Feature engineering
4. Event-window construction
5. Unsupervised anomaly detection
6. Event-based model evaluation
7. Explainable AI interpretation
8. Streamlit dashboard development

The following financial features are used in the analysis:

* Daily Return
* Relative Volume
* Rolling Volatility 10-Day
* 10-Day Moving Average
* Volume Change %
* Price Change %
* Distance From Earnings Event
* Event Window Indicator

---

## Machine Learning Models

The following unsupervised anomaly detection models are implemented:

| Model                | Purpose                                                           |
| -------------------- | ----------------------------------------------------------------- |
| Isolation Forest     | Detects anomalies by isolating unusual observations               |
| One-Class SVM        | Learns the boundary of normal behaviour and identifies deviations |
| Local Outlier Factor | Detects local density-based anomalies                             |

The evaluation is based on event-driven anomaly behaviour rather than supervised classification. Therefore, metrics such as precision, recall, F1-score, ROC-AUC, and confusion matrix are not used because there is no verified labelled fraud or insider trading dataset.

---

## Evaluation Approach

The project applies an unsupervised event-based evaluation framework using:

* Anomaly detection rate within event windows
* Relative anomaly concentration
* Model agreement and consistency
* Anomaly score stability
* False positive analysis outside event windows
* Explainability-based interpretation of detected anomalies

---

## Project Structure

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
│   ├── evaluation/
│   ├── models/
│   └── shap/
│
└── evidence/
    ├── requirements_install_log.txt
    ├── pip_check_log.txt
    ├── frozen_environment_packages.txt
    ├── python_syntax_check_log.txt
    ├── notebook_01_execution_log.txt
    ├── notebook_02_execution_log.txt
    ├── notebook_03_execution_log.txt
    ├── notebook_04_execution_log.txt
    ├── notebook_05_execution_log.txt
    └── executed_notebooks/
```

---

## Reproducibility and Execution Instructions

This repository was tested in a fresh Python virtual environment to verify:

* Dependency installation through `requirements.txt`
* Package compatibility using `pip check`
* Python syntax validity
* Successful notebook execution
* Streamlit dashboard functionality

Execution evidence is provided in the `evidence/` folder.

---

## Fresh Environment Setup

To reproduce the project, first clone the repository:

```bash
git clone https://github.com/Dev2104/explainable-financial-anomaly-detection.git
cd explainable-financial-anomaly-detection
```

Create a fresh virtual environment.

For Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

For macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Verify dependency compatibility:

```bash
pip check
```

During testing, `pip check` returned:

```text
No broken requirements found.
```

---

## Python Syntax Verification

All Python files were checked for syntax issues using Python compilation checks.

The syntax-check evidence is available in:

```text
evidence/python_syntax_check_log.txt
```

The syntax verification returned:

```text
All Python files passed syntax check.
```

---

## Notebook Execution Order

To reproduce the reported results, the notebooks should be executed in the following order:

```text
1. notebooks/01_data_collection_check.ipynb
2. notebooks/02_feature_engineering_check.ipynb
3. notebooks/03_model_testing.ipynb
4. notebooks/04_evaluation_results.ipynb
5. notebooks/05_shap_explainability.ipynb
```

---

## Purpose of Each Notebook

| Notebook                             | Purpose                                                        |
| ------------------------------------ | -------------------------------------------------------------- |
| `01_data_collection_check.ipynb`     | Verifies stock price data and earnings date data collection    |
| `02_feature_engineering_check.ipynb` | Creates event-window data and engineered financial features    |
| `03_model_testing.ipynb`             | Runs Isolation Forest, One-Class SVM, and Local Outlier Factor |
| `04_evaluation_results.ipynb`        | Performs unsupervised event-based model evaluation             |
| `05_shap_explainability.ipynb`       | Applies SHAP explainability to the selected model              |

---

## Notebook Execution Evidence

The notebooks were executed using `jupyter nbconvert` in the fresh environment.

Example command:

```bash
jupyter nbconvert --to notebook --execute "notebooks/01_data_collection_check.ipynb" --output-dir "evidence/executed_notebooks" --output "01_data_collection_check_executed.ipynb" --ExecutePreprocessor.timeout=900
```

All five notebooks completed successfully with exit code `0`.

The execution logs are available in:

```text
evidence/notebook_01_execution_log.txt
evidence/notebook_02_execution_log.txt
evidence/notebook_03_execution_log.txt
evidence/notebook_04_execution_log.txt
evidence/notebook_05_execution_log.txt
```

The executed notebook outputs are available in:

```text
evidence/executed_notebooks/
```

---

## Running the Streamlit Dashboard

The Streamlit dashboard can be launched locally using:

```bash
streamlit run app.py
```

The dashboard provides an interactive presentation of:

* Project overview
* Dataset structure
* Event-window analysis
* Anomaly detection outputs
* Model evaluation results
* Best model selection
* SHAP explainability results

The hosted version of the dashboard is available here:

```text
https://explainable-financial-anomaly-detection-devt2104.streamlit.app/
```

---

## Execution Evidence Folder

The `evidence/` folder contains proof that the project was tested and reproduced in a fresh environment.

| Evidence File                     | Purpose                                                                 |
| --------------------------------- | ----------------------------------------------------------------------- |
| `requirements_install_log.txt`    | Shows dependency installation using `pip install -r requirements.txt`   |
| `pip_check_log.txt`               | Confirms that no broken requirements were found                         |
| `frozen_environment_packages.txt` | Lists the exact package versions installed in the test environment      |
| `python_syntax_check_log.txt`     | Shows Python syntax verification results                                |
| `notebook_01_execution_log.txt`   | Execution log for Notebook 1                                            |
| `notebook_02_execution_log.txt`   | Execution log for Notebook 2                                            |
| `notebook_03_execution_log.txt`   | Execution log for Notebook 3                                            |
| `notebook_04_execution_log.txt`   | Execution log for Notebook 4                                            |
| `notebook_05_execution_log.txt`   | Execution log for Notebook 5                                            |
| `executed_notebooks/`             | Contains executed notebook copies generated during reproduction testing |

---

## Data Sources

The project uses public secondary financial data sources, including:

* Yahoo Finance
* Nasdaq Earnings Calendar
* Investing.com Earnings Calendar

No private, confidential, personal, sensitive, or human participant data is used.

---

## Ethical and Research Limitations

This project is designed for academic research purposes only.

The detected anomalies represent unusual trading behaviour based on statistical and machine learning patterns. They should not be interpreted as evidence of:

* Fraud
* Insider trading
* Market manipulation
* Illegal trading activity

The project does not identify individuals, traders, institutions, or confidential transactions. All analysis is based on public market-level financial data.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* SHAP
* Plotly
* Streamlit
* Jupyter Notebook
* Matplotlib
* GitHub
* Streamlit Cloud

---

## How to Reproduce the Full Project

Follow this order:

```text
1. Clone the GitHub repository
2. Create and activate a fresh Python virtual environment
3. Install dependencies using requirements.txt
4. Run pip check
5. Execute notebooks in the documented order
6. Review generated outputs in the outputs/ folder
7. Launch the Streamlit dashboard using streamlit run app.py
8. Review execution evidence in the evidence/ folder
```

---

## Disclaimer

This repository is part of an MSc Data Science dissertation project. It is intended for academic and educational use only. The findings should not be used as financial advice, trading advice, legal evidence, or regulatory conclusions.
