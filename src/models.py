"""
models.py

This script applies unsupervised anomaly detection models for the dissertation project:

Evaluating Explainable AI-Based Anomaly Detection Models for Financial Trading Surveillance.

The models are used to identify unusual trading behaviour and anomalous market activity
around quarterly earnings announcements.

Models used:
1. Isolation Forest
2. One-Class Support Vector Machine
3. Local Outlier Factor

The project uses unsupervised event-based evaluation, not supervised classification.
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from config import PROCESSED_DATA_PATH, OUTPUT_MODELS_PATH, OUTPUT_TABLES_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / PROCESSED_DATA_PATH
OUTPUT_MODELS_DIR = PROJECT_ROOT / OUTPUT_MODELS_PATH
OUTPUT_TABLES_DIR = PROJECT_ROOT / OUTPUT_TABLES_PATH

FEATURE_DATA_FILE = PROCESSED_DIR / "final_feature_dataset.csv"

MODEL_OUTPUT_FILE = OUTPUT_MODELS_DIR / "model_anomaly_outputs.csv"
MODEL_SUMMARY_FILE = OUTPUT_TABLES_DIR / "model_output_summary.csv"


FEATURE_COLUMNS = [
    "Daily_Return",
    "Relative_Volume",
    "Rolling_Volatility_10D",
    "Moving_Average_10D",
    "Volume_Change_Pct",
    "Price_Change_Pct",
    "Distance_From_Earnings_Event",
]


RANDOM_STATE = 42
CONTAMINATION_RATE = 0.05


def ensure_output_directories() -> None:
    """
    Create required output directories if they do not already exist.
    """
    OUTPUT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)


def load_feature_dataset() -> pd.DataFrame:
    """
    Load the final feature dataset created during feature engineering.
    """
    if not FEATURE_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Final feature dataset not found: {FEATURE_DATA_FILE}. "
            "Please run src/feature_engineering.py first."
        )

    data = pd.read_csv(FEATURE_DATA_FILE)

    required_columns = [
        "Date",
        "Ticker",
        "Event_Window",
    ] + FEATURE_COLUMNS

    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns in feature dataset: {missing_columns}")

    data["Date"] = pd.to_datetime(data["Date"])
    data["Ticker"] = data["Ticker"].astype(str).str.upper().str.strip()
    data["Event_Window"] = data["Event_Window"].astype(int)

    data = data.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    return data


def convert_sklearn_prediction_to_binary(predictions) -> list[int]:
    """
    Convert sklearn anomaly prediction format into dissertation-friendly binary format.

    sklearn:
    -1 = anomaly
     1 = normal

    dissertation output:
     1 = anomaly
     0 = normal
    """
    return [1 if prediction == -1 else 0 for prediction in predictions]


def run_isolation_forest(feature_matrix) -> tuple[list[int], list[float]]:
    """
    Run Isolation Forest and return binary anomaly predictions and anomaly scores.
    """
    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION_RATE,
        random_state=RANDOM_STATE,
    )

    predictions = model.fit_predict(feature_matrix)

    # Lower decision_function values indicate more abnormal observations.
    raw_scores = model.decision_function(feature_matrix)

    anomaly_predictions = convert_sklearn_prediction_to_binary(predictions)
    anomaly_scores = [-score for score in raw_scores]

    return anomaly_predictions, anomaly_scores


def run_one_class_svm(feature_matrix) -> tuple[list[int], list[float]]:
    """
    Run One-Class SVM and return binary anomaly predictions and anomaly scores.
    """
    model = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=CONTAMINATION_RATE,
    )

    predictions = model.fit_predict(feature_matrix)

    # Lower decision_function values indicate more abnormal observations.
    raw_scores = model.decision_function(feature_matrix)

    anomaly_predictions = convert_sklearn_prediction_to_binary(predictions)
    anomaly_scores = [-score for score in raw_scores]

    return anomaly_predictions, anomaly_scores


def run_local_outlier_factor(feature_matrix) -> tuple[list[int], list[float]]:
    """
    Run Local Outlier Factor and return binary anomaly predictions and anomaly scores.
    """
    model = LocalOutlierFactor(
        n_neighbors=20,
        contamination=CONTAMINATION_RATE,
        novelty=False,
    )

    predictions = model.fit_predict(feature_matrix)

    # negative_outlier_factor_ is more negative for more abnormal observations.
    raw_scores = model.negative_outlier_factor_

    anomaly_predictions = convert_sklearn_prediction_to_binary(predictions)
    anomaly_scores = [-score for score in raw_scores]

    return anomaly_predictions, anomaly_scores


def run_models_for_ticker(ticker_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Run all anomaly detection models for one ticker.

    Models are fitted ticker-by-ticker to reduce distortion caused by differences in
    price scale, volume scale, and trading behaviour across companies.
    """
    ticker_data = ticker_data.copy()
    ticker_data = ticker_data.sort_values("Date").reset_index(drop=True)

    feature_matrix = ticker_data[FEATURE_COLUMNS].copy()

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)

    print(f"Running Isolation Forest for {ticker}...")
    if_predictions, if_scores = run_isolation_forest(scaled_features)

    print(f"Running One-Class SVM for {ticker}...")
    ocsvm_predictions, ocsvm_scores = run_one_class_svm(scaled_features)

    print(f"Running Local Outlier Factor for {ticker}...")
    lof_predictions, lof_scores = run_local_outlier_factor(scaled_features)

    ticker_data["IF_Anomaly"] = if_predictions
    ticker_data["IF_Anomaly_Score"] = if_scores

    ticker_data["OCSVM_Anomaly"] = ocsvm_predictions
    ticker_data["OCSVM_Anomaly_Score"] = ocsvm_scores

    ticker_data["LOF_Anomaly"] = lof_predictions
    ticker_data["LOF_Anomaly_Score"] = lof_scores

    ticker_data["Model_Agreement_Count"] = (
        ticker_data["IF_Anomaly"]
        + ticker_data["OCSVM_Anomaly"]
        + ticker_data["LOF_Anomaly"]
    )

    print(
        f"{ticker}: "
        f"IF anomalies={ticker_data['IF_Anomaly'].sum()}, "
        f"OCSVM anomalies={ticker_data['OCSVM_Anomaly'].sum()}, "
        f"LOF anomalies={ticker_data['LOF_Anomaly'].sum()}"
    )

    return ticker_data


def run_all_models(feature_data: pd.DataFrame) -> pd.DataFrame:
    """
    Run all selected anomaly detection models for all tickers.
    """
    all_model_outputs = []

    tickers = sorted(feature_data["Ticker"].unique())

    for ticker in tickers:
        print("-" * 70)
        print(f"Processing ticker: {ticker}")

        ticker_data = feature_data[feature_data["Ticker"] == ticker].copy()

        if ticker_data.empty:
            print(f"WARNING: No feature data found for {ticker}.")
            continue

        ticker_output = run_models_for_ticker(ticker_data, ticker)
        all_model_outputs.append(ticker_output)

    if not all_model_outputs:
        raise ValueError("No model outputs were created.")

    model_output_data = pd.concat(all_model_outputs, ignore_index=True)
    model_output_data = model_output_data.sort_values(["Ticker", "Date"]).reset_index(
        drop=True
    )

    return model_output_data


def create_model_output_summary(model_output_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary table of model anomaly outputs by ticker.
    """
    summary_rows = []

    model_columns = {
        "Isolation Forest": "IF_Anomaly",
        "One-Class SVM": "OCSVM_Anomaly",
        "Local Outlier Factor": "LOF_Anomaly",
    }

    for ticker in sorted(model_output_data["Ticker"].unique()):
        ticker_data = model_output_data[model_output_data["Ticker"] == ticker]

        total_rows = len(ticker_data)
        event_window_days = int(ticker_data["Event_Window"].sum())

        for model_name, anomaly_column in model_columns.items():
            total_anomalies = int(ticker_data[anomaly_column].sum())
            event_window_anomalies = int(
                ticker_data[
                    (ticker_data["Event_Window"] == 1)
                    & (ticker_data[anomaly_column] == 1)
                ].shape[0]
            )
            non_event_window_anomalies = total_anomalies - event_window_anomalies

            anomaly_rate = total_anomalies / total_rows if total_rows > 0 else 0
            event_anomaly_rate = (
                event_window_anomalies / event_window_days
                if event_window_days > 0
                else 0
            )

            summary_rows.append(
                {
                    "Ticker": ticker,
                    "Model": model_name,
                    "Total_Rows": total_rows,
                    "Event_Window_Days": event_window_days,
                    "Total_Anomalies": total_anomalies,
                    "Total_Anomaly_Rate": round(anomaly_rate, 4),
                    "Event_Window_Anomalies": event_window_anomalies,
                    "Event_Window_Anomaly_Rate": round(event_anomaly_rate, 4),
                    "Non_Event_Window_Anomalies": non_event_window_anomalies,
                }
            )

    summary = pd.DataFrame(summary_rows)

    return summary


def validate_model_outputs(model_output_data: pd.DataFrame) -> None:
    """
    Validate model output dataset before saving.
    """
    required_columns = [
        "Date",
        "Ticker",
        "Event_Window",
        "IF_Anomaly",
        "IF_Anomaly_Score",
        "OCSVM_Anomaly",
        "OCSVM_Anomaly_Score",
        "LOF_Anomaly",
        "LOF_Anomaly_Score",
        "Model_Agreement_Count",
    ]

    missing_columns = [col for col in required_columns if col not in model_output_data.columns]

    if missing_columns:
        raise ValueError(f"Model output dataset is missing columns: {missing_columns}")

    anomaly_columns = ["IF_Anomaly", "OCSVM_Anomaly", "LOF_Anomaly"]

    for column in anomaly_columns:
        unique_values = sorted(model_output_data[column].unique())
        if not set(unique_values).issubset({0, 1}):
            raise ValueError(f"{column} contains values other than 0 and 1: {unique_values}")

    print("\nModel output validation passed.")
    print(f"Total model output rows: {len(model_output_data)}")


def save_outputs(model_output_data: pd.DataFrame, summary: pd.DataFrame) -> None:
    """
    Save model anomaly outputs and model summary table.
    """
    data_to_save = model_output_data.copy()
    data_to_save["Date"] = data_to_save["Date"].dt.strftime("%Y-%m-%d")

    data_to_save.to_csv(MODEL_OUTPUT_FILE, index=False)
    summary.to_csv(MODEL_SUMMARY_FILE, index=False)

    print(f"\nModel anomaly outputs saved to: {MODEL_OUTPUT_FILE}")
    print(f"Model output summary saved to: {MODEL_SUMMARY_FILE}")


def main() -> None:
    """
    Main execution function.
    """
    print("=" * 70)
    print("Starting unsupervised anomaly detection model execution")
    print("Models: Isolation Forest, One-Class SVM, Local Outlier Factor")
    print("=" * 70)

    ensure_output_directories()

    feature_data = load_feature_dataset()

    print(f"Loaded final feature dataset rows: {len(feature_data)}")
    print(f"Tickers: {sorted(feature_data['Ticker'].unique())}")
    print(f"Feature columns used: {FEATURE_COLUMNS}")
    print(f"Contamination/nu setting: {CONTAMINATION_RATE}")

    model_output_data = run_all_models(feature_data)

    validate_model_outputs(model_output_data)

    summary = create_model_output_summary(model_output_data)

    print("\nModel Output Summary:")
    print(summary)

    save_outputs(model_output_data, summary)

    print("\nModel execution completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()