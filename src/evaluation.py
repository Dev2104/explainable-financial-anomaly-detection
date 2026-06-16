"""
evaluation.py

This script conducts unsupervised event-based evaluation for the dissertation project:

Evaluating Explainable AI-Based Anomaly Detection Models for Financial Trading Surveillance.

The evaluation does not use supervised classification metrics such as precision, recall,
F1 score, ROC-AUC, or confusion matrix.

Instead, the evaluation focuses on:
1. Anomaly detection rate within pre-earnings event windows
2. Relative anomaly concentration before earnings announcements
3. Model agreement and consistency
4. Anomaly score stability
5. Event-based false positive analysis outside event windows
6. Best model selection for explainability analysis
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config import OUTPUT_MODELS_PATH, OUTPUT_TABLES_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_MODELS_DIR = PROJECT_ROOT / OUTPUT_MODELS_PATH
OUTPUT_TABLES_DIR = PROJECT_ROOT / OUTPUT_TABLES_PATH

MODEL_OUTPUT_FILE = OUTPUT_MODELS_DIR / "model_anomaly_outputs.csv"

EVENT_WINDOW_DETECTION_RATE_FILE = OUTPUT_TABLES_DIR / "event_window_detection_rate.csv"
RELATIVE_ANOMALY_CONCENTRATION_FILE = OUTPUT_TABLES_DIR / "relative_anomaly_concentration.csv"
MODEL_AGREEMENT_SUMMARY_FILE = OUTPUT_TABLES_DIR / "model_agreement_summary.csv"
FALSE_POSITIVE_ANALYSIS_FILE = OUTPUT_TABLES_DIR / "false_positive_analysis.csv"
ANOMALY_SCORE_STABILITY_FILE = OUTPUT_TABLES_DIR / "anomaly_score_stability.csv"
OVERALL_EVALUATION_SUMMARY_FILE = OUTPUT_TABLES_DIR / "overall_model_evaluation_summary.csv"
BEST_MODEL_SELECTION_FILE = OUTPUT_TABLES_DIR / "best_model_selection_summary.csv"


MODEL_COLUMNS = {
    "Isolation Forest": {
        "anomaly_column": "IF_Anomaly",
        "score_column": "IF_Anomaly_Score",
    },
    "One-Class SVM": {
        "anomaly_column": "OCSVM_Anomaly",
        "score_column": "OCSVM_Anomaly_Score",
    },
    "Local Outlier Factor": {
        "anomaly_column": "LOF_Anomaly",
        "score_column": "LOF_Anomaly_Score",
    },
}


def ensure_output_directory() -> None:
    """
    Create output table directory if it does not already exist.
    """
    OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)


def load_model_outputs() -> pd.DataFrame:
    """
    Load model anomaly outputs created by src/models.py.
    """
    if not MODEL_OUTPUT_FILE.exists():
        raise FileNotFoundError(
            f"Model output file not found: {MODEL_OUTPUT_FILE}. "
            "Please run src/models.py first."
        )

    data = pd.read_csv(MODEL_OUTPUT_FILE)

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

    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns in model output file: {missing_columns}")

    data["Date"] = pd.to_datetime(data["Date"])
    data["Ticker"] = data["Ticker"].astype(str).str.upper().str.strip()
    data["Event_Window"] = data["Event_Window"].astype(int)

    anomaly_columns = ["IF_Anomaly", "OCSVM_Anomaly", "LOF_Anomaly"]

    for column in anomaly_columns:
        data[column] = data[column].astype(int)

    data = data.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    return data


def calculate_event_window_detection_rate(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate anomaly detection rate within earnings event windows by ticker and model.

    This measures the proportion of event-window trading days flagged as anomalous.
    """
    rows = []

    for ticker in sorted(data["Ticker"].unique()):
        ticker_data = data[data["Ticker"] == ticker]

        event_data = ticker_data[ticker_data["Event_Window"] == 1]
        non_event_data = ticker_data[ticker_data["Event_Window"] == 0]

        event_window_days = len(event_data)
        non_event_window_days = len(non_event_data)

        for model_name, columns in MODEL_COLUMNS.items():
            anomaly_column = columns["anomaly_column"]

            event_window_anomalies = int(event_data[anomaly_column].sum())
            non_event_window_anomalies = int(non_event_data[anomaly_column].sum())
            total_anomalies = int(ticker_data[anomaly_column].sum())

            event_window_detection_rate = (
                event_window_anomalies / event_window_days
                if event_window_days > 0
                else np.nan
            )

            non_event_window_anomaly_rate = (
                non_event_window_anomalies / non_event_window_days
                if non_event_window_days > 0
                else np.nan
            )

            rows.append(
                {
                    "Ticker": ticker,
                    "Model": model_name,
                    "Total_Trading_Days": len(ticker_data),
                    "Event_Window_Days": event_window_days,
                    "Non_Event_Window_Days": non_event_window_days,
                    "Total_Anomalies": total_anomalies,
                    "Event_Window_Anomalies": event_window_anomalies,
                    "Non_Event_Window_Anomalies": non_event_window_anomalies,
                    "Event_Window_Detection_Rate": round(event_window_detection_rate, 4),
                    "Non_Event_Window_Anomaly_Rate": round(non_event_window_anomaly_rate, 4),
                }
            )

    return pd.DataFrame(rows)


def calculate_relative_anomaly_concentration(
    event_detection_rate: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate relative anomaly concentration.

    Formula:
    Event-window anomaly rate / Non-event-window anomaly rate

    A value above 1 suggests that anomaly alerts are more concentrated during
    pre-earnings event windows than outside those windows.
    """
    concentration_data = event_detection_rate.copy()

    concentration_data["Relative_Anomaly_Concentration"] = concentration_data.apply(
        lambda row: row["Event_Window_Detection_Rate"] / row["Non_Event_Window_Anomaly_Rate"]
        if row["Non_Event_Window_Anomaly_Rate"] > 0
        else np.nan,
        axis=1,
    )

    concentration_data["Relative_Anomaly_Concentration"] = concentration_data[
        "Relative_Anomaly_Concentration"
    ].round(4)

    concentration_data["Concentration_Interpretation"] = concentration_data[
        "Relative_Anomaly_Concentration"
    ].apply(
        lambda value: "Higher concentration inside event windows"
        if value > 1
        else "Higher or similar concentration outside event windows"
    )

    return concentration_data[
        [
            "Ticker",
            "Model",
            "Event_Window_Detection_Rate",
            "Non_Event_Window_Anomaly_Rate",
            "Relative_Anomaly_Concentration",
            "Concentration_Interpretation",
        ]
    ]


def calculate_model_agreement_summary(data: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate agreement between anomaly detection models.

    Model_Agreement_Count:
    0 = no model flagged anomaly
    1 = one model flagged anomaly
    2 = two models flagged anomaly
    3 = all three models flagged anomaly
    """
    agreement_rows = []

    total_rows = len(data)

    for event_status, event_label in [(1, "Event Window"), (0, "Non-Event Window")]:
        subset = data[data["Event_Window"] == event_status]

        for agreement_count in [0, 1, 2, 3]:
            count = int((subset["Model_Agreement_Count"] == agreement_count).sum())

            agreement_rows.append(
                {
                    "Window_Type": event_label,
                    "Model_Agreement_Count": agreement_count,
                    "Rows": count,
                    "Share_Within_Window": round(count / len(subset), 4)
                    if len(subset) > 0
                    else np.nan,
                    "Share_Of_Total_Dataset": round(count / total_rows, 4)
                    if total_rows > 0
                    else np.nan,
                }
            )

    strong_agreement_event = data[
        (data["Event_Window"] == 1) & (data["Model_Agreement_Count"] >= 2)
    ].shape[0]

    strong_agreement_non_event = data[
        (data["Event_Window"] == 0) & (data["Model_Agreement_Count"] >= 2)
    ].shape[0]

    event_window_days = data[data["Event_Window"] == 1].shape[0]
    non_event_window_days = data[data["Event_Window"] == 0].shape[0]

    agreement_rows.append(
        {
            "Window_Type": "Event Window",
            "Model_Agreement_Count": "Strong Agreement >= 2",
            "Rows": strong_agreement_event,
            "Share_Within_Window": round(strong_agreement_event / event_window_days, 4)
            if event_window_days > 0
            else np.nan,
            "Share_Of_Total_Dataset": round(strong_agreement_event / total_rows, 4)
            if total_rows > 0
            else np.nan,
        }
    )

    agreement_rows.append(
        {
            "Window_Type": "Non-Event Window",
            "Model_Agreement_Count": "Strong Agreement >= 2",
            "Rows": strong_agreement_non_event,
            "Share_Within_Window": round(
                strong_agreement_non_event / non_event_window_days, 4
            )
            if non_event_window_days > 0
            else np.nan,
            "Share_Of_Total_Dataset": round(strong_agreement_non_event / total_rows, 4)
            if total_rows > 0
            else np.nan,
        }
    )

    return pd.DataFrame(agreement_rows)


def calculate_false_positive_analysis(data: pd.DataFrame) -> pd.DataFrame:
    """
    Conduct event-based false positive analysis.

    In this dissertation, anomaly alerts outside the pre-earnings event windows are treated
    as potential false positive alerts only from the perspective of the event-based
    evaluation framework. This does not mean those observations are financially irrelevant.
    """
    rows = []

    for ticker in sorted(data["Ticker"].unique()):
        ticker_data = data[data["Ticker"] == ticker]
        non_event_data = ticker_data[ticker_data["Event_Window"] == 0]

        non_event_window_days = len(non_event_data)

        for model_name, columns in MODEL_COLUMNS.items():
            anomaly_column = columns["anomaly_column"]

            total_anomalies = int(ticker_data[anomaly_column].sum())
            outside_event_alerts = int(non_event_data[anomaly_column].sum())

            outside_event_alert_rate = (
                outside_event_alerts / non_event_window_days
                if non_event_window_days > 0
                else np.nan
            )

            outside_alert_share_of_all_alerts = (
                outside_event_alerts / total_anomalies
                if total_anomalies > 0
                else np.nan
            )

            rows.append(
                {
                    "Ticker": ticker,
                    "Model": model_name,
                    "Non_Event_Window_Days": non_event_window_days,
                    "Total_Anomaly_Alerts": total_anomalies,
                    "Event_Based_False_Positive_Alerts": outside_event_alerts,
                    "Event_Based_False_Positive_Rate": round(outside_event_alert_rate, 4),
                    "False_Positive_Share_Of_All_Alerts": round(
                        outside_alert_share_of_all_alerts, 4
                    ),
                    "Interpretation_Note": (
                        "Outside event-window alert under the event-based evaluation framework"
                    ),
                }
            )

    return pd.DataFrame(rows)


def calculate_anomaly_score_stability(data: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate stability of anomaly scores.

    Lower standard deviation suggests more stable anomaly scoring behaviour.
    """
    rows = []

    for ticker in sorted(data["Ticker"].unique()):
        ticker_data = data[data["Ticker"] == ticker]

        for model_name, columns in MODEL_COLUMNS.items():
            score_column = columns["score_column"]

            rows.append(
                {
                    "Ticker": ticker,
                    "Model": model_name,
                    "Score_Mean": round(ticker_data[score_column].mean(), 6),
                    "Score_Std": round(ticker_data[score_column].std(), 6),
                    "Score_Min": round(ticker_data[score_column].min(), 6),
                    "Score_Max": round(ticker_data[score_column].max(), 6),
                    "Score_Range": round(
                        ticker_data[score_column].max() - ticker_data[score_column].min(),
                        6,
                    ),
                }
            )

    return pd.DataFrame(rows)


def create_overall_model_evaluation_summary(
    event_detection_rate: pd.DataFrame,
    relative_concentration: pd.DataFrame,
    false_positive_analysis: pd.DataFrame,
    score_stability: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create one overall evaluation summary across all tickers.
    """
    overall_rows = []

    for model_name in MODEL_COLUMNS.keys():
        model_event_data = event_detection_rate[event_detection_rate["Model"] == model_name]
        model_concentration_data = relative_concentration[
            relative_concentration["Model"] == model_name
        ]
        model_false_positive_data = false_positive_analysis[
            false_positive_analysis["Model"] == model_name
        ]
        model_stability_data = score_stability[score_stability["Model"] == model_name]

        total_trading_days = int(model_event_data["Total_Trading_Days"].sum())
        total_event_days = int(model_event_data["Event_Window_Days"].sum())
        total_non_event_days = int(model_event_data["Non_Event_Window_Days"].sum())

        total_anomalies = int(model_event_data["Total_Anomalies"].sum())
        total_event_anomalies = int(model_event_data["Event_Window_Anomalies"].sum())
        total_non_event_anomalies = int(
            model_event_data["Non_Event_Window_Anomalies"].sum()
        )

        event_detection_rate_overall = (
            total_event_anomalies / total_event_days if total_event_days > 0 else np.nan
        )

        non_event_anomaly_rate_overall = (
            total_non_event_anomalies / total_non_event_days
            if total_non_event_days > 0
            else np.nan
        )

        relative_concentration_overall = (
            event_detection_rate_overall / non_event_anomaly_rate_overall
            if non_event_anomaly_rate_overall > 0
            else np.nan
        )

        overall_rows.append(
            {
                "Model": model_name,
                "Total_Trading_Days": total_trading_days,
                "Total_Event_Window_Days": total_event_days,
                "Total_Non_Event_Window_Days": total_non_event_days,
                "Total_Anomaly_Alerts": total_anomalies,
                "Total_Event_Window_Anomalies": total_event_anomalies,
                "Total_Non_Event_Window_Anomalies": total_non_event_anomalies,
                "Overall_Event_Window_Detection_Rate": round(
                    event_detection_rate_overall, 4
                ),
                "Overall_Non_Event_Window_Anomaly_Rate": round(
                    non_event_anomaly_rate_overall, 4
                ),
                "Overall_Relative_Anomaly_Concentration": round(
                    relative_concentration_overall, 4
                ),
                "Mean_Ticker_Level_Relative_Concentration": round(
                    model_concentration_data["Relative_Anomaly_Concentration"].mean(), 4
                ),
                "Mean_Event_Based_False_Positive_Rate": round(
                    model_false_positive_data["Event_Based_False_Positive_Rate"].mean(), 4
                ),
                "Mean_Anomaly_Score_Std": round(model_stability_data["Score_Std"].mean(), 6),
            }
        )

    return pd.DataFrame(overall_rows)


def create_best_model_selection_summary(
    overall_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rank models using unsupervised event-based evaluation criteria.

    Ranking logic:
    - Higher event-window detection rate is better.
    - Higher relative anomaly concentration is better.
    - Lower event-based false positive rate is better.
    - Lower anomaly score standard deviation is better.

    The selected model is the most suitable model for SHAP explainability analysis.
    """
    selection = overall_summary.copy()

    selection["Rank_Event_Window_Detection"] = selection[
        "Overall_Event_Window_Detection_Rate"
    ].rank(ascending=False, method="min")

    selection["Rank_Relative_Concentration"] = selection[
        "Overall_Relative_Anomaly_Concentration"
    ].rank(ascending=False, method="min")

    selection["Rank_False_Positive_Behaviour"] = selection[
        "Mean_Event_Based_False_Positive_Rate"
    ].rank(ascending=True, method="min")

    selection["Rank_Score_Stability"] = selection["Mean_Anomaly_Score_Std"].rank(
        ascending=True,
        method="min",
    )

    selection["Composite_Rank_Score"] = (
        selection["Rank_Event_Window_Detection"]
        + selection["Rank_Relative_Concentration"]
        + selection["Rank_False_Positive_Behaviour"]
        + selection["Rank_Score_Stability"]
    )

    selection = selection.sort_values(
        ["Composite_Rank_Score", "Rank_Event_Window_Detection"]
    ).reset_index(drop=True)

    selection["Suggested_For_SHAP"] = "No"
    selection.loc[0, "Suggested_For_SHAP"] = "Yes"

    selection["Selection_Note"] = selection["Suggested_For_SHAP"].apply(
        lambda value: (
            "Most suitable model for SHAP explainability based on event-based evaluation"
            if value == "Yes"
            else "Not selected as primary explainability model"
        )
    )

    return selection


def save_outputs(
    event_detection_rate: pd.DataFrame,
    relative_concentration: pd.DataFrame,
    model_agreement_summary: pd.DataFrame,
    false_positive_analysis: pd.DataFrame,
    score_stability: pd.DataFrame,
    overall_summary: pd.DataFrame,
    best_model_selection: pd.DataFrame,
) -> None:
    """
    Save all evaluation output tables.
    """
    event_detection_rate.to_csv(EVENT_WINDOW_DETECTION_RATE_FILE, index=False)
    relative_concentration.to_csv(RELATIVE_ANOMALY_CONCENTRATION_FILE, index=False)
    model_agreement_summary.to_csv(MODEL_AGREEMENT_SUMMARY_FILE, index=False)
    false_positive_analysis.to_csv(FALSE_POSITIVE_ANALYSIS_FILE, index=False)
    score_stability.to_csv(ANOMALY_SCORE_STABILITY_FILE, index=False)
    overall_summary.to_csv(OVERALL_EVALUATION_SUMMARY_FILE, index=False)
    best_model_selection.to_csv(BEST_MODEL_SELECTION_FILE, index=False)

    print(f"\nSaved: {EVENT_WINDOW_DETECTION_RATE_FILE}")
    print(f"Saved: {RELATIVE_ANOMALY_CONCENTRATION_FILE}")
    print(f"Saved: {MODEL_AGREEMENT_SUMMARY_FILE}")
    print(f"Saved: {FALSE_POSITIVE_ANALYSIS_FILE}")
    print(f"Saved: {ANOMALY_SCORE_STABILITY_FILE}")
    print(f"Saved: {OVERALL_EVALUATION_SUMMARY_FILE}")
    print(f"Saved: {BEST_MODEL_SELECTION_FILE}")


def main() -> None:
    """
    Main execution function.
    """
    print("=" * 70)
    print("Starting unsupervised event-based evaluation")
    print("No supervised classification metrics are used.")
    print("=" * 70)

    ensure_output_directory()

    model_output_data = load_model_outputs()

    print(f"Loaded model output rows: {len(model_output_data)}")
    print(f"Tickers: {sorted(model_output_data['Ticker'].unique())}")

    event_detection_rate = calculate_event_window_detection_rate(model_output_data)
    relative_concentration = calculate_relative_anomaly_concentration(event_detection_rate)
    model_agreement_summary = calculate_model_agreement_summary(model_output_data)
    false_positive_analysis = calculate_false_positive_analysis(model_output_data)
    score_stability = calculate_anomaly_score_stability(model_output_data)

    overall_summary = create_overall_model_evaluation_summary(
        event_detection_rate=event_detection_rate,
        relative_concentration=relative_concentration,
        false_positive_analysis=false_positive_analysis,
        score_stability=score_stability,
    )

    best_model_selection = create_best_model_selection_summary(overall_summary)

    print("\nOverall Model Evaluation Summary:")
    print(overall_summary)

    print("\nBest Model Selection Summary:")
    print(best_model_selection)

    selected_model = best_model_selection.loc[
        best_model_selection["Suggested_For_SHAP"] == "Yes", "Model"
    ].iloc[0]

    print(f"\nSuggested model for SHAP explainability: {selected_model}")

    save_outputs(
        event_detection_rate=event_detection_rate,
        relative_concentration=relative_concentration,
        model_agreement_summary=model_agreement_summary,
        false_positive_analysis=false_positive_analysis,
        score_stability=score_stability,
        overall_summary=overall_summary,
        best_model_selection=best_model_selection,
    )

    print("\nEvent-based evaluation completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()