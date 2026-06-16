"""
visualisations.py

This script creates Plotly-based visualisations for the dissertation project:

Evaluating Explainable AI-Based Anomaly Detection Models for Financial Trading Surveillance.

All charts are saved as interactive HTML files to support dissertation analysis and
future Streamlit dashboard integration.

The project focuses on unusual trading behaviour, anomalous market activity,
event-driven financial anomalies, and Explainable AI-based interpretation.
"""

from pathlib import Path

import pandas as pd
import plotly.express as px

from config import OUTPUT_MODELS_PATH, OUTPUT_TABLES_PATH, OUTPUT_CHARTS_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_MODELS_DIR = PROJECT_ROOT / OUTPUT_MODELS_PATH
OUTPUT_TABLES_DIR = PROJECT_ROOT / OUTPUT_TABLES_PATH
OUTPUT_CHARTS_DIR = PROJECT_ROOT / OUTPUT_CHARTS_PATH

MODEL_OUTPUT_FILE = OUTPUT_MODELS_DIR / "model_anomaly_outputs.csv"

MODEL_SUMMARY_FILE = OUTPUT_TABLES_DIR / "model_output_summary.csv"
EVENT_WINDOW_DETECTION_RATE_FILE = OUTPUT_TABLES_DIR / "event_window_detection_rate.csv"
RELATIVE_ANOMALY_CONCENTRATION_FILE = OUTPUT_TABLES_DIR / "relative_anomaly_concentration.csv"
MODEL_AGREEMENT_SUMMARY_FILE = OUTPUT_TABLES_DIR / "model_agreement_summary.csv"
FALSE_POSITIVE_ANALYSIS_FILE = OUTPUT_TABLES_DIR / "false_positive_analysis.csv"
ANOMALY_SCORE_STABILITY_FILE = OUTPUT_TABLES_DIR / "anomaly_score_stability.csv"
OVERALL_EVALUATION_SUMMARY_FILE = OUTPUT_TABLES_DIR / "overall_model_evaluation_summary.csv"
BEST_MODEL_SELECTION_FILE = OUTPUT_TABLES_DIR / "best_model_selection_summary.csv"
SHAP_FEATURE_IMPORTANCE_FILE = OUTPUT_TABLES_DIR / "shap_feature_importance.csv"
SHAP_FEATURE_IMPORTANCE_BY_TICKER_FILE = OUTPUT_TABLES_DIR / "shap_feature_importance_by_ticker.csv"


MODEL_NAME_MAP = {
    "IF_Anomaly": "Isolation Forest",
    "OCSVM_Anomaly": "One-Class SVM",
    "LOF_Anomaly": "Local Outlier Factor",
}

MODEL_SCORE_MAP = {
    "IF_Anomaly_Score": "Isolation Forest",
    "OCSVM_Anomaly_Score": "One-Class SVM",
    "LOF_Anomaly_Score": "Local Outlier Factor",
}


def ensure_output_directory() -> None:
    """
    Create chart output directory if it does not already exist.
    """
    OUTPUT_CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(file_path: Path, required: bool = True) -> pd.DataFrame:
    """
    Load a CSV file safely.
    """
    if not file_path.exists():
        message = f"File not found: {file_path}"
        if required:
            raise FileNotFoundError(message)
        print(f"WARNING: {message}")
        return pd.DataFrame()

    return pd.read_csv(file_path)


def save_chart(fig, filename: str) -> None:
    """
    Save a Plotly chart as an interactive HTML file.
    """
    output_file = OUTPUT_CHARTS_DIR / filename

    fig.update_layout(
        template="plotly_white",
        title_x=0.02,
        margin=dict(l=40, r=40, t=75, b=45),
        legend_title_text="",
    )

    fig.write_html(output_file, include_plotlyjs="cdn")

    print(f"Saved chart: {output_file}")


def prepare_model_output_data() -> pd.DataFrame:
    """
    Load and prepare model output data.
    """
    data = load_csv(MODEL_OUTPUT_FILE)

    data["Date"] = pd.to_datetime(data["Date"])
    data["Ticker"] = data["Ticker"].astype(str).str.upper().str.strip()
    data["Event_Window_Label"] = data["Event_Window"].map(
        {1: "Pre-earnings event window", 0: "Outside event window"}
    )

    return data


def create_closing_price_trend(data: pd.DataFrame) -> None:
    """
    Create closing price trend chart by ticker.
    """
    fig = px.line(
        data,
        x="Date",
        y="Adj Close",
        color="Ticker",
        title="Adjusted Closing Price Trends by Company",
        labels={
            "Date": "Date",
            "Adj Close": "Adjusted closing price",
            "Ticker": "Ticker",
        },
    )

    save_chart(fig, "01_adjusted_closing_price_trends.html")


def create_trading_volume_trend(data: pd.DataFrame) -> None:
    """
    Create trading volume trend chart by ticker.
    """
    fig = px.line(
        data,
        x="Date",
        y="Volume",
        color="Ticker",
        title="Trading Volume Trends by Company",
        labels={
            "Date": "Date",
            "Volume": "Trading volume",
            "Ticker": "Ticker",
        },
    )

    save_chart(fig, "02_trading_volume_trends.html")


def create_event_window_timeline(data: pd.DataFrame) -> None:
    """
    Create chart showing event-window observations over time.
    """
    event_data = data[data["Event_Window"] == 1].copy()

    fig = px.scatter(
        event_data,
        x="Date",
        y="Ticker",
        color="Ticker",
        title="Pre-Earnings Event Window Observations",
        labels={
            "Date": "Date",
            "Ticker": "Company ticker",
        },
        hover_data=["Adj Close", "Volume", "Event_Earnings_Date"],
    )

    save_chart(fig, "03_event_window_timeline.html")


def create_anomaly_counts_by_model(model_summary: pd.DataFrame) -> None:
    """
    Create anomaly count comparison by model and ticker.
    """
    fig = px.bar(
        model_summary,
        x="Ticker",
        y="Total_Anomalies",
        color="Model",
        barmode="group",
        title="Total Anomaly Alerts by Model and Company",
        labels={
            "Ticker": "Company ticker",
            "Total_Anomalies": "Total anomaly alerts",
            "Model": "Model",
        },
        hover_data=["Total_Anomaly_Rate"],
    )

    save_chart(fig, "04_total_anomaly_alerts_by_model.html")


def create_event_vs_non_event_anomalies(event_detection_rate: pd.DataFrame) -> None:
    """
    Create event-window vs non-event-window anomaly comparison.
    """
    plot_data = event_detection_rate.melt(
        id_vars=["Ticker", "Model"],
        value_vars=["Event_Window_Anomalies", "Non_Event_Window_Anomalies"],
        var_name="Window_Type",
        value_name="Anomaly_Count",
    )

    plot_data["Window_Type"] = plot_data["Window_Type"].replace(
        {
            "Event_Window_Anomalies": "Pre-earnings event window",
            "Non_Event_Window_Anomalies": "Outside event window",
        }
    )

    fig = px.bar(
        plot_data,
        x="Ticker",
        y="Anomaly_Count",
        color="Window_Type",
        facet_col="Model",
        barmode="group",
        title="Anomaly Alerts Inside and Outside Event Windows",
        labels={
            "Ticker": "Company ticker",
            "Anomaly_Count": "Anomaly alert count",
            "Window_Type": "Window type",
        },
    )

    fig.update_xaxes(matches=None)

    save_chart(fig, "05_event_vs_non_event_anomalies.html")


def create_event_window_detection_rate_chart(event_detection_rate: pd.DataFrame) -> None:
    """
    Create event-window detection rate chart.
    """
    fig = px.bar(
        event_detection_rate,
        x="Ticker",
        y="Event_Window_Detection_Rate",
        color="Model",
        barmode="group",
        title="Anomaly Detection Rate Within Pre-Earnings Event Windows",
        labels={
            "Ticker": "Company ticker",
            "Event_Window_Detection_Rate": "Event-window detection rate",
            "Model": "Model",
        },
        hover_data=[
            "Event_Window_Anomalies",
            "Event_Window_Days",
            "Non_Event_Window_Anomaly_Rate",
        ],
    )

    save_chart(fig, "06_event_window_detection_rate.html")


def create_relative_anomaly_concentration_chart(relative_concentration: pd.DataFrame) -> None:
    """
    Create relative anomaly concentration chart.
    """
    fig = px.bar(
        relative_concentration,
        x="Ticker",
        y="Relative_Anomaly_Concentration",
        color="Model",
        barmode="group",
        title="Relative Anomaly Concentration Before Earnings Announcements",
        labels={
            "Ticker": "Company ticker",
            "Relative_Anomaly_Concentration": "Relative anomaly concentration",
            "Model": "Model",
        },
        hover_data=[
            "Event_Window_Detection_Rate",
            "Non_Event_Window_Anomaly_Rate",
            "Concentration_Interpretation",
        ],
    )

    fig.add_hline(
        y=1,
        line_dash="dash",
        annotation_text="Equal concentration threshold",
        annotation_position="top left",
    )

    save_chart(fig, "07_relative_anomaly_concentration.html")


def create_model_agreement_chart(model_agreement_summary: pd.DataFrame) -> None:
    """
    Create model agreement chart.
    """
    plot_data = model_agreement_summary.copy()
    plot_data["Model_Agreement_Count"] = plot_data["Model_Agreement_Count"].astype(str)

    fig = px.bar(
        plot_data,
        x="Model_Agreement_Count",
        y="Rows",
        color="Window_Type",
        barmode="group",
        title="Model Agreement Across Event and Non-Event Windows",
        labels={
            "Model_Agreement_Count": "Number of models agreeing",
            "Rows": "Number of observations",
            "Window_Type": "Window type",
        },
        hover_data=["Share_Within_Window", "Share_Of_Total_Dataset"],
    )

    save_chart(fig, "08_model_agreement_summary.html")


def create_false_positive_analysis_chart(false_positive_analysis: pd.DataFrame) -> None:
    """
    Create event-based false positive analysis chart.
    """
    fig = px.bar(
        false_positive_analysis,
        x="Ticker",
        y="Event_Based_False_Positive_Rate",
        color="Model",
        barmode="group",
        title="Event-Based False Positive Rate Outside Pre-Earnings Windows",
        labels={
            "Ticker": "Company ticker",
            "Event_Based_False_Positive_Rate": "Event-based false positive rate",
            "Model": "Model",
        },
        hover_data=[
            "Total_Anomaly_Alerts",
            "Event_Based_False_Positive_Alerts",
            "False_Positive_Share_Of_All_Alerts",
        ],
    )

    save_chart(fig, "09_event_based_false_positive_rate.html")


def create_anomaly_score_stability_chart(score_stability: pd.DataFrame) -> None:
    """
    Create anomaly score stability chart.
    """
    fig = px.bar(
        score_stability,
        x="Ticker",
        y="Score_Std",
        color="Model",
        barmode="group",
        title="Anomaly Score Stability by Model",
        labels={
            "Ticker": "Company ticker",
            "Score_Std": "Standard deviation of anomaly scores",
            "Model": "Model",
        },
        hover_data=["Score_Mean", "Score_Min", "Score_Max", "Score_Range"],
    )

    save_chart(fig, "10_anomaly_score_stability.html")


def create_overall_model_evaluation_chart(overall_summary: pd.DataFrame) -> None:
    """
    Create overall model evaluation summary chart.
    """
    plot_data = overall_summary.melt(
        id_vars=["Model"],
        value_vars=[
            "Overall_Event_Window_Detection_Rate",
            "Overall_Non_Event_Window_Anomaly_Rate",
            "Overall_Relative_Anomaly_Concentration",
            "Mean_Event_Based_False_Positive_Rate",
        ],
        var_name="Evaluation_Metric",
        value_name="Value",
    )

    plot_data["Evaluation_Metric"] = plot_data["Evaluation_Metric"].replace(
        {
            "Overall_Event_Window_Detection_Rate": "Event-window detection rate",
            "Overall_Non_Event_Window_Anomaly_Rate": "Non-event anomaly rate",
            "Overall_Relative_Anomaly_Concentration": "Relative anomaly concentration",
            "Mean_Event_Based_False_Positive_Rate": "Mean event-based false positive rate",
        }
    )

    fig = px.bar(
        plot_data,
        x="Evaluation_Metric",
        y="Value",
        color="Model",
        barmode="group",
        title="Overall Event-Based Model Evaluation Summary",
        labels={
            "Evaluation_Metric": "Evaluation metric",
            "Value": "Metric value",
            "Model": "Model",
        },
    )

    fig.update_xaxes(tickangle=-30)

    save_chart(fig, "11_overall_model_evaluation_summary.html")


def create_best_model_rank_chart(best_model_selection: pd.DataFrame) -> None:
    """
    Create best model selection rank chart.
    """
    fig = px.bar(
        best_model_selection,
        x="Model",
        y="Composite_Rank_Score",
        color="Suggested_For_SHAP",
        title="Best Model Selection for SHAP Explainability",
        labels={
            "Model": "Model",
            "Composite_Rank_Score": "Composite rank score",
            "Suggested_For_SHAP": "Selected for SHAP",
        },
        hover_data=[
            "Rank_Event_Window_Detection",
            "Rank_Relative_Concentration",
            "Rank_False_Positive_Behaviour",
            "Rank_Score_Stability",
            "Selection_Note",
        ],
    )

    save_chart(fig, "12_best_model_selection_rank.html")


def create_anomaly_timeline_for_selected_model(data: pd.DataFrame) -> None:
    """
    Create timeline of Isolation Forest anomaly alerts.
    """
    anomaly_data = data[data["IF_Anomaly"] == 1].copy()

    fig = px.scatter(
        anomaly_data,
        x="Date",
        y="IF_Anomaly_Score",
        color="Ticker",
        symbol="Event_Window_Label",
        title="Isolation Forest Anomaly Alerts Over Time",
        labels={
            "Date": "Date",
            "IF_Anomaly_Score": "Isolation Forest anomaly score",
            "Ticker": "Company ticker",
            "Event_Window_Label": "Window type",
        },
        hover_data=[
            "Ticker",
            "Event_Window_Label",
            "Adj Close",
            "Volume",
            "Relative_Volume",
            "Rolling_Volatility_10D",
        ],
    )

    save_chart(fig, "13_isolation_forest_anomaly_timeline.html")


def create_shap_feature_importance_chart(shap_feature_importance: pd.DataFrame) -> None:
    """
    Create SHAP feature importance chart from saved SHAP output.
    """
    if shap_feature_importance.empty:
        print("WARNING: SHAP feature importance file missing. Skipping SHAP chart.")
        return

    plot_data = shap_feature_importance.sort_values(
        "Mean_Absolute_SHAP_Value",
        ascending=True,
    )

    fig = px.bar(
        plot_data,
        x="Mean_Absolute_SHAP_Value",
        y="Feature",
        orientation="h",
        title="SHAP Feature Importance for Isolation Forest",
        labels={
            "Mean_Absolute_SHAP_Value": "Mean absolute SHAP value",
            "Feature": "Trading feature",
        },
        hover_data=["Importance_Rank", "Mean_SHAP_Value"],
    )

    save_chart(fig, "14_shap_feature_importance.html")


def create_shap_feature_importance_by_ticker_chart(
    shap_feature_importance_by_ticker: pd.DataFrame,
) -> None:
    """
    Create SHAP feature importance chart by ticker.
    """
    if shap_feature_importance_by_ticker.empty:
        print("WARNING: SHAP feature importance by ticker file missing. Skipping chart.")
        return

    fig = px.bar(
        shap_feature_importance_by_ticker,
        x="Feature",
        y="Mean_Absolute_SHAP_Value",
        color="Ticker",
        barmode="group",
        title="SHAP Feature Importance by Company",
        labels={
            "Feature": "Trading feature",
            "Mean_Absolute_SHAP_Value": "Mean absolute SHAP value",
            "Ticker": "Company ticker",
        },
        hover_data=["Importance_Rank_Within_Ticker", "Mean_SHAP_Value"],
    )

    fig.update_xaxes(tickangle=-35)

    save_chart(fig, "15_shap_feature_importance_by_company.html")


def main() -> None:
    """
    Main execution function.
    """
    print("=" * 70)
    print("Starting Plotly visualisation generation")
    print("=" * 70)

    ensure_output_directory()

    data = prepare_model_output_data()

    model_summary = load_csv(MODEL_SUMMARY_FILE)
    event_detection_rate = load_csv(EVENT_WINDOW_DETECTION_RATE_FILE)
    relative_concentration = load_csv(RELATIVE_ANOMALY_CONCENTRATION_FILE)
    model_agreement_summary = load_csv(MODEL_AGREEMENT_SUMMARY_FILE)
    false_positive_analysis = load_csv(FALSE_POSITIVE_ANALYSIS_FILE)
    score_stability = load_csv(ANOMALY_SCORE_STABILITY_FILE)
    overall_summary = load_csv(OVERALL_EVALUATION_SUMMARY_FILE)
    best_model_selection = load_csv(BEST_MODEL_SELECTION_FILE)

    shap_feature_importance = load_csv(SHAP_FEATURE_IMPORTANCE_FILE, required=False)
    shap_feature_importance_by_ticker = load_csv(
        SHAP_FEATURE_IMPORTANCE_BY_TICKER_FILE,
        required=False,
    )

    create_closing_price_trend(data)
    create_trading_volume_trend(data)
    create_event_window_timeline(data)
    create_anomaly_counts_by_model(model_summary)
    create_event_vs_non_event_anomalies(event_detection_rate)
    create_event_window_detection_rate_chart(event_detection_rate)
    create_relative_anomaly_concentration_chart(relative_concentration)
    create_model_agreement_chart(model_agreement_summary)
    create_false_positive_analysis_chart(false_positive_analysis)
    create_anomaly_score_stability_chart(score_stability)
    create_overall_model_evaluation_chart(overall_summary)
    create_best_model_rank_chart(best_model_selection)
    create_anomaly_timeline_for_selected_model(data)
    create_shap_feature_importance_chart(shap_feature_importance)
    create_shap_feature_importance_by_ticker_chart(shap_feature_importance_by_ticker)

    print("\nPlotly visualisation generation completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()