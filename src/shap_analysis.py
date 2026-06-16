"""
shap_analysis.py

This script applies SHAP explainability analysis to the selected anomaly detection model
for the dissertation project:

Evaluating Explainable AI-Based Anomaly Detection Models for Financial Trading Surveillance.

The selected model is identified from the unsupervised event-based evaluation stage.
In this project, Isolation Forest is selected for SHAP analysis.

All visualisations are created using Plotly to support future Streamlit integration.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import shap
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from config import OUTPUT_MODELS_PATH, OUTPUT_TABLES_PATH, OUTPUT_CHARTS_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_MODELS_DIR = PROJECT_ROOT / OUTPUT_MODELS_PATH
OUTPUT_TABLES_DIR = PROJECT_ROOT / OUTPUT_TABLES_PATH
OUTPUT_CHARTS_DIR = PROJECT_ROOT / OUTPUT_CHARTS_PATH

MODEL_OUTPUT_FILE = OUTPUT_MODELS_DIR / "model_anomaly_outputs.csv"
BEST_MODEL_SELECTION_FILE = OUTPUT_TABLES_DIR / "best_model_selection_summary.csv"

SHAP_VALUES_FILE = OUTPUT_TABLES_DIR / "shap_values_dataset.csv"
SHAP_FEATURE_IMPORTANCE_FILE = OUTPUT_TABLES_DIR / "shap_feature_importance.csv"
SHAP_FEATURE_IMPORTANCE_BY_TICKER_FILE = OUTPUT_TABLES_DIR / "shap_feature_importance_by_ticker.csv"
SHAP_SELECTED_ANOMALIES_FILE = OUTPUT_TABLES_DIR / "shap_selected_anomaly_explanations.csv"

SHAP_BAR_HTML_FILE = OUTPUT_CHARTS_DIR / "shap_feature_importance_bar.html"
SHAP_BEESWARM_HTML_FILE = OUTPUT_CHARTS_DIR / "shap_beeswarm_plot.html"
SHAP_TICKER_IMPORTANCE_HTML_FILE = OUTPUT_CHARTS_DIR / "shap_feature_importance_by_ticker.html"


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
    OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def load_model_outputs() -> pd.DataFrame:
    """
    Load model anomaly outputs.
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
    ] + FEATURE_COLUMNS

    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns for SHAP analysis: {missing_columns}")

    data["Date"] = pd.to_datetime(data["Date"])
    data["Ticker"] = data["Ticker"].astype(str).str.upper().str.strip()
    data["Event_Window"] = data["Event_Window"].astype(int)
    data["IF_Anomaly"] = data["IF_Anomaly"].astype(int)

    data = data.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Internal row identifier for safe joining between original data and SHAP values
    data["Row_ID"] = range(len(data))

    return data


def load_selected_model_name() -> str:
    """
    Load the model selected for SHAP from best_model_selection_summary.csv.
    """
    if not BEST_MODEL_SELECTION_FILE.exists():
        raise FileNotFoundError(
            f"Best model selection file not found: {BEST_MODEL_SELECTION_FILE}. "
            "Please run src/evaluation.py first."
        )

    selection_data = pd.read_csv(BEST_MODEL_SELECTION_FILE)

    selected_rows = selection_data[selection_data["Suggested_For_SHAP"] == "Yes"]

    if selected_rows.empty:
        raise ValueError("No model is marked as Suggested_For_SHAP = Yes.")

    selected_model = selected_rows["Model"].iloc[0]

    return selected_model


def train_isolation_forest_for_ticker(
    ticker_data: pd.DataFrame,
) -> tuple[IsolationForest, pd.DataFrame]:
    """
    Re-train Isolation Forest for one ticker using the same configuration used in models.py.

    The model is re-trained here because the modelling step saved predictions and scores,
    but not fitted model objects.
    """
    feature_matrix = ticker_data[FEATURE_COLUMNS].copy()

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)

    scaled_features_df = pd.DataFrame(
        scaled_features,
        columns=FEATURE_COLUMNS,
        index=ticker_data.index,
    )

    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION_RATE,
        random_state=RANDOM_STATE,
    )

    model.fit(scaled_features_df)

    return model, scaled_features_df


def calculate_shap_values(
    model: IsolationForest,
    scaled_features_df: pd.DataFrame,
) -> np.ndarray:
    """
    Calculate SHAP values for the Isolation Forest model.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(scaled_features_df)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]

    return shap_values


def run_shap_for_all_tickers(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run SHAP analysis ticker-by-ticker and combine results.
    """
    all_shap_dataframes = []
    all_scaled_feature_dataframes = []

    for ticker in sorted(data["Ticker"].unique()):
        print(f"Running SHAP analysis for {ticker}...")

        ticker_data = data[data["Ticker"] == ticker].copy()
        ticker_data = ticker_data.sort_values("Date").reset_index(drop=True)

        model, scaled_features_df = train_isolation_forest_for_ticker(ticker_data)
        shap_values = calculate_shap_values(model, scaled_features_df)

        shap_df = pd.DataFrame(shap_values, columns=FEATURE_COLUMNS)

        metadata_columns = [
            "Row_ID",
            "Date",
            "Ticker",
            "Event_Window",
            "IF_Anomaly",
            "IF_Anomaly_Score",
        ]

        for metadata_column in metadata_columns:
            shap_df[metadata_column] = ticker_data[metadata_column].values

        scaled_feature_df = scaled_features_df.copy()

        for metadata_column in metadata_columns:
            scaled_feature_df[metadata_column] = ticker_data[metadata_column].values

        all_shap_dataframes.append(shap_df)
        all_scaled_feature_dataframes.append(scaled_feature_df)

        print(f"{ticker}: SHAP values calculated for {len(ticker_data)} rows.")

    combined_shap_values = pd.concat(all_shap_dataframes, ignore_index=True)
    combined_scaled_features = pd.concat(all_scaled_feature_dataframes, ignore_index=True)

    return combined_shap_values, combined_scaled_features


def create_global_feature_importance(shap_values_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create global SHAP feature importance using mean absolute SHAP values.
    """
    rows = []

    for feature in FEATURE_COLUMNS:
        rows.append(
            {
                "Feature": feature,
                "Mean_Absolute_SHAP_Value": shap_values_data[feature].abs().mean(),
                "Mean_SHAP_Value": shap_values_data[feature].mean(),
            }
        )

    importance_data = pd.DataFrame(rows)
    importance_data = importance_data.sort_values(
        "Mean_Absolute_SHAP_Value",
        ascending=False,
    ).reset_index(drop=True)

    importance_data["Importance_Rank"] = range(1, len(importance_data) + 1)

    return importance_data


def create_feature_importance_by_ticker(shap_values_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create ticker-level SHAP feature importance.
    """
    rows = []

    for ticker in sorted(shap_values_data["Ticker"].unique()):
        ticker_shap = shap_values_data[shap_values_data["Ticker"] == ticker]

        for feature in FEATURE_COLUMNS:
            rows.append(
                {
                    "Ticker": ticker,
                    "Feature": feature,
                    "Mean_Absolute_SHAP_Value": ticker_shap[feature].abs().mean(),
                    "Mean_SHAP_Value": ticker_shap[feature].mean(),
                }
            )

    ticker_importance = pd.DataFrame(rows)

    ticker_importance["Importance_Rank_Within_Ticker"] = ticker_importance.groupby(
        "Ticker"
    )["Mean_Absolute_SHAP_Value"].rank(ascending=False, method="min")

    ticker_importance = ticker_importance.sort_values(
        ["Ticker", "Importance_Rank_Within_Ticker"]
    ).reset_index(drop=True)

    return ticker_importance


def create_selected_anomaly_explanations(
    data: pd.DataFrame,
    shap_values_data: pd.DataFrame,
    top_n: int = 30,
) -> pd.DataFrame:
    """
    Create local explanations for selected high-scoring Isolation Forest anomalies.

    The selected rows are the strongest Isolation Forest anomaly alerts based on
    IF_Anomaly_Score.
    """
    original_columns = [
        "Row_ID",
        "Date",
        "Ticker",
        "Event_Window",
        "IF_Anomaly",
        "IF_Anomaly_Score",
    ] + FEATURE_COLUMNS

    merged_data = data[original_columns].copy()

    shap_feature_data = shap_values_data[["Row_ID"] + FEATURE_COLUMNS].copy()

    shap_feature_data = shap_feature_data.rename(
        columns={feature: f"{feature}_SHAP" for feature in FEATURE_COLUMNS}
    )

    explanation_data = merged_data.merge(
        shap_feature_data,
        on="Row_ID",
        how="left",
    )

    anomaly_data = explanation_data[explanation_data["IF_Anomaly"] == 1].copy()

    anomaly_data = anomaly_data.sort_values(
        "IF_Anomaly_Score",
        ascending=False,
    ).head(top_n)

    rows = []

    for _, row in anomaly_data.iterrows():
        shap_contributions = {
            feature: abs(row[f"{feature}_SHAP"]) for feature in FEATURE_COLUMNS
        }

        top_features = sorted(
            shap_contributions.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:3]

        rows.append(
            {
                "Date": row["Date"].strftime("%Y-%m-%d"),
                "Ticker": row["Ticker"],
                "Event_Window": int(row["Event_Window"]),
                "IF_Anomaly_Score": round(row["IF_Anomaly_Score"], 6),
                "Top_Contributing_Feature_1": top_features[0][0],
                "Top_Contributing_Feature_1_Abs_SHAP": round(top_features[0][1], 6),
                "Top_Contributing_Feature_2": top_features[1][0],
                "Top_Contributing_Feature_2_Abs_SHAP": round(top_features[1][1], 6),
                "Top_Contributing_Feature_3": top_features[2][0],
                "Top_Contributing_Feature_3_Abs_SHAP": round(top_features[2][1], 6),
            }
        )

    return pd.DataFrame(rows)


def save_plotly_feature_importance_bar(feature_importance: pd.DataFrame) -> None:
    """
    Save global SHAP feature importance chart as Plotly HTML.
    """
    chart_data = feature_importance.sort_values(
        "Mean_Absolute_SHAP_Value",
        ascending=True,
    )

    fig = px.bar(
        chart_data,
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

    fig.update_layout(
        template="plotly_white",
        title_x=0.02,
        height=550,
        margin=dict(l=40, r=40, t=70, b=40),
    )

    fig.write_html(SHAP_BAR_HTML_FILE, include_plotlyjs="cdn")

    print(f"Plotly SHAP feature importance chart saved to: {SHAP_BAR_HTML_FILE}")


def create_shap_beeswarm_dataset(
    shap_values_data: pd.DataFrame,
    scaled_feature_data: pd.DataFrame,
    max_points_per_feature: int = 700,
) -> pd.DataFrame:
    """
    Create long-format data for a lightweight Plotly SHAP beeswarm-style chart.

    The full SHAP dataset is large, so the chart uses a controlled sample per feature
    to keep the interactive HTML responsive for Streamlit/dashboard use.
    """
    rows = []

    for feature in FEATURE_COLUMNS:
        temp = pd.DataFrame(
            {
                "Feature": feature,
                "SHAP_Value": shap_values_data[feature].values,
                "Scaled_Feature_Value": scaled_feature_data[feature].values,
                "Ticker": shap_values_data["Ticker"].values,
                "Event_Window": shap_values_data["Event_Window"].values,
                "IF_Anomaly": shap_values_data["IF_Anomaly"].values,
                "Date": pd.to_datetime(shap_values_data["Date"]).dt.strftime(
                    "%Y-%m-%d"
                ).values,
            }
        )

        if len(temp) > max_points_per_feature:
            temp = temp.sample(
                n=max_points_per_feature,
                random_state=RANDOM_STATE,
            )

        rows.append(temp)

    beeswarm_data = pd.concat(rows, ignore_index=True)

    feature_order = (
        beeswarm_data.groupby("Feature")["SHAP_Value"]
        .apply(lambda x: x.abs().mean())
        .sort_values(ascending=False)
        .index.tolist()
    )

    beeswarm_data["Feature"] = pd.Categorical(
        beeswarm_data["Feature"],
        categories=feature_order,
        ordered=True,
    )

    beeswarm_data["Event_Window_Label"] = beeswarm_data["Event_Window"].map(
        {
            1: "Pre-earnings event window",
            0: "Outside event window",
        }
    )

    return beeswarm_data


def save_plotly_shap_beeswarm(
    shap_values_data: pd.DataFrame,
    scaled_feature_data: pd.DataFrame,
) -> None:
    """
    Save a lightweight Plotly SHAP beeswarm-style chart as HTML.
    """
    beeswarm_data = create_shap_beeswarm_dataset(
        shap_values_data=shap_values_data,
        scaled_feature_data=scaled_feature_data,
        max_points_per_feature=700,
    )

    fig = px.scatter(
        beeswarm_data,
        x="SHAP_Value",
        y="Feature",
        color="Scaled_Feature_Value",
        title="SHAP Summary Plot for Isolation Forest",
        labels={
            "SHAP_Value": "SHAP value",
            "Feature": "Trading feature",
            "Scaled_Feature_Value": "Scaled feature value",
        },
        hover_data=[
            "Ticker",
            "Date",
            "Event_Window_Label",
            "IF_Anomaly",
        ],
        render_mode="webgl",
    )

    fig.update_traces(
        marker=dict(
            size=5,
            opacity=0.55,
        )
    )

    fig.update_layout(
        template="plotly_white",
        title_x=0.02,
        height=650,
        margin=dict(l=40, r=40, t=70, b=40),
        yaxis=dict(
            categoryorder="array",
            categoryarray=list(beeswarm_data["Feature"].cat.categories),
        ),
    )

    fig.write_html(SHAP_BEESWARM_HTML_FILE, include_plotlyjs="cdn")

    print(f"Plotly SHAP beeswarm chart saved to: {SHAP_BEESWARM_HTML_FILE}")


def save_plotly_ticker_feature_importance(
    feature_importance_by_ticker: pd.DataFrame,
) -> None:
    """
    Save ticker-wise SHAP feature importance chart as Plotly HTML.
    """
    fig = px.bar(
        feature_importance_by_ticker,
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

    fig.update_layout(
        template="plotly_white",
        title_x=0.02,
        height=650,
        margin=dict(l=40, r=40, t=70, b=120),
        xaxis_tickangle=-35,
    )

    fig.write_html(SHAP_TICKER_IMPORTANCE_HTML_FILE, include_plotlyjs="cdn")

    print(
        "Plotly ticker-level SHAP feature importance chart saved to: "
        f"{SHAP_TICKER_IMPORTANCE_HTML_FILE}"
    )


def save_outputs(
    shap_values_data: pd.DataFrame,
    feature_importance: pd.DataFrame,
    feature_importance_by_ticker: pd.DataFrame,
    selected_anomaly_explanations: pd.DataFrame,
) -> None:
    """
    Save SHAP output tables.
    """
    shap_values_to_save = shap_values_data.copy()
    shap_values_to_save["Date"] = shap_values_to_save["Date"].dt.strftime("%Y-%m-%d")

    shap_values_to_save.to_csv(SHAP_VALUES_FILE, index=False)
    feature_importance.to_csv(SHAP_FEATURE_IMPORTANCE_FILE, index=False)
    feature_importance_by_ticker.to_csv(
        SHAP_FEATURE_IMPORTANCE_BY_TICKER_FILE,
        index=False,
    )
    selected_anomaly_explanations.to_csv(
        SHAP_SELECTED_ANOMALIES_FILE,
        index=False,
    )

    print(f"\nSHAP values dataset saved to: {SHAP_VALUES_FILE}")
    print(f"SHAP feature importance saved to: {SHAP_FEATURE_IMPORTANCE_FILE}")
    print(
        "SHAP feature importance by ticker saved to: "
        f"{SHAP_FEATURE_IMPORTANCE_BY_TICKER_FILE}"
    )
    print(f"Selected anomaly explanations saved to: {SHAP_SELECTED_ANOMALIES_FILE}")


def main() -> None:
    """
    Main execution function.
    """
    print("=" * 70)
    print("Starting Plotly-based SHAP explainability analysis")
    print("=" * 70)

    ensure_output_directories()

    selected_model = load_selected_model_name()

    print(f"Selected model from evaluation stage: {selected_model}")

    if selected_model != "Isolation Forest":
        raise ValueError(
            f"This SHAP script is configured for Isolation Forest, but selected model is: "
            f"{selected_model}"
        )

    data = load_model_outputs()

    print(f"Loaded model output rows: {len(data)}")
    print(f"Tickers: {sorted(data['Ticker'].unique())}")

    shap_values_data, scaled_feature_data = run_shap_for_all_tickers(data)

    feature_importance = create_global_feature_importance(shap_values_data)
    feature_importance_by_ticker = create_feature_importance_by_ticker(shap_values_data)

    selected_anomaly_explanations = create_selected_anomaly_explanations(
        data=data,
        shap_values_data=shap_values_data,
        top_n=30,
    )

    print("\nGlobal SHAP Feature Importance:")
    print(feature_importance)

    print("\nSelected Anomaly Explanations:")
    print(selected_anomaly_explanations.head(10))

    save_outputs(
        shap_values_data=shap_values_data,
        feature_importance=feature_importance,
        feature_importance_by_ticker=feature_importance_by_ticker,
        selected_anomaly_explanations=selected_anomaly_explanations,
    )

    save_plotly_feature_importance_bar(feature_importance)

    save_plotly_shap_beeswarm(
        shap_values_data=shap_values_data,
        scaled_feature_data=scaled_feature_data,
    )

    save_plotly_ticker_feature_importance(feature_importance_by_ticker)

    print("\nPlotly-based SHAP explainability analysis completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()