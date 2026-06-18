from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components


# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Explainable Financial Anomaly Detection Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Project Paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
CSS_PATH = BASE_DIR / "assets" / "style.css"

DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = BASE_DIR / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"


# ============================================================
# Styling
# ============================================================
def load_css(css_path: Path) -> None:
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as file:
            st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)


load_css(CSS_PATH)


# ============================================================
# Data Loading Helpers
# ============================================================
@st.cache_data
def load_csv_file(file_path: Path) -> pd.DataFrame:
    return pd.read_csv(file_path)


def find_csv_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(folder.glob("*.csv"))


def find_chart_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []

    allowed_extensions = [".html", ".png", ".jpg", ".jpeg", ".webp"]

    return sorted(
        [
            file
            for file in folder.iterdir()
            if file.is_file() and file.suffix.lower() in allowed_extensions
        ]
    )


def find_date_column(df: pd.DataFrame) -> str | None:
    possible_columns = [
        "Date",
        "date",
        "Datetime",
        "datetime",
        "Trading_Date",
        "trading_date",
    ]

    for col in possible_columns:
        if col in df.columns:
            return col

    for col in df.columns:
        if "date" in col.lower():
            return col

    return None


def find_ticker_column(df: pd.DataFrame) -> str | None:
    possible_columns = [
        "Ticker",
        "ticker",
        "Symbol",
        "symbol",
        "Company",
        "company",
    ]

    for col in possible_columns:
        if col in df.columns:
            return col

    return None


def find_price_column(df: pd.DataFrame) -> str | None:
    possible_columns = [
        "Adj Close",
        "Adj_Close",
        "Adjusted_Close",
        "Close",
        "close",
        "Price",
        "price",
    ]

    for col in possible_columns:
        if col in df.columns:
            return col

    return None


def find_volume_column(df: pd.DataFrame) -> str | None:
    possible_columns = [
        "Volume",
        "volume",
        "Trading_Volume",
        "trading_volume",
    ]

    for col in possible_columns:
        if col in df.columns:
            return col

    return None


# ============================================================
# UI Helpers
# ============================================================
def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, description: str = "") -> None:
    st.markdown(
        f"""
        <div class="section-card">
            <h3>{title}</h3>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_html_chart(file_name: str, height: int = 650) -> None:
    chart_path = CHARTS_DIR / file_name

    if not chart_path.exists():
        st.warning(f"Chart not found: `{file_name}`")
        return

    suffix = chart_path.suffix.lower()

    if suffix == ".html":
        html_content = chart_path.read_text(encoding="utf-8", errors="ignore")
        components.html(html_content, height=height, scrolling=True)

    elif suffix in [".png", ".jpg", ".jpeg", ".webp"]:
        st.image(str(chart_path), use_container_width=True)

    else:
        st.warning(f"Unsupported chart format: `{file_name}`")


def render_chart_section(
    title: str,
    description: str,
    file_name: str,
    height: int = 650,
) -> None:
    section_header(title, description)
    render_html_chart(file_name, height)


def create_price_chart(
    df: pd.DataFrame,
    date_col: str,
    price_col: str,
    title: str,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df[date_col],
            y=df[price_col],
            mode="lines",
            name="Closing Price",
            line=dict(width=2),
        )
    )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=460,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(5,11,20,0.6)",
        xaxis_title="Date",
        yaxis_title="Price",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    return fig


def create_volume_chart(
    df: pd.DataFrame,
    date_col: str,
    volume_col: str,
    title: str,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df[date_col],
            y=df[volume_col],
            name="Volume",
            opacity=0.75,
        )
    )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=360,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(5,11,20,0.6)",
        xaxis_title="Date",
        yaxis_title="Volume",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    return fig


# ============================================================
# Sidebar Navigation
# ============================================================
st.sidebar.title("Market Surveillance")
st.sidebar.caption("Explainable AI-based anomaly detection")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Market Data Overview",
        "Generated Research Charts",
        "Anomaly Surveillance",
        "Model Comparison",
        "Explainable AI",
        "Research Summary",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Study Scope")
st.sidebar.markdown(
    """
    **Companies:** AAPL, MSFT, NVDA, AMZN, TSLA  
    **Period:** 2021–2025  
    **Event Window:** 10 trading days before earnings  
    **Models:** Isolation Forest, One-Class SVM, LOF  
    """
)


# ============================================================
# Load Processed Dataset
# ============================================================
csv_files = find_csv_files(PROCESSED_DATA_DIR)

selected_file = None
df = pd.DataFrame()
df_view = pd.DataFrame()

date_col = None
ticker_col = None
price_col = None
volume_col = None
selected_ticker = "All"

if csv_files:
    selected_file = st.sidebar.selectbox(
        "Select processed dataset",
        csv_files,
        format_func=lambda path: path.name,
    )

    df = load_csv_file(selected_file)

    date_col = find_date_column(df)
    ticker_col = find_ticker_column(df)
    price_col = find_price_column(df)
    volume_col = find_volume_column(df)

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        df = df.sort_values(date_col)

    if ticker_col:
        available_tickers = sorted(df[ticker_col].dropna().unique().tolist())
        selected_ticker = st.sidebar.selectbox("Select ticker", available_tickers)
        df_view = df[df[ticker_col] == selected_ticker].copy()
    else:
        df_view = df.copy()


# ============================================================
# Home Page
# ============================================================
if page == "Home":
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">
                Explainable Financial Anomaly Detection Dashboard
            </div>
            <div class="hero-subtitle">
                A research-based financial surveillance dashboard for analysing unusual trading behaviour
                around quarterly earnings announcements using unsupervised anomaly detection and Explainable AI.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card("Companies", "5", "AAPL, MSFT, NVDA, AMZN, TSLA")

    with col2:
        metric_card("Study Period", "2021–2025", "Public financial market data")

    with col3:
        metric_card("Event Window", "10 Days", "Pre-earnings trading window")

    with col4:
        metric_card("Models", "3", "IF, OCSVM, LOF")

    st.markdown("### Research Scope")

    st.markdown(
        """
        <div class="section-card">
            This dashboard supports an M.Sc. Data Science dissertation on Explainable AI-based
            anomaly detection for financial trading surveillance. The project analyses whether
            unsupervised anomaly detection models identify unusual price, volume, and volatility
            behaviour around quarterly earnings announcements.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Technical Workflow")

    col1, col2, col3 = st.columns(3)

    with col1:
        section_header(
            "Data Preparation",
            "Public market data and earnings dates are collected, cleaned, merged, and transformed into an event-window dataset.",
        )

    with col2:
        section_header(
            "Anomaly Detection",
            "Isolation Forest, One-Class SVM, and Local Outlier Factor are applied using unsupervised learning.",
        )

    with col3:
        section_header(
            "Explainability",
            "SHAP-based analysis is used to interpret which financial indicators contribute to model-flagged anomalies.",
        )

    st.markdown("### Ethical and Academic Boundary")

    st.markdown(
        """
        <div class="disclaimer">
            This dashboard is developed for academic research purposes only. The results identify unusual trading
            behaviour using unsupervised anomaly detection models and do not represent investment advice, trading
            recommendations, fraud detection, insider trading confirmation, or legal conclusions.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if csv_files:
        st.success(f"Processed data detected: `{selected_file.name}`")
    else:
        st.warning("No CSV files found in `data/processed/` yet.")

    chart_files = find_chart_files(CHARTS_DIR)

    if chart_files:
        st.success(f"Saved chart files detected: `{len(chart_files)}` charts found in `outputs/charts/`.")
    else:
        st.warning("No saved chart files found in `outputs/charts/`.")


# ============================================================
# Market Data Overview Page
# ============================================================
elif page == "Market Data Overview":
    st.title("Market Data Overview")

    if df_view.empty:
        st.warning("No processed dataset available. Add CSV files inside `data/processed/`.")
    else:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            metric_card("Selected Ticker", str(selected_ticker), "Market data filter")

        with col2:
            metric_card("Rows", f"{len(df_view):,}", "Trading observations")

        with col3:
            metric_card("Columns", f"{df_view.shape[1]}", "Available variables")

        with col4:
            if date_col:
                date_range = f"{df_view[date_col].min().date()} → {df_view[date_col].max().date()}"
            else:
                date_range = "N/A"
            metric_card("Date Range", date_range, "Dataset coverage")

        st.markdown("### Dataset Preview")
        st.dataframe(df_view.head(50), use_container_width=True)

        st.markdown("---")
        st.subheader("Interactive Dataset View")

        if date_col and price_col:
            st.plotly_chart(
                create_price_chart(
                    df_view,
                    date_col,
                    price_col,
                    f"{selected_ticker} Price Movement",
                ),
                use_container_width=True,
            )
        else:
            st.info("Price chart requires a date column and a close/price column.")

        if date_col and volume_col:
            st.plotly_chart(
                create_volume_chart(
                    df_view,
                    date_col,
                    volume_col,
                    f"{selected_ticker} Trading Volume",
                ),
                use_container_width=True,
            )
        else:
            st.info("Volume chart requires a date column and a volume column.")

    st.markdown("---")
    st.subheader("Generated Market Research Charts")

    render_chart_section(
        "Adjusted Closing Price Trends",
        "Adjusted closing price trends for the selected companies across the study period.",
        "01_adjusted_closing_price_trends.html",
        720,
    )

    render_chart_section(
        "Trading Volume Trends",
        "Trading volume trends across the selected companies.",
        "02_trading_volume_trends.html",
        720,
    )

    render_chart_section(
        "Event Window Timeline",
        "Timeline showing the 10-day pre-earnings event-window structure used in the research design.",
        "03_event_window_timeline.html",
        650,
    )


# ============================================================
# Generated Research Charts Page
# ============================================================
elif page == "Generated Research Charts":
    st.title("Generated Research Charts")

    st.markdown(
        """
        This section displays the saved visual outputs generated during the anomaly detection,
        event-window evaluation, model comparison, and explainability stages.
        """
    )

    chart_files = find_chart_files(CHARTS_DIR)

    if not chart_files:
        st.warning("No chart files found inside `outputs/charts/`.")
        st.info(
            "Check whether your saved charts are inside `outputs/charts/` and whether they are `.html`, `.png`, `.jpg`, or `.webp` files."
        )
    else:
        selected_chart = st.selectbox(
            "Select saved chart",
            chart_files,
            format_func=lambda path: path.name,
        )

        st.markdown(f"### {selected_chart.stem.replace('_', ' ').title()}")
        render_html_chart(selected_chart.name, 760)

        with st.expander("Chart file details"):
            st.write(f"**File name:** `{selected_chart.name}`")
            st.write(f"**Location:** `{selected_chart}`")
            st.write(f"**Format:** `{selected_chart.suffix}`")


# ============================================================
# Anomaly Surveillance Page
# ============================================================
elif page == "Anomaly Surveillance":
    st.title("Anomaly Surveillance")

    st.markdown(
        """
        This section presents model-flagged anomalous observations and compares anomaly behaviour
        inside and outside the 10-day pre-earnings event window.
        """
    )

    render_chart_section(
        "Total Anomaly Alerts by Model",
        "Comparison of the total number of anomaly alerts generated by Isolation Forest, One-Class SVM, and Local Outlier Factor.",
        "04_total_anomaly_alerts_by_model.html",
        560,
    )

    render_chart_section(
        "Event vs Non-Event Anomalies",
        "Distribution of detected anomalies across pre-earnings event-window periods and ordinary non-event trading days.",
        "05_event_vs_non_event_anomalies.html",
        560,
    )

    render_chart_section(
        "Isolation Forest Anomaly Timeline",
        "Timeline view showing model-flagged anomaly observations across the study period.",
        "13_isolation_forest_anomaly_timeline.html",
        720,
    )


# ============================================================
# Model Comparison Page
# ============================================================
elif page == "Model Comparison":
    st.title("Model Comparison")

    st.markdown(
        """
        The models are compared using unsupervised event-based evaluation criteria.
        This avoids supervised classification metrics such as accuracy, precision, recall, F1-score,
        ROC-AUC, and confusion matrices.
        """
    )

    render_chart_section(
        "Event-Window Detection Rate",
        "Shows how frequently each model detects anomalies within the defined 10-day pre-earnings event windows.",
        "06_event_window_detection_rate.html",
        560,
    )

    render_chart_section(
        "Relative Anomaly Concentration",
        "Compares whether anomaly detections are more concentrated around earnings-related event windows than outside them.",
        "07_relative_anomaly_concentration.html",
        560,
    )

    render_chart_section(
        "Model Agreement Summary",
        "Summarises the consistency and overlap between different unsupervised anomaly detection models.",
        "08_model_agreement_summary.html",
        560,
    )

    render_chart_section(
        "Event-Based False Positive Rate",
        "Analyses anomaly alerts outside the earnings-event windows as part of the false-positive discussion.",
        "09_event_based_false_positive_rate.html",
        560,
    )

    render_chart_section(
        "Anomaly Score Stability",
        "Evaluates the stability of anomaly score behaviour across the selected models.",
        "10_anomaly_score_stability.html",
        560,
    )

    render_chart_section(
        "Overall Model Evaluation Summary",
        "Combines the event-based evaluation indicators to support model comparison.",
        "11_overall_model_evaluation_summary.html",
        560,
    )

    render_chart_section(
        "Best Model Selection Rank",
        "Ranks the models based on the selected unsupervised event-based evaluation criteria.",
        "12_best_model_selection_rank.html",
        560,
    )


# ============================================================
# Explainable AI Page
# ============================================================
elif page == "Explainable AI":
    st.title("Explainable AI")

    st.markdown(
        """
        This section presents SHAP-based explainability outputs for interpreting which financial features
        contributed most strongly to model-flagged anomalous observations.
        """
    )

    render_chart_section(
        "SHAP Feature Importance",
        "Overall SHAP-based feature importance showing the main variables contributing to anomaly detection.",
        "14_shap_feature_importance.html",
        560,
    )

    render_chart_section(
        "SHAP Feature Importance by Company",
        "Company-level comparison of feature contributions across the selected financial instruments.",
        "15_shap_feature_importance_by_company.html",
        560,
    )

    render_chart_section(
        "SHAP Beeswarm Plot",
        "Detailed SHAP beeswarm visualisation showing the direction and strength of feature contributions.",
        "shap_beeswarm_plot.html",
        780,
    )

    render_chart_section(
        "SHAP Feature Importance Bar Chart",
        "Bar-based summary of the most influential features in the explainability analysis.",
        "shap_feature_importance_bar.html",
        560,
    )

    render_chart_section(
        "SHAP Feature Importance by Ticker",
        "Ticker-level SHAP feature contribution comparison.",
        "shap_feature_importance_by_ticker.html",
        560,
    )


# ============================================================
# Research Summary Page
# ============================================================
elif page == "Research Summary":
    st.title("Research Summary")

    st.markdown(
        """
        This section summarises the overall research interpretation, model behaviour,
        limitations, and ethical boundaries of the anomaly detection framework.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        section_header(
            "Research Aim",
            "To evaluate explainable AI-based unsupervised anomaly detection models for identifying unusual trading behaviour around earnings announcements.",
        )

        section_header(
            "Evaluation Framing",
            "The evaluation is event-based and unsupervised. The project does not use supervised classification metrics because no verified ground-truth fraud or insider-trading labels are used.",
        )

    with col2:
        section_header(
            "Explainability Focus",
            "SHAP is used to explain feature contributions behind model-flagged anomalies, especially for the most suitable model.",
        )

        section_header(
            "Ethical Boundary",
            "The project uses public financial data only and does not make legal, regulatory, or investment claims.",
        )

    st.markdown("### Final Academic Disclaimer")

    st.markdown(
        """
        <div class="disclaimer">
            The results shown in this dashboard indicate model-flagged unusual trading behaviour only.
            They should not be interpreted as evidence of fraud, insider trading, manipulation, trading advice,
            investment recommendation, or legal conclusion.
        </div>
        """,
        unsafe_allow_html=True,
    )