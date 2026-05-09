"""
utils/charts.py

Chart generation utilities built on Plotly Express.

All charts use a consistent dark-navy/teal brand palette to match the
executive dashboard aesthetic. Each function returns a Plotly figure
that Streamlit renders with st.plotly_chart().
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Brand palette ──────────────────────────────────────────────────────────────
NAVY = "#0A2342"
TEAL = "#00A8A8"
LIGHT_BLUE = "#4A90D9"
AMBER = "#E67E22"
RED = "#C0392B"
GREEN = "#27AE60"
GRAY = "#8A8A8A"
BG_COLOR = "#F5F7FA"

COLOR_SEQ = [TEAL, LIGHT_BLUE, NAVY, AMBER, GREEN, "#9B59B6", "#1ABC9C", "#E74C3C"]

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, Arial, sans-serif", color=NAVY, size=11),
    margin=dict(t=40, l=10, r=10, b=10),
    legend=dict(
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#D0D7E2",
        borderwidth=1,
        font=dict(size=10),
    ),
)


def bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
    color: str = None,
    orientation: str = "v",
    height: int = 380,
) -> go.Figure:
    fig = px.bar(
        df,
        x=x,
        y=y,
        title=title,
        color=color,
        orientation=orientation,
        color_discrete_sequence=COLOR_SEQ,
        height=height,
    )
    fig.update_layout(**CHART_LAYOUT)
    fig.update_traces(marker_line_width=0)
    return fig


def line_chart(
    df: pd.DataFrame,
    x: str,
    y: str | list,
    title: str = "",
    height: int = 350,
) -> go.Figure:
    y_cols = [y] if isinstance(y, str) else y
    fig = go.Figure()
    for i, col in enumerate(y_cols):
        fig.add_trace(
            go.Scatter(
                x=df[x],
                y=df[col],
                mode="lines+markers",
                name=col.replace("_", " ").title(),
                line=dict(color=COLOR_SEQ[i % len(COLOR_SEQ)], width=2.5),
                marker=dict(size=6),
            )
        )
    fig.update_layout(title=title, height=height, **CHART_LAYOUT)
    return fig


def scatter_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str = None,
    size: str = None,
    title: str = "",
    height: int = 380,
) -> go.Figure:
    fig = px.scatter(
        df, x=x, y=y, color=color, size=size, title=title,
        color_discrete_sequence=COLOR_SEQ, height=height,
    )
    fig.update_layout(**CHART_LAYOUT)
    return fig


def pie_chart(
    df: pd.DataFrame,
    names: str,
    values: str,
    title: str = "",
    height: int = 360,
) -> go.Figure:
    fig = px.pie(
        df, names=names, values=values, title=title,
        color_discrete_sequence=COLOR_SEQ, height=height,
        hole=0.4,   # donut style
    )
    fig.update_layout(**CHART_LAYOUT)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig


def heatmap(
    df: pd.DataFrame,
    title: str = "",
    height: int = 400,
) -> go.Figure:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return go.Figure()

    corr = numeric.corr()
    fig = px.imshow(
        corr,
        title=title,
        color_continuous_scale=[[0, "#C0392B"], [0.5, "white"], [1, TEAL]],
        aspect="auto",
        height=height,
    )
    fig.update_layout(**CHART_LAYOUT)
    return fig


def histogram(
    df: pd.DataFrame,
    column: str,
    bins: int = 30,
    title: str = "",
    height: int = 320,
) -> go.Figure:
    fig = px.histogram(
        df, x=column, nbins=bins, title=title,
        color_discrete_sequence=[TEAL], height=height,
    )
    fig.update_layout(**CHART_LAYOUT)
    fig.update_traces(marker_line_width=0.5, marker_line_color="white")
    return fig


def anomaly_chart(
    series: pd.Series,
    threshold: float = 2.0,
    title: str = "Anomaly Detection",
    height: int = 350,
) -> go.Figure:
    """
    Visualize a time series or numeric series with anomaly thresholds overlaid.
    Values beyond ±threshold std deviations are highlighted in red.
    """
    mean = series.mean()
    std = series.std()
    z_scores = np.abs((series - mean) / std)
    is_anomaly = z_scores >= threshold

    fig = go.Figure()

    # Normal points
    fig.add_trace(go.Scatter(
        x=series.index[~is_anomaly],
        y=series.values[~is_anomaly],
        mode="markers",
        name="Normal",
        marker=dict(color=TEAL, size=7, opacity=0.8),
    ))

    # Anomalous points
    fig.add_trace(go.Scatter(
        x=series.index[is_anomaly],
        y=series.values[is_anomaly],
        mode="markers",
        name="Anomaly",
        marker=dict(color=RED, size=10, symbol="x", line=dict(width=2)),
    ))

    # Mean line
    fig.add_hline(y=mean, line_dash="dash", line_color=GRAY,
                  annotation_text="Mean", annotation_position="right")

    # Threshold bands
    fig.add_hrect(
        y0=mean - threshold * std, y1=mean + threshold * std,
        fillcolor=TEAL, opacity=0.06, line_width=0,
    )

    fig.update_layout(title=title, height=height, **CHART_LAYOUT)
    return fig


def kpi_gauge(
    value: float,
    min_val: float = 0,
    max_val: float = 100,
    title: str = "KPI",
    threshold_low: float = 40,
    threshold_high: float = 70,
    height: int = 250,
) -> go.Figure:
    color = RED if value < threshold_low else (AMBER if value < threshold_high else GREEN)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": title, "font": {"size": 14, "color": NAVY}},
        gauge={
            "axis": {"range": [min_val, max_val], "tickcolor": GRAY},
            "bar": {"color": color},
            "bgcolor": "white",
            "bordercolor": "#D0D7E2",
            "steps": [
                {"range": [min_val, threshold_low], "color": "#FDECEA"},
                {"range": [threshold_low, threshold_high], "color": "#FEF9E7"},
                {"range": [threshold_high, max_val], "color": "#E8F8F0"},
            ],
            "threshold": {
                "line": {"color": NAVY, "width": 3},
                "thickness": 0.75,
                "value": value,
            },
        },
    ))
    fig.update_layout(height=height, **CHART_LAYOUT, margin=dict(t=60, l=20, r=20, b=20))
    return fig


def auto_chart(df: pd.DataFrame, title: str = "", height: int = 380) -> go.Figure:
    """
    Pick the most appropriate chart type automatically based on
    the shape and types of columns in the DataFrame.
    """
    if df is None or df.empty:
        return go.Figure()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if len(numeric_cols) == 0:
        return go.Figure()

    if len(cat_cols) >= 1 and len(numeric_cols) >= 1:
        if df[cat_cols[0]].nunique() <= 12:
            return bar_chart(df, x=cat_cols[0], y=numeric_cols[0], title=title, height=height)

    if len(numeric_cols) >= 2:
        return scatter_chart(df, x=numeric_cols[0], y=numeric_cols[1], title=title, height=height)

    return histogram(df, column=numeric_cols[0], title=title, height=height)
