import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from chart_config import chart_cfg, load_css, page_header, chart_title, kpi, insight, fmt, COLORS
from pdf_export import render_pdf_button


@st.cache_data
def load_data():
    hist = pd.read_csv("data/monthly_revenue_history.csv")
    fore = pd.read_csv("data/revenue_forecast.csv")
    kpi_ = pd.read_csv("data/monthly_kpi_dataset.csv")
    hist["Month"] = pd.to_datetime(hist["Month"])
    fore["Month"] = pd.to_datetime(fore["Month"])
    kpi_["Month"] = pd.to_datetime(kpi_["Month"])
    return hist, fore, kpi_


def revenue_forecast_model():
    load_css()
    hist, fore, kpi_df = load_data()

    last_rev   = hist["Actual_Revenue"].iloc[-1]
    avg_rev    = hist["Actual_Revenue"].mean()
    fore_total = fore["Forecast_Revenue"].sum()
    growth     = (fore["Forecast_Revenue"].iloc[0] - last_rev) / last_rev * 100
    fore["Cumulative"] = fore["Forecast_Revenue"].cumsum()

    page_header("Revenue Forecasting", "ARIMA(1,1,1) model  ·  6-month projection  ·  23 months historical data")
    render_pdf_button("Revenue_Forecasting")

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Latest Monthly Revenue", fmt(last_rev),           "mv-cyan")
    kpi(c2, "Avg Historical Revenue", fmt(avg_rev),            "mv-gold")
    kpi(c3, "6-Month Forecast Total", fmt(fore_total),         "mv-green")
    kpi(c4, "Projected Growth",       f"{growth:+.2f}%",       "mv-green" if growth > 0 else "mv-red")
    st.divider()

    # Full-width forecast chart
    upper = fore["Forecast_Revenue"] * 1.10
    lower = fore["Forecast_Revenue"] * 0.90
    fig1  = go.Figure()
    fig1.add_trace(go.Scatter(x=hist["Month"], y=hist["Actual_Revenue"],
                              mode="lines+markers", name="Historical",
                              line=dict(color="#00E5FF", width=2.2), marker=dict(size=5),
                              fill="tozeroy", fillcolor="rgba(0,229,255,0.05)"))
    fig1.add_trace(go.Scatter(x=fore["Month"], y=fore["Forecast_Revenue"],
                              mode="lines+markers", name="Forecast",
                              line=dict(color="#FFD700", width=2.2, dash="dash"),
                              marker=dict(size=7, symbol="diamond"),
                              fill="tozeroy", fillcolor="rgba(255,215,0,0.05)"))
    fig1.add_trace(go.Scatter(
        x=list(fore["Month"]) + list(fore["Month"][::-1]),
        y=list(upper) + list(lower[::-1]),
        fill="toself", fillcolor="rgba(255,215,0,0.06)",
        line=dict(color="rgba(0,0,0,0)"), name="±10% Band"))
    chart_title("Historical Revenue and 6-Month Forecast")
    fig1.update_layout(**chart_cfg(
        xlabel="Month", ylabel="Revenue (₹)"))
    st.plotly_chart(fig1, use_container_width=True)

    col2, col3 = st.columns(2)
    with col2:
        fg = fore.copy()
        fg["Growth"]    = fg["Forecast_Revenue"].pct_change() * 100
        fg["Month_Str"] = fg["Month"].dt.strftime("%b %Y")
        fg["Color"]     = fg["Growth"].apply(lambda x: "#00FF87" if x >= 0 else "#FF5E5E")
        fig2 = go.Figure(go.Bar(x=fg["Month_Str"], y=fg["Growth"],
                                marker_color=fg["Color"],
                                text=fg["Growth"].apply(lambda x: f"{x:+.1f}%"),
                                textposition="outside"))
        chart_title("Month-over-Month Forecast Growth")
        fig2.update_layout(**chart_cfg(
            xlabel="Month", ylabel="Growth (%)"))
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        kpi_df["Month_Str"] = kpi_df["Month"].dt.strftime("%Y-%m")
        y2c = "#94A3B8"
        chart_title("Monthly Claims Volume vs Leakage")
        cfg3 = chart_cfg(
                         xlabel="Month", ylabel="Claims")
        cfg3["yaxis2"] = dict(overlaying="y", side="right", gridcolor="rgba(0,0,0,0)",
                              tickfont=dict(color=y2c, size=10),
                              title=dict(text="Leakage (₹)", font=dict(color=y2c, size=11)))
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=kpi_df["Month_Str"], y=kpi_df["Total_Claims"],
                              name="Claims", marker_color="rgba(0,229,255,0.55)"))
        fig3.add_trace(go.Scatter(x=kpi_df["Month_Str"], y=kpi_df["Revenue_Leakage"],
                                  name="Leakage", yaxis="y2",
                                  line=dict(color="#FF5E5E", width=1.8), mode="lines+markers"))
        fig3.update_layout(**cfg3)
        st.plotly_chart(fig3, use_container_width=True)

    peak_month = fore.loc[fore["Forecast_Revenue"].idxmax(), "Month"].strftime("%b %Y")
    insight(
        f"Revenue is <strong>{'increasing' if growth > 0 else 'declining'}</strong> "
        f"({growth:+.1f}% projected). Peak: <strong>{peak_month}</strong> at "
        f"<strong>{fmt(fore['Forecast_Revenue'].max())}</strong>. "
        f"6-month total: <strong>{fmt(fore_total)}</strong>.",
        kind="green" if growth > 5 else "" if growth > 0 else "red")

    st.divider()
    st.markdown('<div class="chart-title">6-Month Forecast Table</div>', unsafe_allow_html=True)
    ft = fore[["Month","Forecast_Revenue","Cumulative"]].copy()
    ft["Month"]            = ft["Month"].dt.strftime("%B %Y")
    ft["Forecast_Revenue"] = ft["Forecast_Revenue"].apply(fmt)
    ft["Cumulative"]       = ft["Cumulative"].apply(fmt)
    ft.columns             = ["Month","Forecast Revenue","Cumulative Revenue"]
    st.dataframe(ft, use_container_width=True, hide_index=True)
