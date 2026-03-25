import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from chart_config import chart_cfg, load_css, page_header, chart_title, kpi, insight, fmt, COLORS
from pdf_export import render_pdf_button


@st.cache_data
def load_data():
    df   = pd.read_csv("data/modified_dataset.csv")
    fore = pd.read_csv("data/revenue_forecast.csv")
    kpi_ = pd.read_csv("data/monthly_kpi_dataset.csv")
    df["Claim_Submission_Date"] = pd.to_datetime(
        df["Claim_Submission_Date"], dayfirst=True, errors="coerce")
    df["Month_DT"]  = df["Claim_Submission_Date"].dt.to_period("M").dt.to_timestamp()
    fore["Month"]   = pd.to_datetime(fore["Month"])
    kpi_["Month"]   = pd.to_datetime(kpi_["Month"])
    return df, fore, kpi_


def cfo_strategic():
    load_css()
    if st.session_state.get("role") != "CFO":
        st.error("Access Restricted — CFO credentials required."); return

    df, fore, kpi_df = load_data()

    total_exp   = df["Expected_Revenue"].sum()
    total_act   = df["Actual_Revenue"].sum()
    total_paid  = df["Payment_Received"].sum()
    total_leak  = df["Revenue_Leakage"].sum()
    coll_rate   = total_paid / total_act * 100 if total_act > 0 else 0
    denial_rate = df["Denial_Flag"].mean() * 100
    fore_6m     = fore["Forecast_Revenue"].sum()
    leak_pct    = total_leak / total_exp * 100

    page_header("CFO Strategic Dashboard", "Board-level financial intelligence  ·  Full hospital view")
    render_pdf_button("CFO_Strategic_Dashboard")

    r1, r2, r3, r4 = st.columns(4)
    kpi(r1, "Expected Revenue",  fmt(total_exp),     "mv-cyan")
    kpi(r2, "Actual Revenue",    fmt(total_act),     "mv-green")
    kpi(r3, "Revenue Leakage",   fmt(total_leak),    "mv-red")
    kpi(r4, "Collection Rate",   f"{coll_rate:.1f}%","mv-green")
    st.markdown("<br>", unsafe_allow_html=True)

    r5, r6, r7, r8 = st.columns(4)
    kpi(r5, "Payments Received", fmt(total_paid),    "mv-gold")
    kpi(r6, "Claim Denial Rate", f"{denial_rate:.1f}%","mv-red" if denial_rate > 20 else "mv-gold")
    kpi(r7, "6-Month Forecast",  fmt(fore_6m),       "mv-cyan")
    kpi(r8, "Leakage Rate",      f"{leak_pct:.1f}%", "mv-red" if leak_pct > 15 else "mv-gold")
    st.divider()

    dept_pnl = df.groupby("Department").agg(
        Expected=("Expected_Revenue","sum"), Actual=("Actual_Revenue","sum"),
        Leakage=("Revenue_Leakage","sum"),   Claims=("Claim_ID","count"),
        Denials=("Denial_Flag","sum")).reset_index()
    dept_pnl["Efficiency"]  = dept_pnl["Actual"]  / dept_pnl["Expected"] * 100
    dept_pnl["Denial_Rate"] = dept_pnl["Denials"] / dept_pnl["Claims"]   * 100

    col1, col2 = st.columns(2)
    with col1:
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(x=dept_pnl["Department"], y=dept_pnl["Expected"],
                              name="Expected", marker_color="rgba(0,229,255,0.50)"))
        fig1.add_trace(go.Bar(x=dept_pnl["Department"], y=dept_pnl["Actual"],
                              name="Actual", marker_color="#FFD700", opacity=0.88))
        fig1.add_trace(go.Bar(x=dept_pnl["Department"], y=dept_pnl["Leakage"],
                              name="Leakage", marker_color="#FF5E5E", opacity=0.85))
        chart_title("Dept P&L — Expected vs Actual vs Leakage")
        fig1.update_layout(**chart_cfg(
            xlabel="Department", ylabel="Revenue (₹)"), barmode="group")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        eff = dept_pnl.sort_values("Efficiency")
        fig2 = px.bar(eff, x="Efficiency", y="Department", orientation="h",
                      color="Efficiency",
                      color_continuous_scale=[[0,"#FF5E5E"],[0.6,"#FFD700"],[1,"#00FF87"]])
        fig2.update_coloraxes(showscale=False)
        chart_title("Collection Efficiency by Department")
        fig2.update_layout(**chart_cfg(
            xlabel="Efficiency (%)", ylabel="Department"))
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        monthly = df.groupby("Month_DT")["Actual_Revenue"].sum().reset_index()
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=monthly["Month_DT"], y=monthly["Actual_Revenue"],
                                  name="Historical", line=dict(color="#00E5FF", width=2),
                                  fill="tozeroy", fillcolor="rgba(0,229,255,0.05)"))
        fig3.add_trace(go.Scatter(x=fore["Month"], y=fore["Forecast_Revenue"],
                                  name="Forecast", line=dict(color="#FFD700", width=2, dash="dash")))
        chart_title("Historical Revenue and 6-Month Forecast")
        fig3.update_layout(**chart_cfg(
            xlabel="Month", ylabel="Revenue (₹)"))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        ins = df.groupby("Insurance_Type").agg(
            Revenue=("Actual_Revenue","sum"), Claims=("Claim_ID","count"),
            Denial_Rate=("Denial_Flag","mean")).reset_index()
        ins["Denial_%"] = ins["Denial_Rate"] * 100
        fig4 = px.scatter(ins, x="Revenue", y="Denial_%", size="Claims",
                          text="Insurance_Type", color="Denial_%",
                          color_continuous_scale=[[0,"#00FF87"],[0.5,"#FFD700"],[1,"#FF5E5E"]],
                          size_max=45)
        fig4.update_traces(textposition="top center", textfont_size=11)
        fig4.update_coloraxes(showscale=False)
        chart_title("Revenue vs Denial Rate by Insurance")
        fig4.update_layout(**chart_cfg(
            xlabel="Revenue (₹)", ylabel="Denial Rate (%)"))
        st.plotly_chart(fig4, use_container_width=True)

    best  = dept_pnl.sort_values("Efficiency", ascending=False).iloc[0]
    worst = dept_pnl.sort_values("Leakage", ascending=False).iloc[0]
    insight(
        f"Collection rate: <strong>{coll_rate:.1f}%</strong>. "
        f"Best dept: <strong>{best['Department']}</strong> ({best['Efficiency']:.1f}%). "
        f"Highest leakage: <strong>{worst['Department']}</strong> at <strong>{fmt(worst['Leakage'])}</strong>. "
        f"6-month forecast: <strong>{fmt(fore_6m)}</strong>.",
        kind="red" if leak_pct > 15 else "" if leak_pct > 8 else "green")

    st.divider()
    st.markdown('<div class="chart-title">Department P&L Summary</div>', unsafe_allow_html=True)
    disp = dept_pnl.copy()
    for c_ in ["Expected","Actual","Leakage"]:
        disp[c_] = disp[c_].apply(fmt)
    disp["Efficiency"]  = disp["Efficiency"].apply(lambda x: f"{x:.1f}%")
    disp["Denial_Rate"] = disp["Denial_Rate"].apply(lambda x: f"{x:.1f}%")
    st.dataframe(disp.drop(columns=["Claims","Denials"]), use_container_width=True, hide_index=True)
