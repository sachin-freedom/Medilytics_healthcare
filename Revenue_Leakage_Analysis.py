import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from chart_config import chart_cfg, load_css, page_header, chart_title, kpi, insight, fmt, COLORS
from pdf_export import render_pdf_button


@st.cache_data
def load_data():
    df = pd.read_csv("data/modified_dataset.csv")
    df["Claim_Submission_Date"] = pd.to_datetime(
        df["Claim_Submission_Date"], dayfirst=True, errors="coerce")
    df["Month_Str"] = df["Claim_Submission_Date"].dt.to_period("M").astype(str)
    return df


def revenue():
    load_css()
    data = load_data()
    role = st.session_state.get("role", "User")
    dept = st.session_state.get("department", "All")

    if role == "Department Head":
        data = data[data["Department"] == dept]

    f        = st.session_state.get("filters", {})
    filtered = data.copy()
    if f.get("date_range") and len(f["date_range"]) == 2:
        s, e = pd.to_datetime(f["date_range"][0]), pd.to_datetime(f["date_range"][1])
        filtered = filtered[(filtered["Claim_Submission_Date"] >= s) & (filtered["Claim_Submission_Date"] <= e)]
    if f.get("department_filter") and f["department_filter"] != "All":
        filtered = filtered[filtered["Department"] == f["department_filter"]]
    if f.get("insurance_filter") and f["insurance_filter"] != "All":
        filtered = filtered[filtered["Insurance_Type"] == f["insurance_filter"]]

    if filtered.empty:
        st.error("No data for selected filters."); return

    total_leak   = filtered["Revenue_Leakage"].sum()
    total_exp    = filtered["Expected_Revenue"].sum()
    leak_pct     = total_leak / total_exp * 100 if total_exp > 0 else 0
    avg_leak_idx = filtered["Revenue_Leakage_Index"].mean()
    total_risk   = filtered["Revenue_At_Risk"].sum()

    page_header("Revenue Leakage Analysis", "Gap between expected and actual revenue  ·  60,000 claims")
    render_pdf_button("Revenue_Leakage_Analysis")

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Total Leakage",     fmt(total_leak),       "mv-red")
    kpi(c2, "Leakage Rate",      f"{leak_pct:.1f}%",    "mv-red" if leak_pct > 15 else "mv-gold")
    kpi(c3, "Avg Leakage Index", f"{avg_leak_idx:.1f}%","mv-gold")
    kpi(c4, "Revenue at Risk",   fmt(total_risk),        "mv-purple")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        dept_leak = (filtered.groupby("Department")["Revenue_Leakage"]
                     .sum().reset_index().sort_values("Revenue_Leakage", ascending=True))
        fig1 = px.bar(dept_leak, x="Revenue_Leakage", y="Department", orientation="h",
                      color="Revenue_Leakage",
                      color_continuous_scale=[[0,"#00FF87"],[0.5,"#FFD700"],[1,"#FF5E5E"]])
        fig1.update_coloraxes(showscale=False)
        chart_title("Leakage by Department")
        fig1.update_layout(**chart_cfg(
            xlabel="Revenue Leakage (₹)", ylabel="Department"))
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        trend = (filtered.groupby("Month_Str")
                 .agg(Actual=("Actual_Revenue","sum"), Leakage=("Revenue_Leakage","sum"))
                 .reset_index())
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=trend["Month_Str"], y=trend["Actual"],
                                  name="Actual Revenue", line=dict(color="#00E5FF", width=2),
                                  fill="tozeroy", fillcolor="rgba(0,229,255,0.06)"))
        fig2.add_trace(go.Scatter(x=trend["Month_Str"], y=trend["Leakage"],
                                  name="Leakage", line=dict(color="#FF5E5E", width=1.8, dash="dot"),
                                  fill="tozeroy", fillcolor="rgba(255,94,94,0.08)"))
        chart_title("Revenue vs Leakage Trend")
        fig2.update_layout(**chart_cfg(
            xlabel="Month", ylabel="Amount (₹)"))
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        ins_leak = filtered.groupby("Insurance_Type")["Revenue_Leakage"].sum().reset_index()
        fig3 = px.pie(ins_leak, values="Revenue_Leakage", names="Insurance_Type",
                      hole=0.50, color_discrete_sequence=COLORS)
        fig3.update_traces(textfont_size=12,
                           marker=dict(line=dict(color="rgba(0,0,0,0.15)", width=1.5)))
        chart_title("Leakage Share by Insurance")
        fig3.update_layout(**chart_cfg())
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        cce = (filtered.groupby("Department")["Charge_Capture_Efficiency"]
               .mean().reset_index().sort_values("Charge_Capture_Efficiency"))
        fig4 = px.bar(cce, x="Charge_Capture_Efficiency", y="Department", orientation="h",
                      color="Charge_Capture_Efficiency",
                      color_continuous_scale=[[0,"#FF5E5E"],[0.6,"#FFD700"],[1,"#00FF87"]])
        fig4.update_coloraxes(showscale=False)
        chart_title("Charge Capture Efficiency by Dept")
        fig4.update_layout(**chart_cfg(
            xlabel="Efficiency (%)", ylabel="Department"))
        st.plotly_chart(fig4, use_container_width=True)

    worst = filtered.groupby("Department")["Revenue_Leakage"].sum().sort_values(ascending=False)
    insight(
        f"Total leakage: <strong>{fmt(total_leak)}</strong> ({leak_pct:.1f}%). "
        f"<strong>{worst.index[0]}</strong> highest at <strong>{fmt(worst.iloc[0])}</strong>. "
        f"Revenue at risk: <strong>{fmt(total_risk)}</strong>.",
        kind="red" if leak_pct > 15 else "" if leak_pct > 8 else "green")

    st.divider()
    st.markdown('<div class="chart-title">High-Leakage Claims</div>', unsafe_allow_html=True)
    cols = ["Claim_ID","Department","Doctor_Name","Insurance_Type",
            "Expected_Revenue","Actual_Revenue","Revenue_Leakage","Charge_Capture_Efficiency"]
    st.dataframe(filtered.sort_values("Revenue_Leakage", ascending=False)[cols].head(12),
                 use_container_width=True, hide_index=True)
