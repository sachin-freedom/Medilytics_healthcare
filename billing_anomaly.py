import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from chart_config import chart_cfg, load_css, page_header, chart_title, kpi, insight, fmt, COLORS
from pdf_export import render_pdf_button


@st.cache_data
def load_data():
    df = pd.read_csv("data/updated_cleaned_claim_dataset.csv")
    df["Claim_Submission_Date"] = pd.to_datetime(
        df["Claim_Submission_Date"], dayfirst=True, errors="coerce")
    df["Month_Str"] = df["Claim_Submission_Date"].dt.to_period("M").astype(str)
    cg_mean = df["Claim_Gap"].mean();  cg_std = df["Claim_Gap"].std()
    df["Claim_Gap_Anomaly"] = df["Claim_Gap"] > (cg_mean + 2 * cg_std)
    df["Claim_Gap_Z"]       = (df["Claim_Gap"] - cg_mean) / cg_std
    pg_mean = df["Payment_Gap"].mean(); pg_std = df["Payment_Gap"].std()
    df["Payment_Gap_Anomaly"] = df["Payment_Gap"] > (pg_mean + 2 * pg_std)
    df["Payment_Gap_Z"]       = (df["Payment_Gap"] - pg_mean) / pg_std
    df["Any_Anomaly"] = df["Claim_Gap_Anomaly"] | df["Payment_Gap_Anomaly"] | df["High_Risk_Claim"]
    def sev(row):
        z = max(row["Claim_Gap_Z"], row["Payment_Gap_Z"])
        if z > 3 or row["High_Risk_Claim"]: return "Critical"
        elif z > 2: return "High"
        return "Normal"
    df["Severity"] = df.apply(sev, axis=1)
    return df, cg_mean + 2 * cg_std, pg_mean + 2 * pg_std


def billing_anomaly():
    load_css()
    data, cg_thr, pg_thr = load_data()
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
        st.warning("No data for selected filters."); return

    anom      = filtered[filtered["Any_Anomaly"]]
    cg_anom   = filtered[filtered["Claim_Gap_Anomaly"]]
    pg_anom   = filtered[filtered["Payment_Gap_Anomaly"]]
    anom_pct  = len(anom) / len(filtered) * 100
    cg_pct    = len(cg_anom) / len(filtered) * 100
    pg_pct    = len(pg_anom) / len(filtered) * 100
    total_loss= filtered["Revenue_Loss"].sum()

    page_header("Billing Anomaly Detection", "Claim Gap and Payment Gap outlier analysis  ·  60,000 claims")
    render_pdf_button("Billing_Anomaly_Detection")

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Total Anomalies",       f"{len(anom):,}  ({anom_pct:.1f}%)", "mv-red" if anom_pct > 10 else "mv-gold")
    kpi(c2, "Claim Gap Anomalies",   f"{len(cg_anom):,}  ({cg_pct:.1f}%)", "mv-red")
    kpi(c3, "Payment Gap Anomalies", f"{len(pg_anom):,}  ({pg_pct:.1f}%)", "mv-gold")
    kpi(c4, "Total Revenue Loss",    fmt(total_loss), "mv-red")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        fig1 = go.Figure()
        fig1.add_trace(go.Histogram(
            x=filtered[~filtered["Claim_Gap_Anomaly"]]["Claim_Gap"],
            name="Normal", marker_color="#00E5FF", opacity=0.65, nbinsx=40))
        fig1.add_trace(go.Histogram(
            x=filtered[filtered["Claim_Gap_Anomaly"]]["Claim_Gap"],
            name="Anomaly", marker_color="#FF5E5E", opacity=0.90, nbinsx=40))
        fig1.add_vline(x=cg_thr, line_dash="dash", line_color="#FFD700", line_width=1.8,
                       annotation_text=f"2σ = {fmt(cg_thr)}", annotation_font_color="#FFD700",
                       annotation_font_size=11)
        chart_title("Claim Gap Distribution — Normal vs Anomaly")
        fig1.update_layout(**chart_cfg(
            xlabel="Claim Gap (₹)", ylabel="Count"), barmode="overlay")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=filtered[~filtered["Payment_Gap_Anomaly"]]["Payment_Gap"],
            name="Normal", marker_color="#00E5FF", opacity=0.65, nbinsx=40))
        fig2.add_trace(go.Histogram(
            x=filtered[filtered["Payment_Gap_Anomaly"]]["Payment_Gap"],
            name="Anomaly", marker_color="#FFD700", opacity=0.90, nbinsx=40))
        fig2.add_vline(x=pg_thr, line_dash="dash", line_color="#FF5E5E", line_width=1.8,
                       annotation_text=f"2σ = {fmt(pg_thr)}", annotation_font_color="#FF5E5E",
                       annotation_font_size=11)
        chart_title("Payment Gap Distribution — Normal vs Anomaly")
        fig2.update_layout(**chart_cfg(
            xlabel="Payment Gap (₹)", ylabel="Count"), barmode="overlay")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        dept_gaps = (filtered.groupby("Department")
                     .agg(Avg_Claim_Gap=("Claim_Gap","mean"), Avg_Payment_Gap=("Payment_Gap","mean"))
                     .reset_index().sort_values("Avg_Claim_Gap", ascending=False))
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=dept_gaps["Department"], y=dept_gaps["Avg_Claim_Gap"],
                              name="Avg Claim Gap", marker_color="#FF5E5E", opacity=0.88))
        fig3.add_trace(go.Bar(x=dept_gaps["Department"], y=dept_gaps["Avg_Payment_Gap"],
                              name="Avg Payment Gap", marker_color="#FFD700", opacity=0.88))
        chart_title("Avg Claim Gap and Payment Gap by Department")
        fig3.update_layout(**chart_cfg(
            xlabel="Department", ylabel="Avg Gap (₹)"), barmode="group")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        monthly = (filtered.groupby("Month_Str")
                   .agg(Claim_Gap_Anom=("Claim_Gap_Anomaly","sum"),
                        Payment_Gap_Anom=("Payment_Gap_Anomaly","sum"),
                        High_Risk=("High_Risk_Claim","sum"))
                   .reset_index())
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=monthly["Month_Str"], y=monthly["Claim_Gap_Anom"],
                                  name="Claim Gap Anomaly", mode="lines+markers",
                                  line=dict(color="#FF5E5E", width=2), marker=dict(size=5)))
        fig4.add_trace(go.Scatter(x=monthly["Month_Str"], y=monthly["Payment_Gap_Anom"],
                                  name="Payment Gap Anomaly", mode="lines+markers",
                                  line=dict(color="#FFD700", width=2), marker=dict(size=5)))
        fig4.add_trace(go.Scatter(x=monthly["Month_Str"], y=monthly["High_Risk"],
                                  name="High Risk Claims", mode="lines+markers",
                                  line=dict(color="#A78BFA", width=2, dash="dot"), marker=dict(size=5)))
        chart_title("Monthly Anomaly Count Trend")
        fig4.update_layout(**chart_cfg(
            xlabel="Month", ylabel="Anomaly Count"))
        st.plotly_chart(fig4, use_container_width=True)

    top_cg = filtered.groupby("Department")["Claim_Gap"].mean().sort_values(ascending=False).index[0]
    top_pg = filtered.groupby("Insurance_Type")["Payment_Gap"].mean().sort_values(ascending=False).index[0]
    insight(
        f"<strong>{len(anom):,}</strong> billing anomalies ({anom_pct:.1f}%). "
        f"Claim Gap: <strong>{len(cg_anom):,}</strong> ({cg_pct:.1f}%). "
        f"Payment Gap: <strong>{len(pg_anom):,}</strong> ({pg_pct:.1f}%). "
        f"Revenue loss: <strong>{fmt(total_loss)}</strong>. "
        f"Highest Claim Gap dept: <strong>{top_cg}</strong>. "
        f"Highest Payment Gap insurer: <strong>{top_pg}</strong>.",
        kind="red" if anom_pct > 12 else "" if anom_pct > 6 else "green")

    st.divider()
    st.markdown('<div class="chart-title">Top Anomalous Claims</div>', unsafe_allow_html=True)
    disp = [c for c in ["Claim_ID","Department","Insurance_Type","Claim_Gap",
                         "Payment_Gap","Revenue_Loss","High_Risk_Claim","Severity","Processing_Time"]
            if c in anom.columns]
    st.dataframe(anom[disp].sort_values("Claim_Gap", ascending=False).head(15),
                 use_container_width=True, hide_index=True)
