import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from chart_config import chart_cfg, load_css, page_header, chart_title, kpi, insight, fmt, COLORS
from pdf_export import render_pdf_button


@st.cache_data
def load_data():
    pre  = pd.read_csv("data/pre_processed_data.csv")
    pred = pd.read_csv("data/denial_model_predictions.csv")
    fi   = pd.read_csv("data/denial_feature_importance.csv")
    met  = pd.read_csv("data/denial_model_metrics.csv")
    merged = pre.merge(pred, on="Claim_ID", how="left")
    merged["Claim_Submission_Date"] = pd.to_datetime(
        merged["Claim_Submission_Date"], dayfirst=True, errors="coerce")
    merged["Month_Str"] = merged["Claim_Submission_Date"].dt.to_period("M").astype(str)
    return merged, fi, met


def claim_denial():
    load_css()
    data, fi, met = load_data()
    role = st.session_state.get("role", "User")
    dept = st.session_state.get("department", "All")

    if role == "Department Head":
        data = data[data["Department"] == dept]

    f        = st.session_state.get("filters", {})
    filtered = data.copy()
    if f.get("risk_filter") and f["risk_filter"] != "All":
        filtered = filtered[filtered["Risk_Level"] == f["risk_filter"]]
    if f.get("department_filter") and f["department_filter"] != "All":
        filtered = filtered[filtered["Department"] == f["department_filter"]]
    if f.get("insurance_filter") and f["insurance_filter"] != "All":
        filtered = filtered[filtered["Insurance_Type"] == f["insurance_filter"]]

    if filtered.empty:
        st.error("No data for selected filters."); return

    high          = filtered[filtered["Risk_Level"] == "High"]
    high_pct      = len(high) / len(filtered) * 100
    pot_loss      = high["Claim_Amount"].sum() if not high.empty else 0
    top_risk_dept = high["Department"].value_counts().idxmax() if not high.empty else "N/A"

    page_header("Claim Denial Risk Prediction", "Logistic Regression  ·  ROC-AUC 0.66  ·  60,000 claims")
    render_pdf_button("Claim_Denial_Risk")

    if not met.empty:
        m      = met.iloc[0]
        cols_m = st.columns(len(m))
        for col, (k_, v_) in zip(cols_m, m.items()):
            with col:
                st.markdown(f"""
                <div style="background:var(--bg-card,#0d1b2e);border:1px solid var(--border,rgba(255,255,255,0.07));
                            border-radius:6px;padding:8px 12px;text-align:center;">
                  <div style="font-family:'Inter',sans-serif;font-size:10px;text-transform:uppercase;
                              letter-spacing:0.5px;color:var(--text-muted,#94A3B8);">{k_}</div>
                  <div style="font-family:'Rajdhani',sans-serif;font-size:18px;
                              font-weight:700;color:var(--gold,#FFD700);">{v_:.3f}</div>
                </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "High-Risk Claims",    f"{len(high):,}",    "mv-red")
    kpi(c2, "High-Risk Rate",      f"{high_pct:.1f}%",  "mv-red" if high_pct > 25 else "mv-gold")
    kpi(c3, "Potential Loss",      fmt(pot_loss),        "mv-red")
    kpi(c4, "Top Risk Department", top_risk_dept,       "mv-gold")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        ins_risk = (filtered.groupby("Insurance_Type")["Denial_Probability"]
                    .mean().reset_index().sort_values("Denial_Probability", ascending=False))
        fig1 = px.bar(ins_risk, x="Insurance_Type", y="Denial_Probability",
                      color="Denial_Probability",
                      color_continuous_scale=[[0,"#00FF87"],[0.5,"#FFD700"],[1,"#FF5E5E"]])
        fig1.update_coloraxes(showscale=False)
        chart_title("Denial Probability by Insurance Type")
        fig1.update_layout(**chart_cfg(
            xlabel="Insurance Type", ylabel="Avg Denial Probability"))
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        dept_risk = filtered.groupby(["Department","Risk_Level"]).size().reset_index(name="Count")
        fig2 = px.bar(dept_risk, x="Department", y="Count", color="Risk_Level", barmode="stack",
                      color_discrete_map={"High":"#FF5E5E","Medium":"#FFD700","Low":"#00FF87"})
        chart_title("Risk Level by Department")
        fig2.update_layout(**chart_cfg(
            xlabel="Department", ylabel="Number of Claims"))
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        trend = filtered.groupby("Month_Str")["Denial_Probability"].mean().reset_index()
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=trend["Month_Str"], y=trend["Denial_Probability"],
                                  mode="lines+markers", line=dict(color="#FF5E5E", width=2.2),
                                  marker=dict(size=5), fill="tozeroy",
                                  fillcolor="rgba(255,94,94,0.07)"))
        chart_title("Denial Risk Trend Over Time")
        fig3.update_layout(**chart_cfg(
            xlabel="Month", ylabel="Avg Denial Probability"))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        fi_s = fi.reindex(fi["Coefficient"].abs().sort_values().index).copy()
        fi_s["Color"] = fi_s["Coefficient"].apply(lambda x: "#FF5E5E" if x > 0 else "#00FF87")
        fig4 = go.Figure(go.Bar(x=fi_s["Coefficient"], y=fi_s["Feature"],
                                orientation="h",
                                marker=dict(color=fi_s["Color"], opacity=0.9)))
        fig4.add_vline(x=0, line_color="rgba(255,255,255,0.15)", line_width=1)
        chart_title("Feature Importance (Denial Drivers)")
        fig4.update_layout(**chart_cfg(
            xlabel="Coefficient", ylabel="Feature"))
        st.plotly_chart(fig4, use_container_width=True)

    top_driver = fi.reindex(fi["Coefficient"].abs().sort_values(ascending=False).index).iloc[0]["Feature"]
    insight(
        f"<strong>{high_pct:.1f}%</strong> of claims are High Risk. "
        f"Top risk dept: <strong>{top_risk_dept}</strong>. "
        f"Government denial rate: <strong>26.7%</strong> — key driver: <strong>{top_driver}</strong>. "
        f"Revenue at risk: <strong>{fmt(pot_loss)}</strong>.",
        kind="red" if high_pct > 25 else "" if high_pct > 15 else "green")

    st.divider()
    st.markdown('<div class="chart-title">High-Risk Claims — Priority Review</div>', unsafe_allow_html=True)
    filtered["Denial_Probability"] = pd.to_numeric(filtered["Denial_Probability"], errors="coerce")
    top10 = filtered.sort_values("Denial_Probability", ascending=False).head(10)
    cols_ = [c for c in ["Claim_ID","Department","Insurance_Type","Claim_Amount",
                          "Denial_Probability","Risk_Level"] if c in top10.columns]
    st.dataframe(top10[cols_], use_container_width=True, hide_index=True)
