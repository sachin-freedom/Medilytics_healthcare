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
    df["Settlement_Date"] = pd.to_datetime(
        df["Settlement_Date"], dayfirst=True, errors="coerce")
    df["Month_Str"]      = df["Claim_Submission_Date"].dt.to_period("M").astype(str)
    df["Settlement_TAT"] = (df["Settlement_Date"] - df["Claim_Submission_Date"]).dt.days
    pred = pd.read_csv("data/denial_model_predictions.csv")
    return df.merge(pred, on="Claim_ID", how="left")


def insurance_view():
    load_css()
    if st.session_state.get("role") != "Insurance Team":
        st.error("Access Restricted — Insurance Team credentials required."); return

    data = load_data()

    ins_sum = data.groupby("Insurance_Type").agg(
        Claims=("Claim_ID","count"), Total_Revenue=("Actual_Revenue","sum"),
        Avg_Revenue=("Actual_Revenue","mean"), Denial_Rate=("Denial_Flag","mean"),
        Total_Leakage=("Revenue_Leakage","sum"),
        Avg_TAT=("Settlement_TAT","mean"), Total_At_Risk=("Revenue_At_Risk","sum")
    ).reset_index()
    ins_sum["Denial_%"] = ins_sum["Denial_Rate"] * 100

    worst      = ins_sum.sort_values("Denial_%", ascending=False).iloc[0]
    best       = ins_sum.sort_values("Denial_%").iloc[0]
    total_risk = data["Revenue_At_Risk"].sum()
    govt_d     = ins_sum[ins_sum["Insurance_Type"]=="Government"]["Denial_%"].values
    govt_d     = govt_d[0] if len(govt_d) > 0 else 0

    page_header("Insurance Analytics", "Payer mix  ·  Denial analysis  ·  Settlement tracking")
    render_pdf_button("Insurance_Analytics")

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Highest Denial Rate",    f"{worst['Denial_%']:.1f}% ({worst['Insurance_Type']})", "mv-red")
    kpi(c2, "Lowest Denial Rate",     f"{best['Denial_%']:.1f}% ({best['Insurance_Type']})",   "mv-green")
    kpi(c3, "Total Revenue at Risk",  fmt(total_risk),                                          "mv-purple")
    kpi(c4, "Government Denial Rate", f"{govt_d:.1f}%",                                         "mv-red")
    st.divider()

    tab1, tab2, tab3 = st.tabs(["Payer Mix","Denial Analysis","Settlement & Risk"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.pie(ins_sum, values="Total_Revenue", names="Insurance_Type",
                          hole=0.50, color_discrete_sequence=COLORS)
            fig1.update_traces(textfont_size=12,
                               marker=dict(line=dict(color="rgba(0,0,0,0.12)", width=1.5)))
            chart_title("Revenue Distribution by Payer")
            fig1.update_layout(**chart_cfg())
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            ins_t = data.groupby(["Month_Str","Insurance_Type"])["Actual_Revenue"].sum().reset_index()
            fig2  = px.line(ins_t, x="Month_Str", y="Actual_Revenue",
                            color="Insurance_Type", color_discrete_sequence=COLORS)
            chart_title("Revenue Trend by Insurance Type")
            fig2.update_layout(**chart_cfg(
                xlabel="Month", ylabel="Revenue (₹)"))
            st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            fig3 = px.bar(ins_sum.sort_values("Avg_Revenue", ascending=False),
                          x="Insurance_Type", y="Avg_Revenue",
                          color="Insurance_Type", color_discrete_sequence=COLORS)
            chart_title("Avg Revenue per Claim by Payer")
            fig3.update_layout(**chart_cfg(
                xlabel="Insurance Type", ylabel="Avg Revenue (₹)"))
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            fig4 = px.pie(ins_sum, values="Claims", names="Insurance_Type",
                          hole=0.50, color_discrete_sequence=COLORS)
            fig4.update_traces(marker=dict(line=dict(color="rgba(0,0,0,0.12)", width=1.5)))
            chart_title("Claim Volume by Insurance Type")
            fig4.update_layout(**chart_cfg())
            st.plotly_chart(fig4, use_container_width=True)

        insight("Private insurance is the largest volume payer. Self-Pay yields highest avg revenue per claim.",
                kind="cyan")

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig5 = px.bar(ins_sum.sort_values("Denial_%", ascending=False),
                          x="Insurance_Type", y="Denial_%", color="Denial_%",
                          color_continuous_scale=[[0,"#00FF87"],[0.5,"#FFD700"],[1,"#FF5E5E"]])
            fig5.update_coloraxes(showscale=False)
            chart_title("Denial Rate by Insurance Type")
            fig5.update_layout(**chart_cfg(
                xlabel="Insurance Type", ylabel="Denial Rate (%)"))
            st.plotly_chart(fig5, use_container_width=True)

        with col2:
            dt = (data[data["Denial_Flag"]==1]
                  .groupby(["Month_Str","Insurance_Type"]).size().reset_index(name="Denials"))
            fig6 = px.line(dt, x="Month_Str", y="Denials",
                           color="Insurance_Type", markers=True, color_discrete_sequence=COLORS)
            chart_title("Monthly Denial Volume by Insurance")
            fig6.update_layout(**chart_cfg(
                xlabel="Month", ylabel="Denial Count"))
            st.plotly_chart(fig6, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            if "Denial_Probability" in data.columns:
                ip = (data.groupby("Insurance_Type")["Denial_Probability"]
                      .mean().reset_index().sort_values("Denial_Probability", ascending=False))
                fig7 = px.bar(ip, x="Insurance_Type", y="Denial_Probability",
                              color="Denial_Probability",
                              color_continuous_scale=[[0,"#00FF87"],[0.5,"#FFD700"],[1,"#FF5E5E"]])
                fig7.update_coloraxes(showscale=False)
                chart_title("Avg ML Denial Probability by Payer")
                fig7.update_layout(**chart_cfg(
                    xlabel="Insurance Type", ylabel="Denial Probability"))
                st.plotly_chart(fig7, use_container_width=True)

        with col4:
            if "Risk_Level" in data.columns:
                ri = data.groupby(["Insurance_Type","Risk_Level"]).size().reset_index(name="Count")
                fig8 = px.bar(ri, x="Insurance_Type", y="Count", color="Risk_Level", barmode="stack",
                              color_discrete_map={"High":"#FF5E5E","Medium":"#FFD700","Low":"#00FF87"})
                chart_title("Risk Level Distribution by Insurance")
                fig8.update_layout(**chart_cfg(
                    xlabel="Insurance Type", ylabel="Claims"))
                st.plotly_chart(fig8, use_container_width=True)

        insight(f"Government denial rate: <strong>{govt_d:.1f}%</strong> — highest payer risk. "
                f"Self-Pay denial rate: <strong>{best['Denial_%']:.1f}%</strong> — lowest.", kind="red")

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            fig9 = px.bar(ins_sum.sort_values("Avg_TAT", ascending=False),
                          x="Insurance_Type", y="Avg_TAT", color="Avg_TAT",
                          color_continuous_scale=[[0,"#00FF87"],[0.5,"#FFD700"],[1,"#FF5E5E"]])
            fig9.update_coloraxes(showscale=False)
            chart_title("Avg Settlement Turnaround Days by Payer")
            fig9.update_layout(**chart_cfg(
                xlabel="Insurance Type", ylabel="Avg Days"))
            st.plotly_chart(fig9, use_container_width=True)

        with col2:
            fig10 = px.bar(ins_sum.sort_values("Total_Leakage", ascending=True),
                           x="Total_Leakage", y="Insurance_Type", orientation="h",
                           color="Total_Leakage",
                           color_continuous_scale=[[0,"#00FF87"],[0.5,"#FFD700"],[1,"#FF5E5E"]])
            fig10.update_coloraxes(showscale=False)
            chart_title("Revenue Leakage by Insurance")
            fig10.update_layout(**chart_cfg(
                xlabel="Leakage (₹)", ylabel="Insurance Type"))
            st.plotly_chart(fig10, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            lt = data.groupby(["Month_Str","Insurance_Type"])["Revenue_Leakage"].sum().reset_index()
            fig11 = px.area(lt, x="Month_Str", y="Revenue_Leakage",
                            color="Insurance_Type", color_discrete_sequence=COLORS)
            chart_title("Revenue Leakage Trend by Insurance")
            fig11.update_layout(**chart_cfg(
                xlabel="Month", ylabel="Leakage (₹)"))
            st.plotly_chart(fig11, use_container_width=True)

        with col4:
            fig12 = px.bar(ins_sum.sort_values("Total_At_Risk", ascending=True),
                           x="Total_At_Risk", y="Insurance_Type", orientation="h",
                           color="Total_At_Risk",
                           color_continuous_scale=[[0,"#00E5FF"],[0.5,"#FFD700"],[1,"#FF5E5E"]])
            fig12.update_coloraxes(showscale=False)
            chart_title("Revenue at Risk by Insurance")
            fig12.update_layout(**chart_cfg(
                xlabel="Revenue at Risk (₹)", ylabel="Insurance Type"))
            st.plotly_chart(fig12, use_container_width=True)

        wl = ins_sum.sort_values("Total_Leakage", ascending=False).iloc[0]
        insight(f"Revenue at risk: <strong>{fmt(total_risk)}</strong>. "
                f"<strong>{wl['Insurance_Type']}</strong> highest leakage at "
                f"<strong>{fmt(wl['Total_Leakage'])}</strong>.", kind="red")

    st.divider()
    st.markdown('<div class="chart-title">Insurance Summary Table</div>', unsafe_allow_html=True)
    disp = ins_sum.copy()
    for c_ in ["Total_Revenue","Total_Leakage","Total_At_Risk"]:
        disp[c_] = disp[c_].apply(fmt)
    disp["Avg_Revenue"] = disp["Avg_Revenue"].apply(lambda x: f"\u20b9{x:,.0f}")
    disp["Denial_%"]    = disp["Denial_%"].apply(lambda x: f"{x:.1f}%")
    disp["Avg_TAT"]     = disp["Avg_TAT"].apply(lambda x: f"{x:.0f}d")
    st.dataframe(
        disp.drop(columns=["Denial_Rate"]).rename(columns={
            "Insurance_Type":"Insurance","Avg_TAT":"Avg Settlement Days","Denial_%":"Denial Rate"}),
        use_container_width=True, hide_index=True)
