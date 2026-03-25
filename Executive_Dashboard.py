import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from chart_config import chart_cfg, load_css, page_header, chart_title, kpi, insight, fmt, COLORS
from pdf_export import render_pdf_button


@st.cache_data
def load_data():
    df = pd.read_csv("data/modified_dataset.csv")
    df["Claim_Submission_Date"] = pd.to_datetime(
        df["Claim_Submission_Date"], dayfirst=True, errors="coerce")
    df["Month_DT"]  = df["Claim_Submission_Date"].dt.to_period("M").dt.to_timestamp()
    df["Month_Str"] = df["Claim_Submission_Date"].dt.to_period("M").astype(str)
    return df


def show_dashboard():
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
        st.warning("No data for the selected filters."); return

    total_rev  = filtered["Actual_Revenue"].sum()
    net_recv   = filtered["Payment_Received"].sum()
    total_leak = filtered["Revenue_Leakage"].sum()
    approval   = (1 - filtered["Denial_Flag"].mean()) * 100

    page_header("Executive Overview", "Unified revenue intelligence  ·  Jan 2023 – Jan 2025")
    render_pdf_button("Executive_Overview")

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Total Revenue",   fmt(total_rev),  "mv-gold")
    kpi(c2, "Net Received",    fmt(net_recv),   "mv-green")
    kpi(c3, "Revenue Leakage", fmt(total_leak), "mv-red")
    kpi(c4, "Approval Rate",   f"{approval:.1f}%", "mv-cyan")
    st.divider()

    if role in ["CFO", "RCM"]:
        col1, col2 = st.columns(2)
        with col1:
            monthly = filtered.groupby("Month_DT")["Actual_Revenue"].sum().reset_index()
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=monthly["Month_DT"], y=monthly["Actual_Revenue"],
                mode="lines+markers", line=dict(color="#FFD700", width=2.2),
                marker=dict(size=5), fill="tozeroy", fillcolor="rgba(255,215,0,0.06)"))
            chart_title("Monthly Revenue Trend")
            fig1.update_layout(**chart_cfg(
                xlabel="Month", ylabel="Revenue (₹)"))
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            dept_rev = (filtered.groupby("Department")["Actual_Revenue"]
                        .sum().reset_index().sort_values("Actual_Revenue", ascending=False))
            fig2 = px.bar(dept_rev, x="Department", y="Actual_Revenue",
                          color="Department", color_discrete_sequence=COLORS)
            chart_title("Revenue by Department")
            fig2.update_layout(**chart_cfg(
                xlabel="Department", ylabel="Revenue (₹)"))
            st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            ins_rev = filtered.groupby("Insurance_Type")["Actual_Revenue"].sum().reset_index()
            fig3 = px.pie(ins_rev, values="Actual_Revenue", names="Insurance_Type",
                          hole=0.50, color_discrete_sequence=COLORS)
            fig3.update_traces(textfont_size=12,
                               marker=dict(line=dict(color="rgba(0,0,0,0.15)", width=1.5)))
            chart_title("Revenue by Insurance Type")
            fig3.update_layout(**chart_cfg())
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            denial_dept = (filtered.groupby("Department")
                           .apply(lambda x: x["Denial_Flag"].mean() * 100)
                           .reset_index(name="Denial_Rate"))
            fig4 = px.bar(denial_dept, x="Department", y="Denial_Rate",
                          color="Denial_Rate",
                          color_continuous_scale=[[0,"#00FF87"],[0.5,"#FFD700"],[1,"#FF5E5E"]])
            fig4.update_coloraxes(showscale=False)
            chart_title("Denial Rate by Department")
            fig4.update_layout(**chart_cfg(
                xlabel="Department", ylabel="Denial Rate (%)"))
            st.plotly_chart(fig4, use_container_width=True)

        top      = dept_rev.iloc[0]
        leak_pct = total_leak / filtered["Expected_Revenue"].sum() * 100
        insight(
            f"<strong>{top['Department']}</strong> leads with <strong>{fmt(top['Actual_Revenue'])}</strong>. "
            f"Leakage: <strong>{fmt(total_leak)}</strong> ({leak_pct:.1f}%). "
            f"Approval rate: <strong>{approval:.1f}%</strong>.",
            kind="red" if leak_pct > 15 else "" if leak_pct > 8 else "green")

        st.divider()
        t1, t2 = st.columns(2)
        with t1:
            st.markdown('<div class="chart-title">Top Revenue Departments</div>', unsafe_allow_html=True)
            d = dept_rev.copy(); d["Actual_Revenue"] = d["Actual_Revenue"].apply(fmt)
            st.dataframe(d, use_container_width=True, hide_index=True)
        with t2:
            st.markdown('<div class="chart-title">Outstanding Claims</div>', unsafe_allow_html=True)
            outs = filtered[filtered["Payment_Received"] < filtered["Actual_Revenue"]]
            st.dataframe(outs[["Claim_ID","Department","Insurance_Type",
                                "Actual_Revenue","Payment_Received"]].head(10),
                         use_container_width=True, hide_index=True)
    else:
        st.markdown(f"<h3>{dept} — Department View</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            monthly = filtered.groupby("Month_DT")["Actual_Revenue"].sum().reset_index()
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=monthly["Month_DT"], y=monthly["Actual_Revenue"],
                mode="lines+markers", line=dict(color="#00E5FF", width=2.2),
                fill="tozeroy", fillcolor="rgba(0,229,255,0.07)"))
            chart_title(f"Monthly Revenue — {dept}")
            fig1.update_layout(**chart_cfg(
                xlabel="Month", ylabel="Revenue (₹)"))
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            doc_rev = (filtered.groupby("Doctor_Name")["Actual_Revenue"]
                       .sum().sort_values(ascending=True).reset_index())
            fig2 = px.bar(doc_rev, x="Actual_Revenue", y="Doctor_Name",
                          orientation="h", color_discrete_sequence=["#A78BFA"])
            chart_title("Revenue by Doctor")
            fig2.update_layout(**chart_cfg(
                xlabel="Revenue (₹)", ylabel="Doctor"))
            st.plotly_chart(fig2, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3:
            ins_rev = filtered.groupby("Insurance_Type")["Actual_Revenue"].sum().reset_index()
            fig3 = px.pie(ins_rev, values="Actual_Revenue", names="Insurance_Type",
                          hole=0.50, color_discrete_sequence=COLORS)
            fig3.update_traces(marker=dict(line=dict(color="rgba(0,0,0,0.15)", width=1.5)))
            chart_title("Revenue by Insurance")
            fig3.update_layout(**chart_cfg())
            st.plotly_chart(fig3, use_container_width=True)

        with col4:
            adm = filtered["Admission_Type"].value_counts().reset_index()
            adm.columns = ["Admission_Type", "Count"]
            fig4 = px.pie(adm, values="Count", names="Admission_Type",
                          hole=0.50, color_discrete_sequence=COLORS)
            fig4.update_traces(marker=dict(line=dict(color="rgba(0,0,0,0.15)", width=1.5)))
            chart_title("Admission Type Distribution")
            fig4.update_layout(**chart_cfg())
            st.plotly_chart(fig4, use_container_width=True)

        insight(f"<strong>{dept}</strong>: {len(filtered):,} claims, approval <strong>{approval:.1f}%</strong>. "
                f"Revenue: <strong>{fmt(total_rev)}</strong>. Leakage: <strong>{fmt(total_leak)}</strong>.",
                kind="cyan")
        st.divider()
        st.markdown('<div class="chart-title">Recent Claims</div>', unsafe_allow_html=True)
        st.dataframe(filtered[["Claim_ID","Doctor_Name","Insurance_Type",
                                "Actual_Revenue","Payment_Received","Denial_Flag"]].head(15),
                     use_container_width=True, hide_index=True)
