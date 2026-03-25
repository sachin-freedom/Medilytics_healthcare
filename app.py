import streamlit as st

st.set_page_config(
    page_title="Medilytics",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

from Login import show_login
from Executive_Dashboard import show_dashboard
from sidebar import sidebar

for k, v in {
    "logged_in": False, "page": "executive", "filters": {},
    "role": None, "username": None, "department": None
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

if not st.session_state.logged_in:
    show_login()
else:
    sidebar()
    page = st.session_state.page

    if page == "executive":
        show_dashboard()
    elif page == "revenue_leakage":
        from Revenue_Leakage_Analysis import revenue
        revenue()
    elif page == "claim_denial":
        from Claim_Denial_main import claim_denial
        claim_denial()
    elif page == "billing_anomaly":
        from billing_anomaly import billing_anomaly
        billing_anomaly()
    elif page == "forecast":
        from forecast_dashboard import revenue_forecast_model
        revenue_forecast_model()
    elif page == "cfo_strategic":
        from cfo_strategic import cfo_strategic
        cfo_strategic()
    elif page == "insurance_view":
        from insurance_view import insurance_view
        insurance_view()
    else:
        st.warning(f"Page not found: {page}")
