import streamlit as st
import pandas as pd
import base64
import streamlit.components.v1 as components
from chart_config import load_css

ROLE_PAGES = {
    "CFO":             ["executive","revenue_leakage","claim_denial",
                        "billing_anomaly","forecast","cfo_strategic"],
    "RCM":             ["executive","revenue_leakage","claim_denial",
                        "billing_anomaly","forecast"],
    "Department Head": ["executive","revenue_leakage","claim_denial","billing_anomaly"],
    "Insurance Team":  ["insurance_view"],
}

PAGE_LABELS = {
    "executive":       "Executive Overview",
    "forecast":        "Revenue Forecasting",
    "revenue_leakage": "Revenue Leakage",
    "claim_denial":    "Claim Denial Risk",
    "billing_anomaly": "Billing Anomaly",
    "cfo_strategic":   "CFO Strategic View",
    "insurance_view":  "Insurance Analytics",
}

BADGE = {
    "CFO":            ("rb-cfo",  "CFO"),
    "RCM":            ("rb-rcm",  "RCM"),
    "Department Head":("rb-dept", "Dept Head"),
    "Insurance Team": ("rb-ins",  "Insurance"),
}

# SVG logo — crisp at any size, no base64 corruption possible
_LOGO_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 220 56" width="180" height="46">
  <defs>
    <linearGradient id="sgl" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%"   stop-color="#FFD700"/>
      <stop offset="50%"  stop-color="#FFF8DC"/>
      <stop offset="100%" stop-color="#FFD700"/>
    </linearGradient>
    <filter id="sglow">
      <feGaussianBlur stdDeviation="2.2" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <polyline points="4,34 18,34 22,18 26,46 30,26 34,40 38,34 52,34"
            fill="none" stroke="#FFD700" stroke-width="2"
            stroke-linecap="round" stroke-linejoin="round"
            filter="url(#sglow)" opacity="0.85"/>
  <text x="60" y="38"
        font-family="Rajdhani,Arial Narrow,sans-serif"
        font-size="28" font-weight="700"
        letter-spacing="3"
        fill="url(#sgl)"
        filter="url(#sglow)">MEDILYTICS</text>
  <text x="61" y="50"
        font-family="JetBrains Mono,Courier New,monospace"
        font-size="7.5" font-weight="400"
        letter-spacing="2.5"
        fill="#475569">HEALTHCARE REVENUE INTELLIGENCE</text>
</svg>"""


# ── Theme injection helper ─────────────────────────────────────────────────────
def _inject_theme(theme: str):
    """Stamp data-theme on <html> so CSS variables switch instantly."""
    st.markdown(f"""
    <script>
    (function(){{
      document.documentElement.setAttribute('data-theme', '{theme}');
    }})();
    </script>
    """, unsafe_allow_html=True)


def sidebar():
    load_css()

    # ── Remove leftover login spark artefacts via iframe → parent window ──────
    # st.markdown scripts are sandboxed and can't reach the parent DOM.
    # components.html runs in an iframe whose JS CAN reach window.parent,
    # same mechanism used by the spark injector — so this reliably cleans up.
    components.html("""<!DOCTYPE html><html><body><script>
    (function(){
      var pw   = window.parent;
      var pdoc = pw.document;
      ['medilytics-spark-canvas',
       'medilytics-cursor-dot',
       'medilytics-cursor-override'].forEach(function(id){
        var el = pdoc.getElementById(id);
        if(el) el.remove();
      });
    })();
    </script></body></html>""", height=0, scrolling=False)

    # ── Init theme state ───────────────────────────────────────────────────────
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"

    # Always inject theme attribute on every render
    _inject_theme(st.session_state.theme)

    with st.sidebar:
        # ── LOGO ──────────────────────────────────────────────────────────────
        st.markdown(f"""
        <div style="text-align:center; padding: 8px 0 4px;
                    filter:drop-shadow(0 0 10px rgba(255,215,0,0.3));">
          {_LOGO_SVG}
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── PROFILE ───────────────────────────────────────────────────────────
        role     = st.session_state.get("role", "User")
        username = st.session_state.get("username", "—")
        dept     = st.session_state.get("department", "—")
        cls, lbl = BADGE.get(role, ("rb-rcm", role))

        st.markdown(f"""
        <div style="padding:6px 0 4px;">
          <div class="section-label">Logged in as</div>
          <div style="font-family:'Rajdhani',sans-serif;font-size:18px;
                      font-weight:700;color:#EFF6FF;margin:3px 0;">{username}</div>
          <span class="role-badge {cls}">{lbl}</span>
        </div>""", unsafe_allow_html=True)

        if role == "Department Head":
            st.markdown(f"""
            <div style="font-family:'JetBrains Mono',monospace;font-size:11px;
                        color:#00FF87;margin-top:5px;">{dept}</div>
            """, unsafe_allow_html=True)

        st.divider()

        # ── FILTERS ───────────────────────────────────────────────────────────
        st.markdown('<div class="section-label">Filters</div>', unsafe_allow_html=True)
        page    = st.session_state.get("page", "executive")
        filters = st.session_state.get("filters", {})

        @st.cache_data
        def load_main():
            df = pd.read_csv("data/modified_dataset.csv")
            df["Claim_Submission_Date"] = pd.to_datetime(
                df["Claim_Submission_Date"], dayfirst=True, errors="coerce")
            return df

        @st.cache_data
        def load_pre():
            return pd.read_csv("data/pre_processed_data.csv")

        if page in ["executive","revenue_leakage","billing_anomaly"]:
            df  = load_main()
            mn  = df["Claim_Submission_Date"].min().date()
            mx  = df["Claim_Submission_Date"].max().date()
            filters["date_range"] = st.date_input(
                "Date Range",
                value=filters.get("date_range",(mn,mx)),
                min_value=mn, max_value=mx)
            depts = sorted(df["Department"].dropna().unique().tolist())
            if role in ["CFO","RCM"]:
                filters["department_filter"] = st.selectbox(
                    "Department", ["All"]+depts, index=0)
            elif role == "Department Head":
                filters["department_filter"] = dept
                st.info(dept)
            else:
                filters["department_filter"] = "All"
            ins_opts = sorted(df["Insurance_Type"].dropna().unique().tolist())
            filters["insurance_filter"] = st.selectbox(
                "Insurance Type", ["All"]+ins_opts)

        elif page == "claim_denial":
            df_pre = load_pre()
            depts  = sorted(df_pre["Department"].dropna().unique().tolist())
            filters["risk_filter"] = st.selectbox(
                "Risk Level", ["All","Low","Medium","High"])
            if role in ["CFO","RCM"]:
                filters["department_filter"] = st.selectbox(
                    "Department", ["All"]+depts, index=0)
            elif role == "Department Head":
                filters["department_filter"] = dept
                st.info(dept)
            ins_opts = sorted(df_pre["Insurance_Type"].dropna().unique().tolist())
            filters["insurance_filter"] = st.selectbox(
                "Insurance Type", ["All"]+ins_opts)

        elif page == "forecast":
            st.caption("Showing 6-month ARIMA forecast.")
        elif page in ["cfo_strategic","insurance_view"]:
            st.caption("Role-specific view.")

        st.session_state.filters = filters
        st.divider()

        # ── NAVIGATION ────────────────────────────────────────────────────────
        st.markdown('<div class="section-label">Navigation</div>', unsafe_allow_html=True)
        allowed = ROLE_PAGES.get(role, ["executive"])
        for pg in allowed:
            label = PAGE_LABELS.get(pg, pg)
            if st.button(label, key=f"nav_{pg}", use_container_width=True):
                st.session_state.page    = pg
                st.session_state.filters = {}
                st.rerun()

        st.divider()

        # ── LOGOUT ────────────────────────────────────────────────────────────
        if st.button("Logout", use_container_width=True, key="logout_btn"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
