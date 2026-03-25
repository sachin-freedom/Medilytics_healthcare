# chart_config.py

import streamlit as st

COLORS = ["#FFD700","#00E5FF","#A78BFA","#00FF87","#FF5E5E","#FB923C","#38BDF8","#F472B6"]


def chart_cfg(xlabel: str = "", ylabel: str = "") -> dict:
    """
    Returns a Plotly layout dict.
xlabel → x-axis label
    ylabel → y-axis label
    """
    fc  = "#EFF6FF"
    gc  = "rgba(255,255,255,0.05)"
    lc  = "#94A3B8"
    hbg = "#0d1b2e"
    hb  = "#FFD700"

    xax = dict(gridcolor=gc, zeroline=False, showline=False,
               tickfont=dict(color=fc, size=11))
    yax = dict(gridcolor=gc, zeroline=False, showline=False,
               tickfont=dict(color=fc, size=11))

    if xlabel:
        xax["title"] = dict(text=xlabel, font=dict(color=fc, size=12), standoff=10)
    if ylabel:
        yax["title"] = dict(text=ylabel, font=dict(color=fc, size=12), standoff=10)

    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=fc, family="Rajdhani, Inter, sans-serif", size=12),
        xaxis=xax,
        yaxis=yax,
        legend=dict(
            orientation="h", yanchor="top", y=-0.28,
            xanchor="center", x=0.5,
            font=dict(color=lc, size=11)
        ),
        margin=dict(l=55, r=20, t=12, b=88),
        colorway=COLORS,
        hoverlabel=dict(bgcolor=hbg, font_size=12,
                        font_family="Rajdhani, sans-serif", bordercolor=hb),
    )


    return layout


def styled(fig):
    fig.update_layout(**chart_cfg())
    return fig


def load_css():
    with open("style.css", encoding="utf-8") as f:
        base = f.read()
    st.markdown(f"<style>{base}</style>", unsafe_allow_html=True)


def page_header(title: str, subtitle: str = ""):
    st.markdown(f"""
    <div class="page-header">
      <div class="ph-title">{title}</div>
      {"<div class='ph-sub'>" + subtitle + "</div>" if subtitle else ""}
    </div>""", unsafe_allow_html=True)


def chart_title(label: str):
    """Render a chart title above the chart box (outside Plotly)."""
    st.markdown(f'<div class="chart-title">{label}</div>', unsafe_allow_html=True)


def kpi(col, label: str, value: str, accent: str = ""):
    with col:
        st.markdown(f"""
        <div class="metric-box">
          <div class="metric-label">{label}</div>
          <div class="metric-value {accent}">{value}</div>
        </div>""", unsafe_allow_html=True)


def insight(text: str, kind: str = "gold", title: str = "Key Insight"):
    st.markdown(f"""
    <div class="insight-box {kind if kind != 'gold' else ''}">
      <div class="insight-title">{title}</div>
      {text}
    </div>""", unsafe_allow_html=True)


def fmt(n: float) -> str:
    if abs(n) >= 1e7: return f"\u20b9{n/1e7:.2f} Cr"
    if abs(n) >= 1e5: return f"\u20b9{n/1e5:.1f} L"
    if abs(n) >= 1e3: return f"\u20b9{n/1e3:.1f} K"
    return f"\u20b9{n:.0f}"
