"""LatAm Macro Dashboard — Streamlit app inspired by Dashboard123.
Features: ETF overlays, sparklines in resumen, auto-refresh."""

import time
import base64
import io
import altair as alt
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
from datetime import datetime

from data_loader import (
    load_resumen, load_targets, load_country,
    ALL_COUNTRIES, FULL_COUNTRIES, FULL_VARIABLES, SIMPLE_VARIABLES,
    COUNTRY_NAMES, COUNTRY_FLAGS, COUNTRY_ETF, VARIABLE_COLORS,
)

# ── Theme ──────────────────────────────────────────────────────────────────
COLORS_DARK = {
    "bg": "#0e1117", "bg_card": "#1a1d23", "bg_sidebar": "#14171e",
    "text": "#e6e6e6", "text_header": "#ffffff", "text_muted": "#8b949e",
    "border": "#30363d", "green": "#3fb950", "red": "#f85149",
    "yellow": "#d29922", "blue": "#58a6ff",
}
COLORS_LIGHT = {
    "bg": "#ffffff", "bg_card": "#f6f8fa", "bg_sidebar": "#f0f2f5",
    "text": "#1f2328", "text_header": "#0d1117", "text_muted": "#656d76",
    "border": "#d0d7de", "green": "#1a7f37", "red": "#cf222e",
    "yellow": "#9a6700", "blue": "#0969da",
}


def get_theme_css(theme):
    c = COLORS_DARK if theme == "dark" else COLORS_LIGHT
    return f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
    .stApp {{ background-color: {c['bg']}; font-family: 'IBM Plex Sans', sans-serif; }}
    [data-testid="stSidebar"] {{ background-color: {c['bg_sidebar']}; }}
    .kpi-card {{
        background: {c['bg_card']}; border: 1px solid {c['border']}; border-radius: 8px;
        padding: 12px 16px; text-align: center;
    }}
    .kpi-label {{ font-size: 10px; color: {c['text_muted']}; text-transform: uppercase; letter-spacing: 0.5px; font-family: 'IBM Plex Mono', monospace; }}
    .kpi-value {{ font-size: 20px; font-weight: 700; color: {c['text']}; font-family: 'IBM Plex Mono', monospace; }}
    .signal-badge {{
        display: inline-block; padding: 2px 8px; border-radius: 4px;
        font-size: 11px; font-weight: 600; font-family: 'IBM Plex Mono', monospace;
    }}
    .signal-acelera {{ background: {c['green']}22; color: {c['green']}; }}
    .signal-desacelera {{ background: {c['red']}22; color: {c['red']}; }}
    .signal-estable {{ background: {c['yellow']}22; color: {c['yellow']}; }}
    .section-header {{
        font-size: 16px; font-weight: 600; color: {c['text_header']};
        margin: 12px 0 6px 0; font-family: 'IBM Plex Sans', sans-serif;
    }}
    .country-header {{
        font-size: 14px; font-weight: 600; color: {c['text_header']};
        padding: 6px 0; border-bottom: 1px solid {c['border']}33;
        font-family: 'IBM Plex Sans', sans-serif;
    }}
    .heatmap-table {{ width: 100%; border-collapse: collapse; font-family: 'IBM Plex Mono', monospace; font-size: 12px; }}
    .heatmap-table th {{
        padding: 6px 8px; text-align: center; font-size: 10px; color: {c['text_muted']};
        text-transform: uppercase; letter-spacing: 0.3px; border-bottom: 1px solid {c['border']}44;
    }}
    .heatmap-table td {{ padding: 5px 8px; text-align: center; border-bottom: 1px solid {c['border']}22; }}
    .heatmap-table td:first-child {{ text-align: left; font-weight: 600; color: {c['text']}; }}
    .heatmap-table td:nth-child(2) {{ text-align: left; color: {c['text_muted']}; }}
    .refresh-bar {{
        display: flex; align-items: center; justify-content: flex-end; gap: 8px;
        padding: 4px 8px; font-size: 11px; color: {c['text_muted']};
        font-family: 'IBM Plex Mono', monospace;
    }}
    </style>"""


# ── Helpers ────────────────────────────────────────────────────────────────

def signal_badge(signal):
    s = str(signal).strip()
    if s == "Acelera":
        return '<span class="signal-badge signal-acelera">▲ Acelera</span>'
    elif s == "Desacelera":
        return '<span class="signal-badge signal-desacelera">▼ Desacelera</span>'
    elif s == "Estable":
        return '<span class="signal-badge signal-estable">● Estable</span>'
    return '<span class="signal-badge" style="opacity:0.4">—</span>'


def signal_color(signal, colors):
    s = str(signal).strip()
    if s == "Acelera": return colors["green"]
    if s == "Desacelera": return colors["red"]
    if s == "Estable": return colors["yellow"]
    return colors["text_muted"]


def fmt_pct(v, decimals=1):
    if pd.isna(v): return "—"
    return f"{v*100:.{decimals}f}%"


def fmt_acel(v):
    if pd.isna(v): return "—"
    return f"{v:+.2f} pp"


def kpi_card(label, value, colors, color=None):
    vc = color or colors["text"]
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value" style="color:{vc}">{value}</div>'
        f'</div>'
    )


def style_chart(chart, colors, height=340):
    return (
        chart.properties(height=height)
        .configure_view(strokeWidth=0)
        .configure(background=colors["bg_card"])
        .configure_axis(
            labelColor=colors["text_muted"], titleColor=colors["text_muted"],
            gridColor=f"{colors['border']}60", domainColor=colors["border"],
        )
        .configure_legend(labelColor=colors["text"], titleColor=colors["text"])
    )


# ── ETF Data (yfinance, cached) ───────────────────────────────────────────

@st.cache_data(ttl=900, show_spinner=False)
def fetch_etf_data(ticker: str, period: str = "5y", interval: str = "1wk") -> pd.Series:
    """Fetch ETF weekly close. Returns Series indexed by date."""
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False, timeout=20)
        if data.empty:
            return pd.Series(dtype=float)
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close.index = pd.to_datetime(close.index).tz_localize(None)
        return close
    except Exception:
        return pd.Series(dtype=float)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_etf_daily(ticker: str, period: str = "6mo") -> pd.Series:
    """Fetch ETF daily close for sparklines."""
    try:
        data = yf.download(ticker, period=period, interval="1d", progress=False, timeout=15)
        if data.empty:
            return pd.Series(dtype=float)
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close.index = pd.to_datetime(close.index).tz_localize(None)
        return close
    except Exception:
        return pd.Series(dtype=float)


def build_etf_overlay(etf_ticker: str, colors: dict, date_min=None, date_max=None):
    """Build an Altair ETF line layer for dual-axis overlay."""
    etf_data = fetch_etf_data(etf_ticker)
    if etf_data.empty:
        return None
    if date_min:
        etf_data = etf_data[etf_data.index >= pd.Timestamp(date_min)]
    if date_max:
        etf_data = etf_data[etf_data.index <= pd.Timestamp(date_max)]
    if etf_data.empty:
        return None

    df = etf_data.reset_index()
    df.columns = ["Fecha", etf_ticker]
    return (
        alt.Chart(df)
        .mark_line(strokeWidth=1, opacity=0.35, color=colors["green"])
        .encode(
            x="Fecha:T",
            y=alt.Y(f"{etf_ticker}:Q", title=etf_ticker, axis=alt.Axis(orient="right")),
            tooltip=[alt.Tooltip(f"{etf_ticker}:Q", format=",.1f"), "Fecha:T"],
        )
    )


# ── SVG Sparklines ─────────────────────────────────────────────────────────

def svg_sparkline(values, width=90, height=24, color="#3fb950", neg_color="#f85149"):
    """Generate a tiny inline SVG sparkline from a list of values."""
    vals = [v for v in values if not (pd.isna(v) or np.isinf(v))]
    if len(vals) < 3:
        return ""
    vmin, vmax = min(vals), max(vals)
    rng = vmax - vmin if vmax != vmin else 1.0
    n = len(vals)
    points = []
    for i, v in enumerate(vals):
        x = (i / (n - 1)) * width
        y = height - ((v - vmin) / rng) * (height - 2) - 1
        points.append(f"{x:.1f},{y:.1f}")

    line_color = color if vals[-1] >= vals[0] else neg_color
    polyline = " ".join(points)
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'style="vertical-align:middle;display:inline-block">'
        f'<polyline points="{polyline}" fill="none" stroke="{line_color}" stroke-width="1.5" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
        f'<circle cx="{points[-1].split(",")[0]}" cy="{points[-1].split(",")[1]}" '
        f'r="2" fill="{line_color}"/>'
        f'</svg>'
    )


def make_crec_sparkline(country_code, variable, colors):
    """Build sparkline from the Crecimiento series of a variable for a country."""
    try:
        data = load_country(country_code)
        var_key_map = {
            "Actividad": "Actividad", "Crédito": "Crédito", "Inflación": "Inflación",
            "Tasa de política": "Tasa de política", "Desempleo": "Desempleo",
            "TCRM": "TCRM", "Resultado primario": "Resultado primario",
            "Resultado financiero": "Resultado financiero",
        }
        vk = var_key_map.get(variable, variable)
        if vk not in data:
            return ""
        df = data[vk]
        if "Crecimiento" not in df.columns:
            return ""
        series = df["Crecimiento"].dropna()
        if len(series) < 6:
            return ""
        # Last 24 months
        vals = (series.sort_index().tail(24) * 100).tolist()
        return svg_sparkline(vals, color=colors["green"], neg_color=colors["red"])
    except Exception:
        return ""


def make_etf_sparkline(country_code, colors):
    """Build sparkline from the country ETF daily data."""
    etf = COUNTRY_ETF.get(country_code)
    if not etf:
        return ""
    try:
        series = fetch_etf_daily(etf)
        if series.empty or len(series) < 5:
            return ""
        vals = series.tail(90).tolist()
        return svg_sparkline(vals, width=80, height=20, color=colors["green"], neg_color=colors["red"])
    except Exception:
        return ""


# ── Tab 1: Resumen (Heatmap + Sparklines) ─────────────────────────────────

def render_resumen_tab(colors, theme):
    resumen = load_resumen()
    targets = load_targets()

    st.markdown('<div class="section-header">Señales Macro por País — Último Dato Disponible</div>', unsafe_allow_html=True)
    st.caption("Heatmap de aceleración/desaceleración. Acelera (▲) = expansivo vs. período previo. "
               "Sparklines muestran los últimos 24 meses de la 1ª derivada. ETF sparkline = últimos 90 días.")

    for pais in ["CHL", "BRA", "MEX"]:
        flag = COUNTRY_FLAGS.get(pais, "")
        name = COUNTRY_NAMES.get(pais, pais)
        etf = COUNTRY_ETF.get(pais, "")
        pais_data = resumen[resumen["País"] == pais]
        if pais_data.empty:
            continue

        # ETF sparkline + price for header
        etf_spark = make_etf_sparkline(pais, colors)
        etf_price = ""
        if etf:
            try:
                s = fetch_etf_daily(etf)
                if not s.empty:
                    p = s.iloc[-1]
                    chg = ((s.iloc[-1] / s.iloc[-2]) - 1) * 100 if len(s) > 1 else 0
                    chg_color = colors["green"] if chg >= 0 else colors["red"]
                    etf_price = (
                        f'<span style="font-family:IBM Plex Mono;font-size:12px;margin-left:12px;color:{colors["text_muted"]}">'
                        f'{etf} ${p:.2f} <span style="color:{chg_color}">{chg:+.1f}%</span></span>'
                    )
            except Exception:
                pass

        st.markdown(
            f'<div class="country-header">{flag} {name} {etf_spark} {etf_price}</div>',
            unsafe_allow_html=True,
        )

        html = '<table class="heatmap-table"><tr>'
        html += '<th style="text-align:left">Variable</th><th>Sparkline</th><th>Nivel/Crec.</th><th>Fecha</th>'
        html += '<th>Señal t-1</th><th>Señal t-3</th><th>Señal t-6</th>'
        html += '<th>Acel t-1</th><th>Acel t-3</th><th>Acel t-6</th></tr>'

        for _, row in pais_data.iterrows():
            var = row["Variable"]
            crec = row["Crecimiento"]
            crec_str = fmt_pct(crec) if not pd.isna(crec) else "—"
            fecha_str = row["Fecha"].strftime("%b %Y") if hasattr(row["Fecha"], "strftime") else str(row["Fecha"])[:10]
            spark = make_crec_sparkline(pais, var, colors)
            html += f'<tr><td>{var}</td><td>{spark}</td><td>{crec_str}</td>'
            html += f'<td style="color:{colors["text_muted"]}">{fecha_str}</td>'
            html += f'<td>{signal_badge(row["Señal_t1"])}</td>'
            html += f'<td>{signal_badge(row["Señal_t3"])}</td>'
            html += f'<td>{signal_badge(row["Señal_t6"])}</td>'
            html += f'<td>{fmt_acel(row["Acel_t1"])}</td>'
            html += f'<td>{fmt_acel(row["Acel_t3"])}</td>'
            html += f'<td>{fmt_acel(row["Acel_t6"])}</td></tr>'
        html += '</table>'
        st.markdown(html, unsafe_allow_html=True)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # Targets table
    if not targets.empty:
        st.markdown('<div class="section-header">Metas de Inflación y Tasas de Política</div>', unsafe_allow_html=True)
        html = '<table class="heatmap-table"><tr>'
        html += '<th style="text-align:left">País</th><th>Inflación Actual</th><th>Meta</th><th>Rango</th><th>TPM Actual</th><th>TPM Esperada</th></tr>'
        for _, row in targets.iterrows():
            pais = row["País"]
            flag = COUNTRY_FLAGS.get(pais, "")
            name = COUNTRY_NAMES.get(pais, pais)
            infl_color = colors["red"] if (not pd.isna(row["Inflación_actual"]) and row["Inflación_actual"] > row["Meta_inflación"]) else colors["green"]
            html += f'<tr><td>{flag} {name}</td>'
            html += f'<td style="color:{infl_color};font-weight:600">{fmt_pct(row["Inflación_actual"])}</td>'
            html += f'<td>{fmt_pct(row["Meta_inflación"])}</td>'
            html += f'<td style="color:{colors["text_muted"]}">{row["Rango"]}</td>'
            html += f'<td style="font-weight:600">{fmt_pct(row["TPM_actual"])}</td>'
            html += f'<td style="color:{colors["text_muted"]}">{fmt_pct(row["TPM_esperada"])}</td></tr>'
        html += '</table>'
        st.markdown(html, unsafe_allow_html=True)


# ── Variable Tabs (with ETF overlay) ──────────────────────────────────────

def render_variable_tab(variable, colors, theme):
    st.markdown(f'<div class="section-header">{variable} — Comparación Cross-Country</div>', unsafe_allow_html=True)

    var_key_map = {
        "Actividad": "Actividad", "Crédito": "Crédito", "Inflación": "Inflación",
        "Tasa de política": "Tasa de política", "Desempleo": "Desempleo",
        "TCRM": "TCRM", "Resultado primario": "Resultado primario",
        "Resultado financiero": "Resultado financiero",
    }
    var_key = var_key_map.get(variable, variable)

    if variable in ["TCRM", "Resultado primario", "Resultado financiero"]:
        countries = FULL_COUNTRIES
    else:
        countries = ALL_COUNTRIES

    resumen = load_resumen()
    var_resumen = resumen[resumen["Variable"] == variable]

    if not var_resumen.empty:
        cols = st.columns(len(var_resumen))
        for i, (_, row) in enumerate(var_resumen.iterrows()):
            pais = row["País"]
            flag = COUNTRY_FLAGS.get(pais, "")
            crec = row["Crecimiento"]
            señal = row["Señal_t1"]
            sc = signal_color(señal, colors)
            with cols[i]:
                st.markdown(kpi_card(f"{flag} {COUNTRY_NAMES.get(pais, pais)}", fmt_pct(crec), colors, sc), unsafe_allow_html=True)

    # Cross-country chart
    chart_rows = []
    for code in countries:
        try:
            data = load_country(code)
            if var_key not in data:
                continue
            df = data[var_key]
            if "Crecimiento" not in df.columns:
                continue
            series = df["Crecimiento"].dropna()
            cutoff = series.index.max() - pd.DateOffset(years=5)
            series = series[series.index >= cutoff]
            for dt, val in series.items():
                chart_rows.append({"Fecha": dt, "País": COUNTRY_NAMES.get(code, code), "Crecimiento": val * 100})
        except Exception:
            continue

    if chart_rows:
        df_chart = pd.DataFrame(chart_rows)
        country_list = df_chart["País"].unique().tolist()
        color_palette = ["#3D85C6", "#E06666", "#F6B26B", "#8E7CC3", "#76A5AF", "#93C47D"]
        n = len(country_list)

        lines = (
            alt.Chart(df_chart)
            .mark_line(strokeWidth=1.8)
            .encode(
                x=alt.X("Fecha:T", title=None, axis=alt.Axis(format="%b %Y")),
                y=alt.Y("Crecimiento:Q", title="1ª Derivada (%)"),
                color=alt.Color("País:N",
                    scale=alt.Scale(domain=country_list, range=color_palette[:n]),
                    legend=alt.Legend(orient="top", title=None)),
                tooltip=["País:N", alt.Tooltip("Crecimiento:Q", format=".2f"), "Fecha:T"],
            )
        )
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[6,3], strokeWidth=1, color=colors["text_muted"]).encode(y="y:Q")
        st.altair_chart(style_chart(zero + lines, colors, 320), use_container_width=True)

    # Signal heatmap
    if not var_resumen.empty:
        st.markdown(f'<div class="section-header">Señales de Aceleración — {variable}</div>', unsafe_allow_html=True)
        html = '<table class="heatmap-table"><tr>'
        html += '<th style="text-align:left">País</th><th>t-1 (m/m)</th><th>t-3 (trim.)</th><th>t-6 (sem.)</th></tr>'
        for _, row in var_resumen.iterrows():
            pais = row["País"]
            flag = COUNTRY_FLAGS.get(pais, "")
            html += f'<tr><td>{flag} {COUNTRY_NAMES.get(pais, pais)}</td>'
            html += f'<td>{signal_badge(row["Señal_t1"])} {fmt_acel(row["Acel_t1"])}</td>'
            html += f'<td>{signal_badge(row["Señal_t3"])} {fmt_acel(row["Acel_t3"])}</td>'
            html += f'<td>{signal_badge(row["Señal_t6"])} {fmt_acel(row["Acel_t6"])}</td></tr>'
        html += '</table>'
        st.markdown(html, unsafe_allow_html=True)

    # Individual country level charts WITH ETF overlay
    with st.expander("Niveles Históricos por País (con ETF overlay)", expanded=False):
        nivel_col_map = {
            "Actividad": "Actividad", "Crédito": "Crédito", "Inflación": "Inflación",
            "Tasa de política": "TPM", "Desempleo": "Desempleo", "TCRM": "TCRM",
            "Resultado primario": "Res. Primario", "Resultado financiero": "Res. Financiero",
        }
        for code in countries:
            try:
                data = load_country(code)
                niveles = data.get("niveles")
                if niveles is None:
                    continue
                col = nivel_col_map.get(variable)
                if col is None or col not in niveles.columns:
                    continue
                series = niveles[col].dropna()
                if series.empty:
                    continue
                cutoff = series.index.max() - pd.DateOffset(years=5)
                series = series[series.index >= cutoff]
                if series.empty:
                    continue

                flag = COUNTRY_FLAGS.get(code, "")
                etf = COUNTRY_ETF.get(code, "")
                st.markdown(f'<div class="country-header">{flag} {COUNTRY_NAMES.get(code, code)} — {variable} vs {etf}</div>', unsafe_allow_html=True)

                df_plot = series.reset_index()
                df_plot.columns = ["Fecha", "Nivel"]
                main_line = (
                    alt.Chart(df_plot).mark_line(strokeWidth=1.5, color=VARIABLE_COLORS.get(variable, "#3D85C6"))
                    .encode(
                        x=alt.X("Fecha:T", title=None, axis=alt.Axis(format="%b %Y")),
                        y=alt.Y("Nivel:Q", title=variable, scale=alt.Scale(zero=False)),
                        tooltip=[alt.Tooltip("Nivel:Q", format=",.2f"), "Fecha:T"],
                    )
                )

                # ETF overlay
                d_min = series.index.min().strftime("%Y-%m-%d")
                d_max = series.index.max().strftime("%Y-%m-%d")
                etf_layer = build_etf_overlay(etf, colors, d_min, d_max) if etf else None

                if etf_layer is not None:
                    chart = alt.layer(main_line, etf_layer).resolve_scale(y="independent")
                else:
                    chart = main_line

                st.altair_chart(style_chart(chart, colors, 240), use_container_width=True)
            except Exception:
                continue


# ── Country Deep Dive (with ETF overlay) ──────────────────────────────────

def render_country_tab(colors, theme):
    selected = st.selectbox(
        "Seleccionar país",
        ALL_COUNTRIES,
        format_func=lambda c: f"{COUNTRY_FLAGS.get(c,'')} {COUNTRY_NAMES.get(c,c)}",
        key="country_deep_dive",
    )

    data = load_country(selected)
    is_full = selected in FULL_COUNTRIES
    variables = FULL_VARIABLES if is_full else SIMPLE_VARIABLES
    flag = COUNTRY_FLAGS.get(selected, "")
    name = COUNTRY_NAMES.get(selected, selected)
    etf = COUNTRY_ETF.get(selected, "")

    # ETF KPI
    etf_kpi_html = ""
    if etf:
        try:
            s = fetch_etf_daily(etf)
            if not s.empty:
                p = s.iloc[-1]
                d1 = ((s.iloc[-1] / s.iloc[-2]) - 1) * 100 if len(s) > 1 else 0
                m1 = ((s.iloc[-1] / s.iloc[-22]) - 1) * 100 if len(s) > 22 else 0
                y1 = ((s.iloc[-1] / s.iloc[-252]) - 1) * 100 if len(s) > 252 else 0
                etf_kpi_html = f" — {etf} ${p:.2f}"
        except Exception:
            pass

    st.markdown(f'<div class="section-header">{flag} {name} — Panel Completo{etf_kpi_html}</div>', unsafe_allow_html=True)

    # ETF KPI cards
    if etf:
        try:
            s = fetch_etf_daily(etf)
            if not s.empty and len(s) > 2:
                p = s.iloc[-1]
                d1 = ((s.iloc[-1] / s.iloc[-2]) - 1) * 100 if len(s) > 1 else 0
                m1 = ((s.iloc[-1] / s.iloc[max(-22, -len(s))]) - 1) * 100
                y1 = ((s.iloc[-1] / s.iloc[max(-252, -len(s))]) - 1) * 100
                etf_cols = st.columns(4)
                with etf_cols[0]:
                    st.markdown(kpi_card(f"{etf} Price", f"${p:.2f}", colors), unsafe_allow_html=True)
                with etf_cols[1]:
                    c = colors["green"] if d1 >= 0 else colors["red"]
                    st.markdown(kpi_card("Daily", f"{d1:+.1f}%", colors, c), unsafe_allow_html=True)
                with etf_cols[2]:
                    c = colors["green"] if m1 >= 0 else colors["red"]
                    st.markdown(kpi_card("1M", f"{m1:+.1f}%", colors, c), unsafe_allow_html=True)
                with etf_cols[3]:
                    c = colors["green"] if y1 >= 0 else colors["red"]
                    st.markdown(kpi_card("YTD", f"{y1:+.1f}%", colors, c), unsafe_allow_html=True)
        except Exception:
            pass

    # Macro KPI strip
    resumen = load_resumen()
    pais_data = resumen[resumen["País"] == selected]
    if not pais_data.empty:
        n_vars = len(pais_data)
        cols_per_row = min(n_vars, 4)
        for row_start in range(0, n_vars, cols_per_row):
            row_data = pais_data.iloc[row_start:row_start+cols_per_row]
            cols = st.columns(len(row_data))
            for i, (_, row) in enumerate(row_data.iterrows()):
                var = row["Variable"]
                crec = row["Crecimiento"]
                señal = row["Señal_t1"]
                sc = signal_color(señal, colors)
                with cols[i]:
                    st.markdown(kpi_card(var, fmt_pct(crec), colors, sc), unsafe_allow_html=True)

    # Charts per variable with ETF overlay
    for var in variables:
        if var not in data:
            continue
        df = data[var]
        if df.empty or "Crecimiento" not in df.columns:
            continue

        st.markdown(f'<div class="section-header">{var}</div>', unsafe_allow_html=True)

        series = df["Crecimiento"].dropna()
        if series.empty:
            continue
        cutoff = series.index.max() - pd.DateOffset(years=5)
        series = series[series.index >= cutoff]
        if series.empty:
            continue

        df_plot = (series * 100).reset_index()
        df_plot.columns = ["Fecha", "Crecimiento"]

        var_color = VARIABLE_COLORS.get(var, "#3D85C6")
        line = (
            alt.Chart(df_plot).mark_line(strokeWidth=1.8, color=var_color)
            .encode(
                x=alt.X("Fecha:T", title=None, axis=alt.Axis(format="%b %Y")),
                y=alt.Y("Crecimiento:Q", title="1ª Derivada (%)"),
                tooltip=[alt.Tooltip("Crecimiento:Q", format=".2f"), "Fecha:T"],
            )
        )
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[6,3], strokeWidth=1, color=colors["text_muted"]).encode(y="y:Q")

        # ETF overlay
        d_min = series.index.min().strftime("%Y-%m-%d")
        d_max = series.index.max().strftime("%Y-%m-%d")
        etf_layer = build_etf_overlay(etf, colors, d_min, d_max) if etf else None

        if etf_layer is not None:
            chart = alt.layer(zero, line, etf_layer).resolve_scale(y="independent")
        else:
            chart = zero + line

        st.altair_chart(style_chart(chart, colors, 260), use_container_width=True)

        # Signal strip
        if is_full and "Señal_t1" in df.columns:
            latest = df.dropna(subset=["Señal_t1"]).iloc[0] if not df.dropna(subset=["Señal_t1"]).empty else None
            if latest is not None:
                scols = st.columns(3)
                for j, h in enumerate(["t1", "t3", "t6"]):
                    sk, ak = f"Señal_{h}", f"Acel_{h}"
                    s_val = latest.get(sk, "")
                    a_val = latest.get(ak, np.nan)
                    with scols[j]:
                        labels = {"t1": "Mensual (t-1)", "t3": "Trimestral (t-3)", "t6": "Semestral (t-6)"}
                        st.markdown(
                            f'<div style="text-align:center;font-size:11px;color:{colors["text_muted"]}">{labels[h]}</div>'
                            f'<div style="text-align:center">{signal_badge(s_val)} <span style="font-family:IBM Plex Mono;font-size:12px">{fmt_acel(a_val)}</span></div>',
                            unsafe_allow_html=True
                        )


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="LatAm Macro Monitor",
        page_icon="🌎",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "theme" not in st.session_state:
        st.session_state.theme = "dark"
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.time()

    theme = st.session_state.theme
    colors = COLORS_DARK if theme == "dark" else COLORS_LIGHT
    st.markdown(get_theme_css(theme), unsafe_allow_html=True)

    # Auto-refresh logic
    refresh_minutes = st.session_state.get("refresh_interval", 15)
    if refresh_minutes > 0:
        elapsed = time.time() - st.session_state.last_refresh
        if elapsed >= refresh_minutes * 60:
            st.session_state.last_refresh = time.time()
            st.cache_data.clear()
            st.rerun()

    # Sidebar
    with st.sidebar:
        st.markdown(
            f'<div style="text-align:center;padding:16px 0 8px 0;">'
            f'<div style="font-size:24px;font-weight:700;color:{colors["text_header"]};font-family:IBM Plex Sans">🌎 LatAm Macro</div>'
            f'<div style="font-size:12px;color:{colors["text_muted"]}">Monitor Macroeconómico</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if st.button("🌓 Toggle Theme", use_container_width=True):
            st.session_state.theme = "light" if theme == "dark" else "dark"
            st.rerun()

        if st.button("🔄 Refresh Data", use_container_width=True):
            st.session_state.last_refresh = time.time()
            st.cache_data.clear()
            st.rerun()

        st.markdown("---")

        # Auto-refresh config
        refresh_options = {"Off": 0, "5 min": 5, "10 min": 10, "15 min": 15, "30 min": 30}
        current = st.session_state.get("refresh_interval", 15)
        current_label = next((k for k, v in refresh_options.items() if v == current), "15 min")
        selected_refresh = st.selectbox("Auto-refresh", list(refresh_options.keys()),
                                         index=list(refresh_options.keys()).index(current_label),
                                         key="refresh_select")
        st.session_state.refresh_interval = refresh_options[selected_refresh]

        # Countdown
        if st.session_state.get("refresh_interval", 0) > 0:
            elapsed = time.time() - st.session_state.last_refresh
            remaining = max(0, st.session_state.refresh_interval * 60 - elapsed)
            mins, secs = divmod(int(remaining), 60)
            st.caption(f"⏱ Next refresh: {mins}:{secs:02d}")

        st.markdown("---")
        st.caption("Fuente: Monitor.xlsx (Alphacast)")
        st.caption("ETF data: Yahoo Finance")

        # ETF quick strip
        st.markdown(f'<div style="font-size:11px;font-weight:600;color:{colors["text_header"]};margin:8px 0 4px 0">ETFs LatAm</div>', unsafe_allow_html=True)
        for code in FULL_COUNTRIES:
            etf = COUNTRY_ETF.get(code, "")
            flag = COUNTRY_FLAGS.get(code, "")
            if not etf:
                continue
            try:
                s = fetch_etf_daily(etf)
                if not s.empty and len(s) > 1:
                    p = s.iloc[-1]
                    chg = ((s.iloc[-1] / s.iloc[-2]) - 1) * 100
                    chg_color = colors["green"] if chg >= 0 else colors["red"]
                    spark = make_etf_sparkline(code, colors)
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:6px;font-family:IBM Plex Mono;font-size:11px;padding:2px 0">'
                        f'<span>{flag}</span><span style="color:{colors["text"]};font-weight:600">{etf}</span>'
                        f'<span style="color:{colors["text_muted"]}">${p:.2f}</span>'
                        f'<span style="color:{chg_color}">{chg:+.1f}%</span>'
                        f'{spark}</div>',
                        unsafe_allow_html=True
                    )
            except Exception:
                continue

    # Header
    st.markdown(
        f'<div style="text-align:center;padding:10px 0 4px 0;">'
        f'<div style="font-size:28px;font-weight:700;color:{colors["text_header"]};font-family:IBM Plex Sans">Macro Indicators — Latin America</div>'
        f'<div style="font-size:13px;color:{colors["text_muted"]}">Actividad, inflación, tasas, crédito, empleo, tipo de cambio real y resultado fiscal</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Tabs
    tab_labels = [
        "📊 Resumen", "🏭 Actividad", "💳 Crédito", "📈 Inflación",
        "🏦 Tasas", "👷 Desempleo", "💱 TCRM", "📋 Res. Primario",
        "📋 Res. Financiero", "🔍 País",
    ]
    tabs = st.tabs(tab_labels)

    with tabs[0]: render_resumen_tab(colors, theme)
    with tabs[1]: render_variable_tab("Actividad", colors, theme)
    with tabs[2]: render_variable_tab("Crédito", colors, theme)
    with tabs[3]: render_variable_tab("Inflación", colors, theme)
    with tabs[4]: render_variable_tab("Tasa de política", colors, theme)
    with tabs[5]: render_variable_tab("Desempleo", colors, theme)
    with tabs[6]: render_variable_tab("TCRM", colors, theme)
    with tabs[7]: render_variable_tab("Resultado primario", colors, theme)
    with tabs[8]: render_variable_tab("Resultado financiero", colors, theme)
    with tabs[9]: render_country_tab(colors, theme)


if __name__ == "__main__":
    main()
