"""Stock Market Monitoring Dashboard (Streamlit).

Run with:  streamlit run app.py

Calculations live in src/market_analysis.py; this file only handles the
user interface, charts and session state.
"""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src import market_analysis as ma

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
SAMPLE_PATH = ROOT / "data" / "sample_stock_prices.csv"
LOGO_SVG = ROOT / "assets" / "market_logo.svg"
LOGO_PNG = ROOT / "assets" / "market_logo.png"

C = {
    "bg": "#0B1220",
    "panel": "#111C2E",
    "border": "#1F2D4A",
    "accent": "#3B82F6",
    "pos": "#22C55E",
    "neg": "#EF4444",
    "text": "#E5E7EB",
    "muted": "#94A3B8",
    "warn": "#F59E0B",
}

SAMPLE = "Sample Data"
LIVE = "Live Historical Data"
COMMON_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA"]
OTHER = "Other (type below)"
MAX_COMPARE = 5

DISCLAIMER = (
    "Educational project only. This dashboard is not investment advice and "
    "technical indicators are not trading signals."
)


def _page_icon():
    try:
        from PIL import Image

        if LOGO_PNG.exists():
            return Image.open(LOGO_PNG)
    except Exception:  # noqa: BLE001 - the icon is optional
        pass
    return "📈"


st.set_page_config(
    page_title="Stock Market Monitoring Dashboard",
    page_icon=_page_icon(),
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
    .block-container {{ padding-top: 3.5rem; padding-bottom: 2rem; max-width: 1400px; }}
    .kpi {{ background:{C['panel']}; border:1px solid {C['border']}; border-radius:8px;
            padding:12px 14px; margin-bottom:0.9rem; }}
    .kpi .label {{ color:{C['muted']}; font-size:0.74rem; text-transform:uppercase;
                   letter-spacing:0.05em; }}
    .kpi .value {{ color:{C['text']}; font-size:1.45rem; font-weight:600; line-height:1.35; }}
    .kpi .sub {{ color:{C['muted']}; font-size:0.74rem; }}
    .kpi.pos .value {{ color:{C['pos']}; }}
    .kpi.neg .value {{ color:{C['neg']}; }}
    .badge {{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.74rem;
              font-weight:600; border:1px solid {C['border']}; margin-right:8px; }}
    .badge.sample {{ color:{C['warn']}; border-color:{C['warn']}; }}
    .badge.live {{ color:{C['accent']}; border-color:{C['accent']}; }}
    .meta {{ color:{C['muted']}; font-size:0.85rem; margin-top:4px; }}
    .meta b {{ color:{C['text']}; font-weight:600; }}
    .note {{ color:{C['muted']}; font-size:0.82rem; }}
    h2, h3, h4 {{ padding-top:0.4rem; }}
    div[data-testid="stAlert"] {{ margin-bottom:0.6rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------


def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v))


def fmt_price(v) -> str:
    return "n/a" if _missing(v) else f"{v:,.2f}"


def fmt_signed(v) -> str:
    return "n/a" if _missing(v) else f"{v:+,.2f}"


def fmt_pct(v, signed: bool = True) -> str:
    if _missing(v):
        return "n/a"
    return f"{v * 100:+.2f}%" if signed else f"{v * 100:.2f}%"


def fmt_volume(v) -> str:
    if _missing(v):
        return "n/a"
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= limit:
            return f"{v / limit:,.2f}{suffix}"
    return f"{v:,.0f}"


def tone(v) -> str:
    if _missing(v) or v == 0:
        return "neutral"
    return "pos" if v > 0 else "neg"


def arrow(v) -> str:
    if _missing(v) or v == 0:
        return ""
    return "▲ " if v > 0 else "▼ "


def kpi_card(label: str, value: str, sub: str = "", css: str = "neutral") -> str:
    return (
        f'<div class="kpi {css}"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value)}</div>'
        f'<div class="sub">{html.escape(sub)}&nbsp;</div></div>'
    )


def style_fig(fig, height: int = 420):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=C["panel"],
        plot_bgcolor=C["panel"],
        font=dict(color=C["text"]),
        height=height,
        margin=dict(l=10, r=10, t=70, b=10),
        title=dict(x=0, xanchor="left", y=0.97, yanchor="top", font=dict(size=15)),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"),
    )
    fig.update_xaxes(gridcolor=C["border"], zeroline=False)
    fig.update_yaxes(gridcolor=C["border"], zeroline=False)
    return fig


def color_signed(styler, columns):
    """Colour numbers green/red (pandas renamed applymap -> map in 2.1)."""
    def _c(v):
        if _missing(v) or not isinstance(v, (int, float, np.number)) or v == 0:
            return ""
        return f"color: {C['pos'] if v > 0 else C['neg']}"

    apply = getattr(styler, "map", None) or styler.applymap
    return apply(_c, subset=columns)


# --------------------------------------------------------------------------
# Data access (cached)
# --------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def cached_sample() -> pd.DataFrame:
    return ma.load_sample_data(SAMPLE_PATH)


@st.cache_data(ttl=900, show_spinner=False)
def cached_live(ticker: str, period: str) -> ma.DataResult:
    return ma.fetch_live_data(ticker, period)


def get_ticker_data(ticker, source, period, sample_df) -> ma.DataResult:
    """Return price history for one ticker from the chosen source."""
    symbol = ma.normalize_ticker(ticker)
    if symbol is None:
        return ma.DataResult(None, f"'{ticker}' is not a valid ticker symbol.")
    if source == SAMPLE:
        sub = sample_df[sample_df["Ticker"] == symbol]
        if sub.empty:
            return ma.DataResult(
                None,
                f"{symbol} is not in the synthetic sample dataset. Pick an available "
                "sample symbol or switch to Live Historical Data.",
            )
        return ma.DataResult(sub.reset_index(drop=True), None, str(sub["Company"].iloc[0]))
    return cached_live(symbol, period)


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------

try:
    sample_df = cached_sample()
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load {SAMPLE_PATH.name}: {exc}. Run `python scripts/generate_sample_data.py`.")
    st.stop()

sample_tickers = sorted(sample_df["Ticker"].unique())
company_map = sample_df.drop_duplicates("Ticker").set_index("Ticker")["Company"].to_dict()

if "watchlists" not in st.session_state:
    st.session_state["watchlists"] = {
        SAMPLE: sample_tickers[:4],
        LIVE: ["AAPL", "MSFT", "NVDA"],
    }
if "wl_msg" not in st.session_state:
    st.session_state["wl_msg"] = ""
if "holdings_init" not in st.session_state:
    first_rows = sample_df.sort_values("Date").groupby("Ticker").first()
    st.session_state["holdings_init"] = {
        SAMPLE: pd.DataFrame(
            {
                "Ticker": sample_tickers[:3],
                "Shares": [50.0, 30.0, 100.0],
                "Purchase_Price": [float(first_rows.loc[t, "Close"]) for t in sample_tickers[:3]],
            }
        ),
        LIVE: pd.DataFrame(
            {
                "Ticker": ["AAPL", "MSFT"],
                "Shares": [10.0, 5.0],
                "Purchase_Price": [150.0, 300.0],
            }
        ),
    }


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

sb = st.sidebar
sb.markdown("### Controls")
source = sb.radio(
    "Data source",
    [SAMPLE, LIVE],
    help="Sample Data is a synthetic offline dataset. Live Historical Data uses yfinance "
    "(internet required).",
)

period_label = None
if source == SAMPLE:
    ticker = sb.selectbox(
        "Ticker",
        sample_tickers,
        format_func=lambda t: f"{t} – {company_map.get(t, '')}",
    )
else:
    choice = sb.selectbox("Ticker", COMMON_TICKERS + [OTHER])
    if choice == OTHER:
        typed = sb.text_input("Ticker symbol", value="", placeholder="e.g. NFLX")
        ticker = ma.normalize_ticker(typed)
    else:
        ticker = choice
    period_label = sb.selectbox("Historical period", list(ma.PERIOD_OPTIONS), index=3)
    sb.button("Refresh live data", on_click=cached_live.clear)
period = ma.PERIOD_OPTIONS.get(period_label or "", "1y")

date_box = sb.container()
options_box = sb.container()
watchlist_box = sb.container()

# --------------------------------------------------------------------------
# Load the primary ticker (fall back to sample data if live data fails)
# --------------------------------------------------------------------------

active_source = source
fallback_message = None
if source == LIVE:
    if ticker is None:
        result = ma.DataResult(None, "Enter a valid ticker symbol.")
    else:
        with st.spinner(f"Loading {ticker} from Yahoo Finance..."):
            result = get_ticker_data(ticker, LIVE, period, sample_df)
    if result.data is None:
        fallback_message = result.error
        active_source = SAMPLE
        ticker = sample_tickers[0]
        result = get_ticker_data(ticker, SAMPLE, period, sample_df)
else:
    result = get_ticker_data(ticker, SAMPLE, period, sample_df)

full = result.data
if full is None or full.empty:  # should not happen for sample data
    st.error("No price data available.")
    st.stop()
full = full.sort_values("Date").reset_index(drop=True)

# Chart options
with options_box:
    st.markdown("#### Chart options")
    chart_type = st.radio("Price chart", ["Candlestick", "Line"], horizontal=True)
    ma_short = int(st.number_input("Short moving average (days)", 2, 100, 20))
    ma_long = int(st.number_input("Long moving average (days)", 5, 250, 50))
    show_ma_short = st.checkbox(f"Show {ma_short}-day MA", value=True)
    show_ma_long = st.checkbox(f"Show {ma_long}-day MA", value=True)
    show_volume = st.checkbox("Show volume panel", value=True)
if ma_short >= ma_long:
    sb.warning("Short moving average should be smaller than the long one.")

# Date filter
min_d, max_d = full["Date"].min().date(), full["Date"].max().date()
with date_box:
    picked = st.date_input(
        "Date range",
        value=(min_d, max_d),
        min_value=min_d,
        max_value=max_d,
        key=f"dates_{active_source}_{ticker}_{period_label}",
    )
if isinstance(picked, (tuple, list)) and len(picked) == 2:
    start_d, end_d = picked
else:
    start_d, end_d = min_d, max_d  # user is still choosing the end date

# Watchlist management (per data source so real and fictional tickers never mix)
watchlist = st.session_state["watchlists"][active_source]


def _add_symbol():
    raw = st.session_state.get("wl_input", "")
    symbol = ma.normalize_ticker(raw)
    lst = st.session_state["watchlists"][active_source]
    if symbol is None:
        st.session_state["wl_msg"] = f"'{raw}' is not a valid ticker symbol."
    elif symbol in lst:
        st.session_state["wl_msg"] = f"{symbol} is already in the watchlist."
    else:
        lst.append(symbol)
        st.session_state["wl_msg"] = f"Added {symbol}."
    st.session_state["wl_input"] = ""


def _remove_symbols():
    remove_key = f"wl_remove_{active_source}"
    chosen = st.session_state.get(remove_key, [])
    lst = st.session_state["watchlists"][active_source]
    st.session_state["watchlists"][active_source] = [s for s in lst if s not in chosen]
    st.session_state[remove_key] = []
    st.session_state["wl_msg"] = "Removed selected symbols." if chosen else ""


with watchlist_box:
    st.markdown("#### Watchlist")
    st.caption(f"Symbols for: {active_source}")
    st.text_input("Add symbol", key="wl_input", placeholder="e.g. NVLX or AAPL")
    st.button("Add to watchlist", on_click=_add_symbol)
    st.multiselect("Remove symbols", watchlist, key=f"wl_remove_{active_source}")
    st.button("Remove selected", on_click=_remove_symbols)
    if st.session_state["wl_msg"]:
        st.caption(st.session_state["wl_msg"])

# --------------------------------------------------------------------------
# Indicators for the selected window
# --------------------------------------------------------------------------

ind = ma.add_indicators(full, short=ma_short, long=ma_long)
view = ma.filter_dates(ind, start_d, end_d).reset_index(drop=True)
if view.empty:
    st.warning("No data in the selected date range. Widen the date filter.")
    st.stop()

kpi = ma.kpi_summary(view)
is_sample = active_source == SAMPLE
company = result.company or ticker

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

h_logo, h_text = st.columns([1, 9])
with h_logo:
    if LOGO_PNG.exists():
        st.image(str(LOGO_PNG), width=72)
with h_text:
    st.markdown("## Stock Market Monitoring Dashboard")
    st.markdown(
        '<span class="note">Market overview, price performance, and historical trend analysis.</span>',
        unsafe_allow_html=True,
    )
    badge = (
        '<span class="badge sample">SAMPLE DATA · SYNTHETIC</span>'
        if is_sample
        else '<span class="badge live">LIVE HISTORICAL DATA · YAHOO FINANCE VIA YFINANCE</span>'
    )
    if is_sample:
        freshness = "static synthetic file (no live updates)"
    elif result.fetched_at:
        freshness = f"retrieved {result.fetched_at}"
    else:
        freshness = "unknown"
    st.markdown(
        f'{badge}<div class="meta"><b>{html.escape(str(ticker))}</b> · {html.escape(company)} '
        f'&nbsp;|&nbsp; Range: <b>{start_d:%Y-%m-%d}</b> to <b>{end_d:%Y-%m-%d}</b> '
        f'&nbsp;|&nbsp; Latest data point: <b>{kpi["latest_date"]:%Y-%m-%d}</b> '
        f'&nbsp;|&nbsp; Data freshness: {html.escape(freshness)}'
        + (f" &nbsp;|&nbsp; Period: {html.escape(period_label)}" if not is_sample else "")
        + "</div>",
        unsafe_allow_html=True,
    )

if fallback_message:
    st.warning(
        f"Live data is unavailable: {fallback_message} "
        "Showing the included synthetic sample dataset instead."
    )
if is_sample:
    st.info(
        "These prices are **synthetic demonstration data** generated with a random seed. "
        "They are not real market quotes and every metric below is illustrative only."
    )
else:
    st.caption(
        "Historical daily data from Yahoo Finance via yfinance. It is not guaranteed to be "
        "real-time and may be delayed or incomplete."
    )

# --------------------------------------------------------------------------
# Market overview KPIs
# --------------------------------------------------------------------------

st.markdown("### Market overview")
as_of = f"as of {kpi['latest_date']:%Y-%m-%d}"
r1 = st.columns(4)
r2 = st.columns(4)
r1[0].markdown(kpi_card("Latest close", fmt_price(kpi["latest_close"]), as_of), unsafe_allow_html=True)
r1[1].markdown(
    kpi_card(
        "Daily change",
        arrow(kpi["daily_change"]) + fmt_signed(kpi["daily_change"]),
        "vs previous trading day" if kpi["daily_change"] is not None else "prior day unavailable",
        tone(kpi["daily_change"]),
    ),
    unsafe_allow_html=True,
)
r1[2].markdown(
    kpi_card(
        "Daily % change",
        arrow(kpi["daily_pct"]) + fmt_pct(kpi["daily_pct"]),
        "vs previous trading day" if kpi["daily_pct"] is not None else "prior day unavailable",
        tone(kpi["daily_pct"]),
    ),
    unsafe_allow_html=True,
)
r1[3].markdown(
    kpi_card(
        "Period return",
        arrow(kpi["period_return"]) + fmt_pct(kpi["period_return"]),
        f"{start_d:%Y-%m-%d} to {end_d:%Y-%m-%d}",
        tone(kpi["period_return"]),
    ),
    unsafe_allow_html=True,
)
r2[0].markdown(kpi_card("Period high", fmt_price(kpi["period_high"]), "highest High"), unsafe_allow_html=True)
r2[1].markdown(kpi_card("Period low", fmt_price(kpi["period_low"]), "lowest Low"), unsafe_allow_html=True)
r2[2].markdown(
    kpi_card("Average volume", fmt_volume(kpi["avg_volume"]), f"{kpi['observations']} trading days"),
    unsafe_allow_html=True,
)
latest_vol = float(view["Volume"].iloc[-1])
vol_ratio = latest_vol / kpi["avg_volume"] if kpi["avg_volume"] else None
r2[3].markdown(
    kpi_card(
        "Latest volume",
        fmt_volume(latest_vol),
        f"{vol_ratio:.1f}x the period average" if vol_ratio is not None else "average unavailable",
    ),
    unsafe_allow_html=True,
)
if is_sample:
    st.caption("Sample-data metrics are illustrative and do not describe a real company.")

st.write("")
tabs = st.tabs(
    ["Price & Volume", "Compare", "Returns", "Indicators", "Watchlist", "Portfolio", "About"]
)

# --------------------------------------------------------------------------
# Tab 1: price and volume
# --------------------------------------------------------------------------

with tabs[0]:
    rows = 2 if show_volume else 1
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28] if show_volume else [1.0],
    )
    if chart_type == "Candlestick":
        fig.add_trace(
            go.Candlestick(
                x=view["Date"],
                open=view["Open"],
                high=view["High"],
                low=view["Low"],
                close=view["Close"],
                name="OHLC",
                increasing_line_color=C["pos"],
                decreasing_line_color=C["neg"],
                increasing_fillcolor=C["pos"],
                decreasing_fillcolor=C["neg"],
            ),
            row=1,
            col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(x=view["Date"], y=view["Close"], mode="lines", name="Close",
                       line=dict(color=C["accent"], width=2)),
            row=1,
            col=1,
        )

    ma_notes = []
    for window, show, color in ((ma_short, show_ma_short, C["warn"]), (ma_long, show_ma_long, "#A78BFA")):
        column = f"MA_{window}"
        if not show:
            continue
        if view[column].notna().any():
            fig.add_trace(
                go.Scatter(x=view["Date"], y=view[column], mode="lines", name=f"{window}-day MA",
                           line=dict(color=color, width=1.5)),
                row=1,
                col=1,
            )
        else:
            ma_notes.append(f"{window}-day MA needs at least {window} observations.")

    if show_volume:
        up = view["Close"] >= view["Open"]
        colors = np.where(up, C["pos"], C["neg"]).tolist()
        fig.add_trace(
            go.Bar(x=view["Date"], y=view["Volume"], marker_color=colors, name="Volume"),
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text="Volume", tickformat=".2s", row=2, col=1)

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], rangeslider_visible=False)
    fig.update_xaxes(
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(step="all", label="All"),
            ],
            bgcolor=C["bg"],
            font=dict(color=C["text"]),
            x=0,
            y=1.02,
            yanchor="bottom",
        ),
        row=1,
        col=1,
    )
    style_fig(fig, height=620 if show_volume else 480)
    fig.update_layout(title=None, margin=dict(l=10, r=10, t=50, b=10))
    st.markdown(f"#### {ticker} price history" + (" (synthetic)" if is_sample else ""))
    st.plotly_chart(fig, use_container_width=True)
    for note in ma_notes:
        st.caption(note)
    st.caption(
        "Moving averages smooth past prices. They describe history and are not predictions. "
        "Volume bars are green when Close >= Open and red otherwise. Prices shown are unadjusted closes."
    )

# --------------------------------------------------------------------------
# Tab 2: comparison
# --------------------------------------------------------------------------

with tabs[1]:
    st.markdown(f"#### Compare up to {MAX_COMPARE} stocks")
    if active_source == SAMPLE:
        picks = st.multiselect(
            "Tickers to compare",
            sample_tickers,
            default=sample_tickers[:3],
            max_selections=MAX_COMPARE,
            key="cmp_sample",
        )
    else:
        raw = st.text_input("Comma-separated tickers", value="AAPL, MSFT, NVDA", key="cmp_live")
        picks = []
        for piece in raw.split(","):
            sym = ma.normalize_ticker(piece)
            if sym and sym not in picks:
                picks.append(sym)
        if len(picks) > MAX_COMPARE:
            st.warning(f"Only the first {MAX_COMPARE} tickers are used.")
            picks = picks[:MAX_COMPARE]

    closes = {}
    for sym in picks:
        res = get_ticker_data(sym, active_source, period, sample_df)
        if res.data is None:
            st.warning(f"{sym}: {res.error}")
            continue
        sub = ma.filter_dates(res.data, start_d, end_d)
        if sub.empty:
            st.warning(f"{sym}: no data in the selected date range.")
            continue
        closes[sym] = sub.set_index("Date")["Close"]

    if len(closes) < 2:
        st.info("Select at least two tickers with data to compare performance.")
    else:
        wide = pd.DataFrame(closes)
        norm = ma.normalize_to_base(wide)
        if norm.empty:
            st.warning("The selected tickers share no common trading dates in this range.")
        else:
            fig = go.Figure()
            for sym in norm.columns:
                fig.add_trace(go.Scatter(x=norm.index, y=norm[sym], mode="lines", name=sym))
            fig.add_hline(y=100, line_dash="dot", line_color=C["muted"])
            style_fig(fig, height=460)
            fig.update_layout(title="Normalized performance (start = 100)", yaxis_title="Index (start = 100)")
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
            st.plotly_chart(fig, use_container_width=True)

            summary = pd.DataFrame(
                {
                    "Ticker": norm.columns,
                    "Start date": norm.index[0].strftime("%Y-%m-%d"),
                    "End value": [norm[s].iloc[-1] for s in norm.columns],
                    "Performance %": [norm[s].iloc[-1] - 100 for s in norm.columns],
                }
            ).sort_values("Performance %", ascending=False)
            styled = color_signed(
                summary.style.format({"End value": "{:,.1f}", "Performance %": "{:+.2f}%"}),
                ["Performance %"],
            )
            st.dataframe(styled, hide_index=True, use_container_width=True)
            st.caption(
                "Method: each closing-price series is divided by its first close on the first date "
                "that all selected tickers have data, then multiplied by 100. A value of 120 means "
                "+20% since that date, so stocks with very different share prices can be compared."
            )

# --------------------------------------------------------------------------
# Tab 3: returns
# --------------------------------------------------------------------------

with tabs[2]:
    returns = view["Return"].iloc[1:]  # first row's return uses a price outside the window
    dates_r = view["Date"].iloc[1:]
    if len(returns) < 2:
        st.info("Not enough observations in this range to analyze returns.")
    else:
        if len(returns) < 30:
            st.warning(
                f"Only {len(returns)} daily returns in this range. Distribution statistics from "
                "such a short sample can be misleading."
            )
        gaps = int((view["Date"].diff().dt.days > 5).sum())
        if gaps:
            st.warning(f"{gaps} gap(s) longer than 5 calendar days were found; returns across gaps span several days.")

        left, right = st.columns(2)
        with left:
            fig = px.histogram(x=returns * 100, nbins=40, labels={"x": "Daily return (%)"})
            fig.update_traces(marker_color=C["accent"])
            style_fig(fig, height=380)
            fig.update_layout(title="Distribution of daily returns", yaxis_title="Days", hovermode="closest")
            st.plotly_chart(fig, use_container_width=True)
        with right:
            cum = ma.cumulative_returns(view["Close"]) * 100
            fig = go.Figure(
                go.Scatter(x=view["Date"], y=cum, mode="lines", name="Cumulative return",
                           line=dict(color=C["accent"], width=2), fill="tozeroy",
                           fillcolor="rgba(59,130,246,0.15)")
            )
            style_fig(fig, height=380)
            fig.update_layout(title="Cumulative return since start of range", yaxis_title="Return (%)")
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
            st.plotly_chart(fig, use_container_width=True)

        best_i, worst_i = returns.idxmax(), returns.idxmin()
        stats = pd.DataFrame(
            {
                "Statistic": [
                    "Daily returns", "Mean daily return", "Standard deviation",
                    "Best day", "Worst day", "Days with a gain",
                ],
                "Value": [
                    f"{len(returns)}",
                    fmt_pct(returns.mean()),
                    fmt_pct(returns.std(ddof=1), signed=False),
                    f"{fmt_pct(returns[best_i])} on {view.loc[best_i, 'Date']:%Y-%m-%d}",
                    f"{fmt_pct(returns[worst_i])} on {view.loc[worst_i, 'Date']:%Y-%m-%d}",
                    f"{(returns > 0).mean() * 100:.1f}%",
                ],
            }
        )
        st.dataframe(stats, hide_index=True, use_container_width=True)
        st.caption("Daily return = Close(t) / Close(t-1) - 1, calculated from closing prices.")

# --------------------------------------------------------------------------
# Tab 4: indicators
# --------------------------------------------------------------------------

with tabs[3]:
    latest = view.iloc[-1]
    dd_window = ma.drawdown_series(view.set_index("Date")["Close"])
    mdd, peak_d, trough_d = ma.max_drawdown(view.set_index("Date")["Close"])

    rsi_val = latest["RSI"]
    vol_val = latest["Volatility"]
    if _missing(rsi_val) or pd.isna(rsi_val):
        rsi_text, rsi_sub = "n/a", "needs 15+ observations"
    else:
        rsi_text = f"{rsi_val:.1f}"
        rsi_sub = (
            "above the common 70 reference level" if rsi_val > 70
            else "below the common 30 reference level" if rsi_val < 30
            else "between the 30 and 70 reference levels"
        )
    k = st.columns(3)
    k[0].markdown(kpi_card("RSI (14-day)", rsi_text, rsi_sub), unsafe_allow_html=True)
    k[1].markdown(
        kpi_card(
            "20-day volatility (annualized)",
            "n/a" if pd.isna(vol_val) else fmt_pct(vol_val, signed=False),
            "needs 21+ observations" if pd.isna(vol_val) else "std. dev. of daily returns x sqrt(252)",
        ),
        unsafe_allow_html=True,
    )
    k[2].markdown(
        kpi_card(
            "Maximum drawdown",
            "n/a" if pd.isna(mdd) else fmt_pct(mdd, signed=False),
            "" if pd.isna(mdd) else f"peak {peak_d:%Y-%m-%d} → trough {trough_d:%Y-%m-%d}",
            "neg" if (not pd.isna(mdd) and mdd < 0) else "neutral",
        ),
        unsafe_allow_html=True,
    )

    if view["RSI"].notna().any():
        fig = go.Figure(go.Scatter(x=view["Date"], y=view["RSI"], mode="lines", name="RSI",
                                   line=dict(color=C["accent"], width=2)))
        fig.add_hline(y=70, line_dash="dot", line_color=C["neg"], annotation_text="70")
        fig.add_hline(y=30, line_dash="dot", line_color=C["pos"], annotation_text="30")
        style_fig(fig, height=300)
        fig.update_layout(title="Relative Strength Index (14-day)", yaxis=dict(range=[0, 100]))
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough history in this range to show RSI (needs at least 15 observations).")

    c1, c2 = st.columns(2)
    with c1:
        if view["Volatility"].notna().any():
            fig = go.Figure(go.Scatter(x=view["Date"], y=view["Volatility"] * 100, mode="lines",
                                       line=dict(color=C["warn"], width=2), name="Volatility"))
            style_fig(fig, height=300)
            fig.update_layout(title="20-day historical volatility (annualized, %)")
            fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough history in this range to show volatility (needs 21+ observations).")
    with c2:
        fig = go.Figure(go.Scatter(x=dd_window.index, y=dd_window * 100, mode="lines", fill="tozeroy",
                                   line=dict(color=C["neg"], width=1.5), fillcolor="rgba(239,68,68,0.2)",
                                   name="Drawdown"))
        style_fig(fig, height=300)
        fig.update_layout(title="Drawdown from running peak (%)")
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("How these indicators are calculated"):
        st.markdown(
            f"""
- **Moving average ({ma_short}/{ma_long}-day):** average of the last N closing prices. Shown only once N observations exist.
- **RSI (14-day):** Wilder's Relative Strength Index. `RSI = 100 - 100 / (1 + average gain / average loss)` over 14 days,
  smoothed with Wilder's method. If there were no losses, RSI is 100 (or 50 if the price did not move).
  The 30 and 70 levels are conventions only, **not buy or sell signals**.
- **Historical volatility (20-day):** sample standard deviation of the last 20 daily returns, multiplied by √252 to annualize.
- **Maximum drawdown:** the largest percentage fall from a previous peak close to a later trough close within the selected range.
            """
        )
    st.caption(DISCLAIMER)

# --------------------------------------------------------------------------
# Tab 5: watchlist
# --------------------------------------------------------------------------

with tabs[4]:
    st.markdown(f"#### Watchlist ({active_source})")
    current_list = st.session_state["watchlists"][active_source]
    if not current_list:
        st.info("Your watchlist is empty. Add a symbol from the sidebar.")
    else:
        wl_rows = []
        with st.spinner("Loading watchlist..."):
            for sym in current_list:
                res = get_ticker_data(sym, active_source, period, sample_df)
                row = {"Ticker": sym, "Company": "", "Close": np.nan, "Daily change": np.nan,
                       "Daily %": np.nan, "Period return %": np.nan, "As of": "", "Status": ""}
                if res.data is None:
                    row["Status"] = res.error
                else:
                    sub = ma.filter_dates(res.data, start_d, end_d)
                    summary = ma.kpi_summary(sub)
                    row["Company"] = res.company
                    if summary is None:
                        row["Status"] = "No data in the selected date range"
                    else:
                        row["Close"] = summary["latest_close"]
                        row["Daily change"] = summary["daily_change"] if summary["daily_change"] is not None else np.nan
                        row["Daily %"] = summary["daily_pct"] * 100 if summary["daily_pct"] is not None else np.nan
                        row["Period return %"] = (
                            summary["period_return"] * 100 if summary["period_return"] is not None else np.nan
                        )
                        row["As of"] = f"{summary['latest_date']:%Y-%m-%d}"
                        row["Status"] = "OK" if summary["daily_change"] is not None else "OK (prior day unavailable)"
                wl_rows.append(row)
        wl = pd.DataFrame(wl_rows)
        styled = color_signed(
            wl.style.format(
                {"Close": "{:,.2f}", "Daily change": "{:+,.2f}", "Daily %": "{:+.2f}%",
                 "Period return %": "{:+.2f}%"},
                na_rep="n/a",
            ),
            ["Daily change", "Daily %", "Period return %"],
        )
        st.dataframe(styled, hide_index=True, use_container_width=True)
        if is_sample:
            st.caption(
                "Sample mode only contains fictional symbols. A real ticker such as AAPL is marked "
                "unavailable instead of showing invented prices; switch to Live Historical Data to use it."
            )
        st.caption("Period return uses the date range selected in the sidebar.")

# --------------------------------------------------------------------------
# Tab 6: portfolio
# --------------------------------------------------------------------------

with tabs[5]:
    st.markdown("#### Demonstration portfolio")
    st.caption(
        "Enter hypothetical holdings. Values use the latest available close from the selected data source. "
        "Nothing here is saved outside this browser session."
    )
    edited = st.data_editor(
        st.session_state["holdings_init"][active_source],
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        key=f"holdings_{active_source}",
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker"),
            "Shares": st.column_config.NumberColumn("Shares", min_value=0.0, format="%.2f"),
            "Purchase_Price": st.column_config.NumberColumn(
                "Purchase price per share", min_value=0.0, format="%.2f"
            ),
        },
    )

    latest_prices = {}
    messages = []
    for raw_sym in pd.Series(edited["Ticker"]).dropna().unique():
        sym = ma.normalize_ticker(raw_sym)
        if sym is None or sym in latest_prices:
            continue
        res = get_ticker_data(sym, active_source, period, sample_df)
        if res.data is None:
            latest_prices[sym] = None
            messages.append(f"{sym}: {res.error}")
        else:
            latest_prices[sym] = float(res.data.sort_values("Date")["Close"].iloc[-1])

    table, totals = ma.portfolio_summary(edited, latest_prices)
    if table.empty:
        st.info("Add at least one holding (ticker, shares and purchase price).")
    else:
        pk = st.columns(4)
        pk[0].markdown(kpi_card("Total purchase cost", fmt_price(totals["total_cost"]), "all valid holdings"),
                       unsafe_allow_html=True)
        pk[1].markdown(kpi_card("Market value", fmt_price(totals["market_value"]) if totals["unrealized_pl"] is not None else "n/a",
                                "holdings with prices"), unsafe_allow_html=True)
        pk[2].markdown(kpi_card("Unrealized P/L",
                                arrow(totals["unrealized_pl"]) + fmt_signed(totals["unrealized_pl"]),
                                "excludes unpriced holdings", tone(totals["unrealized_pl"])), unsafe_allow_html=True)
        pk[3].markdown(kpi_card("Unrealized return",
                                arrow(totals["unrealized_return"]) + fmt_pct(totals["unrealized_return"]),
                                "on priced holdings", tone(totals["unrealized_return"])), unsafe_allow_html=True)
        if totals["unpriced_count"]:
            st.warning(f"{totals['unpriced_count']} holding(s) have no price data and are excluded from market value.")
        if totals["invalid_count"]:
            st.warning(f"{totals['invalid_count']} row(s) have invalid input (ticker, shares > 0 and price > 0 are required).")
        for msg in messages:
            st.caption(msg)

        show = table.copy()
        show["Unrealized_Return"] = show["Unrealized_Return"] * 100
        show["Allocation"] = show["Allocation"] * 100
        styled = color_signed(
            show.style.format(
                {"Shares": "{:,.2f}", "Purchase_Price": "{:,.2f}", "Cost": "{:,.2f}",
                 "Latest_Close": "{:,.2f}", "Market_Value": "{:,.2f}", "Unrealized_PL": "{:+,.2f}",
                 "Unrealized_Return": "{:+.2f}%", "Allocation": "{:.1f}%"},
                na_rep="unavailable",
            ),
            ["Unrealized_PL", "Unrealized_Return"],
        )
        st.dataframe(styled, hide_index=True, use_container_width=True)

        alloc = table[table["Status"] == "OK"]
        if not alloc.empty:
            fig = px.pie(alloc, names="Ticker", values="Market_Value", hole=0.55)
            style_fig(fig, height=360)
            fig.update_layout(title="Allocation by market value", hovermode="closest")
            st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Demonstration only. Calculations exclude brokerage fees, taxes, dividends and other transaction costs."
    )

# --------------------------------------------------------------------------
# Tab 7: about
# --------------------------------------------------------------------------

with tabs[6]:
    st.markdown(
        """
#### About this dashboard
- **Sample Data** is a synthetic dataset (fictional companies, random-walk prices, fixed seed). It is for demonstration and offline use only and never represents real market quotes.
- **Live Historical Data** comes from Yahoo Finance through the `yfinance` library. Availability, delays and licensing can vary, and use of Yahoo Finance data is subject to Yahoo's applicable terms. Data is historical (daily bars), not a real-time feed.
- If live data cannot be loaded (no internet, rate limit, unknown symbol), the dashboard falls back to the sample dataset.
- Watchlist and portfolio entries are kept only in the current browser session.
        """
    )
    st.warning(DISCLAIMER)

st.markdown("---")
st.caption(DISCLAIMER)
