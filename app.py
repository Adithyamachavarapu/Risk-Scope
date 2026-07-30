"""
RiskScope  —  A Bloomberg-style Market-Risk Terminal
=====================================================
Single-file Streamlit application for market-risk analysts & risk quants.

Pages
-----
1. TERMINAL         : single-asset deep risk analysis (VaR / ES / Greeks / Monte Carlo / stress tests)
2. PORTFOLIO        : build a basket, weight it, and run portfolio risk + Monte Carlo
3. OPTIONS LAB      : Black-Scholes pricer, full Greeks, and payoff diagrams
4. ABOUT            : methodology & glossary

Author : Adithya Machavarapu
Stack  : Streamlit · yfinance · NumPy · pandas · SciPy · Plotly
Run    : streamlit run app.py
"""

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import norm

# ----------------------------------------------------------------------------- #
#  PAGE CONFIG                                                                   #
# ----------------------------------------------------------------------------- #
st.set_page_config(
    page_title="RiskScope Terminal",
    page_icon="📟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- palette ---------------------------------------------------------------- #
AMBER = "#ff9e00"
GREEN = "#26d07c"
RED = "#ff4d4d"
BG = "#05070a"
PANEL = "#0b0f14"
GRID = "#16202b"
MUTED = "#7a8a99"
TEXT = "#e6ebf0"

TRADING_DAYS = 252

# ----------------------------------------------------------------------------- #
#  GLOBAL CSS  —  the "terminal" look                                           #
# ----------------------------------------------------------------------------- #
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'JetBrains Mono', 'SF Mono', Consolas, 'Roboto Mono', monospace;
    }}
    .stApp {{ background: {BG}; }}
    #MainMenu, footer, header {{ visibility: hidden; }}
    .block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px; }}

    /* ---- brand + section headers ---- */
    .rs-brand {{
        font-size: 26px; font-weight: 700; letter-spacing: 2px; color: {AMBER};
        border-bottom: 2px solid {AMBER}; padding-bottom: 4px; margin-bottom: 2px;
    }}
    .rs-brand span {{ color: {TEXT}; }}
    .rs-tag {{ color: {MUTED}; font-size: 11px; letter-spacing: 3px; text-transform: uppercase; }}
    .rs-section {{
        color: {AMBER}; font-size: 13px; font-weight: 700; letter-spacing: 2px;
        text-transform: uppercase; margin: 18px 0 8px; border-left: 3px solid {AMBER};
        padding-left: 8px;
    }}

    /* ---- ticker ribbon ---- */
    .tbar {{
        display: flex; align-items: baseline; gap: 16px; flex-wrap: wrap;
        background: linear-gradient(90deg, #0e1620 0%, {PANEL} 100%);
        border: 1px solid {GRID}; border-left: 4px solid {AMBER};
        border-radius: 5px; padding: 12px 18px; margin: 6px 0 4px;
    }}
    .tsym  {{ font-size: 30px; font-weight: 700; color: {AMBER}; letter-spacing: 1px; }}
    .tname {{ font-size: 14px; color: {MUTED}; }}
    .tprice {{ font-size: 26px; font-weight: 700; color: {TEXT}; }}
    .tchg  {{ font-size: 15px; font-weight: 600; }}
    .tmeta {{ font-size: 11px; color: {MUTED}; margin-left: auto; text-align: right; }}

    /* ---- metric grid ---- */
    .mgrid {{ display: grid; gap: 8px; margin: 6px 0 12px; }}
    .mcard {{
        background: {PANEL}; border: 1px solid {GRID}; border-left: 3px solid {AMBER};
        border-radius: 4px; padding: 9px 12px; transition: border-color .15s;
    }}
    .mcard:hover {{ border-left-color: {TEXT}; }}
    .mlabel {{ font-size: 9.5px; letter-spacing: 1px; color: {MUTED}; text-transform: uppercase; }}
    .mvalue {{ font-size: 19px; font-weight: 600; color: {TEXT}; margin-top: 3px; }}
    .msub   {{ font-size: 10px; color: {MUTED}; margin-top: 1px; }}
    .pos {{ color: {GREEN}; }}
    .neg {{ color: {RED}; }}
    .amber {{ color: {AMBER}; }}

    /* ---- tabs / radio ---- */
    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
    .stTabs [data-baseweb="tab"] {{
        background: {PANEL}; border: 1px solid {GRID}; border-radius: 4px 4px 0 0;
        color: {MUTED}; font-size: 12px; letter-spacing: 1px;
    }}
    .stTabs [aria-selected="true"] {{ color: {AMBER}; border-bottom: 2px solid {AMBER}; }}

    /* ---- buttons ---- */
    .stButton > button {{
        background: {PANEL}; color: {AMBER}; border: 1px solid {AMBER};
        border-radius: 4px; font-weight: 600; letter-spacing: 1px;
    }}
    .stButton > button:hover {{ background: {AMBER}; color: {BG}; }}

    /* dataframes */
    .stDataFrame {{ border: 1px solid {GRID}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------- #
#  CONSTANTS                                                                     #
# ----------------------------------------------------------------------------- #
PERIODS = {"1M": 30, "6M": 182, "1Y": 365, "3Y": 365 * 3, "5Y": 365 * 5}

CRASH_SCENARIOS = {
    "COVID-19 Crash": ("2020-02-19", "2020-03-23"),
    "Global Financial Crisis (2008)": ("2007-10-09", "2009-03-09"),
    "2022 Rate-Shock Bear Market": ("2022-01-03", "2022-10-12"),
}

PLOT_LAYOUT = dict(
    paper_bgcolor=BG,
    plot_bgcolor=PANEL,
    font=dict(color=TEXT, family="JetBrains Mono, monospace", size=12),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)


# ----------------------------------------------------------------------------- #
#  DATA LAYER  (cached)                                                          #
# ----------------------------------------------------------------------------- #
@st.cache_data(ttl=1800, show_spinner=False)
def load_history(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Download OHLCV history; return a tz-naive DataFrame (empty on failure)."""
    try:
        df = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def load_close(ticker: str, start: str, end: str) -> pd.Series:
    """Adjusted close for an explicit window (used for stress tests)."""
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    except Exception:
        return pd.Series(dtype=float)
    if df is None or df.empty:
        return pd.Series(dtype=float)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    if getattr(close.index, "tz", None) is not None:
        close.index = close.index.tz_localize(None)
    return close.dropna()


@st.cache_data(ttl=1800, show_spinner=False)
def get_name(ticker: str) -> str:
    """Best-effort long name; falls back to the symbol."""
    try:
        info = yf.Ticker(ticker).info
        return info.get("longName") or info.get("shortName") or ticker.upper()
    except Exception:
        return ticker.upper()


# ----------------------------------------------------------------------------- #
#  RISK ENGINE                                                                   #
# ----------------------------------------------------------------------------- #
def daily_returns(close: pd.Series) -> pd.Series:
    return close.pct_change().dropna()


def hist_var(returns: pd.Series, conf: float) -> float:
    """Historical 1-day VaR as a positive loss fraction."""
    if returns.empty:
        return np.nan
    return -np.percentile(returns, (1 - conf) * 100)


def param_var(returns: pd.Series, conf: float) -> float:
    """Parametric (variance-covariance) 1-day VaR, positive loss fraction."""
    if returns.empty:
        return np.nan
    mu, sig = returns.mean(), returns.std()
    return -(mu + norm.ppf(1 - conf) * sig)


def expected_shortfall(returns: pd.Series, conf: float) -> float:
    """Conditional VaR (average loss beyond VaR), positive loss fraction."""
    if returns.empty:
        return np.nan
    cutoff = np.percentile(returns, (1 - conf) * 100)
    tail = returns[returns <= cutoff]
    return -tail.mean() if len(tail) else np.nan


def max_drawdown(close: pd.Series) -> float:
    if close.empty:
        return np.nan
    cummax = close.cummax()
    return (close / cummax - 1).min()


def beta_alpha(asset_ret: pd.Series, bench_ret: pd.Series, rf: float):
    """CAPM beta and annualised Jensen's alpha via OLS on aligned returns."""
    joined = pd.concat([asset_ret, bench_ret], axis=1, join="inner").dropna()
    if len(joined) < 20:
        return np.nan, np.nan
    a, b = joined.iloc[:, 0], joined.iloc[:, 1]
    slope, intercept, *_ = stats.linregress(b, a)
    alpha_ann = (intercept * TRADING_DAYS) - rf * (1 - slope)  # Jensen's alpha (annualised)
    return slope, alpha_ann


def full_metrics(close: pd.Series, bench_ret: pd.Series, rf: float) -> dict:
    r = daily_returns(close)
    if r.empty:
        return {}
    mu_d, sig_d = r.mean(), r.std()
    ann_ret = mu_d * TRADING_DAYS
    ann_vol = sig_d * np.sqrt(TRADING_DAYS)
    downside = r[r < 0].std() * np.sqrt(TRADING_DAYS)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol else np.nan
    sortino = (ann_ret - rf) / downside if downside else np.nan
    mdd = max_drawdown(close)
    calmar = ann_ret / abs(mdd) if mdd else np.nan
    beta, alpha = beta_alpha(r, bench_ret, rf)
    return {
        "daily_ret": mu_d,
        "ann_ret": ann_ret,
        "daily_vol": sig_d,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "mdd": mdd,
        "beta": beta,
        "alpha": alpha,
        "skew": stats.skew(r),
        "kurt": stats.kurtosis(r),  # excess kurtosis
        "var90_h": hist_var(r, 0.90),
        "var95_h": hist_var(r, 0.95),
        "var99_h": hist_var(r, 0.99),
        "var95_p": param_var(r, 0.95),
        "var99_p": param_var(r, 0.99),
        "es95": expected_shortfall(r, 0.95),
        "es99": expected_shortfall(r, 0.99),
        "returns": r,
    }


def monte_carlo(close: pd.Series, horizon: int, n_paths: int, seed: int = 42):
    """Vectorised GBM simulation. Returns (paths[n_paths, horizon+1], terminal_returns)."""
    r = daily_returns(close)
    mu, sig = r.mean(), r.std()
    s0 = float(close.iloc[-1])
    rng = np.random.default_rng(seed)
    shocks = rng.normal(mu, sig, size=(n_paths, horizon))
    growth = np.cumprod(1 + shocks, axis=1)
    paths = np.hstack([np.ones((n_paths, 1)), growth]) * s0
    terminal_ret = paths[:, -1] / s0 - 1
    return paths, terminal_ret


def black_scholes(S, K, T, r, sigma, kind="call") -> dict:
    """Black-Scholes-Merton price + Greeks (theta/day, vega & rho per 1% move)."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return dict.fromkeys(["price", "delta", "gamma", "vega", "theta", "rho", "d1", "d2"], 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    nd1 = norm.pdf(d1)
    gamma = nd1 / (S * sigma * np.sqrt(T))
    vega = S * nd1 * np.sqrt(T) / 100
    if kind == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (-S * nd1 * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        theta = (-S * nd1 * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    return dict(price=price, delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho, d1=d1, d2=d2)


# ----------------------------------------------------------------------------- #
#  UI HELPERS                                                                    #
# ----------------------------------------------------------------------------- #
def pct(x, dp=2):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:.{dp}f}%"


def num(x, dp=2):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{dp}f}"


def money(x, dp=0):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"${x:,.{dp}f}"


def sign_cls(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return "pos" if x >= 0 else "neg"


def metric_grid(items, cols=4):
    """items = list of (label, value_str, css_class, subtext)."""
    cells = ""
    for label, value, cls, sub in items:
        sub_html = f'<div class="msub">{sub}</div>' if sub else ""
        cells += (
            f'<div class="mcard"><div class="mlabel">{label}</div>'
            f'<div class="mvalue {cls}">{value}</div>{sub_html}</div>'
        )
    st.markdown(
        f'<div class="mgrid" style="grid-template-columns:repeat({cols},1fr);">{cells}</div>',
        unsafe_allow_html=True,
    )


def style_fig(fig, height=None, title=None):
    fig.update_layout(**PLOT_LAYOUT)
    if height:
        fig.update_layout(height=height)
    if title:
        fig.update_layout(title=dict(text=title, font=dict(color=AMBER, size=14)))
    return fig


# ----------------------------------------------------------------------------- #
#  SIDEBAR  (global controls)                                                    #
# ----------------------------------------------------------------------------- #
st.sidebar.markdown('<div class="rs-brand">RISK<span>SCOPE</span></div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="rs-tag">market-risk terminal</div>', unsafe_allow_html=True)
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "TERMINAL FUNCTION",
    ["📈  Terminal", "💼  Portfolio", "🧮  Options Lab", "ℹ️  About"],
    label_visibility="collapsed",
)

st.sidebar.markdown('<div class="rs-section">Global Parameters</div>', unsafe_allow_html=True)
rf_rate = st.sidebar.number_input("Risk-free rate (annual %)", 0.0, 20.0, 5.0, 0.25) / 100
benchmark = st.sidebar.text_input("Benchmark ticker", "^GSPC")
notional = st.sidebar.number_input("Position notional ($)", 1000, 100_000_000, 100_000, 1000)
mc_paths = st.sidebar.select_slider("Monte Carlo paths", [1000, 5000, 10000, 25000, 50000], 10000)
mc_horizon = st.sidebar.slider("MC / VaR horizon (trading days)", 5, 252, 21)
st.sidebar.markdown("---")
st.sidebar.caption("Data: Yahoo Finance (delayed). Educational use only — not investment advice.")

# shared benchmark returns (used across pages)
_bench_hist = load_history(benchmark, "5y")
bench_ret = daily_returns(_bench_hist["Close"]) if not _bench_hist.empty else pd.Series(dtype=float)


# ============================================================================= #
#  PAGE 1 — TERMINAL                                                            #
# ============================================================================= #
def page_terminal():
    st.markdown(
        '<div class="rs-brand">RISK<span>SCOPE</span> <span style="font-size:14px;">// TERMINAL</span></div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([4, 1])
    ticker = c1.text_input("Search asset  (stock · ETF · commodity · FX · crypto)",
                           "AAPL", label_visibility="collapsed",
                           placeholder="Search ticker e.g. AAPL, TSLA, GC=F, BTC-USD, ^NSEI").strip().upper()
    period_sel = c2.selectbox("History", list(PERIODS.keys()), index=2, label_visibility="collapsed")

    if not ticker:
        st.info("Enter a ticker to begin. Examples: AAPL · MSFT · GC=F (gold) · CL=F (oil) · BTC-USD · ^NSEI (Nifty 50)")
        return

    with st.spinner(f"Fetching {ticker} …"):
        hist = load_history(ticker, "5y")

    if hist.empty:
        st.error(f"No data for **{ticker}**. Check the symbol (Yahoo Finance format) or your connection.")
        return

    close_full = hist["Close"].dropna()
    name = get_name(ticker)

    # --- ticker ribbon ---
    last = float(close_full.iloc[-1])
    prev = float(close_full.iloc[-2]) if len(close_full) > 1 else last
    chg = last - prev
    chg_pct = chg / prev if prev else 0
    cls = "pos" if chg >= 0 else "neg"
    arrow = "▲" if chg >= 0 else "▼"
    st.markdown(
        f"""<div class="tbar">
        <span class="tsym">{ticker}</span>
        <span class="tname">{name}</span>
        <span class="tprice">{last:,.2f}</span>
        <span class="tchg {cls}">{arrow} {chg:+,.2f} ({chg_pct:+.2%})</span>
        <span class="tmeta">RANGE {close_full.index.min():%d-%b-%Y} → {close_full.index.max():%d-%b-%Y}<br>
        {len(close_full):,} trading days</span>
        </div>""",
        unsafe_allow_html=True,
    )

    # slice for selected performance window
    cutoff = close_full.index.max() - pd.Timedelta(days=PERIODS[period_sel])
    close_win = close_full[close_full.index >= cutoff]

    # metrics on the full 5y sample (stable risk estimates)
    m = full_metrics(close_full, bench_ret, rf_rate)

    left, right = st.columns([2.15, 1])

    # ---------------- LEFT : performance chart ----------------
    with left:
        st.markdown('<div class="rs-section">Price Performance</div>', unsafe_allow_html=True)
        win_ret = (close_win.iloc[-1] / close_win.iloc[0] - 1) if len(close_win) > 1 else 0
        line_color = GREEN if win_ret >= 0 else RED
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=close_win.index, y=close_win.values, mode="lines",
            line=dict(color=line_color, width=1.8), fill="tozeroy",
            fillcolor=f"rgba({'38,208,124' if win_ret>=0 else '255,77,77'},0.08)",
            name=ticker, hovertemplate="%{x|%d-%b-%Y}<br>%{y:,.2f}<extra></extra>",
        ))
        fig.update_layout(showlegend=False)
        fig.update_yaxes(autorange=True)
        style_fig(fig, height=330, title=f"{ticker} · {period_sel}  ({win_ret:+.2%})")
        st.plotly_chart(fig, use_container_width=True)

        # ---- Monte Carlo ----
        st.markdown(
            f'<div class="rs-section">Monte Carlo · {mc_paths:,} paths · {mc_horizon}d horizon</div>',
            unsafe_allow_html=True,
        )
        paths, term_ret = monte_carlo(close_full, mc_horizon, int(mc_paths))
        mc_var95 = -np.percentile(term_ret, 5)
        mc_var99 = -np.percentile(term_ret, 1)
        mc_es95 = -term_ret[term_ret <= np.percentile(term_ret, 5)].mean()

        f1, f2 = st.columns(2)
        with f1:
            fan = go.Figure()
            show = min(120, paths.shape[0])
            xs = np.arange(paths.shape[1])
            for i in range(show):
                fan.add_trace(go.Scatter(x=xs, y=paths[i], mode="lines",
                              line=dict(color=AMBER, width=0.4), opacity=0.12,
                              showlegend=False, hoverinfo="skip"))
            fan.add_trace(go.Scatter(x=xs, y=np.median(paths, axis=0), mode="lines",
                          line=dict(color=TEXT, width=2), name="median"))
            style_fig(fan, height=280, title="Simulated price paths")
            fan.update_layout(showlegend=False)
            st.plotly_chart(fan, use_container_width=True)
        with f2:
            hist_fig = go.Figure()
            hist_fig.add_trace(go.Histogram(x=term_ret * 100, nbinsx=60,
                               marker_color="#1e6f5c", opacity=0.85, name="P&L dist"))
            hist_fig.add_vline(x=-mc_var95 * 100, line=dict(color=AMBER, width=2, dash="dash"),
                               annotation_text="VaR 95", annotation_font_color=AMBER)
            hist_fig.add_vline(x=-mc_var99 * 100, line=dict(color=RED, width=2, dash="dash"),
                               annotation_text="VaR 99", annotation_font_color=RED)
            style_fig(hist_fig, height=280, title=f"Terminal return dist. ({mc_horizon}d)")
            hist_fig.update_layout(showlegend=False, xaxis_title="return %")
            st.plotly_chart(hist_fig, use_container_width=True)

        metric_grid([
            (f"MC VaR 95 · {mc_horizon}d", pct(mc_var95), "neg", money(mc_var95 * notional) + " loss"),
            (f"MC VaR 99 · {mc_horizon}d", pct(mc_var99), "neg", money(mc_var99 * notional) + " loss"),
            (f"MC ES 95 · {mc_horizon}d", pct(mc_es95), "neg", "conditional tail"),
            ("Median outcome", pct(np.median(term_ret)), sign_cls(np.median(term_ret)), "expected drift"),
        ], cols=4)

    # ---------------- RIGHT : risk metric stack ----------------
    with right:
        st.markdown('<div class="rs-section">Risk Metrics · 1-Day</div>', unsafe_allow_html=True)
        metric_grid([
            ("VaR 90% (hist)", pct(m["var90_h"]), "neg", money(m["var90_h"] * notional)),
            ("VaR 95% (hist)", pct(m["var95_h"]), "neg", money(m["var95_h"] * notional)),
            ("VaR 99% (hist)", pct(m["var99_h"]), "neg", money(m["var99_h"] * notional)),
            ("VaR 95% (param)", pct(m["var95_p"]), "neg", "gaussian"),
            ("Exp. Shortfall 95%", pct(m["es95"]), "neg", "CVaR"),
            ("Exp. Shortfall 99%", pct(m["es99"]), "neg", "CVaR"),
        ], cols=2)

        st.markdown('<div class="rs-section">Return / Risk</div>', unsafe_allow_html=True)
        metric_grid([
            ("Ann. Return", pct(m["ann_ret"]), sign_cls(m["ann_ret"]), None),
            ("Ann. Volatility", pct(m["ann_vol"]), "amber", None),
            ("Daily Volatility", pct(m["daily_vol"]), "amber", None),
            ("Max Drawdown", pct(m["mdd"]), "neg", None),
            ("Sharpe", num(m["sharpe"]), sign_cls(m["sharpe"]), None),
            ("Sortino", num(m["sortino"]), sign_cls(m["sortino"]), None),
            ("Calmar", num(m["calmar"]), sign_cls(m["calmar"]), None),
            ("Beta (vs bmk)", num(m["beta"]), "amber", None),
            ("Alpha (ann.)", pct(m["alpha"]), sign_cls(m["alpha"]), None),
            ("Skew", num(m["skew"]), sign_cls(m["skew"]), None),
            ("Excess Kurtosis", num(m["kurt"]), "amber", None),
            ("Sample (days)", f"{len(close_full):,}", "amber", None),
        ], cols=2)

        # option greeks proxy (ATM 30d call on realised vol)
        st.markdown('<div class="rs-section">Greeks · ATM 30d Call</div>', unsafe_allow_html=True)
        g = black_scholes(last, last, 30 / 365, rf_rate, m["ann_vol"], "call")
        metric_grid([
            ("Delta", num(g["delta"], 3), "amber", None),
            ("Gamma", num(g["gamma"], 4), "amber", None),
            ("Vega", num(g["vega"], 3), "amber", "per 1% vol"),
            ("Theta", num(g["theta"], 3), "neg", "per day"),
        ], cols=2)

    # ---------------- STRESS TESTING ----------------
    st.markdown('<div class="rs-section">Historical Stress Testing</div>', unsafe_allow_html=True)
    st.caption("Applies each crisis window to your position. Uses the asset's **actual** return if it "
               "traded then; otherwise a **beta-implied** estimate (β × benchmark move).")
    if st.button("⚡  RUN STRESS TEST", key="stress"):
        beta = m["beta"] if not np.isnan(m["beta"]) else 1.0
        rows = []
        with st.spinner("Replaying historical crises …"):
            for label, (s, e) in CRASH_SCENARIOS.items():
                a = load_close(ticker, s, e)
                b = load_close(benchmark, s, e)
                mkt = (b.iloc[-1] / b.iloc[0] - 1) if len(b) > 1 else np.nan
                if len(a) > 1:
                    shock, basis = a.iloc[-1] / a.iloc[0] - 1, "actual"
                elif not np.isnan(mkt):
                    shock, basis = beta * mkt, f"β-implied (β={beta:.2f})"
                else:
                    shock, basis = np.nan, "no data"
                rows.append({
                    "Scenario": label,
                    "Window": f"{s} → {e}",
                    "Market": pct(mkt),
                    "Asset shock": pct(shock),
                    "Basis": basis,
                    "P&L on notional": money(shock * notional) if not np.isnan(shock) else "—",
                    "_shock": shock,
                })
        df = pd.DataFrame(rows)
        bar = go.Figure(go.Bar(
            x=df["Scenario"], y=df["_shock"] * 100,
            marker_color=[GREEN if v >= 0 else RED for v in df["_shock"].fillna(0)],
            text=[pct(v) for v in df["_shock"]], textposition="outside",
        ))
        style_fig(bar, height=300, title=f"{ticker} — impact under historical crises")
        bar.update_layout(yaxis_title="asset return %", showlegend=False)
        st.plotly_chart(bar, use_container_width=True)
        st.dataframe(df.drop(columns="_shock"), use_container_width=True, hide_index=True)


# ============================================================================= #
#  PAGE 2 — PORTFOLIO                                                           #
# ============================================================================= #
def page_portfolio():
    st.markdown(
        '<div class="rs-brand">RISK<span>SCOPE</span> <span style="font-size:14px;">// PORTFOLIO</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="rs-section">1 · Build your basket</div>', unsafe_allow_html=True)
    st.caption("Enter comma-separated tickers, then set weights. Weights are auto-normalised to 100%.")

    raw = st.text_input("Tickers", "AAPL, MSFT, GC=F, TLT, BTC-USD",
                        help="Mix equities, commodities, bonds, crypto — anything on Yahoo Finance.")
    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
    tickers = list(dict.fromkeys(tickers))  # dedupe, keep order

    if len(tickers) < 2:
        st.info("Add at least **two** assets to build a portfolio.")
        return

    st.markdown('<div class="rs-section">2 · Assign weights</div>', unsafe_allow_html=True)
    cols = st.columns(len(tickers))
    weights = []
    for i, t in enumerate(tickers):
        weights.append(cols[i].number_input(t, 0.0, 100.0, round(100 / len(tickers), 2), 1.0, key=f"w_{t}"))
    w = np.array(weights, dtype=float)
    if w.sum() == 0:
        st.warning("Total weight is zero — assign some weight.")
        return
    w = w / w.sum()
    st.caption("Normalised weights: " + " · ".join(f"{t} {wi:.1%}" for t, wi in zip(tickers, w)))

    if not st.button("🚀  SIMULATE PORTFOLIO", key="sim_pf"):
        st.stop()

    # ---- gather aligned price data ----
    with st.spinner("Loading constituents …"):
        closes = {}
        for t in tickers:
            h = load_history(t, "5y")
            if not h.empty:
                closes[t] = h["Close"]
    if len(closes) < 2:
        st.error("Could not load enough constituents. Check the tickers.")
        return

    prices = pd.DataFrame(closes).dropna()
    if prices.empty or len(prices) < 30:
        st.error("Not enough overlapping history across these assets (they must trade on common dates).")
        return

    used = list(prices.columns)
    w = np.array([weights[tickers.index(t)] for t in used], dtype=float)
    w = w / w.sum()

    rets = prices.pct_change().dropna()
    port_ret = rets.dot(w)
    cov = rets.cov() * TRADING_DAYS

    ann_ret = port_ret.mean() * TRADING_DAYS
    ann_vol = np.sqrt(w @ cov.values @ w)
    daily_vol = port_ret.std()
    sharpe = (ann_ret - rf_rate) / ann_vol if ann_vol else np.nan
    downside = port_ret[port_ret < 0].std() * np.sqrt(TRADING_DAYS)
    sortino = (ann_ret - rf_rate) / downside if downside else np.nan
    p_cum = (1 + port_ret).cumprod()
    mdd = (p_cum / p_cum.cummax() - 1).min()
    v95 = hist_var(port_ret, 0.95)
    v99 = hist_var(port_ret, 0.99)
    es95 = expected_shortfall(port_ret, 0.95)
    es99 = expected_shortfall(port_ret, 0.99)

    st.markdown('<div class="rs-section">Portfolio Risk Dashboard</div>', unsafe_allow_html=True)
    metric_grid([
        ("Ann. Return", pct(ann_ret), sign_cls(ann_ret), None),
        ("Ann. Volatility", pct(ann_vol), "amber", None),
        ("Daily Volatility", pct(daily_vol), "amber", None),
        ("Sharpe", num(sharpe), sign_cls(sharpe), None),
        ("Sortino", num(sortino), sign_cls(sortino), None),
        ("Max Drawdown", pct(mdd), "neg", None),
        ("VaR 95% (1d)", pct(v95), "neg", money(v95 * notional)),
        ("VaR 99% (1d)", pct(v99), "neg", money(v99 * notional)),
        ("CVaR 95% (1d)", pct(es95), "neg", money(es95 * notional)),
        ("CVaR 99% (1d)", pct(es99), "neg", money(es99 * notional)),
        ("Constituents", str(len(used)), "amber", None),
        ("Sample (days)", f"{len(rets):,}", "amber", None),
    ], cols=4)

    a, b = st.columns([1.4, 1])
    with a:
        st.markdown('<div class="rs-section">Growth of $1 (rebased)</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=p_cum.index, y=p_cum.values, mode="lines",
                      line=dict(color=AMBER, width=2), name="Portfolio"))
        for t in used:
            c = (1 + rets[t]).cumprod()
            fig.add_trace(go.Scatter(x=c.index, y=c.values, mode="lines",
                          line=dict(width=0.8), opacity=0.5, name=t))
        style_fig(fig, height=340)
        st.plotly_chart(fig, use_container_width=True)
    with b:
        st.markdown('<div class="rs-section">Correlation Matrix</div>', unsafe_allow_html=True)
        corr = rets.corr()
        heat = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale=[[0, RED], [0.5, BG], [1, GREEN]], zmid=0,
            text=np.round(corr.values, 2), texttemplate="%{text}",
            textfont=dict(size=11), showscale=False,
        ))
        style_fig(heat, height=340)
        st.plotly_chart(heat, use_container_width=True)

    # risk contribution + MC
    st.markdown('<div class="rs-section">Risk Contribution & Monte Carlo</div>', unsafe_allow_html=True)
    mctrl = cov.values @ w
    rc = w * mctrl
    rc_pct = rc / rc.sum() if rc.sum() else rc
    c1, c2 = st.columns(2)
    with c1:
        contrib = go.Figure(go.Bar(
            x=used, y=rc_pct * 100, marker_color=AMBER,
            text=[f"{v:.1%}" for v in rc_pct], textposition="outside",
        ))
        style_fig(contrib, height=300, title="Contribution to portfolio variance")
        contrib.update_layout(yaxis_title="% of risk", showlegend=False)
        st.plotly_chart(contrib, use_container_width=True)
    with c2:
        rng = np.random.default_rng(7)
        mu_d = port_ret.mean()
        sims = rng.normal(mu_d, daily_vol, size=(int(mc_paths), mc_horizon))
        term = np.cumprod(1 + sims, axis=1)[:, -1] - 1
        pf_var95 = -np.percentile(term, 5)
        pf_var99 = -np.percentile(term, 1)
        hf = go.Figure(go.Histogram(x=term * 100, nbinsx=60, marker_color="#1e6f5c", opacity=0.85))
        hf.add_vline(x=-pf_var95 * 100, line=dict(color=AMBER, width=2, dash="dash"),
                     annotation_text="VaR95", annotation_font_color=AMBER)
        hf.add_vline(x=-pf_var99 * 100, line=dict(color=RED, width=2, dash="dash"),
                     annotation_text="VaR99", annotation_font_color=RED)
        style_fig(hf, height=300, title=f"MC {mc_horizon}d portfolio P&L ({int(mc_paths):,} paths)")
        hf.update_layout(showlegend=False, xaxis_title="return %")
        st.plotly_chart(hf, use_container_width=True)
        metric_grid([
            (f"MC VaR95 {mc_horizon}d", pct(pf_var95), "neg", money(pf_var95 * notional)),
            (f"MC VaR99 {mc_horizon}d", pct(pf_var99), "neg", money(pf_var99 * notional)),
        ], cols=2)


# ============================================================================= #
#  PAGE 3 — OPTIONS LAB                                                          #
# ============================================================================= #
def page_options():
    st.markdown(
        '<div class="rs-brand">RISK<span>SCOPE</span> <span style="font-size:14px;">// OPTIONS LAB</span></div>',
        unsafe_allow_html=True,
    )
    st.caption("Black-Scholes-Merton pricer with full Greeks. Pull live spot/vol from a ticker, or set inputs manually.")

    c1, c2, c3 = st.columns([1.4, 1, 1])
    tkr = c1.text_input("Underlying ticker (optional — autofills spot & vol)", "AAPL").strip().upper()
    kind = c2.selectbox("Type", ["call", "put"])
    auto = c3.checkbox("Auto spot/vol", True)

    S_def, sig_def = 100.0, 0.25
    if auto and tkr:
        h = load_history(tkr, "1y")
        if not h.empty:
            S_def = float(h["Close"].iloc[-1])
            sig_def = float(daily_returns(h["Close"]).std() * np.sqrt(TRADING_DAYS))

    a, b, c, d, e = st.columns(5)
    S = a.number_input("Spot (S)", 0.01, 1e6, round(S_def, 2))
    K = b.number_input("Strike (K)", 0.01, 1e6, round(S_def, 2))
    days = c.number_input("Days to expiry", 1, 1825, 30)
    sig = d.number_input("Volatility σ (ann.)", 0.01, 5.0, round(sig_def, 4), 0.01)
    r = e.number_input("Rate r (ann.)", 0.0, 1.0, rf_rate, 0.005)
    T = days / 365

    g = black_scholes(S, K, T, r, sig, kind)
    st.markdown('<div class="rs-section">Valuation & Greeks</div>', unsafe_allow_html=True)
    metric_grid([
        ("Option Price", money(g["price"], 2), "amber", f"{kind.upper()} · K={K:g}"),
        ("Delta", num(g["delta"], 4), "amber", "∂V/∂S"),
        ("Gamma", num(g["gamma"], 5), "amber", "∂Δ/∂S"),
        ("Vega", num(g["vega"], 4), "amber", "per 1% σ"),
        ("Theta", num(g["theta"], 4), "neg", "per day"),
        ("Rho", num(g["rho"], 4), "amber", "per 1% r"),
        ("d1", num(g["d1"], 4), "", None),
        ("d2", num(g["d2"], 4), "", None),
    ], cols=4)

    # payoff + value curve
    st.markdown('<div class="rs-section">Payoff & Value Profile</div>', unsafe_allow_html=True)
    spots = np.linspace(max(0.01, S * 0.4), S * 1.6, 120)
    payoff = np.maximum(spots - K, 0) if kind == "call" else np.maximum(K - spots, 0)
    payoff_net = payoff - g["price"]
    value_now = [black_scholes(s, K, T, r, sig, kind)["price"] for s in spots]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spots, y=payoff_net, mode="lines",
                  line=dict(color=AMBER, width=2), name="P&L at expiry"))
    fig.add_trace(go.Scatter(x=spots, y=np.array(value_now) - g["price"], mode="lines",
                  line=dict(color=GREEN, width=1.6, dash="dot"), name="Value today"))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.add_vline(x=S, line=dict(color=TEXT, width=1, dash="dash"),
                  annotation_text="spot", annotation_font_color=TEXT)
    style_fig(fig, height=380, title=f"{kind.upper()} option · net P&L vs underlying")
    fig.update_layout(xaxis_title="underlying price", yaxis_title="profit / loss")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================= #
#  PAGE 4 — ABOUT                                                                #
# ============================================================================= #
def page_about():
    st.markdown('<div class="rs-brand">RISK<span>SCOPE</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="rs-tag">methodology &nbsp;·&nbsp; glossary</div>', unsafe_allow_html=True)
    st.markdown("""
### What RiskScope does
A Bloomberg-style terminal that turns any tradable asset into a full market-risk report card —
Value-at-Risk, Expected Shortfall, the Greeks, Monte-Carlo simulation, and historical stress tests —
plus a portfolio builder and an options lab.

### How the numbers are computed
| Metric | Method |
|---|---|
| **VaR (historical)** | Empirical percentile of daily returns (1-day horizon). |
| **VaR (parametric)** | Variance-covariance: −(μ + z·σ), Gaussian z-score. |
| **Expected Shortfall / CVaR** | Mean loss *beyond* the VaR cutoff — captures tail severity. |
| **Beta / Alpha** | OLS of asset returns on the benchmark; Jensen's alpha annualised. |
| **Sharpe / Sortino / Calmar** | Excess return over total vol / downside vol / max drawdown. |
| **Monte Carlo** | Geometric Brownian Motion, drift & vol from realised daily returns. |
| **Greeks** | Black-Scholes-Merton closed form (delta, gamma, vega, theta, rho). |
| **Stress tests** | Replays COVID-19, the 2008 GFC, and the 2022 rate shock — actual return if the asset traded then, else β-implied. |

### Ticker formats (Yahoo Finance)
- **Equities/ETFs:** `AAPL`, `MSFT`, `SPY`
- **Indices:** `^GSPC` (S&P 500), `^NSEI` (Nifty 50), `^IXIC` (Nasdaq)
- **Commodities (futures):** `GC=F` (gold), `CL=F` (crude), `SI=F` (silver)
- **FX:** `EURUSD=X`, `USDINR=X`
- **Crypto:** `BTC-USD`, `ETH-USD`

---
*Data is delayed and sourced from Yahoo Finance. RiskScope is an educational / portfolio project — **not investment advice.***
""")


# ----------------------------------------------------------------------------- #
#  ROUTER                                                                        #
# ----------------------------------------------------------------------------- #
if page.startswith("📈"):
    page_terminal()
elif page.startswith("💼"):
    page_portfolio()
elif page.startswith("🧮"):
    page_options()
else:
    page_about()
