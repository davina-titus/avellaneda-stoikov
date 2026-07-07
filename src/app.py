"""
Avellaneda-Stoikov Market Making — Interactive Dashboard
=========================================================
Run:  python src/app.py
Then open http://127.0.0.1:8050
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc

from model import ASParams, simulate, run_monte_carlo, compute_naive_spread

# ──────────────────────────────────────────────
# Colour palette — dark terminal aesthetic 
# ──────────────────────────────────────────────
BG        = "#0D0F14"
PANEL     = "#13161E"
BORDER    = "#1E2330"
TEXT      = "#E2E8F0"
TEXT_DIM  = "#64748B"
BLUE      = "#3B82F6"
GREEN     = "#10B981"
RED       = "#EF4444"
AMBER     = "#F59E0B"
PURPLE    = "#8B5CF6"
TEAL      = "#06B6D4"
GRID      = "#1E2330"
PLOT_BG   = "#0D0F14"

FONT = "JetBrains Mono, Fira Code, monospace"

def card(children, style=None):
    base = dict(background=PANEL, border=f"1px solid {BORDER}",
                borderRadius="8px", padding="20px", marginBottom="16px")
    if style:
        base.update(style)
    return html.Div(children, style=base)

def label(text):
    return html.Div(text, style=dict(
        fontSize="10px", color=TEXT_DIM, textTransform="uppercase",
        letterSpacing="0.1em", marginBottom="4px", fontFamily=FONT))

def metric_box(title, value_id, sub="", color=TEXT):
    return html.Div([
        label(title),
        html.Div(id=value_id, style=dict(
            fontSize="26px", fontWeight="600", color=color,
            fontFamily=FONT, fontVariantNumeric="tabular-nums")),
        html.Div(sub, style=dict(fontSize="11px", color=TEXT_DIM, marginTop="2px")),
    ], style=dict(flex="1", minWidth="120px"))

def slider_row(lbl, sid, mn, mx, step, val, fmt=".3f"):
    return html.Div([
        html.Div([
            label(lbl),
            html.Div(id=f"val-{sid}", style=dict(
                fontSize="13px", color=TEXT, fontFamily=FONT,
                fontVariantNumeric="tabular-nums")),
        ], style=dict(display="flex", justifyContent="space-between", alignItems="center")),
        dcc.Slider(id=sid, min=mn, max=mx, step=step, value=val,
                   marks=None, tooltip={"always_visible": False},
                   updatemode="drag"),
    ], style=dict(marginBottom="14px"))


# ──────────────────────────────────────────────
# App layout
# ──────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG],
                title="A-S Market Maker")
app.layout = html.Div(style=dict(background=BG, minHeight="100vh", padding="24px",
                                  fontFamily=FONT, color=TEXT), children=[

    # Header
    html.Div([
        html.Div([
            html.Div("AVELLANEDA-STOIKOV", style=dict(
                fontSize="11px", color=TEAL, letterSpacing="0.15em",
                textTransform="uppercase", marginBottom="4px")),
            html.Div("Market Making Simulator", style=dict(
                fontSize="28px", fontWeight="700", color=TEXT, letterSpacing="-0.02em")),
            html.Div("Optimal bid/ask quoting under inventory risk · Avellaneda & Stoikov (2008)",
                     style=dict(fontSize="13px", color=TEXT_DIM, marginTop="4px")),
        ]),
        html.Div([
            html.Div(id="status-badge", children="● READY", style=dict(
                fontSize="11px", color=GREEN, letterSpacing="0.08em")),
        ], style=dict(textAlign="right")),
    ], style=dict(display="flex", justifyContent="space-between",
                  alignItems="flex-start", marginBottom="24px")),

    # Main layout — sidebar + content
    html.Div([

        # ── Left sidebar — params ──
        html.Div([
            card([
                html.Div("MODEL PARAMETERS", style=dict(
                    fontSize="11px", color=TEAL, letterSpacing="0.12em",
                    marginBottom="16px")),

                slider_row("Volatility σ", "sl-sigma", 0.5, 5.0, 0.1, 2.0),
                slider_row("Risk aversion γ", "sl-gamma", 0.01, 0.50, 0.01, 0.10),
                slider_row("Order intensity k", "sl-k", 0.5, 8.0, 0.1, 1.5),
                slider_row("Arrival rate A", "sl-A", 10, 300, 10, 140, fmt=".0f"),
                slider_row("Time horizon T", "sl-T", 0.1, 2.0, 0.1, 1.0),
                slider_row("Timestep dt", "sl-dt", 0.0005, 0.005, 0.0005, 0.001),
                slider_row("Initial price S₀", "sl-S0", 0.5, 5.0, 0.5, 1.0),
                slider_row("Max inventory", "sl-maxq", 3, 15, 1, 5),

                html.Div([
                    html.Button("▶  RUN SIMULATION", id="btn-run",
                                style=dict(
                                    width="100%", padding="10px",
                                    background=TEAL, color=BG,
                                    border="none", borderRadius="6px",
                                    fontSize="12px", fontWeight="700",
                                    fontFamily=FONT, letterSpacing="0.08em",
                                    cursor="pointer", marginTop="4px")),
                    html.Button("⟳  MONTE CARLO (200 paths)", id="btn-mc",
                                style=dict(
                                    width="100%", padding="10px",
                                    background="transparent", color=PURPLE,
                                    border=f"1px solid {PURPLE}",
                                    borderRadius="6px", fontSize="12px",
                                    fontWeight="700", fontFamily=FONT,
                                    letterSpacing="0.08em", cursor="pointer",
                                    marginTop="8px")),
                ]),
            ]),

            card([
                html.Div("CORE EQUATIONS", style=dict(
                    fontSize="11px", color=TEAL, letterSpacing="0.12em",
                    marginBottom="12px")),
                *[html.Div(eq, style=dict(
                    fontSize="11px", color=TEXT_DIM, marginBottom="6px",
                    lineHeight="1.6", borderLeft=f"2px solid {BORDER}",
                    paddingLeft="8px"))
                  for eq in [
                    "r(s,q,t) = s − q·γ·σ²·(T−t)",
                    "δ* = γσ²(T−t) + (2/γ)·ln(1+γ/k)",
                    "b* = r − δ*/2   a* = r + δ*/2",
                    "λ±(δ) = A·exp(−k·δ±)",
                  ]],
            ]),
        ], style=dict(width="280px", flexShrink="0")),

        # ── Right — charts & metrics ──
        html.Div([

            # Metrics row
            card([
                html.Div([
                    metric_box("A-S PnL", "met-as-pnl", "optimal quoting", GREEN),
                    metric_box("Naive PnL", "met-naive-pnl", "symmetric fixed spread", RED),
                    metric_box("Edge", "met-edge", "A-S vs naive", AMBER),
                    metric_box("Sharpe (A-S)", "met-sharpe-as", "annualised", TEAL),
                    metric_box("Max DD (A-S)", "met-dd-as", "drawdown", RED),
                    metric_box("Fills (A-S)", "met-fills", "total executions", TEXT),
                    metric_box("Inv risk σ", "met-inv-risk", "std(inventory)", AMBER),
                    metric_box("Avg spread", "met-spread", "A-S vs naive", TEXT),
                ], style=dict(display="flex", gap="16px", flexWrap="wrap")),
            ], style=dict(marginBottom="16px")),

            # Tabs for charts
            dcc.Tabs(id="tabs", value="tab-price", children=[
                dcc.Tab(label="Price & Quotes", value="tab-price"),
                dcc.Tab(label="Inventory", value="tab-inv"),
                dcc.Tab(label="PnL", value="tab-pnl"),
                dcc.Tab(label="Spread", value="tab-spread"),
                dcc.Tab(label="Monte Carlo", value="tab-mc"),
                dcc.Tab(label="Trade Log", value="tab-trades"),
            ], style=dict(borderBottom=f"1px solid {BORDER}"),
               colors=dict(border=BORDER, primary=TEAL, background=PANEL)),

            html.Div(id="tab-content", style=dict(marginTop="4px")),

        ], style=dict(flex="1", minWidth="0")),

    ], style=dict(display="flex", gap="16px", alignItems="flex-start")),

    # Hidden stores
    dcc.Store(id="store-sim"),
    dcc.Store(id="store-mc"),
])


# ──────────────────────────────────────────────
# Plotly figure helpers
# ──────────────────────────────────────────────
def _layout(title="", xtitle="Time", ytitle=""):
    return dict(
        template="plotly_dark",
        paper_bgcolor=PANEL, plot_bgcolor=PLOT_BG,
        font=dict(family=FONT, color=TEXT_DIM, size=11),
        title=dict(text=title, font=dict(size=13, color=TEXT), x=0),
        xaxis=dict(title=xtitle, gridcolor=GRID, zerolinecolor=GRID, title_font_color=TEXT_DIM),
        yaxis=dict(title=ytitle, gridcolor=GRID, zerolinecolor=GRID, title_font_color=TEXT_DIM),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        margin=dict(l=50, r=20, t=40, b=40),
        hovermode="x unified",
    )


def fig_price(r):
    stride = max(1, len(r.times) // 800)
    t  = r.times[::stride]
    s  = r.mid_prices[::stride]
    rp = r.reservation_prices_as[::stride]
    b  = r.bids_as[::stride]
    a  = r.asks_as[::stride]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=a, name="Ask (A-S)", line=dict(color=RED, width=1),
                              fill=None))
    fig.add_trace(go.Scatter(x=t, y=b, name="Bid (A-S)", line=dict(color=GREEN, width=1),
                              fill='tonexty',
                              fillcolor="rgba(16,185,129,0.04)"))
    fig.add_trace(go.Scatter(x=t, y=s, name="Mid price", line=dict(color=BLUE, width=1.5)))
    fig.add_trace(go.Scatter(x=t, y=rp, name="Reservation price",
                              line=dict(color=AMBER, width=1, dash="dot")))

    # Overlay trades
    buys  = [tr for tr in r.trades if tr['strategy']=='A-S' and tr['side']=='buy']
    sells = [tr for tr in r.trades if tr['strategy']=='A-S' and tr['side']=='sell']
    if buys:
        fig.add_trace(go.Scatter(
            x=[tr['t'] for tr in buys], y=[tr['price'] for tr in buys],
            mode="markers", name="Buy fill",
            marker=dict(symbol="triangle-up", color=GREEN, size=7, opacity=0.7)))
    if sells:
        fig.add_trace(go.Scatter(
            x=[tr['t'] for tr in sells], y=[tr['price'] for tr in sells],
            mode="markers", name="Sell fill",
            marker=dict(symbol="triangle-down", color=RED, size=7, opacity=0.7)))

    fig.update_layout(**_layout("Mid Price with Optimal Quotes", "Time", "Price"))
    return fig


def fig_inventory(r):
    stride = max(1, len(r.times) // 800)
    t  = r.times[::stride]
    qa = r.inventory_as[::stride]
    qn = r.inventory_naive[::stride]

    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=BORDER, width=1))
    fig.add_trace(go.Scatter(x=t, y=qn, name="Naive inventory",
                              line=dict(color=TEXT_DIM, width=1, dash="dash")))
    fig.add_trace(go.Scatter(x=t, y=qa, name="A-S inventory",
                              line=dict(color=PURPLE, width=1.5),
                              fill="tozeroy", fillcolor="rgba(139,92,246,0.07)"))
    fig.update_layout(**_layout("Inventory Path", "Time", "Shares held (q)"))
    return fig


def fig_pnl(r):
    stride = max(1, len(r.times) // 800)
    t   = r.times[::stride]
    pa  = r.pnl_as[::stride]
    pn  = r.pnl_naive[::stride]

    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=BORDER, width=1))
    fig.add_trace(go.Scatter(x=t, y=pn, name="Naive PnL",
                              line=dict(color=TEXT_DIM, width=1, dash="dash")))
    fig.add_trace(go.Scatter(x=t, y=pa, name="A-S PnL",
                              line=dict(color=GREEN if pa[-1] >= 0 else RED, width=2),
                              fill="tozeroy",
                              fillcolor=f"rgba(16,185,129,0.06)" if pa[-1] >= 0
                                        else "rgba(239,68,68,0.06)"))
    fig.update_layout(**_layout("Cumulative Mark-to-Market PnL", "Time", "PnL ($)"))
    return fig


def fig_spread(r):
    stride = max(1, len(r.times) // 800)
    t  = r.times[::stride]
    sp = r.spreads_as[::stride]
    ns = np.full_like(sp, r.naive_spread)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=ns, name="Naive spread (fixed)",
                              line=dict(color=TEXT_DIM, width=1, dash="dash")))
    fig.add_trace(go.Scatter(x=t, y=sp, name="A-S optimal spread",
                              line=dict(color=AMBER, width=1.5),
                              fill="tozeroy", fillcolor="rgba(245,158,11,0.06)"))
    fig.update_layout(**_layout("Spread Decomposition Over Time", "Time", "Spread δ*"))
    return fig


def fig_mc(mc):
    fig = make_subplots(rows=2, cols=2, shared_xaxes=False,
                        subplot_titles=["PnL Distribution (A-S vs Naive)",
                                        "Edge Distribution (A-S − Naive)",
                                        "Sharpe Ratio Distribution",
                                        "Max Drawdown Distribution"],
                        vertical_spacing=0.14, horizontal_spacing=0.10)

    def hist(data, name, color, row, col, **kw):
        fig.add_trace(go.Histogram(x=data, name=name, marker_color=color,
                                    opacity=0.7, nbinsx=30, **kw), row=row, col=col)

    hist(mc['pnl_as'],    "A-S PnL",    GREEN,   1, 1)
    hist(mc['pnl_naive'], "Naive PnL",  TEXT_DIM, 1, 1)
    hist(mc['edge'],      "Edge",       AMBER,   1, 2)
    hist(mc['sharpe_as'], "A-S Sharpe", TEAL,    2, 1)
    hist(mc['sharpe_naive'], "Naive Sharpe", TEXT_DIM, 2, 1)
    hist(mc['max_dd_as'], "A-S Max DD", RED,     2, 2)
    hist(mc['max_dd_naive'], "Naive Max DD", TEXT_DIM, 2, 2)

    # Vertical lines for means
    for col_idx, (arr, color) in enumerate([(mc['pnl_as'], GREEN), (mc['edge'], AMBER)], 1):
        fig.add_vline(x=float(arr.mean()), line=dict(color=color, dash="dash", width=1),
                      row=1, col=col_idx)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor=PANEL, plot_bgcolor=PLOT_BG,
        font=dict(family=FONT, color=TEXT_DIM, size=11),
        title=dict(text="Monte Carlo: 200 Simulation Paths", font=dict(size=13, color=TEXT), x=0),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=50, r=20, t=60, b=40),
        showlegend=True,
        barmode="overlay",
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def fig_trades(trades):
    df = pd.DataFrame(trades)
    if df.empty:
        return go.Figure()
    df['color'] = df['side'].map({'buy': GREEN, 'sell': RED})
    fig = go.Figure()
    for side, color in [('buy', GREEN), ('sell', RED)]:
        sub = df[df['side'] == side]
        fig.add_trace(go.Scatter(
            x=sub['t'], y=sub['price'],
            mode="markers",
            name=side.capitalize(),
            marker=dict(color=color, size=6, symbol="triangle-up" if side=='buy' else "triangle-down",
                        opacity=0.8),
            hovertemplate=f"<b>{side}</b><br>t=%{{x:.4f}}<br>price=%{{y:.4f}}<br>inv=%{{customdata}}<extra></extra>",
            customdata=sub['inv'],
        ))
    fig.update_layout(**_layout("Trade Fills (A-S Strategy)", "Time", "Fill Price"))
    return fig


# ──────────────────────────────────────────────
# Callbacks
# ──────────────────────────────────────────────

# Slider labels
for sid, fmt in [("sl-sigma",1),("sl-gamma",2),("sl-k",1),("sl-A",0),
                 ("sl-T",1),("sl-dt",4),("sl-S0",1),("sl-maxq",0)]:
    @app.callback(Output(f"val-{sid}", "children"), Input(sid, "value"),
                  prevent_initial_call=False)
    def _lbl(v, f=fmt): return f"{v:.{f}f}"


@app.callback(
    Output("store-sim", "data"),
    Output("status-badge", "children"),
    Output("status-badge", "style"),
    Input("btn-run", "n_clicks"),
    State("sl-sigma","value"), State("sl-gamma","value"), State("sl-k","value"),
    State("sl-A","value"),     State("sl-T","value"),     State("sl-dt","value"),
    State("sl-S0","value"),    State("sl-maxq","value"),
    prevent_initial_call=False,
)
def run_sim(n, sigma, gamma, k, A, T, dt, S0, maxq):
    p = ASParams(sigma=sigma, gamma=gamma, k=k, A=A, T=T, dt=dt,
                 S0=S0, max_inventory=int(maxq))
    r = simulate(p)

    data = dict(
        times=r.times.tolist(), mid_prices=r.mid_prices.tolist(),
        reservation_prices_as=r.reservation_prices_as.tolist(),
        bids_as=r.bids_as.tolist(), asks_as=r.asks_as.tolist(),
        spreads_as=r.spreads_as.tolist(),
        inventory_as=r.inventory_as.tolist(),
        cash_as=r.cash_as.tolist(), pnl_as=r.pnl_as.tolist(),
        bids_naive=r.bids_naive.tolist(), asks_naive=r.asks_naive.tolist(),
        inventory_naive=r.inventory_naive.tolist(),
        cash_naive=r.cash_naive.tolist(), pnl_naive=r.pnl_naive.tolist(),
        trades=r.trades,
        final_pnl_as=r.final_pnl_as, final_pnl_naive=r.final_pnl_naive,
        edge=r.edge, sharpe_as=r.sharpe_as, sharpe_naive=r.sharpe_naive,
        max_drawdown_as=r.max_drawdown_as, max_drawdown_naive=r.max_drawdown_naive,
        total_trades_as=r.total_trades_as, inventory_risk_as=r.inventory_risk_as,
        avg_spread_as=r.avg_spread_as, naive_spread=r.naive_spread,
    )
    badge_style = dict(fontSize="11px", color=GREEN, letterSpacing="0.08em")
    return data, "● SIMULATED", badge_style


@app.callback(
    Output("store-mc", "data"),
    Input("btn-mc", "n_clicks"),
    State("sl-sigma","value"), State("sl-gamma","value"), State("sl-k","value"),
    State("sl-A","value"),     State("sl-T","value"),     State("sl-dt","value"),
    State("sl-S0","value"),    State("sl-maxq","value"),
    prevent_initial_call=True,
)
def run_mc(n, sigma, gamma, k, A, T, dt, S0, maxq):
    p = ASParams(sigma=sigma, gamma=gamma, k=k, A=A, T=T, dt=dt,
                 S0=S0, max_inventory=int(maxq))
    mc = run_monte_carlo(p, n_paths=200)
    return {k: v.tolist() for k, v in mc.items()}


@app.callback(
    Output("met-as-pnl",   "children"), Output("met-as-pnl",   "style"),
    Output("met-naive-pnl","children"), Output("met-naive-pnl", "style"),
    Output("met-edge",     "children"), Output("met-edge",      "style"),
    Output("met-sharpe-as","children"),
    Output("met-dd-as",    "children"),
    Output("met-fills",    "children"),
    Output("met-inv-risk", "children"),
    Output("met-spread",   "children"),
    Input("store-sim", "data"),
    prevent_initial_call=False,
)
def update_metrics(data):
    if not data:
        return ["—"]*11 + [{}]*3 + ["—"]*8

    def col(v): return dict(fontSize="26px", fontWeight="600",
                             fontFamily=FONT, fontVariantNumeric="tabular-nums",
                             color=GREEN if v >= 0 else RED)
    def fmt(v): return f"{v:+.2f}"

    pnl_as    = data['final_pnl_as']
    pnl_naive = data['final_pnl_naive']
    edge      = data['edge']

    return (
        fmt(pnl_as),    col(pnl_as),
        fmt(pnl_naive), col(pnl_naive),
        fmt(edge),      col(edge),
        f"{data['sharpe_as']:.2f}",
        fmt(data['max_drawdown_as']),
        str(data['total_trades_as']),
        f"{data['inventory_risk_as']:.2f}",
        f"δ*={data['avg_spread_as']:.4f}  naive={data['naive_spread']:.4f}",
    )


@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("store-sim", "data"),
    Input("store-mc",  "data"),
    prevent_initial_call=False,
)
def render_tab(tab, data, mc_data):
    def empty():
        return html.Div("Run a simulation first.",
                        style=dict(color=TEXT_DIM, padding="40px", textAlign="center"))

    wrap = lambda fig: card([dcc.Graph(figure=fig,
                                        config=dict(displayModeBar=True),
                                        style=dict(height="460px"))],
                             style=dict(padding="12px"))

    if tab == "tab-price":
        if not data: return empty()
        from model import SimResult
        import numpy as np
        r = _dict_to_result(data)
        return wrap(fig_price(r))

    elif tab == "tab-inv":
        if not data: return empty()
        r = _dict_to_result(data)
        return wrap(fig_inventory(r))

    elif tab == "tab-pnl":
        if not data: return empty()
        r = _dict_to_result(data)
        return wrap(fig_pnl(r))

    elif tab == "tab-spread":
        if not data: return empty()
        r = _dict_to_result(data)
        return wrap(fig_spread(r))

    elif tab == "tab-mc":
        if not mc_data:
            return html.Div("Click 'Monte Carlo' to run 200 simulation paths.",
                            style=dict(color=TEXT_DIM, padding="40px", textAlign="center"))
        mc = {k: np.array(v) for k, v in mc_data.items()}
        fig = fig_mc(mc)
        summary = html.Div([
            html.Div("Monte Carlo Summary (200 paths)", style=dict(
                fontSize="11px", color=TEAL, letterSpacing="0.1em",
                marginBottom="12px", textTransform="uppercase")),
            html.Div([
                html.Div([
                    html.Div(f"A-S mean PnL:    {mc['pnl_as'].mean():+.4f}", style=dict(color=GREEN)),
                    html.Div(f"Naive mean PnL:  {mc['pnl_naive'].mean():+.4f}", style=dict(color=TEXT_DIM)),
                    html.Div(f"Mean edge:       {mc['edge'].mean():+.4f}", style=dict(color=AMBER)),
                    html.Div(f"Edge > 0:        {(mc['edge'] > 0).mean()*100:.1f}% of paths", style=dict(color=TEXT)),
                ], style=dict(fontSize="12px", fontFamily=FONT, lineHeight="1.9",
                               background=PLOT_BG, padding="12px", borderRadius="6px")),
            ]),
        ], style=dict(padding="0 0 12px 0"))

        return card([summary, dcc.Graph(figure=fig, config=dict(displayModeBar=True),
                                         style=dict(height="500px"))],
                    style=dict(padding="12px"))

    elif tab == "tab-trades":
        if not data: return empty()
        trades = data.get("trades", [])
        if not trades:
            return html.Div("No fills recorded.", style=dict(color=TEXT_DIM, padding="40px"))

        fig = fig_trades(trades)
        df = pd.DataFrame(trades).tail(50).iloc[::-1]

        table = html.Table([
            html.Thead(html.Tr([
                html.Th(c, style=dict(padding="6px 10px", textAlign="left",
                                       fontSize="10px", color=TEXT_DIM,
                                       textTransform="uppercase", letterSpacing="0.08em",
                                       borderBottom=f"1px solid {BORDER}"))
                for c in ["Step","Time","Side","Price","Inventory","Strategy"]
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(tr.get("step",""), style=dict(padding="5px 10px", fontSize="12px")),
                    html.Td(f"{tr.get('t',0):.4f}", style=dict(padding="5px 10px", fontSize="12px")),
                    html.Td(tr.get("side",""),
                            style=dict(padding="5px 10px", fontSize="12px", fontWeight="600",
                                       color=GREEN if tr.get("side")=="buy" else RED)),
                    html.Td(f"{tr.get('price',0):.4f}",
                            style=dict(padding="5px 10px", fontSize="12px", fontFamily=FONT)),
                    html.Td(tr.get("inv",""),
                            style=dict(padding="5px 10px", fontSize="12px")),
                    html.Td(tr.get("strategy",""),
                            style=dict(padding="5px 10px", fontSize="12px", color=TEAL)),
                ], style=dict(borderBottom=f"1px solid {BORDER}"))
                for _, tr in df.iterrows()
            ]),
        ], style=dict(width="100%", borderCollapse="collapse"))

        return card([
            dcc.Graph(figure=fig, config=dict(displayModeBar=True),
                      style=dict(height="300px")),
            html.Div("Last 50 fills (A-S strategy)", style=dict(
                fontSize="10px", color=TEXT_DIM, textTransform="uppercase",
                letterSpacing="0.1em", margin="16px 0 8px 0")),
            html.Div(table, style=dict(overflowX="auto")),
        ], style=dict(padding="12px"))

    return empty()


def _dict_to_result(data):
    """Reconstruct a lightweight result object from JSON store."""
    import numpy as np
    from types import SimpleNamespace
    r = SimpleNamespace()
    for k, v in data.items():
        if isinstance(v, list) and k != "trades":
            setattr(r, k, np.array(v))
        else:
            setattr(r, k, v)
    return r


if __name__ == "__main__":
    print("\n  Avellaneda-Stoikov Market Making Simulator")
    print("  ==========================================")
    print("  Open → http://127.0.0.1:8050\n")
    app.run(debug=True, port=8050)
