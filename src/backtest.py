"""
backtest.py — CLI backtest & parameter sweep
============================================
Usage:
  python src/backtest.py               # single run, print summary
  python src/backtest.py --sweep       # parameter sensitivity sweep
  python src/backtest.py --mc 500      # monte carlo with N paths
  python src/backtest.py --export      # save charts to outputs/
"""
 
import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter

from model import ASParams, simulate, run_monte_carlo

# ── Style ────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0D0F14",
    "axes.facecolor":    "#0D0F14",
    "axes.edgecolor":    "#1E2330",
    "axes.labelcolor":   "#94A3B8",
    "axes.titlecolor":   "#E2E8F0",
    "xtick.color":       "#64748B",
    "ytick.color":       "#64748B",
    "grid.color":        "#1E2330",
    "grid.linewidth":    0.5,
    "text.color":        "#E2E8F0",
    "font.family":       "monospace",
    "font.size":         10,
    "legend.framealpha": 0.0,
    "legend.fontsize":   9,
})

BLUE   = "#3B82F6"
GREEN  = "#10B981"
RED    = "#EF4444"
AMBER  = "#F59E0B"
PURPLE = "#8B5CF6"
DIM    = "#64748B"
TEAL   = "#06B6D4"


def print_summary(r):
    print("\n" + "─"*52)
    print("  AVELLANEDA-STOIKOV  ·  Simulation Summary")
    print("─"*52)
    print(f"  {'Param':20s}  {'Value':>12s}")
    print(f"  {'─'*20}  {'─'*12}")
    p = r.params
    for name, val in [
        ("sigma (σ)",       f"{p.sigma:.4f}"),
        ("gamma (γ)",       f"{p.gamma:.4f}"),
        ("k (intensity)",   f"{p.k:.4f}"),
        ("A (arrival)",     f"{p.A:.4f}"),
        ("T (horizon)",     f"{p.T:.4f}"),
        ("S0 (init price)", f"{p.S0:.2f}"),
        ("steps (N)",       f"{int(p.T/p.dt)}"),
    ]:
        print(f"  {name:20s}  {val:>12s}")
    print("─"*52)
    print(f"  {'Metric':20s}  {'A-S':>10s}  {'Naive':>10s}")
    print(f"  {'─'*20}  {'─'*10}  {'─'*10}")
    for name, v_as, v_naive in [
        ("Final PnL",   f"{r.final_pnl_as:+.4f}", f"{r.final_pnl_naive:+.4f}"),
        ("Sharpe ratio",f"{r.sharpe_as:.3f}",      f"{r.sharpe_naive:.3f}"),
        ("Max drawdown",f"{r.max_drawdown_as:.4f}", f"{r.max_drawdown_naive:.4f}"),
        ("Total fills", f"{r.total_trades_as}",     f"{r.total_trades_naive}"),
        ("Inv risk (σ)",f"{r.inventory_risk_as:.3f}", "—"),
        ("Avg spread",  f"{r.avg_spread_as:.5f}",   f"{r.naive_spread:.5f}"),
    ]:
        print(f"  {name:20s}  {v_as:>10s}  {v_naive:>10s}")
    print(f"\n  Edge (A-S − Naive):  {r.edge:+.4f}")
    print("─"*52 + "\n")


def plot_single(r, save_path=None):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Avellaneda-Stoikov Market Making Simulator",
                 fontsize=14, fontweight="bold", color="#E2E8F0", y=0.98)

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)
    ax_price  = fig.add_subplot(gs[0, :])
    ax_inv    = fig.add_subplot(gs[1, 0])
    ax_pnl    = fig.add_subplot(gs[1, 1])
    ax_spread = fig.add_subplot(gs[2, 0])
    ax_fills  = fig.add_subplot(gs[2, 1])

    for ax in [ax_price, ax_inv, ax_pnl, ax_spread, ax_fills]:
        ax.grid(True, alpha=0.4)

    stride = max(1, len(r.times) // 1000)
    t  = r.times[::stride]
    s  = r.mid_prices[::stride]
    rp = r.reservation_prices_as[::stride]
    b  = r.bids_as[::stride]
    a  = r.asks_as[::stride]
    sp = r.spreads_as[::stride]

    # Price chart
    ax_price.fill_between(t, b, a, alpha=0.08, color=TEAL, label="Spread band")
    ax_price.plot(t, a, color=RED, lw=0.8, label="Ask (A-S)")
    ax_price.plot(t, b, color=GREEN, lw=0.8, label="Bid (A-S)")
    ax_price.plot(t, s, color=BLUE, lw=1.2, label="Mid price")
    ax_price.plot(t, rp, color=AMBER, lw=0.8, ls="--", label="Reservation price")

    buys  = [tr for tr in r.trades if tr["strategy"]=="A-S" and tr["side"]=="buy"]
    sells = [tr for tr in r.trades if tr["strategy"]=="A-S" and tr["side"]=="sell"]
    if buys:
        ax_price.scatter([tr["t"] for tr in buys], [tr["price"] for tr in buys],
                         marker="^", color=GREEN, s=25, zorder=5, alpha=0.6)
    if sells:
        ax_price.scatter([tr["t"] for tr in sells], [tr["price"] for tr in sells],
                         marker="v", color=RED, s=25, zorder=5, alpha=0.6)
    ax_price.set_title("Mid Price & Optimal Quotes", color="#E2E8F0")
    ax_price.set_xlabel("Time")
    ax_price.set_ylabel("Price")
    ax_price.legend(loc="upper left", ncol=5)

    # Inventory
    qa = r.inventory_as[::stride]
    qn = r.inventory_naive[::stride]
    ax_inv.axhline(0, color=DIM, lw=0.6)
    ax_inv.plot(t, qn, color=DIM, lw=1, ls="--", label="Naive")
    ax_inv.fill_between(t, 0, qa, alpha=0.2, color=PURPLE)
    ax_inv.plot(t, qa, color=PURPLE, lw=1.2, label="A-S")
    ax_inv.set_title("Inventory Path", color="#E2E8F0")
    ax_inv.set_xlabel("Time"); ax_inv.set_ylabel("Shares (q)")
    ax_inv.legend()

    # PnL
    pa = r.pnl_as[::stride]
    pn = r.pnl_naive[::stride]
    ax_pnl.axhline(0, color=DIM, lw=0.6)
    ax_pnl.plot(t, pn, color=DIM, lw=1, ls="--", label="Naive PnL")
    c = GREEN if pa[-1] >= 0 else RED
    ax_pnl.fill_between(t, 0, pa, alpha=0.15, color=c)
    ax_pnl.plot(t, pa, color=c, lw=1.5, label="A-S PnL")
    ax_pnl.set_title("Cumulative PnL", color="#E2E8F0")
    ax_pnl.set_xlabel("Time"); ax_pnl.set_ylabel("PnL ($)")
    ax_pnl.legend()

    # Spread
    ns = np.full_like(sp, r.naive_spread)
    ax_spread.fill_between(t, 0, sp, alpha=0.15, color=AMBER)
    ax_spread.plot(t, sp, color=AMBER, lw=1.2, label="A-S spread δ*")
    ax_spread.plot(t, ns, color=DIM, lw=1, ls="--", label="Naive spread")
    ax_spread.set_title("Spread Decay Over Time", color="#E2E8F0")
    ax_spread.set_xlabel("Time"); ax_spread.set_ylabel("Spread δ*")
    ax_spread.legend()

    # Fill distribution
    trades_df = pd.DataFrame(r.trades) if r.trades else pd.DataFrame()
    if not trades_df.empty and "side" in trades_df.columns:
        buys_t  = trades_df[trades_df["side"]=="buy"]["t"].values
        sells_t = trades_df[trades_df["side"]=="sell"]["t"].values
        ax_fills.hist(buys_t,  bins=30, color=GREEN, alpha=0.6, label="Buys")
        ax_fills.hist(sells_t, bins=30, color=RED,   alpha=0.6, label="Sells")
    ax_fills.set_title("Fill Time Distribution", color="#E2E8F0")
    ax_fills.set_xlabel("Time"); ax_fills.set_ylabel("Count")
    ax_fills.legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Saved → {save_path}")
    return fig


def plot_mc(mc, save_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Monte Carlo Analysis — 200 Simulation Paths",
                 fontsize=13, fontweight="bold", color="#E2E8F0")

    def h(ax, data, color, label, bins=35):
        ax.hist(data, bins=bins, color=color, alpha=0.7, label=label, edgecolor="none")
        ax.axvline(data.mean(), color=color, lw=1.5, ls="--",
                   label=f"μ = {data.mean():+.4f}")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    ax = axes[0, 0]
    h(ax, mc["pnl_as"],    GREEN,   "A-S PnL")
    h(ax, mc["pnl_naive"], DIM,     "Naive PnL")
    ax.set_title("PnL Distribution", color="#E2E8F0")
    ax.set_xlabel("Final PnL ($)")

    ax = axes[0, 1]
    h(ax, mc["edge"], AMBER, "Edge (A-S − Naive)")
    ax.axvline(0, color=RED, lw=1, ls=":")
    pct = (mc["edge"] > 0).mean() * 100
    ax.set_title(f"Edge Distribution  ({pct:.1f}% of paths positive)", color="#E2E8F0")
    ax.set_xlabel("Edge ($)")

    ax = axes[1, 0]
    h(ax, mc["sharpe_as"],    TEAL, "A-S Sharpe")
    h(ax, mc["sharpe_naive"], DIM,  "Naive Sharpe")
    ax.set_title("Sharpe Ratio Distribution", color="#E2E8F0")
    ax.set_xlabel("Sharpe")

    ax = axes[1, 1]
    h(ax, mc["max_dd_as"],    RED, "A-S Max DD")
    h(ax, mc["max_dd_naive"], DIM, "Naive Max DD")
    ax.set_title("Max Drawdown Distribution", color="#E2E8F0")
    ax.set_xlabel("Drawdown ($)")

    for ax in axes.flat:
        ax.set_facecolor("#0D0F14"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Saved → {save_path}")
    return fig


def run_sweep(p_base: ASParams, export=False):
    """Sensitivity analysis: vary each parameter, measure PnL edge."""
    print("\n  Parameter Sensitivity Sweep")
    print("─"*48)

    sweeps = {
        "sigma": np.linspace(0.005, 0.08, 12),
        "gamma": np.linspace(0.01, 0.50, 12),
        "k":     np.linspace(0.5,  8.0,  12),
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Parameter Sensitivity (A-S Edge vs Naive)",
                 fontsize=13, fontweight="bold", color="#E2E8F0")

    for ax, (param, values) in zip(axes, sweeps.items()):
        edges, pnls = [], []
        for v in values:
            kw = {**p_base.__dict__, param: v, "seed": 42}
            r  = simulate(ASParams(**kw))
            edges.append(r.edge)
            pnls.append(r.final_pnl_as)
            print(f"  {param}={v:.3f}  edge={r.edge:+.4f}  pnl_as={r.final_pnl_as:+.4f}")

        ax.axhline(0, color=DIM, lw=0.8)
        ax.plot(values, pnls,  color=BLUE, lw=1.2, label="A-S PnL", marker="o", ms=4)
        ax.plot(values, edges, color=AMBER, lw=1.2, label="Edge",   marker="s", ms=4)
        ax.set_xlabel(param); ax.set_ylabel("PnL / Edge ($)")
        ax.set_title(f"Varying {param}", color="#E2E8F0")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_facecolor("#0D0F14")

    plt.tight_layout()
    if export:
        path = "outputs/sweep.png"
        os.makedirs("outputs", exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"\n  Saved → {path}")


def main():
    parser = argparse.ArgumentParser(description="A-S Market Making Backtest")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument("--mc",    type=int, default=0, metavar="N",
                        help="Monte Carlo with N paths")
    parser.add_argument("--export", action="store_true",
                        help="Save charts to outputs/")
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--k",     type=float, default=1.50)
    parser.add_argument("--T",     type=float, default=1.00)
    parser.add_argument("--S0",    type=float, default=1.0)
    parser.add_argument("--seed",  type=int,   default=42)
    args = parser.parse_args()

    p = ASParams(sigma=args.sigma, gamma=args.gamma, k=args.k,
                 T=args.T, S0=args.S0, seed=args.seed)

    print(f"\n  Running single simulation (seed={args.seed})...")
    r = simulate(p)
    print_summary(r)

    if args.export:
        plot_single(r, save_path="outputs/simulation.png")

    if args.sweep:
        run_sweep(p, export=args.export)

    if args.mc > 0:
        print(f"\n  Running Monte Carlo ({args.mc} paths)...")
        mc = run_monte_carlo(p, n_paths=args.mc)
        print(f"  A-S  mean PnL:   {mc['pnl_as'].mean():+.4f} ± {mc['pnl_as'].std():.4f}")
        print(f"  Naive mean PnL:  {mc['pnl_naive'].mean():+.4f} ± {mc['pnl_naive'].std():.4f}")
        print(f"  Mean edge:       {mc['edge'].mean():+.4f}")
        print(f"  Edge > 0:        {(mc['edge'] > 0).mean()*100:.1f}% of paths")
        print(f"  A-S Sharpe:      {mc['sharpe_as'].mean():.3f}")
        if args.export:
            plot_mc(mc, save_path="outputs/montecarlo.png")


if __name__ == "__main__":
    main()
