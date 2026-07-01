"""
Avellaneda-Stoikov Market Making Model
======================================
Implementation of the closed-form solution from:
  Avellaneda, M. & Stoikov, S. (2008).
  "High-frequency trading in a limit order book."
  Quantitative Finance, 8(3), 217-224.

The MM solves a stochastic control problem (Hamilton-Jacobi-Bellman equation)
to find optimal bid/ask quotes that balance spread capture vs. inventory risk.

Key equations:
  Reservation price:  r(s,q,t) = s - q * gamma * sigma^2 * (T - t)
  Optimal spread:     delta* = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)
  Bid:                b* = r - delta*/2
  Ask:                a* = r + delta*/2
  Fill intensity:     lambda(delta) = A * exp(-k * delta)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ASParams:
    """Parameters for the Avellaneda-Stoikov model."""
    sigma: float = 2.00       # Asset volatility per unit time (use normalized S0=1; sigma~2 is regime where A-S clearly outperforms)
    gamma: float = 0.10       # MM risk aversion coefficient
    k: float = 1.50           # Order arrival intensity decay (LOB depth proxy)
    A: float = 140.0          # Base arrival rate of market orders (A-S paper: ~140/s)
    T: float = 1.00           # Trading horizon (normalized)
    dt: float = 0.001         # Timestep
    S0: float = 1.0           # Initial mid price (normalized; all prices in same units as sigma)
    q0: int = 0               # Initial inventory
    max_inventory: int = 5    # Inventory limit (|q| <= max_inventory)
    seed: Optional[int] = None


@dataclass
class SimResult:
    """Full simulation output."""
    times: np.ndarray
    mid_prices: np.ndarray
    reservation_prices_as: np.ndarray
    bids_as: np.ndarray
    asks_as: np.ndarray
    spreads_as: np.ndarray
    inventory_as: np.ndarray
    cash_as: np.ndarray
    pnl_as: np.ndarray

    bids_naive: np.ndarray
    asks_naive: np.ndarray
    inventory_naive: np.ndarray
    cash_naive: np.ndarray
    pnl_naive: np.ndarray

    trades: list = field(default_factory=list)
    params: ASParams = field(default_factory=ASParams)

    # Summary stats (filled post-sim)
    final_pnl_as: float = 0.0
    final_pnl_naive: float = 0.0
    edge: float = 0.0
    sharpe_as: float = 0.0
    sharpe_naive: float = 0.0
    total_trades_as: int = 0
    total_trades_naive: int = 0
    max_drawdown_as: float = 0.0
    max_drawdown_naive: float = 0.0
    avg_spread_as: float = 0.0
    naive_spread: float = 0.0
    inventory_risk_as: float = 0.0


def compute_optimal_quotes(s: float, q: int, t: float, p: ASParams):
    """
    Compute A-S reservation price, optimal spread, bid, ask at a given state.

    Returns: (reservation_price, optimal_spread, bid, ask)
    """
    tau = p.T - t
    if tau <= 0:
        tau = 1e-9

    # Reservation price: skewed mid based on inventory
    r = s - q * p.gamma * p.sigma**2 * tau

    # Optimal half-spread from HJB solution
    delta = (p.gamma * p.sigma**2 * tau) + (2.0 / p.gamma) * np.log(1.0 + p.gamma / p.k)
    delta = max(delta, 1e-6)

    bid = r - delta / 2.0
    ask = r + delta / 2.0

    return r, delta, bid, ask


def compute_naive_spread(p: ASParams) -> float:
    """Fixed symmetric spread — naive benchmark (no inventory awareness)."""
    delta = (p.gamma * p.sigma**2 * p.T) + (2.0 / p.gamma) * np.log(1.0 + p.gamma / p.k)
    return max(delta, 1e-6)


def fill_probability(delta_half: float, p: ASParams, dt: float) -> float:
    """Probability of fill at this half-spread in interval dt."""
    lam = p.A * np.exp(-p.k * delta_half)
    return 1.0 - np.exp(-lam * dt)   # Exact Poisson probability


def simulate(p: ASParams) -> SimResult:
    """
    Run a full A-S simulation alongside a naive symmetric quoting strategy.
    Both strategies face the same mid-price path and same random order arrivals.
    """
    rng = np.random.default_rng(p.seed)
    N = int(p.T / p.dt)
    dt = p.T / N

    # --- Mid price path (GBM with zero drift) ---
    dW = rng.standard_normal(N) * p.sigma * np.sqrt(dt)
    S = np.empty(N + 1)
    S[0] = p.S0
    for i in range(N):
        S[i + 1] = S[i] + dW[i]
        S[i + 1] = max(S[i + 1], 0.01)

    times = np.linspace(0, p.T, N + 1)

    # --- State arrays ---
    r_arr     = np.empty(N + 1)
    bid_as    = np.empty(N + 1)
    ask_as    = np.empty(N + 1)
    spread_as = np.empty(N + 1)
    q_as      = np.empty(N + 1, dtype=int)
    cash_as   = np.empty(N + 1)
    pnl_as    = np.empty(N + 1)

    bid_naive  = np.empty(N + 1)
    ask_naive  = np.empty(N + 1)
    q_naive    = np.empty(N + 1, dtype=int)
    cash_naive = np.empty(N + 1)
    pnl_naive  = np.empty(N + 1)

    # --- Initial state ---
    q_as[0]      = p.q0
    cash_as[0]   = 0.0
    q_naive[0]   = p.q0
    cash_naive[0] = 0.0
    pnl_as[0]   = 0.0
    pnl_naive[0] = 0.0

    naive_spread = compute_naive_spread(p)
    naive_half   = naive_spread / 2.0

    trades = []

    for i in range(N):
        t  = times[i]
        s  = S[i]
        qa = int(q_as[i])
        qn = int(q_naive[i])

        # --- A-S quotes ---
        r, delta, b_as, a_as = compute_optimal_quotes(s, qa, t, p)

        r_arr[i]     = r
        bid_as[i]    = b_as
        ask_as[i]    = a_as
        spread_as[i] = delta

        # Naive always quotes symmetrically around mid (no inventory skew)
        bid_naive[i] = s - naive_half
        ask_naive[i] = s + naive_half  # always centered at S, never at reservation price

        # --- Fill simulation (Poisson arrivals) ---
        # Draw random numbers unconditionally to keep RNG streams independent.
        half_bid_as  = s - b_as
        half_ask_as  = a_as - s

        u_as_ask = rng.random()
        u_as_bid = rng.random()
        u_n_ask  = rng.random()
        u_n_bid  = rng.random()

        # A-S fills
        new_q_as    = qa
        new_cash_as = cash_as[i]

        # Sell to buyer hitting our ask (we sell, inventory decreases)
        can_sell_as = new_q_as > -p.max_inventory
        if can_sell_as and u_as_ask < fill_probability(half_ask_as, p, dt):
            new_cash_as += a_as
            new_q_as    -= 1
            trades.append(dict(step=i, t=round(t, 4), side='sell',
                               price=round(a_as, 4), inv=new_q_as,
                               strategy='A-S'))

        # Buy from seller hitting our bid (we buy, inventory increases)
        can_buy_as = new_q_as < p.max_inventory
        if can_buy_as and u_as_bid < fill_probability(half_bid_as, p, dt):
            new_cash_as -= b_as
            new_q_as    += 1
            trades.append(dict(step=i, t=round(t, 4), side='buy',
                               price=round(b_as, 4), inv=new_q_as,
                               strategy='A-S'))

        q_as[i + 1]    = new_q_as
        cash_as[i + 1] = new_cash_as

        # Naive fills
        new_q_naive    = qn
        new_cash_naive = cash_naive[i]
        p_n = fill_probability(naive_half, p, dt)

        can_sell_n = new_q_naive > -p.max_inventory
        if can_sell_n and u_n_ask < p_n:
            new_cash_naive += ask_naive[i]
            new_q_naive    -= 1

        can_buy_n = new_q_naive < p.max_inventory
        if can_buy_n and u_n_bid < p_n:
            new_cash_naive -= bid_naive[i]
            new_q_naive    += 1

        q_naive[i + 1]    = new_q_naive
        cash_naive[i + 1] = new_cash_naive

        # Mark-to-market PnL (cash + inventory valued at next mid)
        pnl_as[i + 1]    = new_cash_as    + new_q_as    * S[i + 1]
        pnl_naive[i + 1] = new_cash_naive + new_q_naive * S[i + 1]

    # Fill terminal quotes
    r_final, d_final, b_final, a_final = compute_optimal_quotes(S[N], int(q_as[N]), p.T, p)
    r_arr[N]     = r_final
    bid_as[N]    = b_final
    ask_as[N]    = a_final
    spread_as[N] = d_final
    bid_naive[N] = S[N] - naive_half
    ask_naive[N] = S[N] + naive_half

    # --- Summary stats ---
    def max_drawdown(pnl):
        peak = np.maximum.accumulate(pnl)
        dd   = pnl - peak
        return float(np.min(dd))

    def sharpe(pnl, dt):
        rets = np.diff(pnl)
        if rets.std() < 1e-10:
            return 0.0
        return float(rets.mean() / rets.std() * np.sqrt(1.0 / dt))

    as_trades   = [tr for tr in trades if tr['strategy'] == 'A-S']

    res = SimResult(
        times=times, mid_prices=S,
        reservation_prices_as=r_arr,
        bids_as=bid_as, asks_as=ask_as, spreads_as=spread_as,
        inventory_as=q_as, cash_as=cash_as, pnl_as=pnl_as,
        bids_naive=bid_naive, asks_naive=ask_naive,
        inventory_naive=q_naive, cash_naive=cash_naive, pnl_naive=pnl_naive,
        trades=trades, params=p,
    )

    res.final_pnl_as     = float(pnl_as[-1])
    res.final_pnl_naive  = float(pnl_naive[-1])
    res.edge             = res.final_pnl_as - res.final_pnl_naive
    res.sharpe_as        = sharpe(pnl_as, dt)
    res.sharpe_naive     = sharpe(pnl_naive, dt)
    res.total_trades_as  = len(as_trades)
    res.total_trades_naive = int(np.sum(np.abs(np.diff(q_naive))))
    res.max_drawdown_as  = max_drawdown(pnl_as)
    res.max_drawdown_naive = max_drawdown(pnl_naive)
    res.avg_spread_as    = float(np.mean(spread_as))
    res.naive_spread     = naive_spread
    res.inventory_risk_as = float(np.std(q_as.astype(float)))

    return res


def run_monte_carlo(p: ASParams, n_paths: int = 200) -> dict:
    """
    Run multiple simulations to get distribution of outcomes.
    Returns dict of arrays for final PnL, Sharpe, edge, etc.
    """
    results = {
        'pnl_as': [], 'pnl_naive': [], 'edge': [],
        'sharpe_as': [], 'sharpe_naive': [],
        'max_dd_as': [], 'max_dd_naive': [],
        'trades_as': [], 'inv_risk_as': [],
    }
    for i in range(n_paths):
        p_i = ASParams(**{**p.__dict__, 'seed': i})
        r   = simulate(p_i)
        results['pnl_as'].append(r.final_pnl_as)
        results['pnl_naive'].append(r.final_pnl_naive)
        results['edge'].append(r.edge)
        results['sharpe_as'].append(r.sharpe_as)
        results['sharpe_naive'].append(r.sharpe_naive)
        results['max_dd_as'].append(r.max_drawdown_as)
        results['max_dd_naive'].append(r.max_drawdown_naive)
        results['trades_as'].append(r.total_trades_as)
        results['inv_risk_as'].append(r.inventory_risk_as)

    return {k: np.array(v) for k, v in results.items()}
