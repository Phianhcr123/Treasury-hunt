"""Option pricing: Black-Scholes, CRR binomial (American, discrete dividends) and a
real-world "drift tree" that embeds the month-end drift over a short holding window.

Dividends are passed as (time_in_years_from_now, cash_amount) pairs and handled with the
escrowed-dividend model: the tree is built on the stock price minus the present value of
dividends paid before expiry, and early exercise compares against the full cum-dividend price.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, erf, exp, log, sqrt

import numpy as np

Dividends = tuple[tuple[float, float], ...]


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _pv_dividends(divs: Dividends, r: float, t0: float, t1: float) -> float:
    return sum(a * exp(-r * (t - t0)) for t, a in divs if t0 < t <= t1)


def black_scholes(S: float, K: float, T: float, r: float, sigma: float, call: bool = True, dividends: Dividends = ()) -> float:
    """European price; discrete dividends are removed from the spot as their present value."""
    S = S - _pv_dividends(dividends, r, 0.0, T)
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if call else max(K - S, 0.0)
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if call:
        return S * _norm_cdf(d1) - K * exp(-r * T) * _norm_cdf(d2)
    return K * exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def black_scholes_delta(S: float, K: float, T: float, r: float, sigma: float, call: bool = True, dividends: Dividends = ()) -> float:
    S_ex = S - _pv_dividends(dividends, r, 0.0, T)
    if T <= 0 or sigma <= 0:
        return float(S_ex > K) if call else -float(S_ex < K)
    d1 = (log(S_ex / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    return _norm_cdf(d1) if call else _norm_cdf(d1) - 1.0


def crr_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    call: bool = True,
    american: bool = True,
    steps: int = 200,
    dividends: Dividends = (),
) -> float:
    """Cox-Ross-Rubinstein binomial price under the risk-neutral measure."""
    if T <= 0:
        return max(S - K, 0.0) if call else max(K - S, 0.0)
    divs = tuple((t, a) for t, a in dividends if 0 < t <= T)
    sign = 1.0 if call else -1.0
    dt = T / steps
    u = exp(sigma * sqrt(dt))
    d = 1.0 / u
    disc = exp(-r * dt)
    p = (exp(r * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError("Unstable tree: increase steps or check inputs.")

    s0 = S - _pv_dividends(divs, r, 0.0, T)
    j = np.arange(steps + 1)
    v = np.maximum(sign * (s0 * u ** (steps - j) * d**j - K), 0.0)
    for i in range(steps - 1, -1, -1):
        v = disc * (p * v[:-1] + (1.0 - p) * v[1:])
        if american:
            jj = j[: i + 1]
            spot = s0 * u ** (i - jj) * d**jj + _pv_dividends(divs, r, i * dt, T)
            v = np.maximum(v, sign * (spot - K))
    return float(v[0])


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    call: bool = True,
    dividends: Dividends = (),
    steps: int = 100,
    lo: float = 0.005,
    hi: float = 3.0,
    tol: float = 1e-5,
) -> float:
    """Volatility that makes the American CRR price match `price` (bisection). NaN if out of range."""
    f = lambda s: crr_price(S, K, T, r, s, call, True, steps, dividends) - price  # noqa: E731
    f_lo, f_hi = f(lo), f(hi)
    if f_lo > 0 or f_hi < 0:
        return float("nan")
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


@dataclass
class DriftTreeResult:
    price: float
    """Fair (risk-neutral) American price today."""
    expected_exit_value: float
    """Expected option value at the end of the holding window under the drift (real-world) tree."""
    rn_check: float
    """Discounted risk-neutral expectation of the exit value; equals `price` up to tree error."""
    expected_return: float
    """expected_exit_value / price - 1, before trading costs."""
    return_sd: float
    prob_profit: float
    delta: float
    elasticity: float
    """Percent change in the option for a 1% move in the underlying (delta * S / price)."""


def drift_tree(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    horizon: float,
    excess_drift: float,
    call: bool = True,
    dividends: Dividends = (),
    steps_per_day: int = 8,
    value_steps: int = 150,
) -> DriftTreeResult:
    """Value an option over a short holding window using a tree with a temporary drift.

    The first `horizon` years of the tree move with real-world probability
    p = (exp((r + excess_drift) * dt) - d) / (u - d), i.e. the month-end drift is embedded in
    the path. At the end of the window the option is marked at its risk-neutral American
    value for the remaining life, because that is what the market will pay when you sell.
    """
    if not 0 < horizon < T:
        raise ValueError("horizon must be between 0 and T")
    n = max(int(round(horizon * 252 * steps_per_day)), 2)
    dt = horizon / n
    u = exp(sigma * sqrt(dt))
    d = 1.0 / u
    p_q = (exp(r * dt) - d) / (u - d)
    p_p = (exp((r + excess_drift) * dt) - d) / (u - d)
    if not (0 < p_q < 1 and 0 < p_p < 1):
        raise ValueError("Drift too large for this step size; increase steps_per_day.")

    window_divs = tuple((t, a) for t, a in dividends if 0 < t <= horizon)
    later_divs = tuple((t - horizon, a) for t, a in dividends if horizon < t <= T)
    s0 = S - _pv_dividends(window_divs, r, 0.0, horizon)

    j = np.arange(n + 1)
    s_exit = s0 * u ** (n - j) * d**j
    v_exit = np.array([crr_price(s, K, T - horizon, r, sigma, call, True, value_steps, later_divs) for s in s_exit])
    binom = np.array([comb(n, k) for k in j], dtype=float)
    w_p = binom * p_p ** (n - j) * (1 - p_p) ** j
    w_q = binom * p_q ** (n - j) * (1 - p_q) ** j

    price = crr_price(S, K, T, r, sigma, call, True, max(value_steps, 200), dividends)
    e_p = float(w_p @ v_exit)
    rets = v_exit / price - 1.0
    mean_ret = e_p / price - 1.0
    delta = black_scholes_delta(S, K, T, r, sigma, call, dividends)
    return DriftTreeResult(
        price=price,
        expected_exit_value=e_p,
        rn_check=float(exp(-r * horizon) * (w_q @ v_exit)),
        expected_return=mean_ret,
        return_sd=float(np.sqrt(w_p @ (rets - mean_ret) ** 2)),
        prob_profit=float(w_p[rets > 0].sum()),
        delta=delta,
        elasticity=delta * S / price if price > 0 else float("nan"),
    )
