"""Risk management module for Earnings Black Swan strategy.

Implements academic-grade risk controls appropriate for event-driven put options:

1. Kelly criterion for asymmetric payoffs (generalized for options)
2. VIX-regime-aware position scaling (arXiv:2508.16598)
3. Conditional Drawdown-at-Risk (CDaR) scaling (Chekhlov et al. 2005)
4. Concurrent position limits (earnings season clustering)
5. Risk of ruin estimation for asymmetric payoffs (Whelan 2025)
6. Per-trade notional exposure caps

NOT implemented (inappropriate for event-driven puts):
- CPPI portfolio insurance (requires continuous rebalancing)
- Turbulence index (requires intraday portfolio weights)
- Downside risk parity (single-strategy system)
- Full covariance Kelly (insufficient data for N×N estimation)

References:
  - Kelly (1956), "A New Interpretation of Information Rate"
  - Thorp (2006), "The Kelly Criterion in Blackjack, Sports Betting and the Stock Market"
  - Vince (1990), "Portfolio Management Formulas"
  - Chekhlov, Uryasev, Zabarankin (2005), "Drawdown Measure in Portfolio Optimization"
  - Whelan (2025), "Ruin Probabilities for Strategies with Asymmetric Risk"
  - arXiv:2508.16598, "Sizing the Risk: Kelly, VIX, and Hybrid Approaches"
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ============================================================================
# Kelly criterion for asymmetric binary payoffs
# ============================================================================

def kelly_fraction_asymmetric(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = 0.25,
) -> float:
    """Compute fractional Kelly for asymmetric payoffs.

    Generalized Kelly for binary bets where win and loss sizes differ:
        f* = (p * b - q) / b
    where:
        p = win probability
        q = 1 - p
        b = avg_win / avg_loss (odds ratio)

    For EBS with ~44% win rate, +265% avg win, -97% avg loss:
        b = 2.65 / 0.97 = 2.73
        f* = (0.44 * 2.73 - 0.56) / 2.73 = 0.235
        quarter-Kelly = 0.059 (~6% of capital per trade)

    We use fractional Kelly (default 25%) because:
    1. Parameter estimation error (Lopez de Prado 2018)
    2. Non-independent trials (earnings cluster in seasons)
    3. Real-world execution slippage not in model

    Args:
        win_rate: Probability of winning (0 to 1)
        avg_win: Average winning return as positive multiple (e.g., 2.65 for +265%)
        avg_loss: Average losing return as positive multiple (e.g., 0.97 for -97%)
        fraction: Kelly fraction (0.25 = quarter-Kelly, conservative)

    Returns:
        Optimal bet fraction (0 to 1). Returns 0 if no edge.
    """
    if win_rate <= 0 or avg_win <= 0 or avg_loss <= 0:
        return 0.0

    p = win_rate
    q = 1.0 - p
    b = avg_win / avg_loss  # odds ratio

    full_kelly = (p * b - q) / b

    if full_kelly <= 0:
        return 0.0  # no edge — don't bet

    return min(full_kelly * fraction, 0.10)  # cap at 10% regardless


# ============================================================================
# VIX-regime-aware sizing
# ============================================================================

@dataclass
class VIXRegime:
    """VIX-based market regime for position sizing.

    Based on arXiv:2508.16598 — "Sizing the Risk: Kelly, VIX, and Hybrid
    Approaches in Put-Writing on Index Options":
    - Low VIX (<15): calm markets, puts are cheap, reduce size (less edge)
    - Normal VIX (15-25): standard sizing
    - High VIX (25-35): elevated fear, increase size (more mispricing)
    - Extreme VIX (>35): crisis, reduce size (correlated selloffs, not earnings-specific)
    """
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


def vix_regime_scale(vix_level: float) -> float:
    """Scale position size based on VIX regime.

    Rationale (from arXiv:2508.16598, hybrid Kelly-VIX approach):
    - Low VIX: options are cheap but so is edge (0.7x)
    - Normal: baseline (1.0x)
    - High VIX: elevated mispricing, bigger positions (1.3x)
    - Extreme: systemic risk, everything sells off together (0.5x)

    Returns scaling factor (0.5 to 1.3).
    """
    if vix_level < 15:
        return 0.7   # calm — puts cheap but less mispricing
    elif vix_level < 25:
        return 1.0   # normal
    elif vix_level < 35:
        return 1.3   # elevated — more earnings surprises, fatter tails
    else:
        return 0.5   # crisis — correlated selloff, not earnings-specific


# ============================================================================
# Conditional Drawdown-at-Risk (CDaR) scaling
# ============================================================================

def cdar_position_scale(
    equity_curve: list[float],
    current_equity: float,
    dd_threshold_soft: float = 0.10,
    dd_threshold_hard: float = 0.30,
    floor_scale: float = 0.15,
) -> float:
    """CDaR-inspired drawdown scaling for event-driven strategies.

    More aggressive than the previous 25% threshold, based on institutional
    practice (Chekhlov et al. 2005, UPenn drawdown optimization):
    - Soft threshold (10%): begin reducing exposure linearly
    - Hard threshold (30%): minimum exposure (floor)

    This is a simplified CDaR — full CDaR requires LP optimization.
    For event-driven strategies, this heuristic is sufficient because
    positions are discrete (not continuously rebalanced).

    Args:
        equity_curve: Historical equity values (for peak tracking)
        current_equity: Current portfolio value
        dd_threshold_soft: Start reducing at this drawdown (10%)
        dd_threshold_hard: Maximum drawdown before floor (30%)
        floor_scale: Minimum position scale (15% of normal)

    Returns:
        Position scale factor (floor_scale to 1.0)
    """
    if not equity_curve:
        return 1.0

    peak = max(equity_curve)
    if peak <= 0:
        return floor_scale

    current_dd = (current_equity - peak) / peak  # negative when in drawdown

    if current_dd >= -dd_threshold_soft:
        return 1.0  # above soft threshold — full sizing

    if current_dd <= -dd_threshold_hard:
        return floor_scale  # below hard threshold — minimum

    # Linear interpolation between soft and hard
    dd_range = dd_threshold_hard - dd_threshold_soft
    dd_depth = abs(current_dd) - dd_threshold_soft
    return max(floor_scale, 1.0 - (1.0 - floor_scale) * (dd_depth / dd_range))


# ============================================================================
# Concurrent position limits (earnings season clustering)
# ============================================================================

def check_concurrent_positions(
    active_positions: list[dict],
    max_concurrent: int = 5,
    max_sector_concentration: int = 3,
) -> dict:
    """Check concurrent position limits for earnings clustering risk.

    During earnings season (Jan/Apr/Jul/Oct), 20+ companies report
    in the same week. If they're correlated (same sector, same macro
    sensitivity), a single factor shock wipes out all positions.

    Limits:
    - Max 5 concurrent put positions (default)
    - Max 3 in the same sector
    - If at limit, skip new trades until positions expire

    Args:
        active_positions: List of currently held position dicts
            with 'ticker', 'sector', 'expiry_date'
        max_concurrent: Maximum total concurrent positions
        max_sector_concentration: Maximum positions per sector

    Returns:
        dict with 'can_trade', 'reason', 'n_active', 'sector_counts'
    """
    n_active = len(active_positions)

    sector_counts = {}
    for pos in active_positions:
        sector = pos.get("sector", "unknown")
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    if n_active >= max_concurrent:
        return {
            "can_trade": False,
            "reason": f"At max concurrent positions ({max_concurrent})",
            "n_active": n_active,
            "sector_counts": sector_counts,
        }

    max_sector = max(sector_counts.values()) if sector_counts else 0
    if max_sector >= max_sector_concentration:
        top_sector = max(sector_counts, key=sector_counts.get)
        return {
            "can_trade": False,
            "reason": f"Sector {top_sector} at max ({max_sector_concentration})",
            "n_active": n_active,
            "sector_counts": sector_counts,
        }

    return {
        "can_trade": True,
        "reason": "OK",
        "n_active": n_active,
        "sector_counts": sector_counts,
    }


# ============================================================================
# Risk of ruin for asymmetric payoffs
# ============================================================================

def risk_of_ruin(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    bet_fraction: float,
    ruin_threshold: float = 0.50,
    n_simulations: int = 50000,
    n_trades: int = 500,
    seed: int = 42,
) -> dict:
    """Monte Carlo risk of ruin estimation for asymmetric payoffs.

    Standard RoR formulas assume symmetric payoffs (p > 0.5).
    For asymmetric payoffs (44% win rate, +265% vs -97%), we use
    Monte Carlo simulation per Whelan (2025).

    Simulates n_trades sequential bets and counts how many paths
    hit the ruin threshold (e.g., lose 50% of starting capital).

    Args:
        win_rate: Win probability
        avg_win: Average win as positive multiple of bet
        avg_loss: Average loss as positive multiple of bet
        bet_fraction: Fraction of capital bet per trade
        ruin_threshold: Fraction of capital loss that constitutes ruin
        n_simulations: Number of Monte Carlo paths
        n_trades: Trades per simulation path
        seed: Random seed

    Returns:
        dict with 'ruin_probability', 'median_final_capital', 'percentiles'
    """
    rng = np.random.RandomState(seed)

    ruin_count = 0
    final_capitals = []

    for _ in range(n_simulations):
        capital = 1.0
        ruined = False

        for _ in range(n_trades):
            if capital <= (1.0 - ruin_threshold):
                ruined = True
                break

            bet = capital * bet_fraction
            if rng.random() < win_rate:
                capital += bet * avg_win
            else:
                capital -= bet * avg_loss

        if ruined:
            ruin_count += 1
        final_capitals.append(capital)

    final_arr = np.array(final_capitals)

    return {
        "ruin_probability": round(ruin_count / n_simulations, 4),
        "ruin_threshold": ruin_threshold,
        "n_simulations": n_simulations,
        "n_trades_per_path": n_trades,
        "bet_fraction": bet_fraction,
        "median_final_capital": round(float(np.median(final_arr)), 4),
        "percentile_5": round(float(np.percentile(final_arr, 5)), 4),
        "percentile_25": round(float(np.percentile(final_arr, 25)), 4),
        "percentile_75": round(float(np.percentile(final_arr, 75)), 4),
        "percentile_95": round(float(np.percentile(final_arr, 95)), 4),
    }


# ============================================================================
# Composite risk sizer — combines all signals
# ============================================================================

def compute_position_size(
    capital: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    stock_vol: float,
    equity_curve: list[float],
    vix_level: Optional[float] = None,
    kelly_fraction: float = 0.25,
    target_vol: float = 50.0,
    max_position_pct: float = 0.05,
) -> dict:
    """Compute risk-optimal position size combining all signals.

    Layers:
    1. Kelly base: dynamic sizing from strategy statistics
    2. Vol adjustment: scale inversely to stock volatility
    3. CDaR scaling: reduce in drawdowns (10% soft, 30% hard)
    4. VIX regime: scale by market fear level
    5. Cap: never exceed max_position_pct

    Returns dict with position_pct and breakdown of each scaling factor.
    """
    # 1. Kelly base
    kelly_pct = kelly_fraction_asymmetric(win_rate, avg_win, avg_loss, kelly_fraction)

    # 2. Vol adjustment
    stock_vol = max(stock_vol or 30.0, 10.0)
    vol_scale = min(2.0, target_vol / stock_vol)

    # 3. CDaR drawdown scaling
    dd_scale = cdar_position_scale(equity_curve, capital)

    # 4. VIX regime
    vix_scale = vix_regime_scale(vix_level) if vix_level is not None else 1.0

    # Combine
    raw_pct = kelly_pct * vol_scale * dd_scale * vix_scale
    final_pct = min(raw_pct, max_position_pct)
    final_pct = max(final_pct, 0.0)

    return {
        "position_pct": round(final_pct, 6),
        "kelly_base_pct": round(kelly_pct, 6),
        "vol_scale": round(vol_scale, 4),
        "dd_scale": round(dd_scale, 4),
        "vix_scale": round(vix_scale, 4),
        "capped": raw_pct > max_position_pct,
    }
