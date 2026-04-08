"""
EBS Live Signal Generator — Earnings Black Swan put option signals.

Generates buy-put signals for upcoming earnings announcements using a
walk-forward trained logistic regression model on 10 price + earnings
surprise features (Sloan 1996, Beneish 1999, Moskowitz et al. 2012).

Pipeline:
  1. Fetch upcoming earnings calendar (yfinance)
  2. For each upcoming event: compute 10 PIT-safe features
  3. Train/retrain model on all historical events (expanding window)
  4. Predict P(catastrophic drop > 8%) for each upcoming event
  5. Filter signals above threshold (default 0.20)
  6. Size positions via Kelly + CDaR + VIX scaling

Academic basis:
  - Ball & Brown (1968): PEAD (post-earnings announcement drift)
  - Bernard & Thomas (1989): Earnings surprise predictability
  - Moskowitz et al. (2012): Momentum features
  - Lopez de Prado (2018): Walk-forward CV, quarter-Kelly
"""
from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ── Feature names (must match price_earnings_features.py) ──
FEATURE_NAMES = [
    "prior_surprise_pct", "surprise_streak", "surprise_trend",
    "avg_surprise_magnitude", "runup_6m_pct", "runup_3m_pct",
    "runup_1m_pct", "dist_from_52w_high_pct", "realized_vol_60d",
    "vol_regime_change",
]

# ── Default strategy parameters ──
PRED_THRESHOLD = 0.20   # Minimum P(catastrophic) to trigger signal
CAR_THRESHOLD = -8.0    # Definition of catastrophic: -8% in [-1, +2] days
MIN_FEATURES = 5        # Minimum non-zero features per event
MIN_TRAIN_EVENTS = 100  # Minimum events before model can predict

# ── Backtest-validated strategy stats (from earnings_backtest_v2.json) ──
# Strategy A: Model prob >= 0.20
STRATEGY_STATS = {
    "win_rate": 0.442,
    "avg_win": 2.65,    # +265% on winning puts
    "avg_loss": 0.966,  # -96.6% on losing puts
    "n_trades": 886,
    "expectancy": 0.635,
}


@dataclass
class EBSSignal:
    """A single earnings put signal."""
    ticker: str
    earnings_date: str          # YYYY-MM-DD
    pred_prob: float            # P(catastrophic drop)
    position_pct: float         # Recommended allocation (0-5% of capital)
    features: dict              # Feature values for audit
    sector: str = "unknown"
    realized_vol: float = 30.0  # For risk sizing


def compute_features_from_yfinance(
    ticker: str,
    earnings_date: pd.Timestamp,
) -> Optional[dict]:
    """Compute 10 features for a single ticker/date using yfinance.

    Returns dict with feature values, or None if insufficient data.
    """
    import yfinance as yf

    try:
        t = yf.Ticker(ticker)

        # Get earnings history
        try:
            earnings_dates = t.get_earnings_dates(limit=20)
            if earnings_dates is None or earnings_dates.empty:
                return None
        except Exception:
            return None

        # Build earnings history DataFrame
        earnings_rows = []
        for idx, row in earnings_dates.iterrows():
            date = idx.tz_localize(None) if idx.tzinfo else idx
            surprise = row.get("Surprise(%)")
            if pd.notna(surprise):
                earnings_rows.append({
                    "earnings_date": date,
                    "surprise_pct": float(surprise),
                })
        if len(earnings_rows) < 2:
            return None
        earnings_hist = pd.DataFrame(earnings_rows).sort_values("earnings_date")

        # Get price history (1.5 years for 252-day features)
        hist = t.history(period="2y", auto_adjust=True)
        if hist.empty or len(hist) < 126:
            return None
        hist.index = hist.index.tz_localize(None)
        prices = hist["Close"]

        # Compute features
        features = {}
        n_avail = 0

        # Prior earnings
        prior = earnings_hist[earnings_hist["earnings_date"] < earnings_date]
        prior = prior.sort_values("earnings_date")

        if len(prior) >= 1:
            features["prior_surprise_pct"] = float(prior.iloc[-1]["surprise_pct"])
            n_avail += 1

        if len(prior) >= 2:
            streak = 0
            for _, r in prior.iloc[::-1].iterrows():
                s = r["surprise_pct"]
                if s < 0:
                    if streak <= 0: streak -= 1
                    else: break
                elif s > 0:
                    if streak >= 0: streak += 1
                    else: break
                else: break
            features["surprise_streak"] = float(streak)
            n_avail += 1

        if len(prior) >= 4:
            last4 = prior.tail(4)["surprise_pct"].dropna()
            if len(last4) >= 3:
                x = np.arange(len(last4))
                slope = np.polyfit(x, last4.values, 1)[0]
                features["surprise_trend"] = round(float(slope), 4)
                n_avail += 1
            features["avg_surprise_magnitude"] = round(float(last4.abs().mean()), 4)
            n_avail += 1

        # Price features
        pre = prices[prices.index < earnings_date]
        if len(pre) >= 126:
            features["runup_6m_pct"] = round(float((pre.iloc[-1] / pre.iloc[-126] - 1) * 100), 2)
            n_avail += 1
        if len(pre) >= 63:
            features["runup_3m_pct"] = round(float((pre.iloc[-1] / pre.iloc[-63] - 1) * 100), 2)
            n_avail += 1
        if len(pre) >= 21:
            features["runup_1m_pct"] = round(float((pre.iloc[-1] / pre.iloc[-21] - 1) * 100), 2)
            n_avail += 1
        if len(pre) >= 252:
            high = float(pre.iloc[-252:].max())
            if high > 0:
                features["dist_from_52w_high_pct"] = round(float((pre.iloc[-1] - high) / high * 100), 2)
                n_avail += 1

        # Vol features
        if len(pre) >= 61:
            rets = pre.pct_change().dropna()
            if len(rets) >= 60:
                vol60 = float(rets.iloc[-60:].std() * np.sqrt(252) * 100)
                features["realized_vol_60d"] = round(vol60, 2)
                n_avail += 1
                if len(rets) >= 126:
                    vol6m = float(rets.iloc[-126:-66].std() * np.sqrt(252) * 100)
                    if vol6m > 0:
                        features["vol_regime_change"] = round(vol60 / vol6m, 3)
                        n_avail += 1

        if n_avail < MIN_FEATURES:
            return None

        features["_n_available"] = n_avail
        return features

    except Exception as e:
        logger.warning(f"Feature computation failed for {ticker}: {e}")
        return None


def features_to_array(features: dict) -> np.ndarray:
    """Convert feature dict to 10-element array."""
    return np.array([features.get(name, 0.0) for name in FEATURE_NAMES], dtype=np.float64)


class EBSModel:
    """Logistic regression model for earnings catastrophe prediction.

    Wraps sklearn LogisticRegression + StandardScaler with
    save/load for persistence between runs.
    """

    def __init__(self):
        self.model: Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.n_train_events: int = 0
        self.train_date: Optional[str] = None

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Train on historical events.

        Args:
            X: (n_events, 10) feature matrix
            y: (n_events,) binary labels (1=catastrophic, 0=not)

        Returns:
            Training metrics dict.
        """
        # Filter zero-variance features
        variances = np.var(X, axis=0)
        good_cols = [i for i, v in enumerate(variances) if v > 0.001]

        if len(good_cols) < 3:
            raise ValueError(f"Only {len(good_cols)} features with variance > 0.001")

        X_filtered = X[:, good_cols]

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_filtered)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        self.model = LogisticRegression(C=1.0, penalty="l2", max_iter=2000)
        self.model.fit(X_scaled, y)

        self.n_train_events = len(y)
        self.train_date = datetime.now().strftime("%Y-%m-%d")
        self._good_cols = good_cols

        # Metrics
        preds = self.model.predict_proba(X_scaled)[:, 1]
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(y, preds)
        except ValueError:
            auc = 0.5

        return {
            "n_events": int(len(y)),
            "n_positive": int(y.sum()),
            "base_rate": round(float(y.mean()), 4),
            "auc": round(auc, 4),
            "n_features_used": len(good_cols),
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict P(catastrophic) for new events."""
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not trained. Call train() first.")

        X_filtered = X[:, self._good_cols] if hasattr(self, '_good_cols') else X
        X_scaled = self.scaler.transform(X_filtered)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return self.model.predict_proba(X_scaled)[:, 1]

    def save(self, path: Path):
        """Persist model + scaler to disk."""
        state = {
            "model": self.model,
            "scaler": self.scaler,
            "good_cols": getattr(self, "_good_cols", list(range(10))),
            "n_train_events": self.n_train_events,
            "train_date": self.train_date,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"Model saved: {path} ({self.n_train_events} events)")

    def load(self, path: Path) -> bool:
        """Load model from disk. Returns True if successful."""
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
            self.model = state["model"]
            self.scaler = state["scaler"]
            self._good_cols = state["good_cols"]
            self.n_train_events = state["n_train_events"]
            self.train_date = state.get("train_date")
            logger.info(f"Model loaded: {path} ({self.n_train_events} events, trained {self.train_date})")
            return True
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")
            return False


def get_upcoming_earnings(days_ahead: int = 7) -> list[dict]:
    """Fetch earnings announcements in the next N days using yfinance.

    Returns list of {ticker, earnings_date} dicts.
    """
    import yfinance as yf

    # Use a broad set of liquid stocks to scan
    from qrt.data.real_data import REAL_UNIVERSE
    all_tickers = []
    for sector_tickers in REAL_UNIVERSE.values():
        all_tickers.extend(sector_tickers)

    today = pd.Timestamp.now().normalize()
    cutoff = today + pd.Timedelta(days=days_ahead)

    upcoming = []
    for ticker in all_tickers:
        try:
            t = yf.Ticker(ticker)
            dates = t.get_earnings_dates(limit=4)
            if dates is None or dates.empty:
                continue

            for idx in dates.index:
                dt = idx.tz_localize(None) if idx.tzinfo else idx
                if today <= dt <= cutoff:
                    upcoming.append({
                        "ticker": ticker,
                        "earnings_date": dt.strftime("%Y-%m-%d"),
                    })
        except Exception:
            continue

    logger.info(f"Found {len(upcoming)} earnings events in next {days_ahead} days")
    return upcoming


def generate_signals(
    model: EBSModel,
    upcoming: list[dict],
    capital: float = 100_000.0,
    equity_curve: Optional[list[float]] = None,
    vix_level: Optional[float] = None,
    threshold: float = PRED_THRESHOLD,
) -> list[EBSSignal]:
    """Generate buy-put signals for upcoming earnings events.

    Args:
        model: Trained EBSModel
        upcoming: List of {ticker, earnings_date} from get_upcoming_earnings()
        capital: Current portfolio value for position sizing
        equity_curve: Historical equity values for CDaR scaling
        vix_level: Current VIX for regime scaling
        threshold: Minimum P(catastrophic) to generate signal

    Returns:
        List of EBSSignal objects, sorted by pred_prob descending.
    """
    # Import risk sizing from earnings_black_swan
    import sys
    ebs_path = Path(__file__).parent.parent.parent.parent / "earnings_black_swan"
    if ebs_path.exists():
        sys.path.insert(0, str(ebs_path))
    from ebs.risk import compute_position_size

    signals = []
    equity_curve = equity_curve or [capital]

    for event in upcoming:
        ticker = event["ticker"]
        earnings_dt = pd.Timestamp(event["earnings_date"])

        logger.info(f"  Computing features for {ticker} ({event['earnings_date']})...")
        features = compute_features_from_yfinance(ticker, earnings_dt)
        if features is None:
            logger.info(f"    Skipped: insufficient data")
            continue

        # Predict
        X = features_to_array(features).reshape(1, -1)
        try:
            prob = float(model.predict_proba(X)[0])
        except Exception as e:
            logger.warning(f"    Prediction failed: {e}")
            continue

        if prob < threshold:
            logger.info(f"    {ticker}: prob={prob:.3f} < {threshold} — no signal")
            continue

        # Size position
        stock_vol = features.get("realized_vol_60d", 30.0)
        sizing = compute_position_size(
            capital=capital,
            win_rate=STRATEGY_STATS["win_rate"],
            avg_win=STRATEGY_STATS["avg_win"],
            avg_loss=STRATEGY_STATS["avg_loss"],
            stock_vol=stock_vol,
            equity_curve=equity_curve,
            vix_level=vix_level,
            kelly_fraction=0.25,
            max_position_pct=0.05,
        )

        signal = EBSSignal(
            ticker=ticker,
            earnings_date=event["earnings_date"],
            pred_prob=round(prob, 4),
            position_pct=sizing["position_pct"],
            features={k: v for k, v in features.items() if not k.startswith("_")},
            realized_vol=stock_vol,
        )
        signals.append(signal)

        logger.info(
            f"    SIGNAL: {ticker} prob={prob:.3f} "
            f"size={sizing['position_pct']:.2%} "
            f"(kelly={sizing['kelly_base_pct']:.3f} "
            f"vol={sizing['vol_scale']:.2f} "
            f"dd={sizing['dd_scale']:.2f} "
            f"vix={sizing['vix_scale']:.2f})"
        )

    signals.sort(key=lambda s: s.pred_prob, reverse=True)
    return signals
