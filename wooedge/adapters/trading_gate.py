"""
Trading Decision Gate

WOOEdge-based decision safety layer for crypto trading bots.
Gates entries based on uncertainty and hazard, NOT price prediction.

Usage:
    safety = TradingDecisionSafety()
    safety.reset(initial_equity=1000.0)

    for candle in candles:
        obs = TradingObservation.from_candle(candle, prev_candle, state)
        safety.observe(obs)

        bot_wants = TradingAction.LONG_ENTRY
        decision = safety.propose(bot_wants)

        if decision.allowed:
            execute(decision.action)
        else:
            print(f"Blocked: {decision.reason}")
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple, List


class Regime(Enum):
    """Market regime states."""
    TREND = "trend"
    RANGE = "range"
    CHOP = "chop"
    HIGH_VOL = "high_vol"


class TradingAction(Enum):
    """Possible trading actions."""
    HOLD = "hold"
    LONG_ENTRY = "long_entry"
    SHORT_ENTRY = "short_entry"
    EXIT = "exit"
    REDUCE = "reduce"
    SCAN = "scan"  # Wait for more data


@dataclass
class TradingState:
    """
    Current trading state.

    Tracks position, equity, and regime belief distribution.
    """
    position: int = 0  # -1 short, 0 flat, +1 long
    position_size: float = 0.0  # Fraction of equity (0.0-1.0)
    entry_price: float = 0.0

    equity: float = 0.0
    max_equity: float = 0.0
    drawdown: float = 0.0

    last_price: float = 0.0
    atr: float = 0.0
    spread: float = 0.0
    volume: float = 0.0

    regime_belief: Dict[Regime, float] = field(default_factory=lambda: {
        Regime.TREND: 0.25,
        Regime.RANGE: 0.25,
        Regime.CHOP: 0.25,
        Regime.HIGH_VOL: 0.25,
    })
    uncertainty: float = 1.0  # Entropy of regime_belief (0=certain, ~1.39=max)

    def update_drawdown(self) -> None:
        """Update max equity and drawdown."""
        if self.equity > self.max_equity:
            self.max_equity = self.equity
        if self.max_equity > 0:
            self.drawdown = 1.0 - (self.equity / self.max_equity)
        else:
            self.drawdown = 0.0

    def update_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL."""
        if self.position == 0 or self.entry_price == 0:
            return 0.0
        price_change = (current_price - self.entry_price) / self.entry_price
        return self.position * price_change * self.position_size * self.equity


@dataclass
class TradingObservation:
    """
    Observation from latest candle(s).

    Contains features used for regime detection and hazard assessment.
    """
    timestamp: int = 0
    price: float = 0.0
    ret: float = 0.0  # Return: (close - prev_close) / prev_close
    volatility: float = 0.0  # Realized volatility or ATR ratio
    atr: float = 0.0  # Average True Range
    volume_zscore: float = 0.0  # Normalized volume
    spread: float = 0.0  # Bid-ask spread or proxy

    # Computed hazard score
    hazard_score: float = 0.0

    @classmethod
    def from_candle(
        cls,
        candle: Dict,
        prev_candle: Optional[Dict],
        state: TradingState,
        atr_period: int = 14,
    ) -> "TradingObservation":
        """
        Create observation from OHLCV candle data.

        Args:
            candle: Dict with open, high, low, close, volume, timestamp
            prev_candle: Previous candle for return calculation
            state: Current trading state
            atr_period: ATR lookback (used for normalization)

        Returns:
            TradingObservation with computed features
        """
        close = candle.get("close", 0.0)
        high = candle.get("high", close)
        low = candle.get("low", close)
        volume = candle.get("volume", 0.0)
        timestamp = candle.get("timestamp", 0)

        # Return
        prev_close = prev_candle.get("close", close) if prev_candle else close
        ret = (close - prev_close) / prev_close if prev_close > 0 else 0.0

        # Volatility (simplified: range / close)
        volatility = (high - low) / close if close > 0 else 0.0

        # ATR proxy (use state.atr or current range)
        atr = state.atr if state.atr > 0 else (high - low)

        # Volume zscore (simplified: ratio to recent average)
        # In practice, you'd compute this from a rolling window
        volume_zscore = 0.0
        if state.volume > 0:
            volume_zscore = (volume - state.volume) / (state.volume + 1e-8)

        # Spread (from candle or use state)
        spread = candle.get("spread", state.spread)

        # Compute hazard score
        hazard = cls._compute_hazard(
            volatility=volatility,
            spread=spread,
            drawdown=state.drawdown,
            volume_zscore=volume_zscore,
            atr=atr,
            price=close,
        )

        return cls(
            timestamp=timestamp,
            price=close,
            ret=ret,
            volatility=volatility,
            atr=atr,
            volume_zscore=volume_zscore,
            spread=spread,
            hazard_score=hazard,
        )

    @staticmethod
    def _compute_hazard(
        volatility: float,
        spread: float,
        drawdown: float,
        volume_zscore: float,
        atr: float,
        price: float,
    ) -> float:
        """
        Compute hazard score from market conditions.

        Returns value in [0, 1] where higher = more dangerous.
        """
        hazard = 0.0

        # Volatility spike (if vol > 2x normal, add hazard)
        vol_hazard = min(1.0, volatility * 10)  # Scale: 10% range = 1.0
        hazard += vol_hazard * 0.3

        # Spread widening (if spread > 0.5%, add hazard)
        spread_pct = spread / price if price > 0 else 0.0
        spread_hazard = min(1.0, spread_pct * 200)  # Scale: 0.5% = 1.0
        hazard += spread_hazard * 0.2

        # Drawdown risk
        dd_hazard = min(1.0, drawdown * 5)  # Scale: 20% dd = 1.0
        hazard += dd_hazard * 0.3

        # Volume anomaly (extreme volume = uncertainty)
        vol_anom = min(1.0, abs(volume_zscore) / 3)  # Scale: 3 sigma = 1.0
        hazard += vol_anom * 0.2

        return min(1.0, hazard)


@dataclass
class TradingDecision:
    """Result of propose() - what action to take and why."""
    action: TradingAction
    allowed: bool
    reason: str
    original_action: TradingAction
    uncertainty: float
    hazard: float
    suggested_size: float = 1.0  # Position size multiplier (0-1)


class TradingDecisionSafety:
    """
    Decision safety layer for trading bots.

    Gates entries based on uncertainty (regime belief entropy) and hazard.
    Does NOT predict prices - only assesses whether conditions are safe to trade.
    """

    # Regime likelihood parameters (simplified emission model)
    REGIME_FEATURES = {
        Regime.TREND: {"ret_abs": 0.02, "vol": 0.03, "vol_z": 0.5},
        Regime.RANGE: {"ret_abs": 0.005, "vol": 0.015, "vol_z": 0.0},
        Regime.CHOP: {"ret_abs": 0.01, "vol": 0.025, "vol_z": -0.3},
        Regime.HIGH_VOL: {"ret_abs": 0.03, "vol": 0.05, "vol_z": 1.0},
    }

    def __init__(
        self,
        hazard_threshold: float = 0.6,
        uncertainty_threshold: float = 0.8,
        max_drawdown: float = 0.15,
        min_size_multiplier: float = 0.25,
    ):
        """
        Initialize trading decision safety.

        Args:
            hazard_threshold: Block entries above this hazard (0-1)
            uncertainty_threshold: Block entries above this uncertainty (0-~1.39)
            max_drawdown: Force exit above this drawdown
            min_size_multiplier: Minimum position size when uncertain
        """
        self.hazard_threshold = hazard_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.max_drawdown = max_drawdown
        self.min_size_multiplier = min_size_multiplier

        self.state = TradingState()
        self._observation_count = 0
        self._initialized = False

    def reset(self, initial_equity: float = 1000.0) -> None:
        """Reset state for new trading session."""
        self.state = TradingState(
            equity=initial_equity,
            max_equity=initial_equity,
        )
        self._observation_count = 0
        self._initialized = True

    def observe(self, obs: TradingObservation) -> None:
        """
        Update regime belief from observation.

        Uses simple Bayesian update based on feature likelihoods.
        """
        if not self._initialized:
            raise RuntimeError("Call reset() before observe()")

        self._observation_count += 1

        # Update state from observation
        self.state.last_price = obs.price
        self.state.atr = obs.atr if obs.atr > 0 else self.state.atr
        self.state.spread = obs.spread
        self.state.volume = obs.price * max(1, obs.volume_zscore + 1)  # Proxy

        # Bayesian update of regime belief
        self._update_regime_belief(obs)

        # Update uncertainty (entropy)
        self.state.uncertainty = self._compute_entropy(self.state.regime_belief)

        # Update PnL and drawdown
        if self.state.position != 0:
            pnl = self.state.update_pnl(obs.price)
            self.state.equity = self.state.max_equity * (1 - self.state.drawdown) + pnl
        self.state.update_drawdown()

    def _update_regime_belief(self, obs: TradingObservation) -> None:
        """
        Update regime belief using Bayesian inference.

        P(regime|obs) ∝ P(obs|regime) * P(regime)
        """
        likelihoods = {}
        for regime in Regime:
            likelihoods[regime] = self._regime_likelihood(regime, obs)

        # Bayesian update
        new_belief = {}
        total = 0.0
        for regime in Regime:
            prior = self.state.regime_belief[regime]
            likelihood = likelihoods[regime]
            posterior = prior * likelihood
            new_belief[regime] = posterior
            total += posterior

        # Normalize
        if total > 0:
            for regime in Regime:
                new_belief[regime] /= total
        else:
            # Fallback to uniform
            for regime in Regime:
                new_belief[regime] = 0.25

        self.state.regime_belief = new_belief

    def _regime_likelihood(self, regime: Regime, obs: TradingObservation) -> float:
        """
        Compute P(obs|regime) using Gaussian-like likelihood.

        Higher return when observation features match regime expectations.
        """
        expected = self.REGIME_FEATURES[regime]

        # Feature distances
        ret_dist = abs(abs(obs.ret) - expected["ret_abs"])
        vol_dist = abs(obs.volatility - expected["vol"])
        volz_dist = abs(obs.volume_zscore - expected["vol_z"])

        # Gaussian-ish likelihood (smaller distance = higher likelihood)
        ret_lik = math.exp(-ret_dist * 50)
        vol_lik = math.exp(-vol_dist * 30)
        volz_lik = math.exp(-volz_dist * 0.5)

        # Combined likelihood
        return ret_lik * vol_lik * volz_lik + 0.01  # Add floor

    def _compute_entropy(self, belief: Dict[Regime, float]) -> float:
        """Compute Shannon entropy of belief distribution."""
        entropy = 0.0
        for prob in belief.values():
            if prob > 0:
                entropy -= prob * math.log(prob)
        return entropy

    def propose(self, bot_action: TradingAction) -> TradingDecision:
        """
        Evaluate whether bot's proposed action should be allowed.

        Args:
            bot_action: What the trading bot wants to do

        Returns:
            TradingDecision with final action and reasoning
        """
        if not self._initialized:
            raise RuntimeError("Call reset() before propose()")

        hazard = self._get_current_hazard()
        uncertainty = self.state.uncertainty

        # Rule 1: Max drawdown breach => force exit
        if self.state.drawdown >= self.max_drawdown:
            if self.state.position != 0:
                return TradingDecision(
                    action=TradingAction.EXIT,
                    allowed=False,
                    reason=f"MAX_DRAWDOWN_BREACH: {self.state.drawdown:.1%} >= {self.max_drawdown:.1%}",
                    original_action=bot_action,
                    uncertainty=uncertainty,
                    hazard=hazard,
                    suggested_size=0.0,
                )

        # Rule 2: High hazard => no new entries
        is_entry = bot_action in (TradingAction.LONG_ENTRY, TradingAction.SHORT_ENTRY)
        if hazard > self.hazard_threshold and is_entry:
            return TradingDecision(
                action=TradingAction.HOLD,
                allowed=False,
                reason=f"HIGH_HAZARD: {hazard:.2f} > {self.hazard_threshold:.2f}",
                original_action=bot_action,
                uncertainty=uncertainty,
                hazard=hazard,
                suggested_size=0.0,
            )

        # Rule 3: High uncertainty => delay entries, recommend SCAN
        if uncertainty > self.uncertainty_threshold and is_entry:
            return TradingDecision(
                action=TradingAction.SCAN,
                allowed=False,
                reason=f"HIGH_UNCERTAINTY: H={uncertainty:.2f} > {self.uncertainty_threshold:.2f}",
                original_action=bot_action,
                uncertainty=uncertainty,
                hazard=hazard,
                suggested_size=0.0,
            )

        # Compute position size multiplier based on uncertainty
        # Full size at 0 uncertainty, min_size at threshold
        size_mult = 1.0
        if uncertainty > 0:
            norm_uncert = uncertainty / self.uncertainty_threshold
            size_mult = max(self.min_size_multiplier, 1.0 - norm_uncert * 0.5)

        # Rule 4: Allow action with adjusted size
        return TradingDecision(
            action=bot_action,
            allowed=True,
            reason=self._get_regime_summary(),
            original_action=bot_action,
            uncertainty=uncertainty,
            hazard=hazard,
            suggested_size=size_mult,
        )

    def _get_current_hazard(self) -> float:
        """Get current hazard from last observation context."""
        # Combine state-based hazard factors
        hazard = 0.0

        # Drawdown contribution
        hazard += min(0.4, self.state.drawdown * 2)

        # Spread contribution (if tracked)
        if self.state.spread > 0 and self.state.last_price > 0:
            spread_pct = self.state.spread / self.state.last_price
            hazard += min(0.3, spread_pct * 100)

        # ATR contribution (high ATR = volatile)
        if self.state.atr > 0 and self.state.last_price > 0:
            atr_pct = self.state.atr / self.state.last_price
            hazard += min(0.3, atr_pct * 10)

        return min(1.0, hazard)

    def _get_regime_summary(self) -> str:
        """Get human-readable regime summary."""
        best_regime = max(self.state.regime_belief, key=self.state.regime_belief.get)
        prob = self.state.regime_belief[best_regime]
        return f"{best_regime.value}({prob:.0%})"

    def get_state(self) -> TradingState:
        """Get current trading state."""
        return self.state

    def update_position(
        self,
        position: int,
        size: float,
        entry_price: float,
        equity: float,
    ) -> None:
        """
        Update state after trade execution.

        Call this after executing a trade to keep state in sync.
        """
        self.state.position = position
        self.state.position_size = size
        self.state.entry_price = entry_price
        self.state.equity = equity
        self.state.update_drawdown()
