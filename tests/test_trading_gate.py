"""
Tests for Trading Decision Gate.

Tests gating behavior on synthetic scenarios:
- Calm trend => allow entries
- Volatility spike => block entries
- High uncertainty => delay
- Drawdown breach => force exit
"""

import pytest
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wooedge.adapters.trading_gate import (
    TradingState,
    TradingObservation,
    TradingDecisionSafety,
    TradingAction,
    TradingDecision,
    Regime,
)


class TestTradingState:
    """Tests for TradingState dataclass."""

    def test_initial_state(self):
        """Test default state values."""
        state = TradingState()
        assert state.position == 0
        assert state.position_size == 0.0
        assert state.equity == 0.0
        assert state.drawdown == 0.0
        assert state.uncertainty == 1.0

    def test_update_drawdown(self):
        """Test drawdown calculation."""
        state = TradingState(equity=900.0, max_equity=1000.0)
        state.update_drawdown()
        assert state.drawdown == pytest.approx(0.1)

    def test_update_drawdown_new_high(self):
        """Test max equity updates on new high."""
        state = TradingState(equity=1100.0, max_equity=1000.0)
        state.update_drawdown()
        assert state.max_equity == 1100.0
        assert state.drawdown == 0.0

    def test_update_pnl_long(self):
        """Test PnL calculation for long position."""
        state = TradingState(
            position=1,
            position_size=0.5,
            entry_price=100.0,
            equity=1000.0,
        )
        pnl = state.update_pnl(110.0)  # 10% price increase
        assert pnl == pytest.approx(50.0)  # 10% * 0.5 * 1000

    def test_update_pnl_short(self):
        """Test PnL calculation for short position."""
        state = TradingState(
            position=-1,
            position_size=0.5,
            entry_price=100.0,
            equity=1000.0,
        )
        pnl = state.update_pnl(90.0)  # 10% price decrease
        assert pnl == pytest.approx(50.0)  # -10% * -1 * 0.5 * 1000


class TestTradingObservation:
    """Tests for TradingObservation dataclass."""

    def test_from_candle(self):
        """Test creating observation from candle data."""
        candle = {
            'timestamp': 1,
            'open': 100.0,
            'high': 105.0,
            'low': 95.0,
            'close': 102.0,
            'volume': 1000,
            'spread': 0.5,
        }
        prev_candle = {
            'close': 100.0,
        }
        state = TradingState(volume=800)

        obs = TradingObservation.from_candle(candle, prev_candle, state)

        assert obs.timestamp == 1
        assert obs.price == 102.0
        assert obs.ret == pytest.approx(0.02)  # (102-100)/100
        assert obs.volatility == pytest.approx(0.098, rel=0.01)  # (105-95)/102
        assert obs.spread == 0.5
        assert 0 <= obs.hazard_score <= 1

    def test_hazard_score_calm_market(self):
        """Test hazard score in calm market conditions."""
        obs = TradingObservation(
            volatility=0.01,
            spread=0.1,
            hazard_score=TradingObservation._compute_hazard(
                volatility=0.01,
                spread=0.1,
                drawdown=0.0,
                volume_zscore=0.0,
                atr=1.0,
                price=100.0,
            )
        )
        assert obs.hazard_score < 0.3  # Low hazard

    def test_hazard_score_volatile_market(self):
        """Test hazard score in volatile market."""
        hazard = TradingObservation._compute_hazard(
            volatility=0.1,  # 10% range
            spread=1.0,      # 1% spread
            drawdown=0.1,    # 10% drawdown
            volume_zscore=2.0,
            atr=10.0,
            price=100.0,
        )
        assert hazard > 0.5  # High hazard


class TestTradingDecisionSafety:
    """Tests for TradingDecisionSafety class."""

    def test_reset(self):
        """Test reset initializes state correctly."""
        safety = TradingDecisionSafety()
        safety.reset(initial_equity=5000.0)

        assert safety.state.equity == 5000.0
        assert safety.state.max_equity == 5000.0
        assert safety._initialized

    def test_observe_requires_reset(self):
        """Test observe raises if not reset."""
        safety = TradingDecisionSafety()
        obs = TradingObservation(price=100.0)

        with pytest.raises(RuntimeError, match="reset"):
            safety.observe(obs)

    def test_propose_requires_reset(self):
        """Test propose raises if not reset."""
        safety = TradingDecisionSafety()

        with pytest.raises(RuntimeError, match="reset"):
            safety.propose(TradingAction.LONG_ENTRY)

    def test_calm_trend_allows_entry(self):
        """Test that calm trending market allows entries."""
        safety = TradingDecisionSafety(
            hazard_threshold=0.6,
            uncertainty_threshold=0.8,
        )
        safety.reset(initial_equity=1000.0)

        # Simulate calm uptrend
        for i in range(10):
            obs = TradingObservation(
                price=100.0 + i,
                ret=0.01,
                volatility=0.02,
                volume_zscore=0.0,
                spread=0.1,
                hazard_score=0.1,
            )
            safety.observe(obs)

        decision = safety.propose(TradingAction.LONG_ENTRY)

        assert decision.allowed
        assert decision.action == TradingAction.LONG_ENTRY
        assert decision.hazard < 0.6
        assert decision.suggested_size > 0.5

    def test_volatility_spike_blocks_entry(self):
        """Test that volatility spike blocks new entries."""
        safety = TradingDecisionSafety(
            hazard_threshold=0.5,
        )
        safety.reset(initial_equity=1000.0)

        # Set high ATR and spread to trigger hazard
        safety.state.atr = 10.0
        safety.state.spread = 2.0
        safety.state.last_price = 100.0
        safety.state.drawdown = 0.1

        decision = safety.propose(TradingAction.LONG_ENTRY)

        assert not decision.allowed
        assert decision.action == TradingAction.HOLD
        assert "HAZARD" in decision.reason

    def test_high_uncertainty_delays_entry(self):
        """Test that high uncertainty delays entries."""
        safety = TradingDecisionSafety(
            hazard_threshold=0.8,
            uncertainty_threshold=0.5,
        )
        safety.reset(initial_equity=1000.0)

        # Set high uncertainty (uniform belief)
        safety.state.regime_belief = {
            Regime.TREND: 0.25,
            Regime.RANGE: 0.25,
            Regime.CHOP: 0.25,
            Regime.HIGH_VOL: 0.25,
        }
        safety.state.uncertainty = safety._compute_entropy(safety.state.regime_belief)

        decision = safety.propose(TradingAction.LONG_ENTRY)

        assert not decision.allowed
        assert decision.action == TradingAction.SCAN
        assert "UNCERTAINTY" in decision.reason

    def test_drawdown_breach_forces_exit(self):
        """Test that drawdown breach forces exit."""
        safety = TradingDecisionSafety(
            max_drawdown=0.1,
        )
        safety.reset(initial_equity=1000.0)

        # Simulate 15% drawdown
        safety.state.equity = 850.0
        safety.state.max_equity = 1000.0
        safety.state.drawdown = 0.15
        safety.state.position = 1  # Has a position

        decision = safety.propose(TradingAction.HOLD)

        assert not decision.allowed
        assert decision.action == TradingAction.EXIT
        assert "DRAWDOWN" in decision.reason

    def test_position_sizing_scales_with_uncertainty(self):
        """Test that position size decreases with uncertainty."""
        safety = TradingDecisionSafety(
            hazard_threshold=0.9,
            uncertainty_threshold=1.0,
            min_size_multiplier=0.25,
        )
        safety.reset(initial_equity=1000.0)

        # Low uncertainty -> high size
        safety.state.uncertainty = 0.1
        decision_low = safety.propose(TradingAction.LONG_ENTRY)

        # High uncertainty -> low size
        safety.state.uncertainty = 0.8
        decision_high = safety.propose(TradingAction.LONG_ENTRY)

        assert decision_low.suggested_size > decision_high.suggested_size
        assert decision_high.suggested_size >= 0.25

    def test_regime_belief_updates(self):
        """Test that regime belief updates from observations."""
        safety = TradingDecisionSafety()
        safety.reset(initial_equity=1000.0)

        # Observe trending-like data (high return, moderate vol)
        for i in range(5):
            obs = TradingObservation(
                price=100.0 + i * 2,
                ret=0.02,
                volatility=0.03,
                volume_zscore=0.5,
            )
            safety.observe(obs)

        # TREND regime should have higher probability
        beliefs = safety.state.regime_belief
        assert beliefs[Regime.TREND] > 0.25  # Above uniform

    def test_entropy_computation(self):
        """Test entropy computation is correct."""
        safety = TradingDecisionSafety()

        # Uniform distribution -> max entropy
        uniform = {r: 0.25 for r in Regime}
        max_entropy = safety._compute_entropy(uniform)
        assert max_entropy == pytest.approx(math.log(4), rel=0.01)

        # Certain distribution -> zero entropy
        certain = {Regime.TREND: 1.0, Regime.RANGE: 0.0,
                   Regime.CHOP: 0.0, Regime.HIGH_VOL: 0.0}
        min_entropy = safety._compute_entropy(certain)
        assert min_entropy == pytest.approx(0.0)

    def test_update_position(self):
        """Test update_position updates state correctly."""
        safety = TradingDecisionSafety()
        safety.reset(initial_equity=1000.0)

        safety.update_position(
            position=1,
            size=0.5,
            entry_price=100.0,
            equity=1000.0,
        )

        assert safety.state.position == 1
        assert safety.state.position_size == 0.5
        assert safety.state.entry_price == 100.0


class TestTradingGateCLI:
    """Tests for trading_gate CLI command."""

    def test_cli_help(self):
        """Test CLI help shows options."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "wooedge.cli", "trading_gate", "--help"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(__file__))
        )
        assert result.returncode == 0
        assert "--file" in result.stdout
        assert "--hazard-thresh" in result.stdout
        assert "--uncert-thresh" in result.stdout
        assert "--max-dd" in result.stdout

    def test_cli_runs_default_file(self):
        """Test CLI runs with default sample file."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "wooedge.cli", "trading_gate", "-q"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(__file__))
        )
        assert result.returncode == 0
        assert "candles" in result.stdout.lower()
