#!/usr/bin/env python3
"""
Trading Gate Demo

Demonstrates WOOEdge decision safety layer on historical OHLCV data.
Shows how entries are gated based on uncertainty and hazard.

Usage:
    python examples/trading_gate_demo.py
    python examples/trading_gate_demo.py --file path/to/ohlcv.csv
    python examples/trading_gate_demo.py --hazard-thresh 0.5 --uncert-thresh 0.9
"""

import sys
import os
import csv
import argparse
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wooedge.adapters.trading_gate import (
    TradingState,
    TradingObservation,
    TradingDecisionSafety,
    TradingAction,
    Regime,
)


class SimpleBot:
    """
    Simple trend-following bot for demo purposes.

    Generates buy/sell signals based on price momentum.
    """

    def __init__(self, lookback: int = 3):
        self.lookback = lookback
        self.prices: List[float] = []

    def update(self, price: float) -> None:
        """Update price history."""
        self.prices.append(price)
        if len(self.prices) > self.lookback + 1:
            self.prices.pop(0)

    def get_signal(self, current_position: int) -> TradingAction:
        """
        Generate trading signal.

        Returns what the bot WANTS to do (before safety gating).
        """
        if len(self.prices) < self.lookback + 1:
            return TradingAction.HOLD

        # Simple momentum: if price up over lookback, go long
        start_price = self.prices[0]
        end_price = self.prices[-1]
        ret = (end_price - start_price) / start_price

        if ret > 0.02 and current_position <= 0:
            return TradingAction.LONG_ENTRY
        elif ret < -0.02 and current_position >= 0:
            return TradingAction.SHORT_ENTRY
        elif current_position != 0 and abs(ret) < 0.005:
            return TradingAction.EXIT

        return TradingAction.HOLD


def load_ohlcv(filepath: str) -> List[Dict]:
    """Load OHLCV data from CSV file."""
    candles = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            candles.append({
                'timestamp': int(row['timestamp']),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']),
                'spread': float(row.get('spread', 0.1)),
            })
    return candles


def run_backtest(
    candles: List[Dict],
    hazard_threshold: float = 0.6,
    uncertainty_threshold: float = 0.8,
    max_drawdown: float = 0.15,
    initial_equity: float = 1000.0,
    verbose: bool = True,
) -> Dict:
    """
    Run backtest with WOOEdge safety gating.

    Args:
        candles: List of OHLCV candles
        hazard_threshold: Block entries above this hazard
        uncertainty_threshold: Block entries above this uncertainty
        max_drawdown: Force exit above this drawdown
        initial_equity: Starting equity
        verbose: Print detailed output

    Returns:
        Dict with backtest statistics
    """
    # Initialize
    safety = TradingDecisionSafety(
        hazard_threshold=hazard_threshold,
        uncertainty_threshold=uncertainty_threshold,
        max_drawdown=max_drawdown,
    )
    safety.reset(initial_equity=initial_equity)

    bot = SimpleBot(lookback=3)

    # Track statistics
    stats = {
        'total_signals': 0,
        'allowed': 0,
        'blocked_hazard': 0,
        'blocked_uncertainty': 0,
        'blocked_drawdown': 0,
        'trades': 0,
        'final_equity': initial_equity,
        'max_drawdown': 0.0,
    }

    # Print header
    if verbose:
        print("=" * 100)
        print("Trading Gate Demo - WOOEdge Decision Safety")
        print("=" * 100)
        print(f"{'T':>3} {'Price':>8} {'Bot':>12} {'Hazard':>7} {'Uncert':>7} "
              f"{'Decision':>8} {'Final':>12} {'Pos':>4} {'Equity':>10} {'Reason'}")
        print("-" * 100)

    prev_candle = None
    position = 0
    equity = initial_equity
    entry_price = 0.0

    for candle in candles:
        # Create observation
        obs = TradingObservation.from_candle(candle, prev_candle, safety.get_state())

        # Update safety layer
        safety.observe(obs)

        # Get bot's desired action
        bot.update(candle['close'])
        bot_action = bot.get_signal(position)

        # Gate the action
        decision = safety.propose(bot_action)

        # Track statistics
        if bot_action != TradingAction.HOLD:
            stats['total_signals'] += 1
            if decision.allowed:
                stats['allowed'] += 1
            elif 'HAZARD' in decision.reason:
                stats['blocked_hazard'] += 1
            elif 'UNCERTAINTY' in decision.reason:
                stats['blocked_uncertainty'] += 1
            elif 'DRAWDOWN' in decision.reason:
                stats['blocked_drawdown'] += 1

        # Execute allowed action
        final_action = decision.action
        if decision.allowed and final_action in (TradingAction.LONG_ENTRY, TradingAction.SHORT_ENTRY):
            if position == 0:
                position = 1 if final_action == TradingAction.LONG_ENTRY else -1
                entry_price = candle['close']
                stats['trades'] += 1
        elif final_action == TradingAction.EXIT or (not decision.allowed and 'DRAWDOWN' in decision.reason):
            if position != 0:
                # Calculate PnL
                pnl_pct = position * (candle['close'] - entry_price) / entry_price
                equity *= (1 + pnl_pct * decision.suggested_size)
                position = 0
                entry_price = 0.0
                stats['trades'] += 1

        # Update safety state
        safety.update_position(position, 1.0 if position != 0 else 0.0, entry_price, equity)

        # Track max drawdown
        if safety.get_state().drawdown > stats['max_drawdown']:
            stats['max_drawdown'] = safety.get_state().drawdown

        # Print row
        if verbose:
            allowed_str = "ALLOW" if decision.allowed else "BLOCK"
            pos_str = {-1: "SHORT", 0: "FLAT", 1: "LONG"}[position]
            print(f"{candle['timestamp']:>3} {candle['close']:>8.2f} {bot_action.value:>12} "
                  f"{decision.hazard:>7.2f} {decision.uncertainty:>7.2f} "
                  f"{allowed_str:>8} {final_action.value:>12} {pos_str:>5} "
                  f"{equity:>10.2f} {decision.reason[:25]}")

        prev_candle = candle

    stats['final_equity'] = equity

    # Print summary
    if verbose:
        print("-" * 100)
        print("\nSUMMARY")
        print("=" * 50)
        print(f"Total signals:       {stats['total_signals']}")
        print(f"  Allowed:           {stats['allowed']}")
        print(f"  Blocked (hazard):  {stats['blocked_hazard']}")
        print(f"  Blocked (uncert):  {stats['blocked_uncertainty']}")
        print(f"  Blocked (dd):      {stats['blocked_drawdown']}")
        print(f"Total trades:        {stats['trades']}")
        print(f"Final equity:        ${stats['final_equity']:.2f}")
        print(f"Return:              {(stats['final_equity']/initial_equity - 1)*100:.2f}%")
        print(f"Max drawdown:        {stats['max_drawdown']*100:.2f}%")
        print("=" * 50)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Trading Gate Demo")
    parser.add_argument("--file", "-f", type=str,
                        default=os.path.join(os.path.dirname(__file__), "data", "sample_ohlcv.csv"),
                        help="Path to OHLCV CSV file")
    parser.add_argument("--hazard-thresh", type=float, default=0.6,
                        help="Hazard threshold (default 0.6)")
    parser.add_argument("--uncert-thresh", type=float, default=0.8,
                        help="Uncertainty threshold (default 0.8)")
    parser.add_argument("--max-dd", type=float, default=0.15,
                        help="Max drawdown (default 0.15)")
    parser.add_argument("--equity", type=float, default=1000.0,
                        help="Initial equity (default 1000)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Minimal output")

    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        sys.exit(1)

    candles = load_ohlcv(args.file)
    print(f"Loaded {len(candles)} candles from {args.file}\n")

    run_backtest(
        candles,
        hazard_threshold=args.hazard_thresh,
        uncertainty_threshold=args.uncert_thresh,
        max_drawdown=args.max_dd,
        initial_equity=args.equity,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
