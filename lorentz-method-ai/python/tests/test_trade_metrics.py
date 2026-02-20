import math

import pandas as pd

from lc.pipeline.trade_metrics import evaluate_trades_with_metrics


def test_trade_metrics_basic_long_short_cycle() -> None:
    df = pd.DataFrame(
        {
            "close": [100, 105, 103, 99, 101, 98],
            "start_long": [True, False, False, False, False, False],
            "end_long": [False, False, True, False, False, False],
            "start_short": [False, False, False, True, False, False],
            "end_short": [False, False, False, False, False, True],
        }
    )

    trades, summary = evaluate_trades_with_metrics(df)

    assert len(trades) == 2
    assert trades["pnl"].tolist() == [3.0, 1.0]
    assert summary.total_pnl == 4.0
    assert summary.max_drawdown == 0.0
    assert summary.win_rate_pct == 100.0
    assert math.isinf(summary.win_to_lose_ratio)


def test_trade_metrics_handles_losses_and_drawdown() -> None:
    df = pd.DataFrame(
        {
            "close": [100, 95, 94, 96, 97, 102, 105],
            "start_long": [True, False, False, False, True, False, False],
            "end_long": [False, False, True, False, False, False, True],
            "start_short": [False, False, False, True, False, False, False],
            "end_short": [False, False, False, False, False, True, False],
        }
    )

    trades, summary = evaluate_trades_with_metrics(df)

    # long: -6, short: -6, long: +8
    assert trades["pnl"].tolist() == [-6.0, -6.0, 8.0]
    assert summary.total_pnl == -4.0
    assert summary.max_drawdown == 12.0
    assert round(summary.win_rate_pct, 2) == 33.33
    assert summary.win_to_lose_ratio == 0.5


def test_trade_metrics_counts_same_bar_flip_events() -> None:
    # close+open on the same bar should count as two distinct trade events.
    df = pd.DataFrame(
        {
            "close": [100, 102, 101, 99, 98, 97],
            "start_long": [True, False, False, True, False, False],
            "end_long": [False, True, False, False, False, True],
            "start_short": [False, True, False, False, False, False],
            "end_short": [False, False, True, False, False, False],
        }
    )

    trades, summary = evaluate_trades_with_metrics(df)

    # 1) long 100->102 (+2), then same-bar open short at 102
    # 2) short 102->101 (+1)
    # 3) long 99->97 (-2)
    assert len(trades) == 3
    assert trades["pnl"].tolist() == [2.0, 1.0, -2.0]
    assert summary.total_pnl == 1.0
