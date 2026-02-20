from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Literal, Optional

import pandas as pd

TradeSide = Literal["long", "short"]


@dataclass
class TradeRecord:
    side: TradeSide
    entry_index: Any
    exit_index: Any
    entry_price: float
    exit_price: float
    pnl: float


@dataclass
class TradeSummary:
    total_pnl: float
    max_drawdown: float
    win_rate_pct: float
    win_to_lose_ratio: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int


def _to_dataframe(trades: List[TradeRecord]) -> pd.DataFrame:
    return pd.DataFrame([t.__dict__ for t in trades])


def _append_closed_trade(
    trades: List[TradeRecord],
    side: TradeSide,
    entry_index: Any,
    exit_index: Any,
    entry_price: float,
    exit_price: float,
) -> None:
    pnl = (exit_price - entry_price) if side == "long" else (entry_price - exit_price)
    trades.append(
        TradeRecord(
            side=side,
            entry_index=entry_index,
            exit_index=exit_index,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=float(pnl),
        )
    )


def extract_trade_records(df: pd.DataFrame) -> pd.DataFrame:
    """Build closed trade records from adapter-style signal columns.

    Required columns: close, start_long, start_short, end_long, end_short.

    Signal handling order per row:
    1) close an open trade (if its matching end signal is present)
    2) open a new trade (if flat and a start signal is present)

    This allows close+open flips on the same bar to be counted as two trades.
    """
    required = {"close", "start_long", "start_short", "end_long", "end_short"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

    trades: List[TradeRecord] = []
    side: Optional[TradeSide] = None
    entry_price = 0.0
    entry_idx: Any = None

    for idx, row in df.iterrows():
        close = float(row["close"])

        # 1) Close existing trade first.
        if side == "long" and bool(row["end_long"]):
            _append_closed_trade(trades, "long", entry_idx, idx, entry_price, close)
            side = None
            entry_idx = None
        elif side == "short" and bool(row["end_short"]):
            _append_closed_trade(trades, "short", entry_idx, idx, entry_price, close)
            side = None
            entry_idx = None

        # 2) Then open a new one if flat.
        if side is None and bool(row["start_long"]):
            side = "long"
            entry_price = close
            entry_idx = idx
        elif side is None and bool(row["start_short"]):
            side = "short"
            entry_price = close
            entry_idx = idx

    # Force-close last open trade at final close to match existing app behavior.
    if side is not None:
        final_idx = df.index[-1]
        final_close = float(df.iloc[-1]["close"])
        _append_closed_trade(trades, side, entry_idx, final_idx, entry_price, final_close)

    return _to_dataframe(trades)


def summarize_trade_outcomes(trades: pd.DataFrame) -> TradeSummary:
    if trades.empty:
        return TradeSummary(0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0)

    pnl = trades["pnl"].astype(float)
    total_pnl = float(pnl.sum())

    equity = pnl.cumsum()
    running_peak = equity.cummax()
    drawdown = running_peak - equity
    max_drawdown = float(drawdown.max()) if len(drawdown) else 0.0

    wins = int((pnl > 0).sum())
    losses = int((pnl < 0).sum())
    breakeven = int((pnl == 0).sum())
    total = int(len(trades))
    win_rate_pct = float((wins / total) * 100.0) if total else 0.0

    if losses == 0:
        win_to_lose = float("inf") if wins > 0 else 0.0
    else:
        win_to_lose = float(wins / losses)

    return TradeSummary(
        total_pnl=total_pnl,
        max_drawdown=max_drawdown,
        win_rate_pct=win_rate_pct,
        win_to_lose_ratio=win_to_lose,
        total_trades=total,
        winning_trades=wins,
        losing_trades=losses,
        breakeven_trades=breakeven,
    )


def evaluate_trades_with_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, TradeSummary]:
    trades = extract_trade_records(df)
    return trades, summarize_trade_outcomes(trades)
