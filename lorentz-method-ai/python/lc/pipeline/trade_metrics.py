from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

import pandas as pd

TradeSide = Literal["long", "short"]


@dataclass
class TradeRecord:
    side: TradeSide
    entry_index: int
    exit_index: int
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


def extract_trade_records(df: pd.DataFrame) -> pd.DataFrame:
    """Build closed trade records from adapter-style signal columns.

    Required columns: close, start_long, start_short, end_long, end_short.
    """
    required = {"close", "start_long", "start_short", "end_long", "end_short"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

    trades: List[TradeRecord] = []
    side: Optional[TradeSide] = None
    entry_price = 0.0
    entry_idx = -1

    for idx, row in df.iterrows():
        if side is None and bool(row["start_long"]):
            side = "long"
            entry_price = float(row["close"])
            entry_idx = int(idx)
            continue
        if side is None and bool(row["start_short"]):
            side = "short"
            entry_price = float(row["close"])
            entry_idx = int(idx)
            continue

        if side == "long" and bool(row["end_long"]):
            exit_price = float(row["close"])
            trades.append(
                TradeRecord(
                    side="long",
                    entry_index=entry_idx,
                    exit_index=int(idx),
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl=exit_price - entry_price,
                )
            )
            side = None
        elif side == "short" and bool(row["end_short"]):
            exit_price = float(row["close"])
            trades.append(
                TradeRecord(
                    side="short",
                    entry_index=entry_idx,
                    exit_index=int(idx),
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl=entry_price - exit_price,
                )
            )
            side = None

    # Force-close last open trade at final close to match existing app behavior.
    if side is not None:
        exit_idx = int(df.index[-1])
        exit_price = float(df.iloc[-1]["close"])
        pnl = exit_price - entry_price if side == "long" else entry_price - exit_price
        trades.append(
            TradeRecord(
                side=side,
                entry_index=entry_idx,
                exit_index=exit_idx,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=float(pnl),
            )
        )

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
