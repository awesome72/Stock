"""백테스트 성과 지표: CAGR, MDD, Sharpe, Sortino, 승률, 손익비, Profit Factor, 거래횟수."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def cagr(equity_curve: pd.Series) -> float:
    """연복리수익률(CAGR). equity_curve: date 인덱스, 총자산 값."""
    if len(equity_curve) < 2:
        return float("nan")
    start_value = equity_curve.iloc[0]
    end_value = equity_curve.iloc[-1]
    years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    if years <= 0 or start_value <= 0:
        return float("nan")
    return (end_value / start_value) ** (1 / years) - 1


def mdd(equity_curve: pd.Series) -> float:
    """최대낙폭(Maximum Drawdown). 음수로 반환한다(예: -0.23 = -23%)."""
    if len(equity_curve) == 0:
        return float("nan")
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return drawdown.min()


def sharpe_ratio(equity_curve: pd.Series, risk_free_rate: float = 0.0) -> float:
    """샤프비율(연율화). 일간 수익률 기준."""
    returns = equity_curve.pct_change().dropna()
    if len(returns) < 2 or returns.std() == 0:
        return float("nan")
    excess = returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    return float((excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sortino_ratio(equity_curve: pd.Series, risk_free_rate: float = 0.0) -> float:
    """소르티노비율(연율화). 하방(음수) 수익률의 변동성만 분모로 사용한다."""
    returns = equity_curve.pct_change().dropna()
    if len(returns) < 2:
        return float("nan")
    excess = returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return float("nan")
    return float((excess.mean() / downside.std()) * np.sqrt(TRADING_DAYS_PER_YEAR))


def win_rate(trade_returns: pd.Series) -> float:
    """승률(0~1). trade_returns: 거래별 손익(금액 또는 수익률) Series."""
    if len(trade_returns) == 0:
        return float("nan")
    return float((trade_returns > 0).sum() / len(trade_returns))


def payoff_ratio(trade_returns: pd.Series) -> float:
    """손익비: 평균 이익 / 평균 손실(절대값)."""
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]
    if len(wins) == 0 or len(losses) == 0:
        return float("nan")
    return float(wins.mean() / abs(losses.mean()))


def profit_factor(trade_returns: pd.Series) -> float:
    """Profit Factor: 총이익 / 총손실(절대값)."""
    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]
    total_loss = abs(losses.sum())
    if total_loss == 0:
        return float("nan")
    return float(wins.sum() / total_loss)


def trade_count(trade_returns: pd.Series) -> int:
    """총 거래 횟수."""
    return int(len(trade_returns))


def summary(equity_curve: pd.Series, trade_pnl: pd.Series) -> dict[str, float]:
    """성과 지표 8종(CAGR, MDD, Sharpe, Sortino, 승률, 손익비, Profit Factor, 거래횟수)을 한 번에 계산."""
    return {
        "CAGR": cagr(equity_curve),
        "MDD": mdd(equity_curve),
        "Sharpe": sharpe_ratio(equity_curve),
        "Sortino": sortino_ratio(equity_curve),
        "승률": win_rate(trade_pnl),
        "손익비": payoff_ratio(trade_pnl),
        "Profit Factor": profit_factor(trade_pnl),
        "거래횟수": trade_count(trade_pnl),
    }


def walk_forward(
    run_fn: Callable[..., object],
    tickers: list[str],
    start: str,
    end: str,
    train_years: int = 3,
    test_years: int = 1,
    rolls: int = 5,
    **run_kwargs,
) -> pd.DataFrame:
    """학습 train_years / 검증 test_years 롤링 윈도우로 최대 rolls회 반복해 구간별 성과를 표로 반환한다.

    이 프로젝트는 파라미터를 데이터로부터 '학습'하지 않는 규칙 기반 스코어링 엔진이므로,
    여기서 train 구간은 지표 계산에 필요한 과거 데이터를 확보하는 워밍업 용도로만 쓰이고
    (run_fn의 data_start로 전달), 실제 매매/성과 집계는 test 구간에서만 이루어진다.

    run_fn: backtest.engine.run과 동일한 시그니처
      (tickers, start, end, data_start=..., **kwargs) -> BacktestResult.
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    rows = []
    for i in range(rolls):
        train_start = start_ts + pd.DateOffset(years=i)
        test_start = train_start + pd.DateOffset(years=train_years)
        test_end = test_start + pd.DateOffset(years=test_years)
        if test_end > end_ts:
            break

        result = run_fn(
            tickers,
            test_start.strftime("%Y%m%d"),
            test_end.strftime("%Y%m%d"),
            data_start=train_start.strftime("%Y%m%d"),
            **run_kwargs,
        )

        trade_pnl = pd.Series([t.pnl for t in result.trades], dtype=float)
        row = {
            "구간": f"{i + 1}",
            "학습 시작": train_start.date(),
            "검증 시작": test_start.date(),
            "검증 종료": test_end.date(),
        }
        row.update(summary(result.equity_curve, trade_pnl))
        rows.append(row)

    return pd.DataFrame(rows)
