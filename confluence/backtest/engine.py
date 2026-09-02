"""스코어 기반 진입/청산 백테스트 실행 엔진.

핵심 규칙(위반 시 결과를 신뢰할 수 없음):
1. 신호는 반드시 '전일 종가까지'의 데이터로 계산하고, 체결은 '당일 시가'로 한다
   (신호가 발생한 봉의 종가로 체결하지 않는다).
2. 매 시점의 스코어/국면/손절가는 그 시점까지의 데이터만으로 계산된 값을 사용한다
   (confluence.engine.scorer의 벡터화 함수들은 전부 과거 방향으로만 계산되므로,
   이 규칙은 그 지표 계산 함수들의 look-ahead 금지 설계에 의해 이미 보장된다).

포트폴리오 시뮬레이션(현금/보유종목 상태, 점수 우선순위로 진입 제한)은 본질적으로
순차적/상태 의존적이라 벡터화할 수 없으므로, 이 모듈만 예외적으로 날짜별 순방향
루프를 사용한다(indicators/trend.py의 Supertrend와 같은 이유).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import config
from ..data import loader
from ..engine import scorer
from ..indicators import volatility


@dataclass
class Position:
    """진행 중인 보유 포지션의 상태."""

    ticker: str
    shares: float
    entry_date: pd.Timestamp
    entry_price: float
    initial_stop_loss: float
    current_stop_loss: float
    initial_risk: float  # entry_price - initial_stop_loss (1R 단위 금액)
    highest_high_since_entry: float
    trailing_active: bool = False


@dataclass
class Trade:
    """청산 완료된 거래 1건의 기록."""

    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    shares: float
    exit_reason: str  # "stop_loss" | "trailing_stop" | "score_exit" | "period_end"
    pnl: float
    return_pct: float


@dataclass
class BacktestResult:
    """백테스트 1회 실행 결과."""

    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    benchmark_curve: pd.Series | None = None


def _trading_cost(price: float, shares: float, is_sell: bool) -> float:
    """매매 1건의 거래비용(수수료+슬리피지, 매도 시 거래세 추가)."""
    value = price * shares
    cost = value * (config.COMMISSION_RATE + config.SLIPPAGE_RATE)
    if is_sell:
        cost += value * config.TAX_RATE
    return cost


def _mark_to_market(positions: dict[str, Position], ticker_data: dict[str, dict], date) -> float:
    """보유 포지션 전체를 당일 종가(없으면 마지막 매입가) 기준으로 평가한 금액."""
    total = 0.0
    for ticker, pos in positions.items():
        df = ticker_data[ticker]["df"]
        if date in df.index:
            total += pos.shares * df["close"].loc[date]
        else:
            total += pos.shares * pos.entry_price
    return total


def _prepare_ticker_data(
    raw_data: dict[str, pd.DataFrame], benchmark_close: pd.Series | None = None
) -> dict[str, dict]:
    """티커별 원시 OHLCV(+flow) df로부터 백테스트에 필요한 시계열을 미리 계산해둔다.

    score/stop_loss/position_pct/atr을 전체 기간에 대해 한 번씩만(벡터화)
    계산해 시뮬레이션 루프 안에서는 조회(.loc)만 하도록 한다.
    """
    ticker_data: dict[str, dict] = {}
    for ticker, df in raw_data.items():
        if df is None or df.empty:
            continue
        ticker_data[ticker] = {
            "df": df,
            "score": scorer.total_score_series(df, benchmark_close),
            "stop_loss": scorer.stop_loss_series(df),
            "position_pct": scorer.position_size_pct_series(df),
            "atr": volatility.atr(df, period=config.ATR_PERIOD),
        }
    return ticker_data


def _simulate(
    ticker_data: dict[str, dict],
    start,
    end,
    entry_score: float = config.BACKTEST_ENTRY_SCORE,
    exit_score: float = config.BACKTEST_EXIT_SCORE,
    initial_cash: float = config.BACKTEST_INITIAL_CASH,
    max_positions: int = config.MAX_CONCURRENT_POSITIONS,
) -> BacktestResult:
    """이미 준비된 ticker_data(스코어/국면/손절가 등이 계산된 상태)로 시뮬레이션만 실행한다.

    데이터 조회에 의존하지 않는 순수 함수라 합성 데이터로 직접 테스트하기 좋다.
    """
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

    trading_dates: set[pd.Timestamp] = set()
    for data in ticker_data.values():
        trading_dates.update(data["df"].loc[start_ts:end_ts].index)
    trading_dates = sorted(trading_dates)

    cash = initial_cash
    positions: dict[str, Position] = {}
    trades: list[Trade] = []
    equity_records: list[tuple[pd.Timestamp, float]] = []

    for date in trading_dates:
        # --- (a) 보유 종목 청산 판정: 손절/트레일링(당일 저가) > 스코어청산(전일 종가 기준, 당일 시가 체결) ---
        for ticker in list(positions.keys()):
            data = ticker_data[ticker]
            df = data["df"]
            if date not in df.index:
                continue
            pos = positions[ticker]
            idx = df.index.get_loc(date)
            low = df["low"].loc[date]
            high = df["high"].loc[date]
            open_ = df["open"].loc[date]
            close = df["close"].loc[date]

            exit_price = None
            exit_reason = None

            if low <= pos.current_stop_loss:
                # 1) 손절 또는 3) 트레일링(먼저 온 것이 우선 - 같은 로직으로 통합 처리됨.
                # current_stop_loss는 트레일링이 활성화되면 상향 조정되므로, 항상 그 시점의
                # 유효한 손절선을 의미한다)
                exit_price = pos.current_stop_loss
                exit_reason = "trailing_stop" if pos.trailing_active else "stop_loss"
            else:
                # 트레일링 갱신: 수익 1R 도달 시 본전 이동 후 3×ATR 추적
                pos.highest_high_since_entry = max(pos.highest_high_since_entry, high)
                if not pos.trailing_active and close >= pos.entry_price + pos.initial_risk:
                    pos.trailing_active = True
                    pos.current_stop_loss = max(pos.current_stop_loss, pos.entry_price)
                if pos.trailing_active:
                    atr_value = data["atr"].loc[date]
                    if not pd.isna(atr_value):
                        trail_level = pos.highest_high_since_entry - config.TRAILING_STOP_ATR_MULTIPLIER * atr_value
                        pos.current_stop_loss = max(pos.current_stop_loss, trail_level)

                # 2) 스코어 청산: 전일 종가 기준 계산된 스코어가 exit_score 미만이면 당일 시가 청산
                if idx > 0:
                    prev_date = df.index[idx - 1]
                    prev_score = data["score"].loc[prev_date]
                    if not pd.isna(prev_score) and prev_score < exit_score:
                        exit_price = open_
                        exit_reason = "score_exit"

            if exit_price is not None:
                shares = pos.shares
                sell_cost = _trading_cost(exit_price, shares, is_sell=True)
                cash += exit_price * shares - sell_cost
                buy_cost = _trading_cost(pos.entry_price, shares, is_sell=False)
                pnl = (exit_price - pos.entry_price) * shares - buy_cost - sell_cost
                trades.append(
                    Trade(
                        ticker=ticker,
                        entry_date=pos.entry_date,
                        entry_price=pos.entry_price,
                        exit_date=date,
                        exit_price=exit_price,
                        shares=shares,
                        exit_reason=exit_reason,
                        pnl=pnl,
                        return_pct=pnl / (pos.entry_price * shares),
                    )
                )
                del positions[ticker]

        # --- (b) 신규 진입 판정: 전일 종가 기준 스코어가 entry_score를 상향 돌파 -> 당일 시가 체결 ---
        if len(positions) < max_positions:
            candidates: list[tuple[float, str]] = []
            for ticker, data in ticker_data.items():
                if ticker in positions:
                    continue
                df = data["df"]
                if date not in df.index:
                    continue
                idx = df.index.get_loc(date)
                if idx < 2:
                    continue
                prev_date = df.index[idx - 1]
                prev2_date = df.index[idx - 2]
                score_prev = data["score"].loc[prev_date]
                score_prev2 = data["score"].loc[prev2_date]
                if pd.isna(score_prev) or pd.isna(score_prev2):
                    continue
                crossed_up = (score_prev2 < entry_score) and (score_prev >= entry_score)
                if crossed_up:
                    candidates.append((score_prev, ticker))

            candidates.sort(key=lambda item: item[0], reverse=True)  # 점수 높은 순으로 우선 배정

            for _, ticker in candidates:
                if len(positions) >= max_positions:
                    break
                data = ticker_data[ticker]
                df = data["df"]
                idx = df.index.get_loc(date)
                prev_date = df.index[idx - 1]
                open_price = df["open"].loc[date]
                stop_loss_price = data["stop_loss"].loc[prev_date]
                position_pct = data["position_pct"].loc[prev_date]

                if pd.isna(open_price) or pd.isna(stop_loss_price) or pd.isna(position_pct):
                    continue
                if stop_loss_price >= open_price or position_pct <= 0:
                    continue

                equity_now = cash + _mark_to_market(positions, ticker_data, date)
                position_value = equity_now * position_pct
                shares = np.floor(position_value / open_price)
                if shares <= 0:
                    continue

                buy_cost = _trading_cost(open_price, shares, is_sell=False)
                total_cost = open_price * shares + buy_cost
                if total_cost > cash:
                    affordable_shares = np.floor(
                        cash / (open_price * (1 + config.COMMISSION_RATE + config.SLIPPAGE_RATE))
                    )
                    if affordable_shares <= 0:
                        continue
                    shares = affordable_shares
                    buy_cost = _trading_cost(open_price, shares, is_sell=False)
                    total_cost = open_price * shares + buy_cost
                    if total_cost > cash:
                        continue

                cash -= total_cost
                initial_risk = open_price - stop_loss_price
                positions[ticker] = Position(
                    ticker=ticker,
                    shares=shares,
                    entry_date=date,
                    entry_price=open_price,
                    initial_stop_loss=stop_loss_price,
                    current_stop_loss=stop_loss_price,
                    initial_risk=initial_risk,
                    highest_high_since_entry=df["high"].loc[date],
                )

        # --- (c) 당일 종가 기준 총자산 기록 ---
        equity_records.append((date, cash + _mark_to_market(positions, ticker_data, date)))

    # 기간 종료 시점까지 남은 포지션은 마지막 날 종가로 청산 처리(성과 집계를 위함)
    if positions and trading_dates:
        last_date = trading_dates[-1]
        for ticker, pos in list(positions.items()):
            df = ticker_data[ticker]["df"]
            if last_date not in df.index:
                continue
            exit_price = df["close"].loc[last_date]
            sell_cost = _trading_cost(exit_price, pos.shares, is_sell=True)
            buy_cost = _trading_cost(pos.entry_price, pos.shares, is_sell=False)
            pnl = (exit_price - pos.entry_price) * pos.shares - buy_cost - sell_cost
            trades.append(
                Trade(
                    ticker=ticker,
                    entry_date=pos.entry_date,
                    entry_price=pos.entry_price,
                    exit_date=last_date,
                    exit_price=exit_price,
                    shares=pos.shares,
                    exit_reason="period_end",
                    pnl=pnl,
                    return_pct=pnl / (pos.entry_price * pos.shares),
                )
            )

    equity_curve = pd.Series({d: v for d, v in equity_records}, dtype=float).sort_index()
    return BacktestResult(trades=trades, equity_curve=equity_curve)


def run(
    tickers: list[str],
    start: str,
    end: str,
    entry_score: float = config.BACKTEST_ENTRY_SCORE,
    exit_score: float = config.BACKTEST_EXIT_SCORE,
    data_start: str | None = None,
    initial_cash: float = config.BACKTEST_INITIAL_CASH,
    max_positions: int = config.MAX_CONCURRENT_POSITIONS,
    benchmark_close: pd.Series | None = None,
    include_flow: bool = True,
) -> BacktestResult:
    """지정 종목들에 대해 [start, end] 구간 백테스트를 실행한다.

    data_start(미지정 시 start - BACKTEST_WARMUP_DAYS일)부터 데이터를 가져와 지표를
    미리 워밍업시키지만, 실제 매매/성과 집계는 [start, end] 구간에서만 이루어진다.
    include_flow=True이고 KRX_ID/KRX_PW가 설정되어 있으면 수급 데이터도 병합해
    수급 카테고리를 포함한 스코어를 계산한다(없으면 자동으로 수급 카테고리 제외 계산 불가 -
    해당 카테고리가 NaN이 되어 그 시점 스코어 전체가 계산 불가 처리되므로,
    수급 데이터 없이 백테스트하려면 별도 개선이 필요하다는 점에 유의).
    """
    start_ts = pd.Timestamp(start)
    data_start_ts = pd.Timestamp(data_start) if data_start else start_ts - pd.DateOffset(days=config.BACKTEST_WARMUP_DAYS)
    data_start_str = data_start_ts.strftime("%Y%m%d")

    raw_data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = loader.fetch_ohlcv(ticker, data_start_str, end)
        if df.empty:
            continue
        if include_flow:
            try:
                flow_df = loader.fetch_investor_flow(ticker, data_start_str, end)
                df = df.join(flow_df, how="left")
            except RuntimeError:
                pass  # KRX 미인증 등: 수급 데이터 없이 진행(수급 카테고리는 계산 불가로 처리됨)
        raw_data[ticker] = df

    if not raw_data:
        raise ValueError("유효한 OHLCV 데이터를 가져온 종목이 없습니다.")

    ticker_data = _prepare_ticker_data(raw_data, benchmark_close)
    result = _simulate(
        ticker_data, start, end, entry_score, exit_score, initial_cash, max_positions
    )
    result.benchmark_curve = benchmark_close
    return result
