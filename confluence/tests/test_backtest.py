"""백테스트 엔진 테스트: 체결가/시점 규칙 + look-ahead 방지 검증."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from confluence.backtest import engine, metrics


def _make_hand_crafted_ticker_data(score_values: list[float], open_override: dict[int, float]) -> tuple[dict, pd.DatetimeIndex]:
    """score/stop_loss/position_pct/atr을 실제 지표 계산 없이 직접 주입한 단일 종목 데이터.

    체결 시점/가격 규칙만 순수하게 검증하기 위해 실제 지표 파이프라인을 우회한다.
    """
    n = len(score_values)
    dates = pd.bdate_range("2024-01-02", periods=n)
    close = pd.Series([100.0] * n, index=dates)
    open_ = close.copy()
    for i, v in open_override.items():
        open_.iloc[i] = v
    high = pd.concat([open_, close], axis=1).max(axis=1) + 1
    low = pd.concat([open_, close], axis=1).min(axis=1) - 1
    volume = pd.Series([1_000.0] * n, index=dates)
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=dates)

    ticker_data = {
        "TEST": {
            "df": df,
            "score": pd.Series(score_values, index=dates, dtype=float),
            "regime": pd.Series(["RANGING"] * n, index=dates),
            "stop_loss": pd.Series([1.0] * n, index=dates),  # 매우 낮게 설정 -> 손절 미발동
            "position_pct": pd.Series([0.1] * n, index=dates),
            "atr": pd.Series([1.0] * n, index=dates),
        }
    }
    return ticker_data, dates


def test_entry_executes_at_next_bar_open_not_signal_close():
    # score: ... 79(idx3) -> 85(idx4)로 상향 돌파. 신호일은 idx4, 체결은 idx5의 시가여야 한다.
    scores = [50, 50, 50, 79, 85, 85, 85, 85, 85, 85]
    ticker_data, dates = _make_hand_crafted_ticker_data(scores, open_override={5: 999.0})

    result = engine._simulate(
        ticker_data, dates[0], dates[-1], entry_score=80, exit_score=45, initial_cash=1_000_000.0, max_positions=8
    )

    assert len(result.trades) == 1  # 기간 종료 시 강제 청산되어 거래 1건으로 기록됨
    trade = result.trades[0]
    assert trade.entry_date == dates[5]
    assert trade.entry_price == pytest.approx(999.0)  # 신호일(dates[4]) 종가(100)가 아님


def test_score_exit_executes_at_next_bar_open_not_signal_close():
    # idx4에서 진입(80 상향돌파, idx5 시가 체결) 후 idx7에서 45 미만으로 하락 -> idx8 시가 청산.
    scores = [50, 50, 50, 79, 85, 85, 85, 40, 40, 40]
    ticker_data, dates = _make_hand_crafted_ticker_data(scores, open_override={5: 200.0, 8: 555.0})

    result = engine._simulate(
        ticker_data, dates[0], dates[-1], entry_score=80, exit_score=45, initial_cash=1_000_000.0, max_positions=8
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.entry_date == dates[5]
    assert trade.exit_date == dates[8]
    assert trade.exit_reason == "score_exit"
    assert trade.exit_price == pytest.approx(555.0)  # 신호일(dates[7]) 종가(100)가 아님


def test_stop_loss_executes_at_stop_price():
    n = 10
    dates = pd.bdate_range("2024-01-02", periods=n)
    close = pd.Series([100.0] * n, index=dates)
    open_ = close.copy()
    open_.iloc[3] = 100.0
    high = close + 1
    low = close - 1
    low.iloc[5] = 80.0  # idx5에서 저가가 손절선(90)을 이탈

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": pd.Series([1000.0] * n, index=dates)},
        index=dates,
    )
    scores = pd.Series([50, 50, 79, 85, 85, 85, 85, 85, 85, 85], index=dates, dtype=float)
    ticker_data = {
        "TEST": {
            "df": df,
            "score": scores,
            "regime": pd.Series(["RANGING"] * n, index=dates),
            "stop_loss": pd.Series([90.0] * n, index=dates),
            "position_pct": pd.Series([0.1] * n, index=dates),
            "atr": pd.Series([1.0] * n, index=dates),
        }
    }

    result = engine._simulate(
        ticker_data, dates[0], dates[-1], entry_score=80, exit_score=45, initial_cash=1_000_000.0, max_positions=8
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_date == dates[5]
    assert trade.exit_reason == "stop_loss"
    assert trade.exit_price == pytest.approx(90.0)


def _make_trending_ohlcv(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n)
    returns = rng.normal(0.0006, 0.014, n)  # 완만한 상승 드리프트 + 노이즈 -> 진입/청산 신호 발생 유도
    close = 100 * np.cumprod(1 + returns)
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    volume = rng.integers(10_000, 500_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=dates
    )


def test_engine_never_uses_data_beyond_end():
    """end 이후의 데이터가 존재해도 [start, end] 구간의 결과에는 영향을 주지 않아야 한다."""
    df_full = _make_trending_ohlcv(300, seed=11)
    df_short = df_full.iloc[:250]

    start = df_short.index[100]
    end = df_short.index[-1]  # 두 실행 모두 동일한 end 사용 (df_full은 그 뒤에 50봉이 더 있음)

    data_full = engine._prepare_ticker_data({"TEST": df_full})
    data_short = engine._prepare_ticker_data({"TEST": df_short})

    result_full = engine._simulate(data_full, start, end, initial_cash=1_000_000.0)
    result_short = engine._simulate(data_short, start, end, initial_cash=1_000_000.0)

    pd.testing.assert_series_equal(result_full.equity_curve, result_short.equity_curve, check_names=False)

    assert len(result_full.trades) == len(result_short.trades)
    for t_full, t_short in zip(result_full.trades, result_short.trades):
        assert t_full.entry_date == t_short.entry_date
        assert t_full.entry_price == pytest.approx(t_short.entry_price)
        assert t_full.exit_date == t_short.exit_date
        assert t_full.exit_price == pytest.approx(t_short.exit_price)
        assert t_full.exit_reason == t_short.exit_reason


def test_walk_forward_rolls_and_date_slicing():
    """walk_forward가 run_fn을 올바른 [test_start, test_end], data_start=train_start로 호출하는지 확인."""
    calls: list[dict] = []

    def fake_run(tickers, start, end, data_start=None, **kwargs):
        calls.append({"start": pd.Timestamp(start), "end": pd.Timestamp(end), "data_start": pd.Timestamp(data_start)})
        dates = pd.bdate_range(start, end)
        equity = pd.Series(np.linspace(100.0, 110.0, len(dates)), index=dates)
        return engine.BacktestResult(trades=[], equity_curve=equity)

    result = metrics.walk_forward(
        fake_run, ["X"], "20150101", "20250101", train_years=3, test_years=1, rolls=5
    )

    assert len(result) == 5  # 2015+3+1=2019 ... 2019+4=2023, 전부 2025 이내라 5구간 모두 포함
    assert len(calls) == 5
    assert calls[0]["data_start"] == pd.Timestamp("2015-01-01")
    assert calls[0]["start"] == pd.Timestamp("2018-01-01")
    assert calls[0]["end"] == pd.Timestamp("2019-01-01")
    assert calls[1]["data_start"] == pd.Timestamp("2016-01-01")
    assert calls[1]["start"] == pd.Timestamp("2019-01-01")


def test_metrics_summary_basic_sanity():
    dates = pd.bdate_range("2023-01-02", periods=5)
    equity = pd.Series([100.0, 110.0, 90.0, 120.0, 130.0], index=dates)
    trade_pnl = pd.Series([10.0, -5.0, 20.0, -2.0])

    result = metrics.summary(equity, trade_pnl)

    assert result["거래횟수"] == 4
    assert result["승률"] == pytest.approx(0.5)
    assert result["MDD"] < 0
    assert result["Profit Factor"] == pytest.approx(30 / 7)
