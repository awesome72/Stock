"""지표 모듈 테스트: 수기 계산값 대조 + look-ahead 방지 검증."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from confluence.indicators import flow, momentum, trend, volatility, volume


# ---------------------------------------------------------------------------
# 수기 계산값 대조 (RSI, MACD, ATR)
# ---------------------------------------------------------------------------


def test_rsi_known_values():
    # closes = [100, 102, 101, 105, 103], period=3 (Wilder, alpha=1/3, ewm adjust=False)
    # gain=[nan,2,0,4,0], loss=[nan,0,1,0,2]
    # avg_gain: idx1=2, idx2=4/3, idx3=20/9, idx4=40/27 (min_periods=3 -> idx0,1,2 NaN)
    # avg_loss: idx1=0, idx2=1/3, idx3=2/9, idx4=22/27
    # RSI[3] = 100 - 100/(1 + (20/9)/(2/9)) = 100 - 100/11 = 1000/11
    # RSI[4] = 100 - 100/(1 + (40/27)/(22/27)) = 100 - 1100/31 = 2000/31
    dates = pd.bdate_range("2024-01-02", periods=5)
    df = pd.DataFrame({"close": [100.0, 102.0, 101.0, 105.0, 103.0]}, index=dates)

    result = momentum.rsi(df, period=3)

    assert result.iloc[0:3].isna().all()
    assert result.iloc[3] == pytest.approx(1000 / 11)
    assert result.iloc[4] == pytest.approx(2000 / 31)


def test_rsi_monotonic_bounds():
    dates = pd.bdate_range("2024-01-02", periods=10)
    up = pd.DataFrame({"close": np.arange(100, 110, dtype=float)}, index=dates)
    down = pd.DataFrame({"close": np.arange(110, 100, -1, dtype=float)}, index=dates)

    assert momentum.rsi(up, period=3).iloc[-1] == pytest.approx(100.0)
    assert momentum.rsi(down, period=3).iloc[-1] == pytest.approx(0.0)


def test_macd_known_values():
    # closes = [10, 12, 15, 11, 14], fast=2(alpha=2/3), slow=3(alpha=1/2), signal=2(alpha=2/3)
    # ema_fast: e0=10, e1=34/3, e2=124/9, e3=322/27, e4=1078/81 (min_periods=2 -> idx0 NaN)
    # ema_slow: e0=10, e1=11, e2=13, e3=12, e4=13 (min_periods=3 -> idx0,1 NaN)
    # macd_line(둘다 유효한 구간만): idx2=124/9-13=7/9, idx3=322/27-12=-2/27, idx4=1078/81-13=25/81
    # signal_line(macd_line에 ewm, min_periods=2, idx2에서 시작): idx2=7/9(1개=NaN),
    #   idx3=(2/3)(-2/27)+(1/3)(7/9)=17/81, idx4=(2/3)(25/81)+(1/3)(17/81)=67/243
    # histogram = macd_line - signal_line: idx3=-2/27-17/81=-23/81, idx4=25/81-67/243=8/243
    dates = pd.bdate_range("2024-01-02", periods=5)
    df = pd.DataFrame({"close": [10.0, 12.0, 15.0, 11.0, 14.0]}, index=dates)

    result = trend.macd(df, fast=2, slow=3, signal=2)

    assert result["macd"].iloc[2] == pytest.approx(7 / 9)
    assert result["macd"].iloc[3] == pytest.approx(-2 / 27)
    assert result["macd"].iloc[4] == pytest.approx(25 / 81)
    assert pd.isna(result["signal"].iloc[2])
    assert result["signal"].iloc[3] == pytest.approx(17 / 81)
    assert result["signal"].iloc[4] == pytest.approx(67 / 243)
    assert pd.isna(result["histogram"].iloc[2])
    assert result["histogram"].iloc[3] == pytest.approx(-23 / 81)
    assert result["histogram"].iloc[4] == pytest.approx(8 / 243)


def test_atr_known_values():
    # H=[10,11,12,11,13], L=[8,9,10,9,10], C=[9,10,11,10,12], period=3
    # TR = [2,2,2,2,3] (계산 근거: idx0은 H-L만 유효(2), 이후는 |H-prevC|,|L-prevC| 포함해도 최대값이 동일하게 2 또는 마지막에 3)
    # ATR(Wilder, alpha=1/3): idx0,1 NaN(min_periods=3), idx2=2.0, idx3=2.0, idx4=7/3
    dates = pd.bdate_range("2024-01-02", periods=5)
    df = pd.DataFrame(
        {
            "high": [10.0, 11.0, 12.0, 11.0, 13.0],
            "low": [8.0, 9.0, 10.0, 9.0, 10.0],
            "close": [9.0, 10.0, 11.0, 10.0, 12.0],
        },
        index=dates,
    )

    result = volatility.atr(df, period=3)

    assert result.iloc[0:2].isna().all()
    assert result.iloc[2] == pytest.approx(2.0)
    assert result.iloc[3] == pytest.approx(2.0)
    assert result.iloc[4] == pytest.approx(7 / 3)


# ---------------------------------------------------------------------------
# look-ahead 방지 테스트 (핵심)
# 데이터 마지막 20봉을 잘라내고 계산해도, 겹치는 구간의 지표값은 전체 데이터로
# 계산한 값과 동일해야 한다. 다르다면 어딘가에서 미래 데이터를 참조했다는 뜻이다.
# ---------------------------------------------------------------------------


def _make_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n)
    returns = rng.normal(0, 0.01, n)
    close = 100 * np.cumprod(1 + returns)
    open_ = close * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
    vol = rng.integers(1_000, 100_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=dates,
    )


def _make_flow(n: int = 200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n)
    foreign_net = rng.normal(0, 1e8, n)
    institution_net = rng.normal(0, 1e8, n)
    individual_net = -(foreign_net + institution_net)
    return pd.DataFrame(
        {"foreign_net": foreign_net, "institution_net": institution_net, "individual_net": individual_net},
        index=dates,
    )


OHLCV_INDICATORS = [
    ("sma_alignment", trend.sma_alignment, {}),
    ("macd", trend.macd, {}),
    ("adx_dmi", trend.adx_dmi, {}),
    ("ichimoku", trend.ichimoku, {}),
    ("supertrend", trend.supertrend, {}),
    ("rsi", momentum.rsi, {}),
    ("stochastic", momentum.stochastic, {}),
    ("cci", momentum.cci, {}),
    ("rsi_divergence", momentum.rsi_divergence, {}),
    ("bollinger", volatility.bollinger, {}),
    ("atr", volatility.atr, {}),
    ("keltner", volatility.keltner, {}),
    ("squeeze_flag", volatility.squeeze_flag, {}),
    ("obv", volume.obv, {}),
    ("vwap", volume.vwap, {}),
    ("mfi", volume.mfi, {}),
    ("volume_profile_poc", volume.volume_profile_poc, {}),
]

FLOW_INDICATORS = [
    ("foreign_net_streak", flow.foreign_net_streak, {}),
    ("institution_net_streak", flow.institution_net_streak, {}),
    ("flow_score_5d", flow.flow_score_5d, {}),
    ("flow_score_20d", flow.flow_score_20d, {}),
]


def _assert_overlap_equal(result_full, result_trunc, overlap_index):
    full_overlap = result_full.loc[overlap_index]
    trunc_overlap = result_trunc.loc[overlap_index]
    if isinstance(full_overlap, pd.DataFrame):
        pd.testing.assert_frame_equal(full_overlap, trunc_overlap, rtol=1e-9, atol=1e-9)
    else:
        pd.testing.assert_series_equal(
            full_overlap, trunc_overlap, rtol=1e-9, atol=1e-9, check_names=False
        )


@pytest.mark.parametrize("name,func,kwargs", OHLCV_INDICATORS, ids=[c[0] for c in OHLCV_INDICATORS])
def test_lookahead_ohlcv_indicators(name, func, kwargs):
    df_full = _make_ohlcv()
    df_trunc = df_full.iloc[:-20]

    result_full = func(df_full, **kwargs)
    result_trunc = func(df_trunc, **kwargs)

    _assert_overlap_equal(result_full, result_trunc, df_trunc.index)


@pytest.mark.parametrize("name,func,kwargs", FLOW_INDICATORS, ids=[c[0] for c in FLOW_INDICATORS])
def test_lookahead_flow_indicators(name, func, kwargs):
    df_full = _make_flow()
    df_trunc = df_full.iloc[:-20]

    result_full = func(df_full, **kwargs)
    result_trunc = func(df_trunc, **kwargs)

    _assert_overlap_equal(result_full, result_trunc, df_trunc.index)
