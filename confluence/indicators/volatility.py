"""변동성 지표: 볼린저밴드, ATR, Keltner Channel, Squeeze 판별."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config


def atr(df: pd.DataFrame, period: int = config.ATR_PERIOD) -> pd.Series:
    """ATR(Average True Range, Wilder 방식).

    Requires: df['high'], df['low'], df['close']
    Returns: ATR Series. 워밍업 구간(첫 period-1개)은 NaN.
    """
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)  # shift(1): 전일 종가(과거) 참조
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _pct_rank_last(arr: np.ndarray) -> float:
    """배열의 마지막 값이 배열 내에서 차지하는 백분위(0~1, 자기 자신 포함)."""
    last = arr[-1]
    return float((arr <= last).sum()) / len(arr)


def bollinger(
    df: pd.DataFrame,
    period: int = config.BB_PERIOD,
    num_std: float = config.BB_STD,
    width_lookback: int = config.REGIME_BB_WIDTH_LOOKBACK,
) -> pd.DataFrame:
    """볼린저밴드 + 상대 밴드폭 및 그 백분위(Squeeze 판별용).

    Requires: df['close']
    Returns: DataFrame with columns mid, upper, lower, bandwidth, bandwidth_percentile(0~1).
    bandwidth_percentile은 미래 데이터를 쓰지 않도록 과거 width_lookback봉 내에서의
    상대 순위로 계산한다(전체 기간 percentile이 아님 - look-ahead 금지 규칙).
    """
    mid = df["close"].rolling(period).mean()
    std_dev = df["close"].rolling(period).std()
    upper = mid + num_std * std_dev
    lower = mid - num_std * std_dev
    bandwidth = (upper - lower) / mid

    bandwidth_percentile = bandwidth.rolling(width_lookback).apply(_pct_rank_last, raw=True)

    return pd.DataFrame(
        {
            "mid": mid,
            "upper": upper,
            "lower": lower,
            "bandwidth": bandwidth,
            "bandwidth_percentile": bandwidth_percentile,
        }
    )


def keltner(
    df: pd.DataFrame,
    ema_period: int = config.KELTNER_EMA_PERIOD,
    atr_period: int = config.ATR_PERIOD,
    multiplier: float = config.KELTNER_MULTIPLIER,
) -> pd.DataFrame:
    """Keltner Channel.

    Requires: df['high'], df['low'], df['close']
    Returns: DataFrame with columns mid, upper, lower.
    """
    mid = df["close"].ewm(span=ema_period, adjust=False, min_periods=ema_period).mean()
    atr_series = atr(df, period=atr_period)
    upper = mid + multiplier * atr_series
    lower = mid - multiplier * atr_series
    return pd.DataFrame({"mid": mid, "upper": upper, "lower": lower})


def squeeze_flag(
    df: pd.DataFrame,
    bb_period: int = config.BB_PERIOD,
    bb_std: float = config.BB_STD,
    keltner_ema_period: int = config.KELTNER_EMA_PERIOD,
    atr_period: int = config.ATR_PERIOD,
    keltner_multiplier: float = config.KELTNER_MULTIPLIER,
) -> pd.Series:
    """볼린저밴드가 Keltner Channel 내부에 완전히 들어온 상태(Squeeze) 여부.

    Requires: df['high'], df['low'], df['close']
    Returns: nullable boolean Series. 계산 불가 구간은 0으로 채우지 않고 <NA>.
    """
    bb = bollinger(df, period=bb_period, num_std=bb_std)
    kc = keltner(df, ema_period=keltner_ema_period, atr_period=atr_period, multiplier=keltner_multiplier)

    flag = ((bb["upper"] < kc["upper"]) & (bb["lower"] > kc["lower"])).astype("boolean")
    invalid = bb["upper"].isna() | bb["lower"].isna() | kc["upper"].isna() | kc["lower"].isna()
    return flag.mask(invalid)
