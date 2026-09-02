"""추세 지표: SMA 정배열, MACD, ADX/DMI, 일목균형표, Supertrend."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from .volatility import atr as _atr


def sma_alignment(df: pd.DataFrame, windows: tuple[int, ...] = config.SMA_WINDOWS) -> pd.Series:
    """이동평균 정배열 점수.

    windows(오름차순, 예: 5/20/60/120)의 모든 쌍에 대해 '짧은 기간 SMA > 긴 기간 SMA'이면 +1,
    반대면 -1을 합산한다. 완전 정배열이면 +C(n,2), 완전 역배열이면 -C(n,2).

    Requires: df['close']
    Returns: 점수 Series. 가장 긴 window의 rolling 계산이 끝나기 전까지는 NaN.
    """
    smas = [df["close"].rolling(w).mean() for w in windows]
    n = len(windows)
    score = pd.Series(0.0, index=df.index)
    for i in range(n - 1):
        for j in range(i + 1, n):
            score = score + np.sign(smas[i] - smas[j])
    return score.where(smas[-1].notna())


def macd(
    df: pd.DataFrame,
    fast: int = config.MACD_FAST,
    slow: int = config.MACD_SLOW,
    signal: int = config.MACD_SIGNAL,
) -> pd.DataFrame:
    """MACD, Signal, Histogram.

    Requires: df['close']
    Returns: DataFrame with columns macd, signal, histogram.
    """
    ema_fast = df["close"].ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "histogram": histogram})


def adx_dmi(df: pd.DataFrame, period: int = config.ADX_PERIOD) -> pd.DataFrame:
    """ADX / +DI / -DI (Wilder 방식).

    Requires: df['high'], df['low'], df['close']
    Returns: DataFrame with columns adx, plus_di, minus_di.
    """
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)  # shift(1): 전일 종가(과거) 참조
    prev_high = high.shift(1)  # shift(1): 전일 고가(과거) 참조
    prev_low = low.shift(1)  # shift(1): 전일 저가(과거) 참조

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    alpha = 1.0 / period
    tr_smooth = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    return pd.DataFrame({"adx": adx, "plus_di": plus_di, "minus_di": minus_di})


def ichimoku(
    df: pd.DataFrame,
    tenkan: int = config.ICHIMOKU_TENKAN,
    kijun: int = config.ICHIMOKU_KIJUN,
    senkou_b_period: int = config.ICHIMOKU_SENKOU_B,
) -> pd.DataFrame:
    """일목균형표: 전환선/기준선/선행스팬A·B/후행스팬 시그널.

    Requires: df['high'], df['low'], df['close']
    Returns: DataFrame with columns tenkan, kijun, senkou_a, senkou_b, chikou_signal.

    선행스팬(senkou_a, senkou_b)은 관례대로 kijun기간만큼 shift(kijun)하여, 오늘 화면에
    실제로 표시되는 구름 값과 맞춘다(shift(kijun)은 i-kijun 시점에 계산된 값을 i 시점으로
    가져오는 것이므로 과거 데이터만 사용 - 미래 참조 아님).
    후행스팬(전통적 정의: 종가를 kijun기간만큼 shift(-kijun)하여 과거 위치에 표시)은
    시점 t의 값에 t+kijun의 데이터를 노출시켜 이 프로젝트의 look-ahead 금지 규칙을
    위반하므로 사용하지 않는다. 대신 동일한 정보를 인과적으로 담은
    chikou_signal = close[t] - close[t-kijun] (오늘 종가와 kijun일 전 종가의 비교)로 대체한다.
    """
    high, low, close = df["high"], df["low"], df["close"]

    tenkan_line = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_line = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    senkou_a = ((tenkan_line + kijun_line) / 2).shift(kijun)  # shift(kijun): 과거 계산값을 오늘 위치로 이동 (안전)
    senkou_b = (
        (high.rolling(senkou_b_period).max() + low.rolling(senkou_b_period).min()) / 2
    ).shift(kijun)  # 위와 동일하게 과거 방향 shift
    chikou_signal = close - close.shift(kijun)  # shift(kijun): kijun일 전 종가(과거)와 비교

    return pd.DataFrame(
        {
            "tenkan": tenkan_line,
            "kijun": kijun_line,
            "senkou_a": senkou_a,
            "senkou_b": senkou_b,
            "chikou_signal": chikou_signal,
        }
    )


def supertrend(
    df: pd.DataFrame,
    atr_period: int = config.SUPERTREND_ATR_PERIOD,
    multiplier: float = config.SUPERTREND_MULTIPLIER,
) -> pd.DataFrame:
    """Supertrend: 추세 추종 밴드 및 방향.

    Requires: df['high'], df['low'], df['close']
    Returns: DataFrame with columns supertrend(밴드 값), direction(1=상승추세, -1=하락추세).

    밴드가 직전 봉의 밴드/종가에 따라 조건부로 유지되는 재귀적 정의라 rolling/ewm 같은
    벡터화 연산으로 표현할 수 없다(pandas-ta, ta-lib 등 표준 구현도 동일하게 순차 루프 사용).
    따라서 이 함수만 예외적으로 단일 순방향 루프를 사용한다. 각 시점은 자신과 바로 직전
    시점의 값만 참조하므로 미래 데이터를 사용하지 않는다(look-ahead 아님).
    """
    atr_series = _atr(df, period=atr_period)
    mid = (df["high"] + df["low"]) / 2
    upper_basic = (mid + multiplier * atr_series).to_numpy()
    lower_basic = (mid - multiplier * atr_series).to_numpy()
    close = df["close"].to_numpy()
    n = len(df)

    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    trend = np.full(n, np.nan)
    direction = np.full(n, np.nan)

    first_valid = atr_series.first_valid_index()
    if first_valid is None:
        return pd.DataFrame({"supertrend": trend, "direction": direction}, index=df.index)

    start = df.index.get_loc(first_valid)
    upper_band[start] = upper_basic[start]
    lower_band[start] = lower_basic[start]
    direction[start] = 1.0
    trend[start] = lower_band[start]

    for i in range(start + 1, n):
        upper_band[i] = (
            upper_basic[i]
            if (upper_basic[i] < upper_band[i - 1] or close[i - 1] > upper_band[i - 1])
            else upper_band[i - 1]
        )
        lower_band[i] = (
            lower_basic[i]
            if (lower_basic[i] > lower_band[i - 1] or close[i - 1] < lower_band[i - 1])
            else lower_band[i - 1]
        )
        if close[i] > upper_band[i - 1]:
            direction[i] = 1.0
        elif close[i] < lower_band[i - 1]:
            direction[i] = -1.0
        else:
            direction[i] = direction[i - 1]
        trend[i] = lower_band[i] if direction[i] == 1.0 else upper_band[i]

    return pd.DataFrame({"supertrend": trend, "direction": direction}, index=df.index)
