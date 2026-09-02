"""모멘텀 지표: RSI, 스토캐스틱, CCI, RSI 다이버전스."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config


def rsi(df: pd.DataFrame, period: int = config.RSI_PERIOD) -> pd.Series:
    """RSI(Wilder 방식).

    Requires: df['close']
    Returns: RSI Series(0~100). 워밍업 구간은 NaN.
    """
    delta = df["close"].diff()  # diff(): 전일 대비 변화량 (과거 방향)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def stochastic(
    df: pd.DataFrame,
    k_period: int = config.STOCH_K_PERIOD,
    smooth_k: int = config.STOCH_SMOOTH_K,
    d_period: int = config.STOCH_D_PERIOD,
) -> pd.DataFrame:
    """슬로우 스토캐스틱.

    Requires: df['high'], df['low'], df['close']
    Returns: DataFrame with columns k, d (0~100).
    """
    lowest_low = df["low"].rolling(k_period).min()
    highest_high = df["high"].rolling(k_period).max()
    raw_k = 100 * (df["close"] - lowest_low) / (highest_high - lowest_low)
    k = raw_k.rolling(smooth_k).mean()
    d = k.rolling(d_period).mean()
    return pd.DataFrame({"k": k, "d": d})


def cci(df: pd.DataFrame, period: int = config.CCI_PERIOD) -> pd.Series:
    """CCI(Commodity Channel Index).

    Requires: df['high'], df['low'], df['close']
    Returns: CCI Series.
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp = typical_price.rolling(period).mean()
    mean_abs_dev = typical_price.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (typical_price - sma_tp) / (0.015 * mean_abs_dev)


def rsi_divergence(
    df: pd.DataFrame,
    lookback: int = config.RSI_DIVERGENCE_LOOKBACK,
    pivot_window: int = config.RSI_DIVERGENCE_PIVOT_WINDOW,
    rsi_period: int = config.RSI_PERIOD,
) -> pd.Series:
    """RSI 다이버전스.

    최근 lookback봉 내에서 가장 최근 가격 스윙 고점 2개를 찾아 해당 시점 RSI와 비교한다.
    가격 고점 상승 + RSI 고점 하락 = bearish divergence(-1).
    가격 고점 하락 + RSI 고점 상승 = bullish divergence(+1). 그 외 0.

    스윙 고점은 좌우 대칭 피벗(확정을 위해 미래 1봉이 더 필요)이 아니라, 오직 과거
    pivot_window봉만 보는 'trailing 신고가 갱신 시점'으로 정의한다
    (high[i]가 직전 pivot_window봉 중 최고가일 때). 미래 데이터를 전혀 참조하지 않는다.

    Requires: df['high'], df['close']
    Returns: {-1, 0, 1} Series.
    """
    high = df["high"]
    rsi_series = rsi(df, period=rsi_period)

    rolling_max = high.rolling(pivot_window).max()
    is_swing_high = high >= rolling_max  # 과거 pivot_window봉 기준 신고가 갱신 시점 (미래 데이터 없음)

    swing_high_price = high.where(is_swing_high)
    swing_high_rsi = rsi_series.where(is_swing_high)
    price_arr = swing_high_price.to_numpy()
    rsi_arr = swing_high_rsi.to_numpy()

    # raw=False로 매 윈도우마다 pandas Series를 새로 만들고 .loc으로 조회하면
    # 종목당 수천 번 호출되어 눈에 띄게 느려진다(실측 ~2.4초/종목). 대신 윈도우의
    # 정수 위치(row position)만 raw=True로 받아 미리 뽑아둔 numpy 배열을 직접
    # 인덱싱한다 - 결과는 동일하고 훨씬 빠르다.
    def _last_two_divergence(idx_window: np.ndarray) -> float:
        idx = idx_window.astype(np.int64)
        prices_w = price_arr[idx]
        valid = np.nonzero(~np.isnan(prices_w))[0]
        if len(valid) < 2:
            return 0.0
        i1, i2 = valid[-2], valid[-1]
        p1, p2 = prices_w[i1], prices_w[i2]
        r1, r2 = rsi_arr[idx[i1]], rsi_arr[idx[i2]]
        if np.isnan(r1) or np.isnan(r2):
            return 0.0
        if p2 > p1 and r2 < r1:
            return -1.0
        if p2 < p1 and r2 > r1:
            return 1.0
        return 0.0

    position = pd.Series(np.arange(len(df), dtype=float), index=df.index)
    return position.rolling(lookback, min_periods=1).apply(_last_two_divergence, raw=True)
