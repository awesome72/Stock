"""거래량 지표: OBV, VWAP, MFI, 거래량 프로파일(매물대)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config


def obv(df: pd.DataFrame) -> pd.Series:
    """OBV(On Balance Volume).

    전일 대비 종가 상승/하락 방향으로 거래량을 더하거나 뺀 누적합.
    Requires: df['close'], df['volume']
    Returns: 누적 OBV Series.
    """
    direction = np.sign(df["close"].diff())  # diff(): 전일 대비 (과거 방향)
    signed_volume = direction * df["volume"]
    return signed_volume.cumsum()


def vwap(df: pd.DataFrame, window: int = config.VWAP_WINDOW) -> pd.Series:
    """롤링 거래량가중평균가격.

    이 프로그램은 일봉 데이터만 사용하므로 진짜 일중 VWAP가 아니라, 최근 window일의
    거래량가중평균가로 근사한 지지선 개념이다.
    Requires: df['high'], df['low'], df['close'], df['volume']
    Returns: VWAP Series.
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    pv = typical_price * df["volume"]
    return pv.rolling(window).sum() / df["volume"].rolling(window).sum()


def mfi(df: pd.DataFrame, period: int = config.MFI_PERIOD) -> pd.Series:
    """MFI(Money Flow Index, 거래량 가중 RSI).

    Requires: df['high'], df['low'], df['close'], df['volume']
    Returns: MFI Series(0~100).
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    raw_flow = typical_price * df["volume"]
    prev_typical = typical_price.shift(1)  # shift(1): 전일 대비 비교 (과거 방향)
    positive_flow = raw_flow.where(typical_price > prev_typical, 0.0)
    negative_flow = raw_flow.where(typical_price < prev_typical, 0.0)
    positive_sum = positive_flow.rolling(period).sum()
    negative_sum = negative_flow.rolling(period).sum()
    money_ratio = positive_sum / negative_sum
    return 100 - (100 / (1 + money_ratio))


def volume_profile_poc(
    df: pd.DataFrame,
    window: int = config.VOLUME_PROFILE_WINDOW,
    bins: int = config.VOLUME_PROFILE_BINS,
) -> pd.Series:
    """거래량 프로파일 매물대(Point Of Control).

    최근 window봉을 bins개 가격구간으로 나눠 거래량이 가장 많이 쌓인 가격대의 중간값을 반환한다.
    Requires: df['high'], df['low'], df['close'], df['volume']
    Returns: POC 가격 Series.
    """
    typical_price = ((df["high"] + df["low"] + df["close"]) / 3).to_numpy()
    vol = df["volume"].to_numpy()

    # raw=False로 매 윈도우마다 pandas Series를 새로 만들면 종목당 수천 번 호출되어
    # 눈에 띄게 느려진다(실측 ~3초/종목, 백테스트에서 지배적 비용). 대신 윈도우의
    # 정수 위치(row position)만 raw=True로 받아 미리 numpy 배열로 뽑아둔 typical_price/
    # volume을 직접 인덱싱한다 - 결과는 동일하고 훨씬 빠르다.
    def _poc(idx_window: np.ndarray) -> float:
        idx = idx_window.astype(np.int64)
        tp_w = typical_price[idx]
        lo, hi = tp_w.min(), tp_w.max()
        if hi <= lo:
            return np.nan
        vol_w = vol[idx]
        edges = np.linspace(lo, hi, bins + 1)
        bucket = np.clip(np.digitize(tp_w, edges) - 1, 0, bins - 1)
        bucket_vol = np.bincount(bucket, weights=vol_w, minlength=bins)
        best = bucket_vol.argmax()
        return (edges[best] + edges[best + 1]) / 2

    position = pd.Series(np.arange(len(df), dtype=float), index=df.index)
    return position.rolling(window).apply(_poc, raw=True)
