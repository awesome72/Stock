"""시장 국면(Regime) 판별: 추세상승/추세하락/횡보/변동성확대."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from ..indicators import trend, volatility

REGIME_LABEL_KO: dict[str, str] = {
    "TRENDING_UP": "추세 상승",
    "TRENDING_DOWN": "추세 하락",
    "RANGING": "횡보",
    "VOLATILE": "변동성 확대",
}


def classify_regime(df: pd.DataFrame) -> pd.Series:
    """각 시점의 시장 국면을 판별한다.

    - TRENDING_UP:   ADX >= REGIME_ADX_TRENDING and +DI > -DI
    - TRENDING_DOWN: ADX >= REGIME_ADX_TRENDING and -DI > +DI
    - RANGING:       ADX < REGIME_ADX_RANGING and 볼린저밴드폭이 최근 120일 하위 30%
    - VOLATILE:      ATR이 20일 평균의 1.5배 이상
    - 우선순위: VOLATILE > TRENDING_UP/TRENDING_DOWN > RANGING > 기본값(RANGING)

    Requires: df['high'], df['low'], df['close']
    Returns: config.REGIME_LABELS 값을 갖는 문자열 Series. 계산 불가 구간은 NaN.
    """
    adx_df = trend.adx_dmi(df, period=config.ADX_PERIOD)
    atr_series = volatility.atr(df, period=config.ATR_PERIOD)
    bb = volatility.bollinger(
        df, period=config.BB_PERIOD, num_std=config.BB_STD, width_lookback=config.REGIME_BB_WIDTH_LOOKBACK
    )

    adx = adx_df["adx"]
    plus_di = adx_df["plus_di"]
    minus_di = adx_df["minus_di"]

    atr_avg = atr_series.rolling(config.REGIME_ATR_LOOKBACK).mean()
    is_volatile = atr_series >= config.REGIME_ATR_VOLATILE_MULTIPLIER * atr_avg

    is_trending_up = (adx >= config.REGIME_ADX_TRENDING) & (plus_di > minus_di)
    is_trending_down = (adx >= config.REGIME_ADX_TRENDING) & (minus_di > plus_di)
    is_ranging = (adx < config.REGIME_ADX_RANGING) & (
        bb["bandwidth_percentile"] <= config.REGIME_BB_WIDTH_PERCENTILE
    )

    # np.select: 앞에 있는 조건일수록 우선순위가 높다 (첫 매치가 채택됨)
    conditions = [is_volatile, is_trending_up, is_trending_down, is_ranging]
    choices = ["VOLATILE", "TRENDING_UP", "TRENDING_DOWN", "RANGING"]
    regime = pd.Series(np.select(conditions, choices, default="RANGING"), index=df.index)

    invalid = adx.isna() | atr_avg.isna() | bb["bandwidth_percentile"].isna()
    return regime.mask(invalid)
