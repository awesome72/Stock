"""국면별 가중치를 적용한 0~100점 스코어링 엔진."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import config
from ..indicators import flow as flow_indicators
from ..indicators import momentum, trend, volatility, volume
from . import regime as regime_engine


@dataclass
class ScoreCard:
    """단일 종목·시점의 스코어카드."""

    ticker: str
    date: pd.Timestamp
    total: float
    grade: str
    regime: str
    category_scores: dict[str, dict[str, float]]
    evidences: list[str] = field(default_factory=list)
    invalidation: str = ""
    stop_loss: float = float("nan")
    position_size_pct: float = 0.0


def grade(total: float) -> str:
    """점수 구간을 등급 문자열로 변환한다.

    45~64 구간은 반드시 '진입 금지'로 표기한다('약한 매수' 등으로 바꾸지 말 것 -
    애매한 구간에 매수 뉘앙스의 이름을 붙이면 사람은 반드시 진입하기 때문).
    """
    if total >= config.GRADE_STRONG_CONFLUENCE:
        return "강한 합의"
    if total >= config.GRADE_WATCH:
        return "관심"
    if total >= config.GRADE_NO_ENTRY:
        return "진입 금지"
    return "회피"


def _streak_as_of(bool_series: pd.Series, date) -> int:
    """date 시점까지의 데이터만으로, 마지막 상태(True/False)가 연속된 일수를 구한다."""
    s = bool_series.loc[:date]
    group_id = (s != s.shift(1)).cumsum()  # shift(1): 직전값과 비교 (과거 방향)
    streak = s.groupby(group_id).cumcount() + 1
    return int(streak.iloc[-1])


def _combine(raw_scores: dict[str, float], regime_label: str) -> tuple[float, dict[str, dict[str, float]]]:
    """카테고리별 원점수(0~1)에 국면별 가중치를 곱해 합산한다.

    모든 국면의 가중치 합은 100이므로, raw_scores가 전부 1.0이면 total은 정확히 100이다.
    """
    weights = config.REGIME_CATEGORY_WEIGHTS[regime_label]
    category_scores = {
        cat: {"earned": raw_scores[cat] * weights[cat], "max": float(weights[cat])} for cat in config.CATEGORIES
    }
    total = sum(v["earned"] for v in category_scores.values())
    return total, category_scores


def _trend_score(df: pd.DataFrame, date) -> tuple[float, list[tuple[float, str]]]:
    """추세 카테고리 원점수(0~1). 내부 배점: 정배열 15, MACD 10, 일목 10 (합 35)."""
    sma_series = trend.sma_alignment(df)
    macd_df = trend.macd(df)
    ichimoku_df = trend.ichimoku(df)

    sma_value = sma_series.loc[date]
    macd_value = macd_df["macd"].loc[date]
    signal_value = macd_df["signal"].loc[date]
    hist_value = macd_df["histogram"].loc[date]
    tenkan_value = ichimoku_df["tenkan"].loc[date]
    kijun_value = ichimoku_df["kijun"].loc[date]
    senkou_a_value = ichimoku_df["senkou_a"].loc[date]
    senkou_b_value = ichimoku_df["senkou_b"].loc[date]
    chikou_value = ichimoku_df["chikou_signal"].loc[date]
    close_value = df["close"].loc[date]

    required = [
        sma_value, macd_value, signal_value, hist_value,
        tenkan_value, kijun_value, senkou_a_value, senkou_b_value, chikou_value,
    ]
    if any(pd.isna(v) for v in required):
        return float("nan"), []

    n = len(config.SMA_WINDOWS)
    max_sma_score = n * (n - 1) / 2
    sma_norm = (sma_value + max_sma_score) / (2 * max_sma_score)

    if macd_value > signal_value and hist_value > 0:
        macd_score = 1.0
    elif macd_value > signal_value:
        macd_score = 0.75
    elif hist_value > 0:
        macd_score = 0.25
    else:
        macd_score = 0.0

    cloud_top = max(senkou_a_value, senkou_b_value)
    cloud_bottom = min(senkou_a_value, senkou_b_value)
    cloud_score = 1.0 if close_value > cloud_top else (0.0 if close_value < cloud_bottom else 0.5)
    cross_score = 1.0 if tenkan_value > kijun_value else 0.0
    chikou_score = 1.0 if chikou_value > 0 else 0.0
    ichimoku_score = (cloud_score + cross_score + chikou_score) / 3

    trend_raw = (15 * sma_norm + 10 * macd_score + 10 * ichimoku_score) / 35

    evidences: list[tuple[float, str]] = []
    if sma_norm != 0.5:
        direction = "정배열" if sma_value > 0 else "역배열"
        evidences.append((abs(sma_norm - 0.5), f"이동평균 {direction} (정배열점수 {sma_value:+.0f})"))

    is_bullish_cross = macd_value > signal_value
    streak = _streak_as_of(macd_df["macd"] > macd_df["signal"], date)
    cross_label = "골든크로스" if is_bullish_cross else "데드크로스"
    evidences.append((abs(macd_score - 0.5), f"MACD {cross_label} {streak}일차"))

    if cloud_score != 0.5:
        position = "구름대 위" if cloud_score == 1.0 else "구름대 아래"
        evidences.append((abs(ichimoku_score - 0.5), f"일목균형표 {position}"))

    return trend_raw, evidences


def _momentum_score(df: pd.DataFrame, date, current_regime: str) -> tuple[float, list[tuple[float, str]]]:
    """모멘텀 카테고리 원점수(0~1). 내부 배점: RSI 10, 스토캐스틱 5, 다이버전스 5 (합 20).

    RSI는 국면에 따라 해석이 달라진다: 추세장/변동성 확대 국면에서는 과매수(70+)를
    매도 신호로 보지 않고 모멘텀 강도로 해석하며, 횡보장에서만 평균회귀(과매수=약세,
    과매도=강세) 오실레이터로 해석한다.
    """
    rsi_series = momentum.rsi(df, period=config.RSI_PERIOD)
    stoch = momentum.stochastic(df)
    divergence_series = momentum.rsi_divergence(df)

    rsi_value = rsi_series.loc[date]
    k_value = stoch["k"].loc[date]
    divergence_value = divergence_series.loc[date]

    if any(pd.isna(v) for v in [rsi_value, k_value, divergence_value]):
        return float("nan"), []

    if current_regime == "RANGING":
        rsi_score = float(np.clip(1 - rsi_value / 100, 0, 1))
    else:
        rsi_score = float(np.clip(rsi_value / 100, 0, 1))

    stoch_score = float(np.clip(1 - k_value / 100, 0, 1))  # 스토캐스틱은 항상 평균회귀식 해석(횡보장 전용 지표)
    divergence_score = (divergence_value + 1) / 2  # -1,0,1 -> 0,0.5,1

    momentum_raw = (10 * rsi_score + 5 * stoch_score + 5 * divergence_score) / 20

    rsi_label = "과매수" if rsi_value >= 70 else "과매도" if rsi_value <= 30 else "중립"
    evidences: list[tuple[float, str]] = [
        (abs(rsi_score - 0.5), f"RSI {rsi_value:.0f} ({rsi_label})"),
        (abs(stoch_score - 0.5), f"스토캐스틱 %K {k_value:.0f}"),
    ]
    if divergence_value == -1:
        evidences.append((0.5, "RSI 약세 다이버전스 감지"))
    elif divergence_value == 1:
        evidences.append((0.5, "RSI 강세 다이버전스 감지"))

    return momentum_raw, evidences


def _volume_score(df: pd.DataFrame, date) -> tuple[float, list[tuple[float, str]]]:
    """거래량 카테고리 원점수(0~1). 내부 배점: OBV 8, 매물대 돌파 7, 거래량 급증 5 (합 20)."""
    obv_series = volume.obv(df)
    poc_series = volume.volume_profile_poc(df)
    vol_avg = df["volume"].rolling(config.VOLUME_SURGE_LOOKBACK).mean()

    obv_value = obv_series.loc[date]
    obv_prev = obv_series.shift(config.OBV_TREND_LOOKBACK).loc[date]  # shift(N): N일 전(과거) 값과 비교
    poc_value = poc_series.loc[date]
    close_value = df["close"].loc[date]
    prev_close_value = df["close"].shift(1).loc[date]  # shift(1): 전일(과거) 종가
    volume_value = df["volume"].loc[date]
    vol_avg_value = vol_avg.loc[date]

    required = [obv_value, obv_prev, poc_value, prev_close_value, vol_avg_value]
    if any(pd.isna(v) for v in required):
        return float("nan"), []

    obv_rising = obv_value > obv_prev
    obv_score = 1.0 if obv_rising else 0.0

    poc_breakout = close_value > poc_value
    poc_score = 1.0 if poc_breakout else 0.0

    is_surge = volume_value > config.VOLUME_SURGE_MULTIPLIER * vol_avg_value
    if is_surge and close_value > prev_close_value:
        surge_score = 1.0
    elif is_surge and close_value < prev_close_value:
        surge_score = 0.0
    else:
        surge_score = 0.5

    volume_raw = (8 * obv_score + 7 * poc_score + 5 * surge_score) / 20

    evidences: list[tuple[float, str]] = [
        (abs(obv_score - 0.5), f"OBV {'상승' if obv_rising else '하락'}"),
        (abs(poc_score - 0.5), f"매물대 돌파 {'확인' if poc_breakout else '미확인'}"),
    ]
    if surge_score != 0.5:
        evidences.append((abs(surge_score - 0.5), f"거래량 급증 동반 {'상승' if surge_score == 1.0 else '하락'}"))

    return volume_raw, evidences


def _flow_score(df: pd.DataFrame, date) -> tuple[float, list[tuple[float, str]]]:
    """수급 카테고리 원점수(0~1). 내부 배점: 외국인 8, 기관 7 (합 15).

    df에 foreign_net/institution_net 컬럼이 없으면 계산 불가(NaN)로 처리한다.
    """
    if "foreign_net" not in df.columns or "institution_net" not in df.columns:
        return float("nan"), []

    foreign_streak_series = flow_indicators.foreign_net_streak(df)
    institution_streak_series = flow_indicators.institution_net_streak(df)

    foreign_streak = foreign_streak_series.loc[date]
    institution_streak = institution_streak_series.loc[date]

    if pd.isna(foreign_streak) or pd.isna(institution_streak):
        return float("nan"), []

    foreign_component = float(np.clip(foreign_streak / config.FLOW_STREAK_FULL_SCORE_DAYS, 0, 1))
    institution_component = float(np.clip(institution_streak / config.FLOW_STREAK_FULL_SCORE_DAYS, 0, 1))

    flow_raw = (8 * foreign_component + 7 * institution_component) / 15

    evidences: list[tuple[float, str]] = []
    if foreign_streak >= 1:
        evidences.append((foreign_component, f"외국인 {int(foreign_streak)}일 연속 순매수"))
    if institution_streak >= 1:
        evidences.append((institution_component, f"기관 {int(institution_streak)}일 연속 순매수"))

    return flow_raw, evidences


def _relative_strength_score(
    df: pd.DataFrame, date, benchmark_close: pd.Series | None
) -> tuple[float, list[tuple[float, str]]]:
    """상대강도 카테고리 원점수(0~1).

    진짜 오닐식 RS Rating(전 종목 대비 백분위 1~99)은 유니버스 전체의 동시 수익률
    비교가 필요해 단일 종목 스코어카드에서는 계산할 수 없다(P5 스크리너 영역).
    이 단계에서는 benchmark_close(예: KOSPI 종가 Series)가 주어질 때만 '벤치마크
    대비 초과수익률' 근사치를 계산하고, 주어지지 않으면 중립값 0.5로 처리한다
    (현재 데이터 파이프라인에 지수 벤치마크 수집기가 아직 없음).
    """
    if benchmark_close is None:
        return 0.5, []

    stock_return = df["close"].pct_change(config.RS_WINDOW)
    bench_return = benchmark_close.pct_change(config.RS_WINDOW)

    stock_value = stock_return.loc[date]
    bench_value = bench_return.loc[date] if date in bench_return.index else float("nan")

    if pd.isna(stock_value) or pd.isna(bench_value):
        return float("nan"), []

    excess = stock_value - bench_value
    rs_raw = float(np.clip(0.5 + excess / (2 * config.RS_SCALE), 0, 1))

    evidences = [(abs(rs_raw - 0.5), f"KOSPI 대비 {excess * 100:+.1f}%p ({'강세' if excess > 0 else '약세'})")]
    return rs_raw, evidences


def score(
    df: pd.DataFrame,
    date,
    benchmark_close: pd.Series | None = None,
    ticker: str = "",
) -> ScoreCard:
    """국면별 가중치를 적용한 0~100점 스코어카드를 계산한다.

    Requires: df에 open/high/low/close/volume 컬럼과 date를 포함하는 DatetimeIndex.
      수급 카테고리를 계산하려면 foreign_net/institution_net 컬럼도 필요하다
      (없으면 해당 카테고리는 0으로 채우는 대신 계산 불가 에러를 던진다).

    date 시점까지의 데이터만 사용하도록 df를 내부적으로 date까지 슬라이싱한다
    (look-ahead 금지 - 이 함수에 미래 데이터가 포함된 df를 통째로 넘겨도 안전).
    """
    if date not in df.index:
        raise ValueError(f"{date}가 df 인덱스에 없습니다.")

    df_hist = df.loc[:date]  # date 이후 데이터 차단

    regime_series = regime_engine.classify_regime(df_hist)
    current_regime = regime_series.loc[date]
    if pd.isna(current_regime):
        raise ValueError(f"{date}: 국면을 판별하기에 데이터가 부족합니다(워밍업 구간).")

    trend_raw, trend_ev = _trend_score(df_hist, date)
    momentum_raw, momentum_ev = _momentum_score(df_hist, date, current_regime)
    volume_raw, volume_ev = _volume_score(df_hist, date)
    flow_raw, flow_ev = _flow_score(df_hist, date)
    rs_raw, rs_ev = _relative_strength_score(df_hist, date, benchmark_close)

    raw_scores = {
        "trend": trend_raw,
        "momentum": momentum_raw,
        "volume": volume_raw,
        "flow": flow_raw,
        "relative_strength": rs_raw,
    }
    for cat, raw in raw_scores.items():
        if pd.isna(raw):
            raise ValueError(f"{date}: '{cat}' 카테고리 점수를 계산할 데이터가 부족합니다.")

    total, category_scores = _combine(raw_scores, current_regime)

    all_evidences = trend_ev + momentum_ev + volume_ev + flow_ev + rs_ev
    all_evidences.sort(key=lambda item: item[0], reverse=True)
    evidences = [sentence for _, sentence in all_evidences[:5]]

    close_value = df_hist["close"].loc[date]
    atr_series = volatility.atr(df_hist, period=config.ATR_PERIOD)
    atr_value = atr_series.loc[date]
    if pd.isna(atr_value):
        raise ValueError(f"{date}: ATR을 계산할 데이터가 부족해 손절가를 산출할 수 없습니다.")
    stop_loss_price = close_value - config.STOP_LOSS_ATR_MULTIPLIER * atr_value

    risk_per_price_pct = (close_value - stop_loss_price) / close_value
    position_size_pct = (
        min(config.MAX_POSITION_PCT, config.RISK_PER_TRADE_PCT / risk_per_price_pct)
        if risk_per_price_pct > 0
        else 0.0
    )

    sma20 = df_hist["close"].rolling(20).mean().loc[date]
    invalidation_text = (
        f"종가 기준 20일선({sma20:,.0f}원) 이탈 시" if not pd.isna(sma20) else "20일선 계산 불가(데이터 부족)"
    )

    return ScoreCard(
        ticker=ticker,
        date=pd.Timestamp(date),
        total=round(total, 1),
        grade=grade(total),
        regime=current_regime,
        category_scores=category_scores,
        evidences=evidences,
        invalidation=invalidation_text,
        stop_loss=round(float(stop_loss_price), 0),
        position_size_pct=round(float(position_size_pct), 4),
    )
