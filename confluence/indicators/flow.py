"""수급 지표(한국 전용): 외국인/기관 순매수 연속일수, 수급 점수."""

from __future__ import annotations

import pandas as pd


def _positive_streak(series: pd.Series) -> pd.Series:
    """양수(순매수)가 지속된 연속 영업일수. 순매도/보합이면 0, 입력 결측이면 NaN."""
    is_positive = series > 0
    # 이전 값과 부호(양/비양)가 달라질 때마다 새 그룹 시작 (shift(1): 직전값과 비교, 과거 방향)
    group_id = (is_positive != is_positive.shift(1)).cumsum()
    streak = is_positive.groupby(group_id).cumcount() + 1
    result = streak.where(is_positive, 0).astype(float)
    return result.where(series.notna())


def foreign_net_streak(df: pd.DataFrame) -> pd.Series:
    """외국인 연속 순매수일수.

    Requires: df['foreign_net']
    Returns: 연속일수 Series(순매도/보합은 0, 결측 입력은 NaN).
    """
    return _positive_streak(df["foreign_net"])


def institution_net_streak(df: pd.DataFrame) -> pd.Series:
    """기관 연속 순매수일수.

    Requires: df['institution_net']
    Returns: 연속일수 Series(순매도/보합은 0, 결측 입력은 NaN).
    """
    return _positive_streak(df["institution_net"])


def _flow_score(df: pd.DataFrame, window: int) -> pd.Series:
    """최근 window거래일 동안 외국인·기관이 순매수했던 날의 비율 평균(0~1)."""
    foreign_positive = (df["foreign_net"] > 0).astype(float)
    institution_positive = (df["institution_net"] > 0).astype(float)
    foreign_ratio = foreign_positive.rolling(window).mean()
    institution_ratio = institution_positive.rolling(window).mean()
    return (foreign_ratio + institution_ratio) / 2


def flow_score_5d(df: pd.DataFrame) -> pd.Series:
    """최근 5거래일 수급 점수(0~1). (외국인 순매수일 비율 + 기관 순매수일 비율) / 2.

    Requires: df['foreign_net'], df['institution_net']
    """
    return _flow_score(df, window=5)


def flow_score_20d(df: pd.DataFrame) -> pd.Series:
    """flow_score_5d와 동일한 정의를 20거래일 창에 적용.

    Requires: df['foreign_net'], df['institution_net']
    """
    return _flow_score(df, window=20)
