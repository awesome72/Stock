"""국면 판별(regime.py) + 스코어링(scorer.py) 테스트."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from confluence import config
from confluence.engine import regime, scorer


def _make_trending_up(n: int = 150) -> pd.DataFrame:
    """꾸준히 우상향하는 저노이즈 데이터. ADX가 확실히 25 이상으로 올라간다."""
    dates = pd.bdate_range("2023-01-02", periods=n)
    close = 100 + np.arange(n) * 0.5
    high = close + 0.3
    low = close - 0.3
    open_ = close - 0.1
    volume = np.full(n, 10_000.0)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=dates)


def _make_ranging(n: int = 150) -> pd.DataFrame:
    """진폭이 점점 좁아지는(스퀴즈) 오실레이션 데이터. 추세 없이 ADX가 낮게 유지된다."""
    dates = pd.bdate_range("2023-01-02", periods=n)
    t = np.arange(n)
    amplitude = np.linspace(3.0, 0.2, n)
    close = 100 + amplitude * np.sin(t * 2 * np.pi / 10)
    high = close + 0.1
    low = close - 0.1
    open_ = close
    volume = np.full(n, 10_000.0)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=dates)


def test_trending_up_regime_and_high_trend_score():
    df = _make_trending_up()
    date = df.index[-1]

    regime_series = regime.classify_regime(df)
    assert regime_series.loc[date] == "TRENDING_UP"

    trend_raw, _ = scorer._trend_score(df, date)
    assert trend_raw > 0.7


def test_ranging_regime_gets_momentum_weight_40():
    df = _make_ranging()
    date = df.index[-1]

    regime_series = regime.classify_regime(df)
    assert regime_series.loc[date] == "RANGING"

    raw_scores = {cat: 0.5 for cat in config.CATEGORIES}
    _, category_scores = scorer._combine(raw_scores, regime_series.loc[date])
    assert category_scores["momentum"]["max"] == 40


@pytest.mark.parametrize("regime_label", list(config.REGIME_CATEGORY_WEIGHTS.keys()))
def test_perfect_category_scores_total_100(regime_label):
    raw_scores = {cat: 1.0 for cat in config.CATEGORIES}
    total, category_scores = scorer._combine(raw_scores, regime_label)

    assert total == pytest.approx(100.0)
    assert sum(v["max"] for v in category_scores.values()) == 100


def test_grade_boundaries():
    assert scorer.grade(80) == "강한 합의"
    assert scorer.grade(79.9) == "관심"
    assert scorer.grade(65) == "관심"
    assert scorer.grade(64.9) == "진입 금지"
    assert scorer.grade(45) == "진입 금지"
    assert scorer.grade(44.9) == "회피"
