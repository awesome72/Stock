"""P3 완료 조건 확인 스크립트: 삼성전자 최근 날짜 스코어카드를 PRD 형식대로 출력.

수급(foreign_net/institution_net) 데이터는 KRX_ID/KRX_PW가 설정된 경우에만 병합한다.
"""

from __future__ import annotations

import logging
import os
import sys

import pandas as pd

from confluence import config
from confluence.data import loader
from confluence.engine import regime as regime_engine
from confluence.engine import scorer
from confluence.indicators import trend

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

TICKER = "005930"
NAME = "삼성전자"
START = "20150101"
END = "20250101"


def _bar(earned: float, max_: float, width: int = 10) -> str:
    filled = round(width * earned / max_) if max_ else 0
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def main() -> None:
    df = loader.fetch_ohlcv(TICKER, START, END)

    if os.getenv("KRX_ID") and os.getenv("KRX_PW"):
        flow_df = loader.fetch_investor_flow(TICKER, START, END)
        df = df.join(flow_df, how="left")

    date = df.index[-1]
    try:
        sc = scorer.score(df, date, ticker=TICKER)
    except ValueError as exc:
        print(f"스코어카드 계산 실패: {exc}")
        if "flow" in str(exc):
            print("(KRX_ID / KRX_PW 환경변수를 설정하면 수급 데이터가 포함됩니다: https://data.krx.co.kr)")
        return

    df_hist = df.loc[:date]
    adx_value = trend.adx_dmi(df_hist, period=config.ADX_PERIOD)["adx"].loc[date]
    regime_ko = regime_engine.REGIME_LABEL_KO[sc.regime]

    print(f"{NAME} ({TICKER})  |  국면: {regime_ko} (ADX {adx_value:.1f})")
    print("─" * 46)
    print(f"종합 점수    {sc.total:.0f} / 100        [ {sc.grade} ]")
    print()

    for cat in config.CATEGORIES:
        cs = sc.category_scores[cat]
        label = config.CATEGORY_LABEL_KO[cat]
        bar = _bar(cs["earned"], cs["max"])
        print(f"{label:6s} {cs['earned']:4.0f}/{cs['max']:.0f}  {bar}")

    print()
    if sc.evidences:
        print("근거: " + ", ".join(sc.evidences))
        print()
    print(f"⚠️ 반증 조건: {sc.invalidation}")
    close_value = df["close"].loc[date]
    stop_pct = (sc.stop_loss - close_value) / close_value * 100
    print(f"🛑 손절선: {sc.stop_loss:,.0f}원 (진입가 -2×ATR, {stop_pct:+.1f}%)")
    print(f"📐 권장 비중: {sc.position_size_pct * 100:.1f}%")


if __name__ == "__main__":
    main()
