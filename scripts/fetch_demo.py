"""P1 완료 조건 확인 스크립트.

1) 삼성전자(005930) 10년치 OHLCV를 수집하고, 재조회가 캐시로 1초 이내에 끝나는지 측정한다.
2) 수집된 행 수와 결측(추정) 영업일 수를 출력한다.
3) fetch_investor_flow / fetch_universe는 KRX_ID·KRX_PW가 설정된 경우에만 시도한다.
"""

from __future__ import annotations

import logging
import os
import sys
import time

import pandas as pd

from confluence.data import loader

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stdout)

TICKER = "005930"
START = "20150101"
END = "20250101"


def check_ohlcv() -> None:
    t0 = time.perf_counter()
    df = loader.fetch_ohlcv(TICKER, START, END)
    t1 = time.perf_counter()
    print(f"[OHLCV] 1차 수집: {len(df)}행, {t1 - t0:.2f}초")

    t2 = time.perf_counter()
    df_cached = loader.fetch_ohlcv(TICKER, START, END)
    t3 = time.perf_counter()
    print(f"[OHLCV] 재조회(캐시): {len(df_cached)}행, {t3 - t2:.2f}초")

    expected_days = len(pd.bdate_range(START, END))
    missing = expected_days - len(df_cached)
    print(f"[OHLCV] 예상 영업일수: {expected_days}, 실제 수집 행수: {len(df_cached)}, 결측(추정): {missing}일")


def check_flow_and_universe() -> None:
    if not (os.getenv("KRX_ID") and os.getenv("KRX_PW")):
        print("[Flow/Universe] KRX_ID / KRX_PW 미설정 - 건너뜀 (https://data.krx.co.kr 계정 필요)")
        return

    flow = loader.fetch_investor_flow(TICKER, "20250101", "20250201")
    print(f"[Flow] {len(flow)}행 수집")

    universe = loader.fetch_universe("KOSPI", top_n=200)
    print(f"[Universe] KOSPI 시가총액 상위 {len(universe)}종목 수집")


if __name__ == "__main__":
    check_ohlcv()
    check_flow_and_universe()
