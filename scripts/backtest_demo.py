"""P4 완료 조건 확인 스크립트: KOSPI 50종목 2015~2025 백테스트 + 워크포워드 + 벤치마크 비교.

KRX_ID/KRX_PW 환경변수가 필요하다(유니버스/수급/지수 데이터 전부 KRX 인증 필요).
"""

from __future__ import annotations

import logging
import sys
import time

import pandas as pd

from confluence import config
from confluence.backtest import engine, metrics
from confluence.data import loader

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

START = "20150101"
END = "20250101"
UNIVERSE_SIZE = 50


def main() -> None:
    print(f"KOSPI 시가총액 상위 {UNIVERSE_SIZE}종목 조회 중...")
    tickers = loader.fetch_universe("KOSPI", top_n=UNIVERSE_SIZE)
    print(f"{len(tickers)}종목 확보: {tickers[:5]}...")

    print("벤치마크(KOSPI 종합지수) 조회 중...")
    index_df = loader.fetch_index_ohlcv(config.KOSPI_INDEX_CODE, START, END)
    benchmark_close = index_df["close"]

    print(f"백테스트 실행 중 ({START}~{END}, entry={config.BACKTEST_ENTRY_SCORE:.0f}, exit={config.BACKTEST_EXIT_SCORE:.0f})...")
    t0 = time.perf_counter()
    result = engine.run(tickers, START, END, benchmark_close=benchmark_close)
    t1 = time.perf_counter()
    print(f"완료: {t1 - t0:.1f}초, 거래 {len(result.trades)}건")

    trade_pnl = pd.Series([t.pnl for t in result.trades], dtype=float)
    perf = metrics.summary(result.equity_curve, trade_pnl)

    print()
    print("=== 성과 지표 8종 ===")
    for k, v in perf.items():
        if k in ("CAGR", "MDD", "승률"):
            print(f"{k:15s} {v * 100:+.2f}%")
        elif k == "거래횟수":
            print(f"{k:15s} {v}")
        else:
            print(f"{k:15s} {v:.2f}")

    print()
    print("=== 벤치마크(KOSPI) 대비 누적수익 ===")
    strategy_cum = result.equity_curve / result.equity_curve.iloc[0] - 1
    bench_aligned = benchmark_close.reindex(result.equity_curve.index).ffill()
    bench_cum = bench_aligned / bench_aligned.iloc[0] - 1
    print(f"전략 누적수익:   {strategy_cum.iloc[-1] * 100:+.2f}%")
    print(f"KOSPI 누적수익:  {bench_cum.iloc[-1] * 100:+.2f}%")
    print(f"초과수익:        {(strategy_cum.iloc[-1] - bench_cum.iloc[-1]) * 100:+.2f}%p")

    print()
    print("=== 워크포워드 검증 (학습 3년 / 검증 1년, 최대 5구간) ===")
    wf = metrics.walk_forward(
        engine.run, tickers, START, END, train_years=3, test_years=1, rolls=5, benchmark_close=benchmark_close
    )
    with pd.option_context("display.width", 140, "display.max_columns", None):
        print(wf.to_string(index=False))


if __name__ == "__main__":
    main()
