"""P6: 전략이 우연이 아닌지 검증한다.

1. 파라미터 민감도 테스트 (RSI기간/MACD단기/ADX임계값/entry_score/ATR손절배수 ±20%)
2. 국면 가중치 무력화 대조군 (균등가중 vs 국면별가중)
3. 랜덤 진입 대조군 (같은 거래횟수/보유기간, 무작위 종목·시점 1000회 시뮬레이션 p-value)
4. 신호별 실제 승률 테이블 (신호 발생 후 20거래일 수익률)

결과를 reports/validation_YYYYMMDD.md 로 저장한다.

검증 구간은 2020-01-01~2025-12-31로 고정한다. 직전 타이밍 점검에서 이미 이
구간의 OHLCV/투자자별 수급 데이터를 SQLite에 캐시해뒀으므로(KRX 재인증 없이),
아래에서 반복되는 12회 이상의 백테스트 실행이 전부 네트워크 호출 없이
캐시로만 처리되어 빠르게 끝난다. KRX_ID/KRX_PW 환경변수가 필요하다
(캐시가 없는 새 환경에서 처음 실행할 때만).
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from confluence import config
from confluence.backtest import engine as bt
from confluence.backtest import metrics
from confluence.data import loader
from confluence.indicators import flow as flow_indicators
from confluence.indicators import momentum, trend, volume

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

UNIVERSE_SIZE = 50
START = "20200101"
END = "20251231"
DATA_START = "20190101"
START_TS = pd.Timestamp(START)
END_TS = pd.Timestamp(END)

# 아래 둘은 전략의 튜닝 파라미터가 아니라 이 검증 스크립트 자체의 실행 설정이므로
# config.py가 아니라 여기 상수로 둔다(CLAUDE.md 규칙4의 "파라미터 5개"는 전략 파라미터 한정).
RANDOM_SIMULATIONS = 1000
RANDOM_SEED = 42
FORWARD_HORIZON = 20  # 신호별 승률 테이블: 신호 발생 후 관찰할 거래일 수

ROUND_TRIP_COST_PCT = 2 * (config.COMMISSION_RATE + config.SLIPPAGE_RATE) + config.TAX_RATE

REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"


@contextmanager
def _patched(module, name: str, value):
    """module.name을 임시로 value로 바꾸고 블록 종료 시 원복한다."""
    original = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, original)


def _trade_pnl(result: bt.BacktestResult) -> pd.Series:
    return pd.Series([t.pnl for t in result.trades], dtype=float)


def _fmt_pct(v: float) -> str:
    return "N/A" if pd.isna(v) else f"{v * 100:+.2f}%"


def _fmt_num(v: float) -> str:
    return "N/A" if pd.isna(v) else f"{v:.2f}"


def _df_to_md(df: pd.DataFrame) -> str:
    """tabulate 의존성 없이 DataFrame을 마크다운 표 문자열로 변환한다."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 공통 데이터 준비
# ---------------------------------------------------------------------------


def fetch_context() -> tuple[list[str], pd.Series, dict[str, pd.DataFrame]]:
    print(f"KOSPI 시가총액 상위 {UNIVERSE_SIZE}종목 조회 중...")
    tickers = loader.fetch_universe("KOSPI", top_n=UNIVERSE_SIZE)

    print("벤치마크(KOSPI 종합지수) 조회 중...")
    index_df = loader.fetch_index_ohlcv(config.KOSPI_INDEX_CODE, DATA_START, END)
    benchmark_close = index_df["close"]

    print(f"{len(tickers)}종목 원시 OHLCV+수급 데이터 조회 중 (캐시 활용)...")
    raw_data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = loader.fetch_ohlcv(ticker, DATA_START, END)
        if df.empty:
            continue
        try:
            flow_df = loader.fetch_investor_flow(ticker, DATA_START, END)
            df = df.join(flow_df, how="left")
        except RuntimeError:
            pass
        raw_data[ticker] = df

    return tickers, benchmark_close, raw_data


# ---------------------------------------------------------------------------
# 1. 파라미터 민감도 테스트
# ---------------------------------------------------------------------------

SENSITIVITY_CONFIG_PARAMS = [
    ("RSI_PERIOD", "RSI 기간", lambda v: int(round(v))),
    ("MACD_FAST", "MACD 단기", lambda v: int(round(v))),
    ("REGIME_ADX_TRENDING", "ADX 추세 임계값", lambda v: round(v, 1)),
    ("STOP_LOSS_ATR_MULTIPLIER", "ATR 손절 배수", lambda v: round(v, 2)),
]


def _sensitivity_row(label, base_value, base_perf, low_value, low_perf, high_value, high_perf) -> dict:
    base_cagr, low_cagr, high_cagr = base_perf["CAGR"], low_perf["CAGR"], high_perf["CAGR"]
    cagr_values = [c for c in (base_cagr, low_cagr, high_cagr) if not pd.isna(c)]
    if len(cagr_values) < 2 or abs(base_cagr) < 1e-6:
        swing_pct = float("nan")
    else:
        swing_pct = (max(cagr_values) - min(cagr_values)) / abs(base_cagr) * 100

    flag = "N/A" if pd.isna(swing_pct) else ("⚠️ 과최적화 의심" if swing_pct > 30 else "OK")

    return {
        "파라미터": label,
        "기본값": base_value,
        "-20%": low_value,
        "+20%": high_value,
        "기본 CAGR": _fmt_pct(base_cagr),
        "-20% CAGR": _fmt_pct(low_cagr),
        "+20% CAGR": _fmt_pct(high_cagr),
        "기본 Sharpe": _fmt_num(base_perf["Sharpe"]),
        "-20% Sharpe": _fmt_num(low_perf["Sharpe"]),
        "+20% Sharpe": _fmt_num(high_perf["Sharpe"]),
        "CAGR 변동폭": "N/A" if pd.isna(swing_pct) else f"{swing_pct:.0f}%",
        "판정": flag,
    }


def run_sensitivity(tickers: list[str], benchmark_close: pd.Series, base_result: bt.BacktestResult) -> pd.DataFrame:
    base_perf = metrics.summary(base_result.equity_curve, _trade_pnl(base_result))
    rows = []

    for attr, label, rounder in SENSITIVITY_CONFIG_PARAMS:
        base_value = getattr(config, attr)
        low_value = rounder(base_value * 0.8)
        high_value = rounder(base_value * 1.2)
        print(f"  민감도: {label} {low_value}/{high_value} (기본 {base_value}) 실행 중...")

        with _patched(config, attr, low_value):
            low_result = bt.run(tickers, START, END, data_start=DATA_START, benchmark_close=benchmark_close)
        with _patched(config, attr, high_value):
            high_result = bt.run(tickers, START, END, data_start=DATA_START, benchmark_close=benchmark_close)

        low_perf = metrics.summary(low_result.equity_curve, _trade_pnl(low_result))
        high_perf = metrics.summary(high_result.equity_curve, _trade_pnl(high_result))
        rows.append(_sensitivity_row(label, base_value, base_perf, low_value, low_perf, high_value, high_perf))

    base_entry = config.BACKTEST_ENTRY_SCORE
    low_entry, high_entry = round(base_entry * 0.8, 1), round(base_entry * 1.2, 1)
    print(f"  민감도: 진입 점수(entry_score) {low_entry}/{high_entry} (기본 {base_entry}) 실행 중...")
    low_result = bt.run(tickers, START, END, data_start=DATA_START, entry_score=low_entry, benchmark_close=benchmark_close)
    high_result = bt.run(tickers, START, END, data_start=DATA_START, entry_score=high_entry, benchmark_close=benchmark_close)
    low_perf = metrics.summary(low_result.equity_curve, _trade_pnl(low_result))
    high_perf = metrics.summary(high_result.equity_curve, _trade_pnl(high_result))
    rows.append(_sensitivity_row("진입 점수(entry_score)", base_entry, base_perf, low_entry, low_perf, high_entry, high_perf))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. 국면 가중치 무력화 대조군
# ---------------------------------------------------------------------------


def run_regime_neutralization(tickers: list[str], benchmark_close: pd.Series, base_result: bt.BacktestResult) -> tuple[pd.DataFrame, bt.BacktestResult]:
    equal_weights = {cat: 100 // len(config.CATEGORIES) for cat in config.CATEGORIES}
    remainder = 100 - sum(equal_weights.values())
    equal_weights[config.CATEGORIES[0]] += remainder  # 정수 배분 나머지 보정(합 100 유지)
    equal_regime_weights = {label: dict(equal_weights) for label in config.REGIME_LABELS}

    print(f"  국면 무력화: 균등가중치 {equal_weights} 로 실행 중...")
    with _patched(config, "REGIME_CATEGORY_WEIGHTS", equal_regime_weights):
        equal_result = bt.run(tickers, START, END, data_start=DATA_START, benchmark_close=benchmark_close)

    base_perf = metrics.summary(base_result.equity_curve, _trade_pnl(base_result))
    equal_perf = metrics.summary(equal_result.equity_curve, _trade_pnl(equal_result))

    rows = [
        {"지표": k, "국면별 가중치(현재)": (_fmt_pct(base_perf[k]) if k in ("CAGR", "MDD", "승률") else _fmt_num(base_perf[k]) if k != "거래횟수" else str(base_perf[k])),
         "균등 가중치(대조군)": (_fmt_pct(equal_perf[k]) if k in ("CAGR", "MDD", "승률") else _fmt_num(equal_perf[k]) if k != "거래횟수" else str(equal_perf[k]))}
        for k in base_perf
    ]
    return pd.DataFrame(rows), equal_result


# ---------------------------------------------------------------------------
# 3. 랜덤 진입 대조군
# ---------------------------------------------------------------------------


def _holding_bars(raw_data: dict[str, pd.DataFrame], trades) -> list[int]:
    bars = []
    for t in trades:
        df = raw_data.get(t.ticker)
        if df is None or t.entry_date not in df.index or t.exit_date not in df.index:
            continue
        bars.append(int(df.index.get_loc(t.exit_date) - df.index.get_loc(t.entry_date)))
    return bars


def _build_return_pools(raw_data: dict[str, pd.DataFrame], holding_values: set[int]) -> dict[int, np.ndarray]:
    pools: dict[int, list[np.ndarray]] = {h: [] for h in holding_values}
    for df in raw_data.values():
        window = df.loc[START_TS:END_TS]
        opens = window["open"].to_numpy()
        closes = window["close"].to_numpy()
        n = len(window)
        for h in holding_values:
            max_start = n - h - 2
            if max_start < 0:
                continue
            entry = opens[1 : max_start + 2]
            exit_ = closes[1 + h : max_start + 2 + h]
            rets = (exit_ - entry) / entry - ROUND_TRIP_COST_PCT
            pools[h].append(rets)
    return {h: (np.concatenate(v) if v else np.array([])) for h, v in pools.items()}


def run_random_control(raw_data: dict[str, pd.DataFrame], base_result: bt.BacktestResult) -> dict:
    trades = base_result.trades
    holding = _holding_bars(raw_data, trades)
    if len(holding) == 0:
        return {"error": "실제 거래에서 보유기간을 산출할 수 없어 랜덤 대조군을 건너뜀"}

    actual_returns = np.array([t.return_pct for t in trades], dtype=float)
    actual_stat = float(np.nanmean(actual_returns))

    unique_h, counts = np.unique(holding, return_counts=True)
    pools = _build_return_pools(raw_data, set(unique_h.tolist()))

    rng = np.random.default_rng(RANDOM_SEED)
    total = np.zeros(RANDOM_SIMULATIONS)
    valid_trade_count = 0
    for h, cnt in zip(unique_h, counts):
        pool = pools.get(int(h), np.array([]))
        if len(pool) == 0:
            continue
        samples = rng.choice(pool, size=(RANDOM_SIMULATIONS, int(cnt)), replace=True)
        total += samples.sum(axis=1)
        valid_trade_count += int(cnt)

    if valid_trade_count == 0:
        return {"error": "보유기간에 대응하는 랜덤 표본을 구성할 수 없어 랜덤 대조군을 건너뜀"}

    sim_stats = total / valid_trade_count
    p_value = (1 + np.sum(sim_stats >= actual_stat)) / (RANDOM_SIMULATIONS + 1)
    percentile = float(np.mean(sim_stats < actual_stat) * 100)

    return {
        "실제 거래 평균수익률": actual_stat,
        "랜덤 시뮬레이션 평균": float(np.mean(sim_stats)),
        "랜덤 시뮬레이션 표준편차": float(np.std(sim_stats)),
        "랜덤 대비 백분위": percentile,
        "p-value": float(p_value),
        "유의(p<0.05)": bool(p_value < 0.05),
        "시뮬레이션 횟수": RANDOM_SIMULATIONS,
        "거래 횟수": len(trades),
    }


# ---------------------------------------------------------------------------
# 4. 신호별 실제 승률 테이블
# ---------------------------------------------------------------------------

# (표시명, 강세(True)/약세(False) 신호 여부 - 약세 신호는 하락이 '성공'이므로 별도 표시만 하고
#  승률 정의(양의 수익률 비율) 자체는 모든 신호에 동일하게 적용한다(PRD 명세 그대로).
BEARISH_SIGNALS = {"RSI 약세 다이버전스"}


def _compute_signals(df: pd.DataFrame) -> dict[str, pd.Series]:
    signals: dict[str, pd.Series] = {}

    macd_df = trend.macd(df)
    golden_cross = (macd_df["macd"] > macd_df["signal"]) & (macd_df["macd"].shift(1) <= macd_df["signal"].shift(1))
    signals["MACD 골든크로스"] = golden_cross.fillna(False)

    poc = volume.volume_profile_poc(df)
    poc_breakout = (df["close"] > poc) & (df["close"].shift(1) <= poc.shift(1))
    signals["매물대 상향 돌파"] = poc_breakout.fillna(False)

    divergence = momentum.rsi_divergence(df)
    signals["RSI 강세 다이버전스"] = divergence == 1
    signals["RSI 약세 다이버전스"] = divergence == -1

    if "foreign_net" in df.columns:
        f_streak = flow_indicators.foreign_net_streak(df)
        signals["외국인 5일 연속 순매수"] = f_streak == config.FLOW_STREAK_FULL_SCORE_DAYS
    if "institution_net" in df.columns:
        i_streak = flow_indicators.institution_net_streak(df)
        signals["기관 5일 연속 순매수"] = i_streak == config.FLOW_STREAK_FULL_SCORE_DAYS

    return signals


def run_signal_winrate(raw_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records: dict[str, list[np.ndarray]] = {}
    for df in raw_data.values():
        window = df.loc[START_TS:END_TS]
        if len(window) < FORWARD_HORIZON + 2:
            continue
        opens = window["open"].to_numpy()
        closes = window["close"].to_numpy()
        n = len(window)
        for name, bool_series in _compute_signals(window).items():
            idx = np.nonzero(bool_series.to_numpy())[0]
            idx = idx[idx <= n - FORWARD_HORIZON - 2]
            if len(idx) == 0:
                continue
            entry = opens[idx + 1]
            exit_ = closes[idx + 1 + FORWARD_HORIZON]
            rets = (exit_ - entry) / entry - ROUND_TRIP_COST_PCT
            records.setdefault(name, []).append(rets)

    rows = []
    for name, arrs in records.items():
        rets = np.concatenate(arrs)
        win_rate = float(np.mean(rets > 0))
        rows.append(
            {
                "신호": name + (" (약세 신호)" if name in BEARISH_SIGNALS else ""),
                "발생 횟수": len(rets),
                "평균 수익률": _fmt_pct(float(np.mean(rets))),
                "중앙값 수익률": _fmt_pct(float(np.median(rets))),
                "승률(양의 수익률 비율)": f"{win_rate * 100:.1f}%",
                "판정": "제거 후보" if (win_rate < 0.5 and name not in BEARISH_SIGNALS) else "-",
            }
        )
    return pd.DataFrame(rows).sort_values("승률(양의 수익률 비율)")


# ---------------------------------------------------------------------------
# 리포트 작성
# ---------------------------------------------------------------------------

PRD_CRITERIA = {
    "CAGR": ("벤치마크(KOSPI) 초과", None),
    "MDD": ("-25% 이내", -0.25),
    "Sharpe": ("1.0 이상", 1.0),
    "승률": ("45% 이상", 0.45),
    "손익비": ("1.8 이상", 1.8),
    "Profit Factor": ("1.5 이상", 1.5),
    "거래횟수": ("100회 이상", 100),
}


def _baseline_table(perf: dict, benchmark_cagr: float) -> pd.DataFrame:
    rows = []
    for key, (desc, threshold) in PRD_CRITERIA.items():
        value = perf[key]
        if key == "CAGR":
            passed = (not pd.isna(value)) and (not pd.isna(benchmark_cagr)) and value > benchmark_cagr
        elif key == "MDD":
            passed = (not pd.isna(value)) and value >= threshold
        elif key == "거래횟수":
            passed = value >= threshold
        else:
            passed = (not pd.isna(value)) and value >= threshold

        display = _fmt_pct(value) if key in ("CAGR", "MDD", "승률") else (str(value) if key == "거래횟수" else _fmt_num(value))
        rows.append({"지표": key, "값": display, "PRD 기준": desc, "충족": "충족" if passed else "미충족"})
    return pd.DataFrame(rows)


def build_report(
    base_result: bt.BacktestResult,
    benchmark_close: pd.Series,
    sensitivity_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    random_stats: dict,
    signal_df: pd.DataFrame,
) -> str:
    base_perf = metrics.summary(base_result.equity_curve, _trade_pnl(base_result))
    bench_aligned = benchmark_close.reindex(base_result.equity_curve.index).ffill()
    benchmark_cagr = metrics.cagr(bench_aligned)
    baseline_df = _baseline_table(base_perf, benchmark_cagr)
    prd_pass_count = (baseline_df["충족"] == "충족").sum()

    overopt_flags = sensitivity_df["판정"].astype(str).str.contains("과최적화").sum()

    lines: list[str] = []
    lines.append(f"# Confluence 검증 리포트 ({date.today().isoformat()})")
    lines.append("")
    lines.append("## 0. 검증 조건")
    lines.append(f"- 유니버스: KOSPI 시가총액 상위 {UNIVERSE_SIZE}종목")
    lines.append(f"- 검증 구간: {START} ~ {END} (지표 워밍업: {DATA_START}부터)")
    lines.append("- 벤치마크: KOSPI 종합지수")
    lines.append(f"- 진입/청산 기본 점수: {config.BACKTEST_ENTRY_SCORE:.0f} / {config.BACKTEST_EXIT_SCORE:.0f}")
    lines.append("")
    lines.append("## 1. 기준 성과 (Baseline)")
    lines.append(f"KOSPI 벤치마크 CAGR: {_fmt_pct(benchmark_cagr)}")
    lines.append("")
    lines.append(_df_to_md(baseline_df))
    lines.append("")
    lines.append(f"**PRD 최소 기준 {len(PRD_CRITERIA)}개 중 {prd_pass_count}개 충족.**")
    lines.append("")
    lines.append("## 2. 파라미터 민감도 테스트 (±20%)")
    lines.append("판정 기준: CAGR 변동폭(=(최대-최소)/|기본값|) > 30% 이면 '과최적화 의심'.")
    lines.append("")
    lines.append(_df_to_md(sensitivity_df))
    lines.append("")
    if overopt_flags > 0:
        lines.append(f"⚠️ **{overopt_flags}개 파라미터에서 과최적화 의심 신호가 발견되었다.** 위 표에서 해당 항목을 확인할 것.")
    else:
        lines.append("모든 파라미터가 ±20% 변경에도 CAGR 변동폭 30% 이내로, 과최적화 의심 신호는 발견되지 않았다.")
    lines.append("")
    lines.append("## 3. 국면 가중치 무력화 대조군")
    lines.append("모든 국면에 동일한 균등가중치를 적용한 버전과 현재의 국면별 가중치 버전을 비교한다.")
    lines.append("")
    lines.append(_df_to_md(regime_df))
    lines.append("")
    try:
        current_cagr_str = regime_df.loc[regime_df["지표"] == "CAGR", "국면별 가중치(현재)"].iloc[0]
        equal_cagr_str = regime_df.loc[regime_df["지표"] == "CAGR", "균등 가중치(대조군)"].iloc[0]
        improved = float(current_cagr_str.strip("%+")) > float(equal_cagr_str.strip("%+"))
    except (IndexError, ValueError):
        improved = None
    if improved is True:
        lines.append("국면별 가중치 버전이 균등가중치 대조군보다 CAGR이 높다 — 국면 적응형 가중치가 성과에 기여한다는 근거가 있다.")
    elif improved is False:
        lines.append(
            "⚠️ **국면별 가중치 버전이 균등가중치 대조군보다 CAGR이 낮거나 같다.** "
            "이는 '국면에 따라 가중치를 다르게 적용하면 성과가 개선된다'는 이 프로젝트의 핵심 가설이 "
            "이 백테스트 표본에서는 뒷받침되지 않는다는 뜻이므로 명확히 보고한다."
        )
    else:
        lines.append("두 버전의 CAGR을 비교할 수 없어(계산 실패) 결론을 내리지 못했다.")
    lines.append("")
    lines.append("## 4. 랜덤 진입 대조군")
    if "error" in random_stats:
        lines.append(f"랜덤 대조군을 계산하지 못했다: {random_stats['error']}")
    else:
        lines.append(
            f"- 실제 전략의 거래당 평균 수익률: {_fmt_pct(random_stats['실제 거래 평균수익률'])}\n"
            f"- 랜덤 진입 {random_stats['시뮬레이션 횟수']}회 시뮬레이션 평균: {_fmt_pct(random_stats['랜덤 시뮬레이션 평균'])} "
            f"(표준편차 {_fmt_pct(random_stats['랜덤 시뮬레이션 표준편차'])})\n"
            f"- 실제 전략은 랜덤 분포의 상위 {100 - random_stats['랜덤 대비 백분위']:.1f}% 안에 위치한다.\n"
            f"- p-value = {random_stats['p-value']:.4f} "
            f"({'통계적으로 유의함 (p<0.05)' if random_stats['유의(p<0.05)'] else '통계적으로 유의하지 않음 (p>=0.05)'})"
        )
        if not random_stats["유의(p<0.05)"]:
            lines.append(
                "\n⚠️ **전략의 거래당 평균수익률이 무작위 진입과 통계적으로 유의하게 다르다고 보기 어렵다.** "
                "이는 신호 선별 자체의 실효성에 의문을 제기하는 결과이므로 명확히 보고한다."
            )
    lines.append("")
    lines.append("## 5. 신호별 실제 승률 테이블")
    lines.append(f"신호 발생 익일 시가 진입 → {FORWARD_HORIZON}거래일 후 종가 청산(왕복비용 반영) 기준 수익률이다.")
    lines.append("'RSI 약세 다이버전스'는 하락을 예상하는 신호이므로, 승률(양의 수익률 비율)이 낮은 것이 오히려 신호가 유효하다는 뜻일 수 있다 — 이 신호는 '제거 후보' 판정에서 제외했다.")
    lines.append("")
    lines.append(_df_to_md(signal_df))
    lines.append("")
    removal_candidates = signal_df[signal_df["판정"] == "제거 후보"]["신호"].tolist()
    if removal_candidates:
        lines.append(f"⚠️ **승률 50% 미만으로 제거 후보로 표시된 신호:** {', '.join(removal_candidates)}")
    else:
        lines.append("승률 50% 미만인 신호는 없었다(약세 다이버전스 제외).")
    lines.append("")
    lines.append("## 6. 종합 결론")
    lines.append(f"- PRD 최소 성과 기준 {len(PRD_CRITERIA)}개 중 {prd_pass_count}개 충족.")
    lines.append(f"- 파라미터 민감도: {overopt_flags}개 항목에서 과최적화 의심.")
    if improved is False:
        lines.append("- 국면별 가중치가 균등가중치 대비 개선을 보이지 못함 — 핵심 가설 재검토 필요.")
    elif improved is True:
        lines.append("- 국면별 가중치가 균등가중치 대비 CAGR을 개선함.")
    if "error" not in random_stats and not random_stats["유의(p<0.05)"]:
        lines.append("- 랜덤 진입 대비 통계적 유의성 확보 실패.")
    elif "error" not in random_stats:
        lines.append("- 랜덤 진입 대비 통계적으로 유의미한 우위 확인.")
    if removal_candidates:
        lines.append(f"- 제거 후보 신호: {', '.join(removal_candidates)}.")
    lines.append("")
    lines.append(
        "이 리포트는 성과를 좋게 보이도록 임의로 조정하지 않고, 위 4개 검증에서 나온 결과를 그대로 기록한 것이다."
    )

    return "\n".join(lines)


def main() -> None:
    tickers, benchmark_close, raw_data = fetch_context()

    print(f"기준(Baseline) 백테스트 실행 중 ({START}~{END})...")
    t0 = time.perf_counter()
    base_result = bt.run(tickers, START, END, data_start=DATA_START, benchmark_close=benchmark_close)
    print(f"  완료: {time.perf_counter() - t0:.1f}초, 거래 {len(base_result.trades)}건")

    print("1. 파라미터 민감도 테스트 실행 중...")
    sensitivity_df = run_sensitivity(tickers, benchmark_close, base_result)

    print("2. 국면 가중치 무력화 대조군 실행 중...")
    regime_df, _ = run_regime_neutralization(tickers, benchmark_close, base_result)

    print("3. 랜덤 진입 대조군 계산 중...")
    random_stats = run_random_control(raw_data, base_result)

    print("4. 신호별 실제 승률 테이블 계산 중...")
    signal_df = run_signal_winrate(raw_data)

    print("리포트 작성 중...")
    report = build_report(base_result, benchmark_close, sensitivity_df, regime_df, random_stats, signal_df)

    REPORT_DIR.mkdir(exist_ok=True)
    report_path = REPORT_DIR / f"validation_{date.today().strftime('%Y%m%d')}.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"완료: {report_path}")


if __name__ == "__main__":
    main()
