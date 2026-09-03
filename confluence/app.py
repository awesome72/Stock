"""Streamlit 진입점: 종목 분석 / 스크리너 / 백테스트 3개 탭.

`streamlit run confluence/app.py`로 실행한다(리포지토리 루트가 아닌 다른 위치에서
실행하거나, 배포 환경에서 sys.path에 리포지토리 루트가 없을 수도 있으므로
아래에서 명시적으로 추가한다 - confluence 패키지 내부 파일이 곧 실행 스크립트인
특수한 경우라 상대 import 대신 절대 import를 쓴다).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from confluence import config
from confluence.backtest import engine as bt_engine
from confluence.backtest import metrics as bt_metrics
from confluence.data import loader
from confluence.engine import regime as regime_engine
from confluence.engine import scorer
from confluence.indicators import momentum, trend, volatility

st.set_page_config(page_title="Confluence", layout="wide")

TODAY = pd.Timestamp.today().strftime("%Y%m%d")
# 종목분석/스크리너 탭은 백테스트가 아니라 '현재 스코어카드' 조회가 목적이므로,
# config.BACKTEST_WARMUP_DAYS(백테스트용 워밍업 버퍼)보다 넉넉한 3년치를 받아와
# SMA120/일목(52+26)/RSI다이버전스(60봉) 등 가장 긴 지표도 충분히 워밍업되게 한다.
ANALYSIS_START = (pd.Timestamp.today() - pd.DateOffset(years=3)).strftime("%Y%m%d")


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_universe(market: str, top_n: int) -> list[str]:
    return loader.fetch_universe(market, top_n=top_n)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_benchmark(start: str, end: str) -> pd.Series | None:
    try:
        return loader.fetch_index_ohlcv(config.KOSPI_INDEX_CODE, start, end)["close"]
    except RuntimeError:
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def _cached_ticker_name(ticker: str) -> str:
    return loader.fetch_ticker_name(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_ohlcv_with_flow(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = loader.fetch_ohlcv(ticker, start, end)
    if df.empty:
        return df
    try:
        flow_df = loader.fetch_investor_flow(ticker, start, end)
        df = df.join(flow_df, how="left")
    except RuntimeError:
        pass  # KRX 미인증: 수급 데이터 없이 진행 (해당 종목의 수급 카테고리는 계산 불가로 처리됨)
    return df


# 지표/카테고리 설명 (UI 표시용 정적 텍스트 - 튜닝 파라미터가 아니므로 config.py가 아닌 여기 둔다)

OVERLAY_HELP: dict[str, str] = {
    "이동평균": "일정 기간(SMA5/20/60/120) 종가의 평균을 이은 선. 가격이 이평선 위에 있고 "
    "짧은 이평선이 긴 이평선 위에 있으면(정배열) 상승 추세로 해석한다.",
    "볼린저밴드": "이동평균에 표준편차를 더하고 뺀 상단/하단 밴드. 밴드 폭은 변동성을, "
    "밴드 이탈은 과매수·과매도 가능성을 나타낸다.",
    "일목균형표": "전환선·기준선·구름대(선행스팬A/B)·후행스팬으로 구성된 추세 지표. "
    "가격이 구름대 위/아래에 있는지로 추세 방향을, 구름 두께로 지지·저항 강도를 가늠한다.",
    "거래량": "일별 체결 주식 수. 가격 변동에 실린 매매 강도를 보여준다.",
}

CHART_INDICATOR_HELP: dict[str, str] = {
    "MACD": "단기(12일)·장기(26일) 이동평균의 차이(MACD선)와 그 신호선(Signal, 9일 평균)의 "
    "교차로 추세 전환 시점을 포착한다. 막대(Histogram)는 둘의 차이로, 커질수록 추세가 강하다.",
    "RSI": "최근 상승폭과 하락폭의 비율을 0~100으로 나타낸 오실레이터. 통상 70 이상은 "
    "과매수, 30 이하는 과매도 구간으로 참고하되, 강한 추세장에서는 과매수 상태가 "
    "오히려 모멘텀 강도를 뜻할 수 있다.",
}

CATEGORY_HELP: dict[str, str] = {
    "trend": "정배열(이동평균), MACD, 일목균형표로 추세의 방향과 강도를 평가한다.",
    "momentum": "RSI, 스토캐스틱, RSI 다이버전스로 상승·하락 탄력을 평가한다. 국면에 따라 "
    "과매수 해석이 달라진다(추세장=모멘텀 강도, 횡보장=반전 신호).",
    "volume": "OBV(누적거래량), 매물대(POC) 돌파, 거래량 급증 여부로 가격 움직임에 실린 "
    "거래 강도를 평가한다.",
    "flow": "외국인·기관 순매수 연속일수로 수급 주체의 매수/매도 압력을 평가한다. "
    "KRX 인증 정보가 없거나 최근 데이터가 없으면 계산하지 않는다.",
    "relative_strength": "KOSPI 지수 대비 초과수익률로 시장 대비 상대적 강도를 평가한다. "
    "벤치마크 데이터가 없으면 중립값(0.5)으로 처리한다.",
}


def _regime_ko(label) -> str:
    if pd.isna(label):
        return "판별불가"
    return regime_engine.REGIME_LABEL_KO.get(label, str(label))


def _fmt_metric(key: str, value: float) -> str:
    if pd.isna(value):
        return "N/A"
    if key in ("CAGR", "MDD", "승률"):
        return f"{value * 100:+.2f}%"
    if key == "거래횟수":
        return str(int(value))
    return f"{value:.2f}"


# ---------------------------------------------------------------------------
# 탭1: 종목 분석
# ---------------------------------------------------------------------------


def _build_price_figure(df: pd.DataFrame, overlays: dict[str, bool]) -> go.Figure:
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.175, 0.175],
        subplot_titles=("가격", "거래량", "MACD", "RSI"),
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="가격"
        ),
        row=1,
        col=1,
    )

    if overlays.get("이동평균"):
        for w in config.SMA_WINDOWS:
            sma = df["close"].rolling(w).mean()
            fig.add_trace(go.Scatter(x=df.index, y=sma, name=f"SMA{w}", line=dict(width=1)), row=1, col=1)

    if overlays.get("볼린저밴드"):
        bb = volatility.bollinger(df)
        fig.add_trace(go.Scatter(x=df.index, y=bb["upper"], name="BB상단", line=dict(width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=bb["lower"], name="BB하단", line=dict(width=1, dash="dot")), row=1, col=1)

    if overlays.get("일목균형표"):
        ich = trend.ichimoku(df)
        fig.add_trace(go.Scatter(x=df.index, y=ich["senkou_a"], name="선행스팬A", line=dict(width=1)), row=1, col=1)
        fig.add_trace(
            go.Scatter(x=df.index, y=ich["senkou_b"], name="선행스팬B", line=dict(width=1), fill="tonexty"),
            row=1,
            col=1,
        )
        fig.add_trace(go.Scatter(x=df.index, y=ich["tenkan"], name="전환선", line=dict(width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=ich["kijun"], name="기준선", line=dict(width=1)), row=1, col=1)

    if overlays.get("거래량"):
        fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="거래량", showlegend=False), row=2, col=1)

    macd_df = trend.macd(df)
    fig.add_trace(go.Scatter(x=df.index, y=macd_df["macd"], name="MACD", line=dict(width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=macd_df["signal"], name="Signal", line=dict(width=1)), row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=macd_df["histogram"], name="Histogram", showlegend=False), row=3, col=1)

    rsi_series = momentum.rsi(df)
    fig.add_trace(go.Scatter(x=df.index, y=rsi_series, name="RSI", line=dict(width=1)), row=4, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="gray", row=4, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="gray", row=4, col=1)

    fig.update_layout(height=800, xaxis_rangeslider_visible=False, margin=dict(t=30, b=10))
    return fig


def render_stock_analysis_tab() -> None:
    st.subheader("종목 분석")
    default_ticker = st.session_state.get("selected_ticker", "005930")
    ticker = st.text_input("티커(6자리 종목코드)", value=default_ticker, key="ticker_input")
    if not ticker:
        return

    company_name = _cached_ticker_name(ticker)
    if company_name:
        st.markdown(f"#### {company_name} ({ticker})")
    else:
        st.warning(f"{ticker}: 종목명을 찾을 수 없습니다. 티커를 확인하세요(상장폐지 종목일 수 있음).")

    overlay_cols = st.columns(4)
    overlays = {
        "이동평균": overlay_cols[0].checkbox("이동평균", value=True, help=OVERLAY_HELP["이동평균"]),
        "볼린저밴드": overlay_cols[1].checkbox("볼린저밴드", value=False, help=OVERLAY_HELP["볼린저밴드"]),
        "일목균형표": overlay_cols[2].checkbox("일목균형표", value=False, help=OVERLAY_HELP["일목균형표"]),
        "거래량": overlay_cols[3].checkbox("거래량", value=True, help=OVERLAY_HELP["거래량"]),
    }
    with st.expander("차트 지표 설명"):
        for name, desc in {**OVERLAY_HELP, **CHART_INDICATOR_HELP}.items():
            st.markdown(f"- **{name}**: {desc}")

    with st.spinner(f"{ticker} 데이터 조회 중..."):
        df = _cached_ohlcv_with_flow(ticker, ANALYSIS_START, TODAY)
        benchmark_close = _cached_benchmark(ANALYSIS_START, TODAY)

    if df.empty:
        st.error(f"{ticker}: 데이터를 가져오지 못했습니다. 종목코드를 확인하세요.")
        return

    chart_col, score_col = st.columns([3, 1])

    with chart_col:
        fig = _build_price_figure(df, overlays)
        st.plotly_chart(fig, width="stretch")

    with score_col:
        last_date = df.index[-1]
        try:
            card = scorer.score(df, last_date, benchmark_close=benchmark_close, ticker=ticker)
        except ValueError as exc:
            # CLAUDE.md 규칙: 데이터가 없을 때 임의 값으로 채우지 않는다 - 계산 불가 사유를 그대로 노출.
            st.warning(f"스코어 계산 불가 ({last_date.date()}): {exc}")
            return

        st.metric("총점", f"{card.total:.1f} / 100", card.grade)
        st.caption(f"국면: {_regime_ko(card.regime)}  ·  기준일: {last_date.date()}")

        st.markdown("**카테고리별 점수**")
        for cat, scores in card.category_scores.items():
            ratio = scores["earned"] / scores["max"] if scores["max"] else 0.0
            st.progress(
                min(1.0, max(0.0, ratio)),
                text=f"{config.CATEGORY_LABEL_KO[cat]}  {scores['earned']:.1f} / {scores['max']:.0f}",
            )
        with st.expander("카테고리 설명"):
            for cat in card.category_scores:
                st.markdown(f"- **{config.CATEGORY_LABEL_KO[cat]}**: {CATEGORY_HELP[cat]}")

        st.markdown("**근거**")
        if card.evidences:
            for ev in card.evidences:
                st.write(f"- {ev}")
        else:
            st.write("- (해당 없음)")

        st.markdown(f"**반증 조건**\n\n{card.invalidation}")
        st.markdown(f"**손절가**: {card.stop_loss:,.0f}원")
        st.markdown(f"**권장 비중**: {card.position_size_pct * 100:.1f}%")


# ---------------------------------------------------------------------------
# 탭2: 스크리너
# ---------------------------------------------------------------------------


def _run_scan(tickers: list[str], benchmark_close: pd.Series | None) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict] = []
    errors: list[str] = []
    progress = st.progress(0.0, text="스캔 준비 중...")
    total = len(tickers)
    for i, ticker in enumerate(tickers):
        try:
            df = _cached_ohlcv_with_flow(ticker, ANALYSIS_START, TODAY)
            if not df.empty:
                score_series = scorer.total_score_series(df, benchmark_close)
                regime_series = regime_engine.classify_regime(df)
                last_valid = score_series.last_valid_index()
                if last_valid is not None:
                    total_score = score_series.loc[last_valid]
                    rows.append(
                        {
                            "티커": ticker,
                            "종목명": _cached_ticker_name(ticker),
                            "점수": round(float(total_score), 1),
                            "등급": scorer.grade(total_score),
                            "국면": _regime_ko(regime_series.loc[last_valid]),
                            "기준일": last_valid.date(),
                        }
                    )
        except Exception as exc:  # noqa: BLE001 - 스캔 중 개별 종목 오류로 전체 스캔이 멈추면 안 됨
            errors.append(f"{ticker}: {exc}")
        progress.progress((i + 1) / total, text=f"스캔 중... ({i + 1}/{total})")
    progress.empty()
    return pd.DataFrame(rows), errors


def render_screener_tab() -> None:
    st.subheader("스크리너")
    col1, col2, col3 = st.columns(3)
    universe_size = col1.slider(
        "스캔 종목 수(시가총액 순위 상위 N)",
        10,
        200,
        50,
        step=10,
        help="실제 시가총액(원화) 값이 아니라 시가총액 순위 기준 상위 N종목을 스캔 대상으로 삼는다.",
    )
    min_score = col2.slider("최소 점수", 0, 100, 65)
    regime_options = list(regime_engine.REGIME_LABEL_KO.values())
    regime_filter = col3.multiselect("국면 필터", regime_options, default=regime_options)

    st.caption(
        "첫 스캔은 KRX 조회 때문에 종목당 최대 1초 정도 걸릴 수 있다(요청 사이 0.3초 지연 포함). "
        "이후 1시간 동안은 캐시된 결과를 사용한다."
    )
    with st.expander("점수·등급·국면 설명"):
        st.markdown("**점수(0~100)**: 아래 5개 카테고리 점수의 가중합(국면에 따라 가중치가 달라진다).")
        for cat in config.CATEGORIES:
            st.markdown(f"- **{config.CATEGORY_LABEL_KO[cat]}**: {CATEGORY_HELP[cat]}")
        st.markdown(
            f"**등급**: {config.GRADE_STRONG_CONFLUENCE}점 이상 강한 합의 · "
            f"{config.GRADE_WATCH}~{config.GRADE_STRONG_CONFLUENCE - 1}점 관심 · "
            f"{config.GRADE_NO_ENTRY}~{config.GRADE_WATCH - 1}점 진입 금지 · "
            f"{config.GRADE_NO_ENTRY}점 미만 회피"
        )
        st.markdown("**국면**: 추세/변동성 지표로 판별한 시장 국면(추세 상승·추세 하락·횡보·변동성 확대)에 따라 카테고리별 가중치가 달라진다.")

    if st.button("스캔 실행", type="primary"):
        tickers = _cached_universe("KOSPI", universe_size)
        benchmark_close = _cached_benchmark(ANALYSIS_START, TODAY)
        result_df, errors = _run_scan(tickers, benchmark_close)
        st.session_state["screener_result"] = result_df
        st.session_state["screener_errors"] = errors

    result = st.session_state.get("screener_result")
    if result is None:
        st.info("스캔 실행 버튼을 눌러 유니버스를 스캔하세요.")
        return
    if result.empty:
        st.warning("스캔 결과가 없습니다(전 종목 데이터 부족 또는 워밍업 미완료).")
        return

    filtered = result[(result["점수"] >= min_score) & (result["국면"].isin(regime_filter))]
    filtered = filtered.sort_values("점수", ascending=False)
    st.dataframe(filtered, width="stretch", hide_index=True)

    errors = st.session_state.get("screener_errors") or []
    if errors:
        with st.expander(f"스캔 중 오류 발생 종목 {len(errors)}건"):
            for e in errors:
                st.write(f"- {e}")

    if not filtered.empty:
        # Streamlit 탭은 코드에서 프로그래밍적으로 전환할 수 없으므로(공식 API 없음),
        # session_state에 선택값을 저장해두고 '종목 분석' 탭이 그 값을 기본값으로 읽게 한다.
        name_by_ticker = dict(zip(filtered["티커"], filtered["종목명"]))
        chosen = st.selectbox(
            "종목 분석 탭에서 확인할 티커",
            filtered["티커"].tolist(),
            format_func=lambda t: f"{t} ({name_by_ticker.get(t, '')})",
        )
        if st.button("종목 분석 탭으로 전달"):
            st.session_state["selected_ticker"] = chosen
            st.success(f"{chosen}을(를) '종목 분석' 탭에 전달했습니다. 상단의 '종목 분석' 탭을 클릭하세요.")


# ---------------------------------------------------------------------------
# 탭3: 백테스트
# ---------------------------------------------------------------------------


def render_backtest_tab() -> None:
    st.subheader("백테스트")
    col1, col2, col3, col4 = st.columns(4)
    start_date = col1.date_input("시작일", value=pd.Timestamp.today() - pd.DateOffset(years=5))
    end_date = col2.date_input("종료일", value=pd.Timestamp.today())
    entry_score = col3.number_input("진입 점수", 0.0, 100.0, config.BACKTEST_ENTRY_SCORE, step=1.0)
    exit_score = col4.number_input("청산 점수", 0.0, 100.0, config.BACKTEST_EXIT_SCORE, step=1.0)
    universe_size = st.slider("유니버스(시가총액 순위 상위 N종목)", 10, 200, 50, step=10)

    if st.button("백테스트 실행", type="primary"):
        if start_date >= end_date:
            st.error("시작일은 종료일보다 빨라야 합니다.")
            return
        start_str = pd.Timestamp(start_date).strftime("%Y%m%d")
        end_str = pd.Timestamp(end_date).strftime("%Y%m%d")
        with st.spinner("백테스트 실행 중... (신규 구간은 KRX 조회로 시간이 걸릴 수 있음)"):
            tickers = _cached_universe("KOSPI", universe_size)
            warmup_start = (pd.Timestamp(start_date) - pd.DateOffset(days=config.BACKTEST_WARMUP_DAYS)).strftime(
                "%Y%m%d"
            )
            benchmark_close = _cached_benchmark(warmup_start, end_str)
            result = bt_engine.run(
                tickers,
                start_str,
                end_str,
                entry_score=entry_score,
                exit_score=exit_score,
                benchmark_close=benchmark_close,
            )
        st.session_state["backtest_result"] = result

    result: bt_engine.BacktestResult | None = st.session_state.get("backtest_result")
    if result is None:
        st.info("백테스트 실행 버튼을 눌러 결과를 확인하세요.")
        return
    if len(result.equity_curve) == 0:
        st.warning("백테스트 결과가 비어 있습니다(거래가 발생하지 않았거나 데이터 부족).")
        return

    trade_pnl = pd.Series([t.pnl for t in result.trades], dtype=float)
    perf = bt_metrics.summary(result.equity_curve, trade_pnl)

    metric_cols = st.columns(4)
    for i, (k, v) in enumerate(perf.items()):
        metric_cols[i % 4].metric(k, _fmt_metric(k, v))

    equity = result.equity_curve
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    mdd_end = drawdown.idxmin()
    mdd_start = equity.loc[:mdd_end].idxmax()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity / equity.iloc[0] - 1, name="전략"))
    if result.benchmark_curve is not None:
        bench = result.benchmark_curve.reindex(equity.index).ffill()
        if bench.notna().any():
            first_valid = bench.first_valid_index()
            bench_norm = bench / bench.loc[first_valid] - 1
            fig.add_trace(go.Scatter(x=bench_norm.index, y=bench_norm, name="KOSPI"))
    fig.add_vrect(
        x0=mdd_start, x1=mdd_end, fillcolor="red", opacity=0.12, line_width=0, annotation_text="MDD 구간"
    )
    fig.update_layout(height=450, yaxis_tickformat=".0%", margin=dict(t=30, b=10))
    st.plotly_chart(fig, width="stretch")

    trades_df = pd.DataFrame(
        [
            {
                "티커": t.ticker,
                "종목명": _cached_ticker_name(t.ticker),
                "진입일": t.entry_date.date(),
                "청산일": t.exit_date.date(),
                "수익률": f"{t.return_pct * 100:+.2f}%",
                "청산사유": t.exit_reason,
            }
            for t in result.trades
        ]
    )
    st.dataframe(trades_df, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------


def main() -> None:
    st.title("Confluence")
    st.caption("다중 이론 합의 기반 기술적 분석 엔진 — 예측이 아닌 근거(evidence) 기반 스코어")

    if not (os.environ.get("KRX_ID") and os.environ.get("KRX_PW")):
        st.info(
            "KRX_ID/KRX_PW가 설정되지 않았습니다. 개별 종목 시세는 조회되지만, "
            "수급(외국인/기관)·유니버스·벤치마크 조회는 건너뜁니다."
        )

    tab1, tab2, tab3 = st.tabs(["종목 분석", "스크리너", "백테스트"])
    with tab1:
        render_stock_analysis_tab()
    with tab2:
        render_screener_tab()
    with tab3:
        render_backtest_tab()


if __name__ == "__main__":
    main()
