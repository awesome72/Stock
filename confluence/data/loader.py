"""pykrx를 통한 시세·수급 데이터 수집.

수급(fetch_investor_flow) 및 시가총액(fetch_universe) 조회는 KRX가 2025년경
data.krx.co.kr API에 인증을 요구하도록 변경하면서, 환경변수 KRX_ID / KRX_PW가
설정되어 있어야 정상 동작한다 (pykrx가 내부적으로 자동 로그인한다).
계정은 https://data.krx.co.kr 에서 무료로 발급받을 수 있다.
개별 종목 OHLCV(fetch_ohlcv)는 인증 없이 동작한다.
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from . import store

logger = logging.getLogger(__name__)

_PYKRX_SLEEP_SEC = 0.3
_MAX_RETRIES = 3

_KRX_AUTH_HINT = (
    "KRX_ID / KRX_PW 환경변수가 설정되어 있는지 확인하세요. "
    "https://data.krx.co.kr 에서 무료 계정을 발급받은 뒤 설정해야 합니다."
)


def _retry_call(func, *args, **kwargs):
    """pykrx 호출을 최대 _MAX_RETRIES회 재시도하고, 모두 실패하면 명시적 예외를 던진다."""
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # pykrx는 네트워크/파싱 오류를 일반 Exception으로 던진다
            last_exc = exc
            logger.warning("pykrx 호출 실패 (%d/%d): %s", attempt, _MAX_RETRIES, exc)
            time.sleep(_PYKRX_SLEEP_SEC)
    raise RuntimeError(f"pykrx 호출이 {_MAX_RETRIES}회 재시도 후에도 실패했습니다: {last_exc}") from last_exc


def fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """종목의 수정주가 일봉 OHLCV를 조회한다. SQLite 캐시를 우선 사용한다.

    Parameters
    ----------
    ticker : 6자리 종목코드 (예: "005930")
    start, end : "YYYYMMDD" 형식 문자열

    Returns
    -------
    date를 인덱스로 하는 DataFrame. 컬럼: open, high, low, close, volume.
    비영업일/결측 구간은 forward-fill 하지 않고 행 자체를 비워 둔다.
    거래정지 등으로 데이터가 짧으면 예외를 던지지 않고 로그 경고 후 그대로 반환한다.
    """
    from pykrx import stock

    conn = store.get_connection()
    try:
        start_n, end_n = store.normalize_date(start), store.normalize_date(end)
        fetched_start, fetched_end = store.get_fetch_range(conn, "ohlcv", ticker)

        if fetched_start is not None and fetched_start <= start_n and fetched_end >= end_n:
            return store.query_ohlcv(conn, ticker, start, end)

        logger.info("pykrx에서 %s OHLCV 수집: %s~%s", ticker, start, end)
        raw = _retry_call(stock.get_market_ohlcv_by_date, start, end, ticker, adjusted=True)
        time.sleep(_PYKRX_SLEEP_SEC)

        if raw is None or raw.empty:
            logger.warning("%s: 수집된 OHLCV가 없습니다 (%s~%s)", ticker, start, end)
            store.expand_fetch_range(conn, "ohlcv", ticker, start, end)
            return store.query_ohlcv(conn, ticker, start, end)

        raw = raw.rename(
            columns={"시가": "open", "고가": "high", "저가": "low", "종가": "close", "거래량": "volume"}
        )[["open", "high", "low", "close", "volume"]]

        # pykrx는 장기 거래정지 구간을 행을 비우는 대신 OHLC=0으로 채워서 반환하는 경우가 있다.
        # 0을 실제 가격으로 저장하면 이후 포지션 사이징(포지션가치/진입가)에서 0으로 나눠
        # cash가 inf/NaN으로 오염되므로, 결측과 동일하게 취급해 행 자체를 제거한다.
        invalid = (raw[["open", "high", "low", "close"]] <= 0).any(axis=1)
        if invalid.any():
            logger.warning(
                "%s: 가격이 0 이하인 행 %d개 제외 (거래정지 구간으로 추정)", ticker, int(invalid.sum())
            )
            raw = raw[~invalid]

        store.upsert_ohlcv(conn, ticker, raw)
        store.expand_fetch_range(conn, "ohlcv", ticker, start, end)

        expected_days = len(pd.bdate_range(start_n, end_n))
        if len(raw) < expected_days:
            logger.warning(
                "%s: 영업일 약 %d일 중 %d일만 수집됨 (거래정지/상장폐지 가능성)",
                ticker, expected_days, len(raw),
            )

        return store.query_ohlcv(conn, ticker, start, end)
    finally:
        conn.close()


def fetch_investor_flow(ticker: str, start: str, end: str) -> pd.DataFrame:
    """종목의 투자자별(외국인/기관/개인) 순매수 금액을 일별로 조회한다. SQLite 캐시를 우선 사용한다.

    KRX_ID / KRX_PW 환경변수가 설정되어 있지 않으면 pykrx가 빈 결과를 반환하므로,
    이 함수는 그 빈 결과를 임의 값(0 등)으로 채우지 않고 명시적 예외를 던진다.

    Parameters
    ----------
    ticker : 6자리 종목코드
    start, end : "YYYYMMDD" 형식 문자열

    Returns
    -------
    date를 인덱스로 하는 DataFrame. 컬럼: foreign_net, institution_net, individual_net (원화 순매수 금액).
    """
    from pykrx import stock

    conn = store.get_connection()
    try:
        start_n, end_n = store.normalize_date(start), store.normalize_date(end)
        fetched_start, fetched_end = store.get_fetch_range(conn, "flow", ticker)

        if fetched_start is not None and fetched_start <= start_n and fetched_end >= end_n:
            return store.query_flow(conn, ticker, start, end)

        logger.info("pykrx에서 %s 투자자별 순매수 수집: %s~%s", ticker, start, end)
        raw = _retry_call(stock.get_market_trading_value_by_date, start, end, ticker)
        time.sleep(_PYKRX_SLEEP_SEC)

        if raw is None or raw.empty:
            raise RuntimeError(f"{ticker}: 투자자별 순매수 데이터를 가져오지 못했습니다. {_KRX_AUTH_HINT}")

        raw = raw.rename(
            columns={"외국인합계": "foreign_net", "기관합계": "institution_net", "개인": "individual_net"}
        )[["foreign_net", "institution_net", "individual_net"]]

        store.upsert_flow(conn, ticker, raw)
        store.expand_fetch_range(conn, "flow", ticker, start, end)
        return store.query_flow(conn, ticker, start, end)
    finally:
        conn.close()


def fetch_universe(market: str = "KOSPI", top_n: int = 200) -> list[str]:
    """시가총액 상위 top_n 종목의 티커 리스트를 반환한다.

    KRX_ID / KRX_PW 환경변수가 설정되어 있지 않으면 pykrx가 빈 결과를 반환하므로,
    이 함수는 그 경우 임의의 티커 목록을 만들어내지 않고 명시적 예외를 던진다.

    Parameters
    ----------
    market : "KOSPI" | "KOSDAQ" | "KONEX" | "ALL"
    top_n : 반환할 종목 수

    Returns
    -------
    시가총액 내림차순으로 정렬된 6자리 종목코드 리스트.
    """
    from pykrx import stock

    ref_date = _retry_call(stock.get_nearest_business_day_in_a_week)
    time.sleep(_PYKRX_SLEEP_SEC)
    cap = _retry_call(stock.get_market_cap_by_ticker, ref_date, market=market)
    time.sleep(_PYKRX_SLEEP_SEC)

    if cap is None or cap.empty:
        raise RuntimeError(f"{market} 시가총액 데이터를 가져오지 못했습니다 (기준일 {ref_date}). {_KRX_AUTH_HINT}")

    return cap.sort_values("시가총액", ascending=False).head(top_n).index.tolist()


def fetch_ticker_name(ticker: str) -> str:
    """티커에 해당하는 종목명을 조회한다.

    KRX 상장/상장폐지 종목 목록(공개, 인증 불필요)에서 조회하며, 시세 조회와 달리
    KRX_ID/KRX_PW가 없어도 동작한다.

    Parameters
    ----------
    ticker : 6자리 종목코드

    Returns
    -------
    종목명 문자열. 목록에 없는 티커면 빈 문자열을 반환한다(예외를 던지지 않음 -
    화면에 티커만 표시하고 넘어갈 수 있게 하기 위함).
    """
    from pykrx import stock

    name = stock.get_market_ticker_name(ticker)
    return name if isinstance(name, str) else ""


def fetch_index_ohlcv(index_code: str, start: str, end: str) -> pd.DataFrame:
    """지수(예: KOSPI 종합지수)의 일봉 OHLCV를 조회한다. SQLite 캐시를 우선 사용한다.

    fetch_investor_flow/fetch_universe와 마찬가지로 KRX_ID/KRX_PW가 필요하다.

    Parameters
    ----------
    index_code : pykrx 지수 코드 (예: "1001" = KOSPI 종합지수, config.KOSPI_INDEX_CODE)
    start, end : "YYYYMMDD" 형식 문자열

    Returns
    -------
    date를 인덱스로 하는 DataFrame. 컬럼: open, high, low, close, volume.
    """
    from pykrx import stock

    cache_key = f"IDX_{index_code}"  # ohlcv 테이블은 (ticker, date) PK라 종목코드 자리에 지수 전용 키를 사용
    conn = store.get_connection()
    try:
        start_n, end_n = store.normalize_date(start), store.normalize_date(end)
        fetched_start, fetched_end = store.get_fetch_range(conn, "ohlcv", cache_key)

        if fetched_start is not None and fetched_start <= start_n and fetched_end >= end_n:
            return store.query_ohlcv(conn, cache_key, start, end)

        logger.info("pykrx에서 지수 %s OHLCV 수집: %s~%s", index_code, start, end)
        raw = _retry_call(stock.get_index_ohlcv_by_date, start, end, index_code)
        time.sleep(_PYKRX_SLEEP_SEC)

        if raw is None or raw.empty:
            raise RuntimeError(f"지수 {index_code} OHLCV 데이터를 가져오지 못했습니다. {_KRX_AUTH_HINT}")

        raw = raw.rename(
            columns={"시가": "open", "고가": "high", "저가": "low", "종가": "close", "거래량": "volume"}
        )[["open", "high", "low", "close", "volume"]]

        store.upsert_ohlcv(conn, cache_key, raw)
        store.expand_fetch_range(conn, "ohlcv", cache_key, start, end)
        return store.query_ohlcv(conn, cache_key, start, end)
    finally:
        conn.close()
