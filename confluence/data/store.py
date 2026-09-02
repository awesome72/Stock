"""SQLite 기반 시세·수급 캐시 저장소.

같은 (ticker, date) 구간을 두 번 요청해도 pykrx를 다시 호출하지 않도록
캐시 존재 여부를 확인하는 데 사용한다. 모든 쓰기는 upsert(INSERT OR REPLACE)로
처리해 (ticker, date) 복합 기본키 충돌 시 최신 값으로 갱신한다.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "confluence.db"


def get_connection(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    """SQLite 커넥션을 열고 필요한 테이블이 없으면 생성해 반환한다."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    _create_tables(conn)
    return conn


def _create_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ohlcv (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (ticker, date)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS flow (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            foreign_net REAL,
            institution_net REAL,
            individual_net REAL,
            PRIMARY KEY (ticker, date)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fetch_range (
            ticker TEXT NOT NULL,
            kind TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            PRIMARY KEY (ticker, kind)
        )
        """
    )
    conn.commit()


def normalize_date(date_str: str) -> str:
    """'YYYYMMDD' 또는 'YYYY-MM-DD' 입력을 'YYYY-MM-DD'로 통일한다."""
    s = date_str.replace("-", "")
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"


def upsert_ohlcv(conn: sqlite3.Connection, ticker: str, df: pd.DataFrame) -> None:
    """date 인덱스, open/high/low/close/volume 컬럼을 가진 df를 ohlcv 테이블에 upsert.

    pandas/pykrx는 numpy 스칼라(float64/int64)를 사용하는데, sqlite3는 이를 native
    Python 타입으로 인식하지 못하고 원시 바이트(BLOB)로 저장해버리므로 float()/int()로
    명시적으로 변환해야 한다.
    """
    rows = [
        (
            ticker,
            normalize_date(str(idx)),
            float(row.open),
            float(row.high),
            float(row.low),
            float(row.close),
            int(row.volume),
        )
        for idx, row in df.iterrows()
    ]
    conn.executemany(
        """
        INSERT OR REPLACE INTO ohlcv (ticker, date, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()


def upsert_flow(conn: sqlite3.Connection, ticker: str, df: pd.DataFrame) -> None:
    """date 인덱스, foreign_net/institution_net/individual_net 컬럼을 가진 df를 flow 테이블에 upsert.

    upsert_ohlcv와 동일한 이유로 numpy 스칼라를 float()로 명시 변환한다.
    """
    rows = [
        (
            ticker,
            normalize_date(str(idx)),
            float(row.foreign_net),
            float(row.institution_net),
            float(row.individual_net),
        )
        for idx, row in df.iterrows()
    ]
    conn.executemany(
        """
        INSERT OR REPLACE INTO flow (ticker, date, foreign_net, institution_net, individual_net)
        VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()


def query_ohlcv(conn: sqlite3.Connection, ticker: str, start: str, end: str) -> pd.DataFrame:
    """캐시에서 [start, end] 구간의 OHLCV를 date 인덱스 DataFrame으로 반환한다."""
    df = pd.read_sql_query(
        """
        SELECT date, open, high, low, close, volume FROM ohlcv
        WHERE ticker = ? AND date BETWEEN ? AND ?
        ORDER BY date
        """,
        conn,
        params=(ticker, normalize_date(start), normalize_date(end)),
        parse_dates=["date"],
    )
    return df.set_index("date")


def query_flow(conn: sqlite3.Connection, ticker: str, start: str, end: str) -> pd.DataFrame:
    """캐시에서 [start, end] 구간의 투자자별 순매수를 date 인덱스 DataFrame으로 반환한다."""
    df = pd.read_sql_query(
        """
        SELECT date, foreign_net, institution_net, individual_net FROM flow
        WHERE ticker = ? AND date BETWEEN ? AND ?
        ORDER BY date
        """,
        conn,
        params=(ticker, normalize_date(start), normalize_date(end)),
        parse_dates=["date"],
    )
    return df.set_index("date")


def cached_date_range(conn: sqlite3.Connection, table: str, ticker: str) -> tuple[str | None, str | None]:
    """해당 ticker에 대해 테이블에 캐시된 최소/최대 date를 반환한다. 캐시가 없으면 (None, None)."""
    if table not in ("ohlcv", "flow"):
        raise ValueError(f"알 수 없는 테이블: {table}")
    cur = conn.execute(f"SELECT MIN(date), MAX(date) FROM {table} WHERE ticker = ?", (ticker,))
    row = cur.fetchone()
    return (row[0], row[1]) if row else (None, None)


def get_fetch_range(conn: sqlite3.Connection, kind: str, ticker: str) -> tuple[str | None, str | None]:
    """이 ticker에 대해 이미 API로 요청 완료된 [start,end] 요청 구간(envelope)을 반환한다.

    실제 저장된 데이터의 날짜 범위(cached_date_range)와 달리, 공휴일 등으로 거래일이
    없는 날짜가 end로 요청되어도 캐시 적중을 정확히 판별할 수 있도록 "요청했던 구간" 자체를
    별도로 기록해 둔 값이다. 캐시가 없으면 (None, None).
    """
    cur = conn.execute(
        "SELECT start_date, end_date FROM fetch_range WHERE ticker = ? AND kind = ?",
        (ticker, kind),
    )
    row = cur.fetchone()
    return (row[0], row[1]) if row else (None, None)


def expand_fetch_range(conn: sqlite3.Connection, kind: str, ticker: str, start: str, end: str) -> None:
    """요청 구간 [start,end]를 기존 envelope과 합쳐(min/max) 기록한다."""
    start_n, end_n = normalize_date(start), normalize_date(end)
    prev_start, prev_end = get_fetch_range(conn, kind, ticker)
    if prev_start is not None:
        start_n = min(start_n, prev_start)
        end_n = max(end_n, prev_end)
    conn.execute(
        """
        INSERT OR REPLACE INTO fetch_range (ticker, kind, start_date, end_date)
        VALUES (?, ?, ?, ?)
        """,
        (ticker, kind, start_n, end_n),
    )
    conn.commit()
