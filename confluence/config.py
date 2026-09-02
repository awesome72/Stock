"""Confluence 전역 상수: 지표 파라미터, 국면 임계값, 국면별 가중치, 거래비용, 리스크 규칙.

하드코딩 금지 원칙에 따라 모든 튜닝 가능한 숫자는 이 파일에서만 정의한다.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 지표 파라미터
# ---------------------------------------------------------------------------

RSI_PERIOD: int = 14

MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9

BB_PERIOD: int = 20
BB_STD: float = 2.0

ATR_PERIOD: int = 14

ADX_PERIOD: int = 14

ICHIMOKU_TENKAN: int = 9
ICHIMOKU_KIJUN: int = 26
ICHIMOKU_SENKOU_B: int = 52

SMA_WINDOWS: tuple[int, int, int, int] = (5, 20, 60, 120)

SUPERTREND_ATR_PERIOD: int = 10
SUPERTREND_MULTIPLIER: float = 3.0

KELTNER_EMA_PERIOD: int = 20
KELTNER_MULTIPLIER: float = 2.0

STOCH_K_PERIOD: int = 14
STOCH_SMOOTH_K: int = 3
STOCH_D_PERIOD: int = 3

CCI_PERIOD: int = 20

MFI_PERIOD: int = 14
VWAP_WINDOW: int = 20  # 일봉 데이터를 사용하므로 진짜 일중 VWAP가 아닌 롤링 근사치
VOLUME_PROFILE_WINDOW: int = 60
VOLUME_PROFILE_BINS: int = 20

RSI_DIVERGENCE_LOOKBACK: int = 60
RSI_DIVERGENCE_PIVOT_WINDOW: int = 5  # look-ahead 없이 과거 N봉 기준 신고가 갱신 시점을 스윙 고점으로 정의

# ---------------------------------------------------------------------------
# 스코어링 엔진 파라미터 (engine/scorer.py)
# ---------------------------------------------------------------------------

OBV_TREND_LOOKBACK: int = 20  # OBV가 N일 전보다 높은지로 상승/하락 판별
VOLUME_SURGE_LOOKBACK: int = 20  # 거래량 급증 판별 기준 이동평균 기간
VOLUME_SURGE_MULTIPLIER: float = 1.5  # 이 배수 이상이면 거래량 급증
FLOW_STREAK_FULL_SCORE_DAYS: int = 5  # 연속 순매수일수가 이 값 이상이면 수급 서브점수 만점
RS_WINDOW: int = 20  # 상대강도 계산에 사용할 수익률 비교 기간(거래일)
RS_SCALE: float = 0.10  # 벤치마크 대비 초과수익률을 0~1 점수로 매핑할 때의 정규화 폭(±10%p)

# ---------------------------------------------------------------------------
# 국면(Regime) 판별 임계값
# ---------------------------------------------------------------------------

REGIME_ADX_TRENDING: float = 25.0  # ADX >= 이 값이면 추세장
REGIME_ADX_RANGING: float = 20.0  # ADX < 이 값이면 횡보장 후보

REGIME_BB_WIDTH_LOOKBACK: int = 120  # 밴드폭 백분위 계산 기간
REGIME_BB_WIDTH_PERCENTILE: float = 0.30  # 하위 30% 이내면 횡보(Squeeze) 조건 충족

REGIME_ATR_LOOKBACK: int = 20  # ATR 평균 계산 기간
REGIME_ATR_VOLATILE_MULTIPLIER: float = 1.5  # 20일 평균 ATR의 이 배수 이상이면 변동성 확대

REGIME_LABELS: tuple[str, str, str, str] = (
    "TRENDING_UP",
    "TRENDING_DOWN",
    "RANGING",
    "VOLATILE",
)

# 국면 판별 우선순위: VOLATILE > TRENDING(UP/DOWN) > RANGING > 기본값(RANGING)
REGIME_PRIORITY: tuple[str, str, str, str] = (
    "VOLATILE",
    "TRENDING_UP",
    "TRENDING_DOWN",
    "RANGING",
)

# ---------------------------------------------------------------------------
# 카테고리 배점: 기본값 및 국면별 재배분 (총 100점)
# ---------------------------------------------------------------------------

CATEGORIES: tuple[str, str, str, str, str] = (
    "trend",
    "momentum",
    "volume",
    "flow",
    "relative_strength",
)

CATEGORY_LABEL_KO: dict[str, str] = {
    "trend": "추세",
    "momentum": "모멘텀",
    "volume": "거래량",
    "flow": "수급",
    "relative_strength": "상대강도",
}

BASE_CATEGORY_WEIGHTS: dict[str, int] = {
    "trend": 35,
    "momentum": 20,
    "volume": 20,
    "flow": 15,
    "relative_strength": 10,
}

REGIME_CATEGORY_WEIGHTS: dict[str, dict[str, int]] = {
    "TRENDING_UP": {
        "trend": 45,
        "momentum": 10,
        "volume": 20,
        "flow": 15,
        "relative_strength": 10,
    },
    "RANGING": {
        "trend": 15,
        "momentum": 40,
        "volume": 20,
        "flow": 15,
        "relative_strength": 10,
    },
    "TRENDING_DOWN": {
        "trend": 40,
        "momentum": 10,
        "volume": 20,
        "flow": 20,
        "relative_strength": 10,
    },
    "VOLATILE": {
        "trend": 25,
        "momentum": 15,
        "volume": 30,
        "flow": 20,
        "relative_strength": 10,
    },
}

# ---------------------------------------------------------------------------
# 스코어 등급 구간
# ---------------------------------------------------------------------------

GRADE_STRONG_CONFLUENCE: int = 80  # 80~100: 강한 합의
GRADE_WATCH: int = 65  # 65~79: 관심
GRADE_NO_ENTRY: int = 45  # 45~64: 진입 금지 (반드시 이 명칭 유지)
# 0~44: 회피

# ---------------------------------------------------------------------------
# 거래비용
# ---------------------------------------------------------------------------

COMMISSION_RATE: float = 0.00015  # 수수료 0.015%
TAX_RATE: float = 0.0018  # 세금(거래세) 0.18%
SLIPPAGE_RATE: float = 0.001  # 슬리피지 0.1%

# ---------------------------------------------------------------------------
# 리스크 관리
# ---------------------------------------------------------------------------

RISK_PER_TRADE_PCT: float = 0.02  # 1회 최대 손실: 계좌의 2%
STOP_LOSS_ATR_MULTIPLIER: float = 2.0  # 손절선: 진입가 - 2*ATR
MAX_POSITION_PCT: float = 0.20  # 종목당 최대 비중 20%
MAX_CONCURRENT_POSITIONS: int = 8  # 동시 보유 최대 종목 수 (5~8)
TRAILING_STOP_TRIGGER_R: float = 1.0  # 수익 1R 도달 시 본전 이동
TRAILING_STOP_ATR_MULTIPLIER: float = 3.0  # 이후 3*ATR 추적
