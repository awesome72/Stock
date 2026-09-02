# Confluence — 다중 이론 합의 기반 기술적 분석 엔진

## 프로젝트 목적
개별 기술적 지표가 충돌할 때, 시장 국면(Regime)에 따라 가중치를 다르게 적용해
0~100점 스코어를 산출하고, 그 규칙이 과거에 유효했는지 백테스트로 검증하는 프로그램.

## 절대 규칙 (위반 시 코드 거부)
1. **Look-ahead bias 금지**: 시점 t의 신호 계산에 t 이후 데이터를 절대 사용하지 않는다.
   신호는 종가 확정 후 생성, 체결은 익일 시가 기준.
2. **손절 없는 진입 신호 금지**: 모든 매수 신호는 ATR 기반 손절가를 함께 반환해야 한다.
3. **"예측" 금지**: 함수명·변수명·출력 문구에 predict/forecast 사용 금지.
   probability, score, signal, evidence 를 사용한다.
4. **파라미터 5개 초과 금지**: 전략의 튜닝 가능한 파라미터는 최대 5개.
5. **수정주가 사용**: 배당락/액면분할 미조정 데이터 사용 금지.

## 기술 스택
- Python 3.11+
- 데이터: pykrx (국내 시세·수급), FinanceDataReader
- 연산: pandas, numpy, pandas-ta
- 저장: SQLite (파일 1개, 별도 서버 없음)
- UI: Streamlit + Plotly
- 테스트: pytest

## 코드 스타일
- 모든 지표 함수는 `pd.DataFrame`을 받아 `pd.Series` 또는 `pd.DataFrame`을 반환
- 반복문 대신 벡터 연산 (for 루프로 지표 계산 금지)
- 타입 힌트 필수
- 함수 docstring에 **입력 컬럼 요구사항과 반환값 의미**를 명시
- 하드코딩된 숫자 금지 → `config.py`의 상수로 분리

## 디렉토리 구조
confluence/
├── config.py            # 모든 파라미터·가중치 상수
├── data/
│   ├── loader.py        # pykrx 수집
│   └── store.py         # SQLite 캐시
├── indicators/
│   ├── trend.py         # SMA, MACD, ADX, 일목, Supertrend
│   ├── momentum.py      # RSI, Stochastic, CCI, 다이버전스
│   ├── volatility.py    # Bollinger, ATR, Keltner
│   ├── volume.py        # OBV, VWAP, MFI, 매물대
│   └── flow.py          # 외국인/기관 수급 (한국 전용)
├── engine/
│   ├── regime.py        # 국면 판별
│   └── scorer.py        # 가중 스코어링
├── backtest/
│   ├── engine.py
│   └── metrics.py
├── app.py               # Streamlit 진입점
└── tests/

## 하지 말 것
- 지표를 요청 없이 추가하지 말 것
- 백테스트 성과를 근거 없이 좋게 보이도록 조정하지 말 것
- 데이터가 없을 때 임의 값으로 채우지 말 것 (명시적 에러를 낼 것)
- UI를 먼저 예쁘게 만들지 말 것 (P5까지 기능 우선)
