# `Confluence` Vibe Coding 프롬프트 세트

> PRD를 실제 코드로 옮기기 위한 단계별 프롬프트 모음.
> **한 프롬프트 = 한 커밋 = 한 검증** 원칙으로 진행할 것.

---

## 0. 사용 규칙 (먼저 읽을 것)

| 규칙 | 이유 |
|---|---|
| 프롬프트 하나 실행 후 **반드시 직접 실행해보고** 다음으로 | AI가 "완성했습니다"라고 해도 안 돌아가는 경우가 태반 |
| 각 단계 끝나면 커밋 | 롤백 지점 없으면 3시간 뒤 복구 불가 |
| 완료 조건(Acceptance)을 프롬프트에 항상 포함 | 없으면 AI가 그럴듯한 껍데기만 만듦 |
| 한 프롬프트에서 파일 3개 이상 건드리지 않기 | 컨텍스트 붕괴 방지 |
| 에러는 **전체 스택트레이스를 그대로** 붙여넣기 | 요약하면 진단 정확도 급락 |

---

## 1. 프로젝트 헌법 — `CLAUDE.md`

> 프로젝트 루트에 `CLAUDE.md`로 저장. Claude Code가 매 세션 자동으로 읽는다.
> (Cursor 사용 시 `.cursorrules`, 그 외에는 매 세션 첫 메시지로 붙여넣기)

```markdown
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
```

---

## 2. Phase별 구현 프롬프트

### P0 — 스캐폴딩

```
CLAUDE.md에 정의된 디렉토리 구조로 프로젝트 뼈대를 만들어줘.

요구사항:
1. 위 구조대로 폴더와 빈 모듈 파일 생성 (각 파일에 docstring만)
2. config.py에 아래 상수를 정의:
   - 지표 파라미터 (RSI 14, MACD 12/26/9, BB 20/2, ATR 14, ADX 14, 일목 9/26/52)
   - 국면 4종의 판별 임계값
   - 국면별 카테고리 가중치 딕셔너리 (추세/모멘텀/거래량/수급/RS)
   - 거래비용: 수수료 0.015%, 세금 0.18%, 슬리피지 0.1%
   - 리스크: 1회 최대손실 2%, 손절 2×ATR, 종목당 최대비중 20%
3. requirements.txt
4. .gitignore (data/*.db, __pycache__, .venv)
5. README.md는 3줄 이내로 간단히

아직 로직은 구현하지 마. 구조와 상수만.

완료 조건: `python -c "import confluence.config"` 가 에러 없이 실행됨
```

---

### P1 — 데이터 파이프라인

```
데이터 레이어를 구현해줘.

data/loader.py:
- fetch_ohlcv(ticker, start, end) -> DataFrame
  pykrx로 수정주가 일봉 수집. 컬럼: date(index), open, high, low, close, volume
- fetch_investor_flow(ticker, start, end) -> DataFrame
  투자자별 순매수(외국인/기관/개인) 금액 기준
- fetch_universe(market='KOSPI') -> list[str]
  시가총액 상위 200종목 티커

data/store.py:
- SQLite 캐시. 같은 구간을 두 번 요청하면 API를 다시 호출하지 않아야 함
- 테이블: ohlcv(ticker, date, ...), flow(ticker, date, ...)
- (ticker, date) 복합 유니크 인덱스
- upsert 방식 (중복 시 갱신)

필수 처리:
- pykrx API 호출 사이에 0.3초 sleep (차단 방지)
- 네트워크 실패 시 3회 재시도 후 명시적 예외
- 결측 구간을 절대 forward-fill 하지 말 것. 결측은 결측으로 남기고 로그로 경고
- 거래정지/상장폐지로 데이터가 짧은 종목은 예외를 던지지 말고 반환하되, 길이를 로그에 남길 것

완료 조건:
- 삼성전자(005930) 10년치 수집 후 재조회 시 1초 이내 반환
- 수집된 행 수와 결측일 수를 출력하는 간단한 스크립트 포함
```

---

### P2 — 지표 엔진

```
indicators/ 하위 5개 모듈을 구현해줘. pandas-ta를 쓰되, 없는 것은 직접 구현.

trend.py: sma_alignment(5/20/60/120 정배열 점수), macd, adx_dmi, ichimoku, supertrend
momentum.py: rsi, stochastic, cci, rsi_divergence
volatility.py: bollinger(+밴드폭 백분위), atr, keltner, squeeze_flag
volume.py: obv, vwap, mfi, volume_profile_poc(매물대 최대거래 가격대)
flow.py: foreign_net_streak(외국인 연속순매수일), institution_net_streak,
        flow_score_5d, flow_score_20d

공통 규칙:
- 모든 함수: (df: pd.DataFrame, **params) -> pd.Series | pd.DataFrame
- 계산 불가 구간은 NaN (0으로 채우지 말 것)
- shift() 사용 시 반드시 과거 방향인지 주석으로 명시
- for 루프 금지, 벡터 연산만

rsi_divergence 상세:
- 최근 60봉 내 가격 스윙 고점 2개 vs 해당 시점 RSI 비교
- 가격 고점 상승 + RSI 고점 하락 = bearish divergence (-1)
- 반대는 +1, 없으면 0

tests/test_indicators.py:
- 알려진 값으로 검증. 최소 RSI, MACD, ATR 3개는 수기 계산값과 대조
- look-ahead 테스트: 데이터를 마지막 20봉 잘라내고 계산했을 때,
  겹치는 구간의 지표값이 전체 데이터로 계산한 값과 동일해야 함 ← 이 테스트가 핵심

완료 조건: pytest 전부 통과 + look-ahead 테스트 통과
```

---

### P3 — 국면 판별 + 스코어링 (핵심)

```
engine/regime.py 와 engine/scorer.py 를 구현해줘. 이 프로젝트의 핵심이다.

regime.py:
- classify_regime(df) -> pd.Series  (각 시점의 국면 라벨)
  - TRENDING_UP:   ADX >= 25 and +DI > -DI
  - TRENDING_DOWN: ADX >= 25 and -DI > +DI
  - RANGING:       ADX < 20 and 볼린저밴드폭이 최근 120일 하위 30%
  - VOLATILE:      ATR이 20일 평균의 1.5배 이상
  - 우선순위: VOLATILE > TRENDING > RANGING > (기본값 RANGING)

scorer.py:
- score(df, date) -> ScoreCard (dataclass)
  필드: total(0~100), regime, category_scores(dict), evidences(list[str]),
        invalidation(str), stop_loss(float), position_size_pct(float)

- 5개 카테고리 원점수(0~1)를 각각 계산한 뒤, config의 국면별 가중치를 곱해 합산
- category_scores에는 획득점수/배점 을 함께 담을 것

- evidences: 점수에 기여한 근거를 사람이 읽을 문장으로 (최대 5개)
  예: "MACD 골든크로스 3일차", "외국인 5일 연속 순매수"

- invalidation: 이 신호가 무효화되는 조건을 구체적 가격으로
  예: "종가 기준 20일선(68,400원) 이탈 시"

- stop_loss: 종가 - 2*ATR(14)
- position_size_pct: (2% 리스크) / ((종가-손절가)/종가), 최대 20%로 캡

- grade(total) -> str: 80+ 강한합의 / 65~79 관심 / 45~64 진입금지 / 44- 회피
  ※ 45~64 구간의 라벨은 반드시 "진입 금지"여야 한다. "약한 매수" 등으로 바꾸지 말 것.

테스트:
- 인위적으로 만든 상승추세 데이터에서 TRENDING_UP + 추세 카테고리 고득점 확인
- 횡보 데이터에서 모멘텀 가중치가 40으로 적용되는지 확인
- 모든 카테고리 만점 시 total이 정확히 100인지 확인

완료 조건: 삼성전자 최근 날짜 스코어카드를 콘솔에 PRD 형식대로 출력
```

---

### P4 — 백테스트 (스크리너보다 먼저!)

```
backtest/ 를 구현해줘. 이게 없으면 뒤의 모든 결과를 믿을 수 없다.

engine.py:
- run(tickers, start, end, entry_score=80, exit_score=45) -> BacktestResult
- 진입: 스코어가 entry_score 이상으로 상향 돌파한 다음 날 시가 매수
- 청산 조건 (셋 중 먼저 오는 것):
  1) 손절: 저가가 stop_loss 이탈 → 해당 가격에 체결
  2) 스코어 exit_score 미만 → 다음 날 시가
  3) 트레일링: 수익 1R 도달 후 3×ATR 추적
- 포지션 사이징은 scorer의 position_size_pct 사용
- 동시 보유 최대 8종목, 현금 부족 시 스코어 높은 순 우선
- 거래비용 반드시 반영

metrics.py:
- CAGR, MDD, Sharpe, Sortino, 승률, 손익비(Payoff), Profit Factor, 거래횟수
- walk_forward(train_years=3, test_years=1, rolls=5) -> 구간별 성과 표

절대 규칙:
- 매 시점 스코어 계산에 해당 시점까지의 데이터만 슬라이싱해서 넘길 것
- 체결가는 신호 발생 봉의 종가가 아니라 **다음 봉 시가**
- 이 두 가지를 위반하지 않았음을 검증하는 테스트를 반드시 작성할 것

완료 조건:
- KOSPI 50종목 2015~2025 백테스트 실행
- 성과 지표 8종 + 워크포워드 5구간 표 출력
- 벤치마크(KOSPI) 대비 누적수익 비교 출력
```

---

### P5 — 스크리너 + Streamlit UI

```
app.py에 Streamlit 앱을 만들어줘. 화면은 3개 탭.

탭1 "종목 분석":
- 티커 입력
- Plotly 캔들차트 + 지표 오버레이 토글 (이동평균/볼린저/일목/거래량)
- 하단에 MACD, RSI 서브플롯
- 우측에 스코어카드: 총점, 국면, 카테고리별 막대, 근거 리스트, 반증조건, 손절가, 권장 비중

탭2 "스크리너":
- 유니버스 전체 스캔 → 점수순 테이블
- 필터: 최소점수 슬라이더, 국면 멀티셀렉트, 시가총액 하한
- 행 클릭 시 탭1로 이동
- 스캔 결과는 st.cache_data로 캐싱 (TTL 1시간)

탭3 "백테스트":
- 기간·진입점수·청산점수 입력
- 실행 버튼 → 누적수익 곡선(벤치마크 동시 표시), MDD 구간 음영
- 성과 지표 카드 8개
- 거래 내역 테이블 (진입일, 청산일, 수익률, 청산사유)

성능:
- 200종목 스캔이 30초를 넘으면 멀티프로세싱 적용
- 진행률 표시 필수

UI는 기본 테마 그대로. 꾸미는 데 시간 쓰지 말 것.

완료 조건: streamlit run app.py 로 3개 탭 모두 동작
```

---

### P6 — 검증 및 과최적화 점검

```
전략이 우연이 아닌지 검증하는 스크립트를 작성해줘.

scripts/validate.py:

1. 파라미터 민감도 테스트
   - RSI기간, MACD단기, ADX임계값, entry_score, ATR배수 각각을 ±20% 변경
   - 각 변경마다 백테스트 실행 → CAGR/MDD/Sharpe 변동폭 표로 출력
   - 판정: CAGR 변동폭이 30%를 넘으면 "과최적화 의심" 경고

2. 국면 가중치 무력화 테스트
   - 모든 국면에 동일 가중치(균등)를 적용한 버전과 성과 비교
   - 국면별 가중치가 실제로 성과를 개선하는지 확인
   - 개선이 없다면 이 프로젝트의 핵심 가설이 틀린 것이므로 명확히 보고할 것

3. 랜덤 진입 대조군
   - 같은 횟수/보유기간으로 무작위 진입한 1000회 시뮬레이션
   - 전략 성과가 랜덤 분포의 상위 5% 안에 드는지 (p-value)

4. 신호별 실제 승률 테이블
   - 각 근거(골든크로스, 외국인 순매수, RSI 다이버전스 등)별로
     발생 후 20일 수익률의 평균/중앙값/승률을 집계
   - 승률 50% 미만인 신호는 "제거 후보"로 표시

결과를 reports/validation_YYYYMMDD.md 로 저장.

솔직하게 보고할 것. 성과가 나쁘면 나쁘다고 쓸 것.
```

---

## 3. 재사용 프롬프트

### 에러 해결

```
아래 에러가 발생했어. 전체 스택트레이스야.

[스택트레이스 전문 붙여넣기]

추측으로 고치지 말고:
1. 원인이 되는 코드 라인을 먼저 지목
2. 왜 그 라인에서 발생하는지 설명
3. 그 다음에 수정

같은 종류의 에러가 다른 곳에도 있는지 함께 확인해줘.
```

### look-ahead 감사

```
전체 코드베이스에서 look-ahead bias 가능성이 있는 부분을 감사해줘.

점검 항목:
- shift(-n) 또는 음수 shift 사용처
- rolling().mean() 이후 center=True 사용
- 전체 데이터로 정규화/스케일링 후 시계열 분할
- 백테스트 루프에서 미래 데이터가 포함된 df를 넘기는 곳
- 지표 계산에 마지막 행 종가를 쓰는데 체결도 같은 날 종가인 경우

발견된 각 건에 대해 파일:라인, 위험도(높음/중간/낮음), 수정안을 표로 정리해줘.
```

### 리팩터링

```
[파일명]이 너무 길어졌어. 아래 원칙으로 리팩터링해줘.

- 함수 하나는 한 화면(50줄) 이내
- 중복 로직 추출
- 하드코딩 숫자는 config.py로 이동
- 동작은 절대 바꾸지 말 것

리팩터링 전후로 기존 테스트가 모두 통과하는지 확인하고,
바뀐 게 없음을 백테스트 결과 동일성으로 증명해줘.
```

### 코드 리뷰

```
방금 작성한 코드를 냉정하게 리뷰해줘.
칭찬은 빼고, 다음만 지적해:

1. CLAUDE.md의 절대 규칙 5개 중 위반한 것
2. 조용히 실패할 수 있는 지점 (에러 없이 잘못된 값을 반환하는 곳)
3. 성능 병목
4. 테스트가 없는 핵심 로직

각 지적에 심각도(치명/주의/사소)를 붙여줘.
```

---

## 4. 자주 빠지는 함정

| 증상 | 원인 | 대응 프롬프트 |
|---|---|---|
| 백테스트 수익률이 비현실적으로 높음 | look-ahead bias | "look-ahead 감사" 프롬프트 실행 |
| AI가 지표를 계속 추가함 | 완료 조건 부재 | "요청하지 않은 지표를 추가하지 마. 지금 있는 것만으로 완료 조건을 충족시켜줘" |
| 코드가 점점 커지며 통제 불능 | 한 번에 너무 많이 요청 | 롤백 후 프롬프트를 3개로 쪼갤 것 |
| "완성했습니다"인데 안 돌아감 | 실행 검증 누락 | "실제로 실행해서 출력을 보여줘. 안 되면 될 때까지 고쳐줘" |
| 결측 데이터를 0으로 채움 | 기본 행동 | CLAUDE.md 규칙 재강조 + 해당 라인 지적 |

---

## 5. 진행 체크리스트

```
[ ] P0  스캐폴딩          → import 성공
[ ] P1  데이터            → 10년치 수집, 재조회 1초 이내
[ ] P2  지표              → pytest 통과 + look-ahead 테스트 통과
[ ] P3  스코어링          → 스코어카드 콘솔 출력
[ ] P4  백테스트          → 성과 지표 8종 + 워크포워드
[ ] P5  UI                → 3개 탭 동작
[ ] P6  검증              → 민감도/대조군 리포트

각 단계 완료 시: git commit -m "P{n}: {요약}"
```
