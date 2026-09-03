# Confluence

다중 이론 합의(Confluence) 기반 기술적 분석 엔진. 개별 기술적 지표가 서로 충돌할 때
시장 국면(Regime)에 따라 가중치를 다르게 적용해 0~100점의 근거(evidence) 기반 스코어를
산출하고, 그 규칙이 과거 데이터에서 실제로 유효했는지 백테스트·통계 검증으로 확인한다.

- 상세 기획: [`technical-analysis-program-prd.md`](technical-analysis-program-prd.md)
- 개발 규칙(반드시 준수): [`CLAUDE.md`](CLAUDE.md) — look-ahead 금지, 손절 없는 신호 금지,
  predict/forecast 표현 금지, 파라미터 5개 초과 금지, 수정주가 사용
- 배포된 앱: https://awesome72-stock.streamlit.app/

## 현재 상태 (2026-09-03)

PRD 로드맵 P0~P6까지 전부 구현됨. 단, **검증 결과 실전 배포에는 아직 미달**이다 —
아래 "검증 결과" 절 참고. 스코어링 로직 자체를 신뢰하기 전에 반드시 읽을 것.

| 단계 | 내용 | 상태 |
|---|---|---|
| P0 | 프로젝트 스캐폴딩, config.py | 완료 |
| P1 | 데이터 파이프라인 (pykrx 수집 + SQLite 캐시) | 완료 |
| P2 | 지표 엔진 (추세/모멘텀/변동성/거래량/수급) | 완료 |
| P3 | 국면 판별 + 가중 스코어링 엔진 | 완료 |
| P4 | 백테스트 엔진 | 완료 |
| P5 | Streamlit UI (종목분석/스크리너/백테스트) | 완료 |
| P6 | 파라미터 민감도·대조군·통계 검증 | 완료 — 결과는 혼재 |

## 기능

### 종목 분석
- 티커 입력 → 회사명 자동 표시, 캔들차트(이동평균/볼린저밴드/일목균형표/거래량 오버레이) +
  MACD + RSI
- 모든 차트 지표와 스코어 카테고리에 설명(툴팁/expander) 제공
- 국면별 가중 스코어카드: 총점, 등급(강한 합의/관심/진입 금지/회피), 카테고리별 배점,
  근거(evidence) 목록, 반증 조건, ATR 기반 손절가, 권장 포지션 비중
- **종합 분석**: 위 스코어카드를 사람이 읽는 요약 문장으로 재구성(가장 강한/약한 근거,
  손절가 대비 리스크%, 권장 비중)하고 최근 120거래일 점수 추이 차트를 함께 보여준다.
  예측이나 매매 권유가 아니라는 면책 문구를 항상 동반한다.

### 스크리너
- KOSPI 시가총액 상위 N종목을 스캔해 점수·등급·국면 순으로 정렬
- 결과에 종목명 병기, 점수/등급/국면 설명 expander 제공
- 선택한 종목을 '종목 분석' 탭으로 전달(Streamlit에 탭 전환 API가 없어 `session_state`로 우회)

### 백테스트
- 기간·유니버스·진입/청산 점수를 지정해 실행, 자본금 곡선(MDD 구간 표시), 8개 성과지표,
  거래 내역(종목명 포함) 표 제공

## 검증 결과 요약 (`reports/validation_20260903.md`)

KOSPI 상위 50종목, 2020~2025 백테스트 기준:

- PRD 최소 성과 기준 7개 중 **3개만 충족** (Sharpe/MDD/승률/Profit Factor 미달)
- `entry_score`, `ADX 임계값` 파라미터에서 **과최적화 의심**(±20% 변경 시 CAGR 변동폭 >30%)
- 실제 전략 대비 **랜덤 진입 대조군과 통계적으로 유의한 차이를 확보하지 못함**(p=0.14)
- 국면별 가중치는 균등가중치 대조군보다 확실히 우수(CAGR +21.4% vs +8.1%) — 이 부분만 근거 있음
- '외국인 5일 연속 순매수' 신호는 승률 49.9%로 제거 후보

**결론: 국면 적응형 가중치라는 아이디어 자체는 균등가중치보다 낫다는 근거가 있지만, 현재
파라미터/신호 조합을 그대로 실거래에 쓰기엔 이르다.** 다음 개선 방향은 아래 "다음에 개선할 것"
참고.

## 디렉토리 구조

```
confluence/
├── config.py            # 모든 파라미터·가중치 상수 (하드코딩 금지 원칙)
├── data/
│   ├── loader.py         # pykrx 수집 (OHLCV/수급/유니버스/지수/종목명)
│   └── store.py          # SQLite 캐시 (data/confluence.db, git 추적 안 함)
├── indicators/
│   ├── trend.py          # SMA, MACD, 일목균형표
│   ├── momentum.py       # RSI, 스토캐스틱, 다이버전스
│   ├── volatility.py     # 볼린저밴드, ATR
│   ├── volume.py         # OBV, 매물대(POC)
│   └── flow.py           # 외국인/기관 수급 연속일수
├── engine/
│   ├── regime.py         # 국면 판별 (추세상승/추세하락/횡보/변동성확대)
│   └── scorer.py         # 국면별 가중 스코어링 + 근거 생성
├── backtest/
│   ├── engine.py          # 신호 기반 시뮬레이션 (익일 시가 체결)
│   └── metrics.py         # CAGR/MDD/Sharpe/승률/PF 등
├── app.py                # Streamlit 진입점 (3탭)
└── tests/                 # pytest 38개

scripts/                   # 헤드리스 데모/검증 스크립트 (python -m scripts.<name>로 실행)
├── fetch_demo.py
├── scorecard_demo.py
├── backtest_demo.py
└── validate.py            # P6 검증 리포트 생성 → reports/validation_YYYYMMDD.md
reports/                    # validate.py 산출물
```

## 로컬 실행

```bash
pip install -r requirements.txt
export KRX_ID=발급받은ID
export KRX_PW=발급받은비밀번호
streamlit run confluence/app.py
```

- `KRX_ID`/`KRX_PW`: https://data.krx.co.kr 무료 가입 후 발급. 개별 종목 시세 조회는 인증
  없이도 되지만, **스크리너(유니버스/시가총액)·벤치마크(KOSPI 지수)·수급 데이터는 인증 필수**다.
  없으면 임의 값으로 채우지 않고 명시적 에러/경고를 낸다(CLAUDE.md 규칙).
- 테스트: `pytest`
- P6 검증 리포트 재생성: `python -m scripts.validate`
- 프로젝트 스크립트는 절대 임포트(`from confluence import config`)를 쓰므로 반드시
  `python -m scripts.<name>` 형태로 실행할 것(`python scripts/<name>.py`는 임포트 에러).

## 배포 (Streamlit Community Cloud)

계정 `awesome72`, 저장소 `awesome72/Stock`, 브랜치 `master`, 진입점 `confluence/app.py`,
Python 3.11 고정. `master`에 푸시하면 자동 재배포된다. KRX_ID/KRX_PW는 앱 Secrets에 등록되어
있음.

`requirements.txt`의 `pandas`/`pykrx` 버전은 **의도적으로 고정**되어 있다 —
`pandas<3.0` (pykrx 1.2.x가 요구) / `pykrx>=1.2.8`. 이걸 풀어두면 Cloud의 `uv` 리졸버가
`pandas`를 최신으로 올렸다가 `pykrx`를 그 제약이 없는 아주 오래된 버전(1.0.51)으로
후퇴시켜 KRX API 파싱이 전부 깨지는 문제가 실제로 있었다(고침: 커밋 `959df1a`). 버전 제약을
건드릴 때는 이 배경을 먼저 확인할 것.

당일 데이터 관련: OHLCV는 장 마감 후 바로 나오지만 수급(외국인/기관) 데이터는 하루 늦게
집계되는 경우가 있다. 이 경우 종목분석 탭이 "'flow' 카테고리 점수를 계산할 데이터가
부족합니다"를 표시하는데, 이는 버그가 아니라 CLAUDE.md의 "데이터 없을 때 임의값 채우지
않기" 규칙에 따른 의도된 동작이다.

## 다음에 개선할 것

우선순위 순:

1. **전략 자체의 실효성 재검토** — 랜덤 대조군 대비 유의성을 못 얻은 원인 분석. 카테고리
   가중치(`config.REGIME_CATEGORY_WEIGHTS`)나 신호 조합을 바꿔가며 P6 검증(`scripts/validate.py`)을
   반복해서, 과최적화 없이 유의성을 확보하는 조합을 찾을 것. `entry_score`/`ADX 임계값`
   과최적화 의심 항목부터.
2. **'외국인 5일 연속 순매수' 신호 제거 또는 재정의** — 승률 49.9%로 무의미. `flow.py`/
   `scorer._flow_score` 수정.
3. **MDD 개선** — 현재 -43.6%로 PRD 기준(-25%) 대비 크게 초과. 청산 로직(`backtest/engine.py`)의
   트레일링 스탑(`config.TRAILING_STOP_*`) 활용 여부 점검.
4. **종목분석 탭의 "당일 스코어 계산 불가" 완화** — 스크리너는 이미 `last_valid_index()`로
   가장 최근 유효한 날짜를 쓰는데, 종목분석 탭은 `df.index[-1]`(오늘)을 고집한다. 스크리너
   방식으로 통일하면 수급 데이터가 하루 늦게 들어와도 전날 기준으로 바로 보여줄 수 있다
   (`app.py`의 `render_stock_analysis_tab`).
5. **상대강도(RS) 카테고리 고도화** — 현재는 벤치마크 대비 초과수익률 근사치. PRD가 원하는
   진짜 오닐식 RS Rating(전 유니버스 백분위)은 스크리너 스캔 결과를 재활용하면 가능.
6. 파라미터는 CLAUDE.md 규칙상 전략당 최대 5개이므로, 새 파라미터를 추가하려면 기존 것을
   빼거나 통합해야 한다.

세션 간 이어가기용 상세 배경(원인 조사 과정, 이미 시도했다가 폐기한 가설 등)은 이 저장소를
다루는 Claude Code 세션의 프로젝트 메모리에 별도로 기록되어 있다.
