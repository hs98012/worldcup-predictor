# 2026 World Cup Predictor

국제 축구 경기 데이터를 기반으로 2026 월드컵 경기 결과와 토너먼트 진출 확률을 예측하는 프로젝트다.

과거 국가대표 경기 데이터를 이용해 경기별 승·무·패 확률을 계산하고, 반복 시뮬레이션으로 조별리그 통과 확률과 토너먼트 진출·우승 확률을 산출한다. 데이터 갱신부터 예측 결과 생성, Vue 대시보드 반영까지 GitHub Actions로 자동화했다.

- Dashboard: https://hs98012.github.io/worldcup-predictor/
- Prediction: `HOME_WIN / DRAW / AWAY_WIN`
- Simulation: 10,000회 반복

## 주요 기능

- 국제 경기 데이터 전처리
- Elo Rating 및 최근 경기력 기반 피처 생성
- Logistic Regression 기반 승·무·패 확률 예측
- 확률 보정 및 모델 성능 비교
- 조별리그 및 토너먼트 시뮬레이션
- 실제 경기 결과 기반 예측 검증
- Vue 대시보드 시각화
- GitHub Actions 기반 데이터·예측 결과 자동 갱신

## 기술 스택

| 영역 | 기술 |
| --- | --- |
| Data / ML | Python, pandas, NumPy, scikit-learn |
| Model | Logistic Regression, Elo Rating |
| Frontend | Vue.js, Vite |
| Automation | GitHub Actions |
| Deployment | GitHub Pages |

## 데이터 처리 흐름

```text
국제 경기 데이터
      |
      v
완료 경기 / 예정 경기 분리
      |
      v
Elo 및 최근 경기력 계산
      |
      v
모델 학습 및 승·무·패 확률 예측
      |
      +--------------------+
      |                    |
      v                    v
실제 경기 결과 반영     10,000회 시뮬레이션
      |                    |
      +----------+---------+
                 |
                 v
          대시보드용 JSON 생성
                 |
                 v
             Vue 화면 반영
```

## 모델 구성

국가대표 경기 데이터를 바탕으로 다음 정보를 주요 피처로 사용했다.

- Elo Rating
- 최근 5·10경기 경기력
- 평균 득점·실점
- 최근 골득실
- 상대 팀 전력을 반영한 최근 경기력

랜덤 분할 대신 시간 순서 기준으로 오래된 경기 80%를 학습 데이터로 사용하고 최근 20%를 평가 데이터로 사용했다.

모델 후보는 Accuracy뿐 아니라 Log Loss도 함께 비교해 확률 예측 품질을 확인했다.

## 최근 경기력 보정

단순 최근 승률만 사용할 경우 약한 상대를 상대로 한 연승이 과대평가될 수 있다.

이를 줄이기 위해 최근 경기 당시 상대 팀의 Elo를 함께 반영하고, recent form이 전체 전력에 미치는 영향을 제한했다.

```text
장기 전력
Elo Rating

        +

최근 경기력
상대 Elo를 반영한 최근 성적

        ↓

최종 경기 전력
```

최근 경기력 보정값에는 상한을 두어 단기간 결과가 장기 Elo보다 지나치게 큰 영향을 주지 않도록 했다.

## 무승부 예측 보정

초기 모델에서는 무승부 recall이 낮게 나타났다.

전력이 비슷한 경기에서 Elo 차이, 최근 경기력 차이, 평균 득점·실점 차이가 작은 경우를 대상으로 draw probability 보정을 실험했다.

보정은 다음 조건을 만족하는 경우에만 적용했다.

- DRAW recall 개선
- Accuracy 하락 0.01 이하
- Log Loss 증가 0.02 이하

선택된 보정 결과는 별도 JSON으로 저장해 변경 전후를 확인할 수 있도록 했다.

## 2026 월드컵 선수단 전력 반영

과거 경기 결과만으로는 대회 시점의 현재 선수단 수준을 충분히 반영하기 어렵다고 판단했다.

이를 보완하기 위해 선수단 시장가치, 주요 리그 소속 선수 수, FIFA 랭킹 등을 이용한 `team_strength_score`를 2026 월드컵 예측에 추가했다.

이 값은 과거 경기 모델의 학습 피처로 사용하지 않고, 모델이 계산한 경기 확률에 대한 대회 전용 후처리 보정값으로만 사용했다.

이를 통해 미래 시점의 선수단 정보가 과거 학습 데이터에 포함되는 데이터 누수를 피하면서 현재 전력을 반영하도록 했다.

## 자동화

GitHub Actions에서 매일 데이터 변경 여부를 확인한다.

변경이 있으면 다음 과정을 순서대로 실행한다.

```text
데이터 다운로드
      ↓
변경 여부 확인
      ↓
전처리
      ↓
모델 학습
      ↓
경기 예측
      ↓
시뮬레이션
      ↓
대시보드 JSON 생성
      ↓
GitHub Pages 반영
```

예측 결과는 단순히 최신 값으로 덮어쓰지 않고 예측 당시의 확률을 `prediction_history.json`에 저장한다.

이후 실제 경기 결과가 입력되면 과거 예측값과 비교해 누적 정확도와 최근 검증 결과를 확인할 수 있다.

## 주요 산출물

| 파일 | 설명 |
| --- | --- |
| `fixture_predictions.json` | 예정 경기 승·무·패 확률 |
| `prediction_history.json` | 예측 당시 확률 기록 |
| `prediction_evaluation.json` | 실제 결과와 예측 비교 |
| `model_metrics.json` | 모델 성능 및 보정 결과 |
| `draw_calibration.json` | 무승부 보정 결과 |
| `group_simulation.json` | 조별리그 결과 시뮬레이션 |
| `tournament_simulation.json` | 토너먼트 진출·우승 확률 |
| `dashboard_summary.json` | 화면 표시용 주요 결과 |

## 프로젝트 구조

```text
worldcup-predictor/
├── data/
│   ├── raw/
│   └── processed/
├── frontend/
│   ├── public/data/
│   └── src/
├── models/
├── scripts/
│   ├── utils/
│   └── 99_run_daily_pipeline.py
├── .github/workflows/
│   ├── daily-pipeline.yml
│   └── deploy-frontend.yml
└── requirements.txt
```

## 실행

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/99_run_daily_pipeline.py
```

Vue 개발 서버는 다음과 같이 실행한다.

```bash
cd frontend
npm ci
npm run dev
```

## 현재 상태

데이터 갱신, 모델 학습, 경기별 확률 예측, 조별리그·토너먼트 시뮬레이션, 대시보드 반영까지 하나의 파이프라인으로 연결했다.

현재는 실제 경기 결과가 추가될 때마다 기존 예측과 비교할 수 있도록 예측 이력과 검증 데이터를 함께 관리하고 있다.
