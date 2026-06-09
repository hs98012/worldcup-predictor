# 2026 World Cup Predictor

국제 축구 경기 결과 데이터를 기반으로 2026 월드컵 경기 결과와 대회 진행
확률을 예측하는 프로젝트입니다.

머신러닝 모델로 경기별 승·무·패 확률을 계산하고, 반복 시뮬레이션을 통해
조별리그 통과 확률과 토너먼트 진출·우승 확률을 산출합니다. 생성된 결과는
Vue 대시보드에서 시각화하며, GitHub Actions를 통해 데이터 갱신 과정을
매일 자동 실행합니다.

- Dashboard: https://hs98012.github.io/worldcup-predictor/
- Prediction task: `HOME_WIN / DRAW / AWAY_WIN` 다중 분류
- Simulation: 조별리그 및 토너먼트 10,000회 반복

## 프로젝트 개요

이 프로젝트는 단일 모델 학습에 그치지 않고 다음 과정을 하나의 흐름으로
연결하는 것을 목표로 했습니다.

1. 국제 축구 원본 데이터 변경 감지
2. 완료 경기와 예정 경기 분리 및 전처리
3. 머신러닝 모델 학습과 남은 경기 확률 예측
4. 실제 월드컵 결과 기반 조별 순위 계산
5. 조별리그와 토너먼트 시뮬레이션
6. Vue 대시보드용 JSON 생성 및 동기화
7. GitHub Actions를 통한 일일 자동 갱신

## 주요 기능

### 데이터 처리

- 원본 `results.csv`의 SHA-256 해시, 행 수, 최신 경기 날짜 변경 감지
- 스코어 존재 여부에 따른 완료 경기와 예정 경기 분리
- 팀 이름 표기 차이를 정규화해 모델과 시뮬레이션에서 일관되게 사용

### 예측 및 시뮬레이션

- scikit-learn `LogisticRegression` 기반 승·무·패 다중 분류
- Elo와 최근 경기력을 활용한 피처 생성
- 예정된 월드컵 경기의 홈 승·무승부·원정 승 확률 생성
- 무승부 예측 보정 로직 적용
- 조별리그 통과 확률 및 토너먼트 단계별 진출 확률 계산
- 10,000회 시뮬레이션 기반 우승 확률 산출

### 실제 경기 반영

- 완료된 2026 월드컵 경기의 날짜, 팀, 스코어, 결과 저장
- 실제 결과를 반영한 조별리그 현재 순위 계산
- 승점, 골득실, 다득점, 팀명 순으로 순위 정렬
- 아직 완료된 경기가 없어도 12개 조와 48개 팀의 초기 순위 생성

### 대시보드

- 완료된 월드컵 경기 결과
- 조별리그 현재 순위
- 남은 경기 승·무·패 예측
- 조별리그 통과 확률
- 토너먼트 진출 및 우승 확률 Top 10
- 대한민국 진출 확률 강조 표시
- 데스크톱과 모바일 반응형 UI

## 기술 스택

| 영역 | 기술 |
| --- | --- |
| Data / ML | Python, pandas, NumPy, scikit-learn, joblib |
| Model | LogisticRegression, Elo rating |
| Frontend | Vue.js, Vite |
| Automation | GitHub Actions |
| Deployment | GitHub Pages |

## 데이터 파이프라인

```text
외부 국제 경기 results.csv
          |
          v
원본 변경 감지
(해시 / 행 수 / 최신 날짜)
          |
          v
완료 경기와 예정 경기 분리
          |
          +-------------------------------+
          |                               |
          v                               v
completed_matches.csv             upcoming_fixtures.csv
          |                               |
          v                               v
Elo 및 최근 경기력 계산             남은 경기 승/무/패 예측
          |                               |
          +---------------+---------------+
                          |
                          v
                 모델 학습 및 확률 보정
                          |
          +---------------+----------------+
          |                                |
          v                                v
실제 경기 결과 및 현재 순위         조별리그 / 토너먼트 시뮬레이션
          |                                |
          +---------------+----------------+
                          |
                          v
               대시보드용 JSON 산출물 생성
                          |
                          v
 data/processed/*.json -> frontend/public/data/*.json
                          |
                          v
               Vue + Vite 대시보드
                          |
                          v
                    GitHub Pages
```

## 프로젝트 구조

```text
worldcup-predictor/
├── data/
│   ├── raw/                    # 원본 국제 경기 데이터
│   └── processed/              # 전처리, 예측, 시뮬레이션 산출물
├── frontend/
│   ├── public/data/            # Vue에서 fetch하는 JSON
│   └── src/                    # Vue 대시보드
├── models/                     # 학습된 모델
├── scripts/
│   ├── utils/                  # 팀명 정규화, 월드컵 조 편성
│   └── 99_run_daily_pipeline.py
├── .github/workflows/
│   ├── daily-pipeline.yml      # 일일 데이터 갱신
│   └── deploy-frontend.yml     # GitHub Pages 배포
└── requirements.txt
```

## 주요 산출물

| 파일 | 설명 |
| --- | --- |
| `completed_matches.csv` | 스코어가 입력된 완료 경기 |
| `upcoming_fixtures.csv` | 아직 스코어가 없는 예정 경기 |
| `fixture_predictions.json` | 남은 경기의 승·무·패 확률 |
| `worldcup_completed_results.json` | 완료된 2026 월드컵 실제 결과 |
| `worldcup_group_standings.json` | 실제 결과 기반 현재 조별 순위 |
| `group_simulation.json` | 팀별 조 순위 및 32강 진출 확률 |
| `tournament_simulation.json` | 토너먼트 단계별 진출 및 우승 확률 |
| `dashboard_summary.json` | 모델과 주요 예측 결과 요약 |

산출물은 `data/processed/`에 생성됩니다. Vue에서 사용하는 JSON은
`scripts/09_sync_frontend_data.py`를 통해 `frontend/public/data/`로
복사됩니다.

## 로컬 실행

### Python 파이프라인

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/99_run_daily_pipeline.py
```

실제 결과와 조별 순위 JSON만 개별 생성할 수도 있습니다.

```bash
python scripts/10_create_worldcup_completed_results.py
python scripts/11_create_group_standings.py
python scripts/09_sync_frontend_data.py
```

### Vue 개발 서버

```bash
cd frontend
npm ci
npm run dev
```

기본 개발 서버 주소는 `http://localhost:5173/worldcup-predictor/`입니다.

### Vue 프로덕션 빌드

```bash
cd frontend
npm run build
```

빌드 결과는 `frontend/dist/`에 생성됩니다.

## 자동 업데이트

`.github/workflows/daily-pipeline.yml`은 매일 `00:00 UTC`
(한국 시간 `09:00`)에 실행되며 수동 실행도 지원합니다.

1. 원본 국제 경기 CSV 다운로드
2. 해시, 행 수, 최신 날짜를 기존 데이터와 비교
3. 변경된 경우 전처리 데이터 갱신
4. Elo 및 머신러닝 모델 재학습
5. 남은 경기 예측과 시뮬레이션 재실행
6. 실제 결과와 현재 조별 순위 JSON 생성
7. `frontend/public/data/`로 대시보드 데이터 동기화
8. 변경된 데이터, 모델, 프론트엔드 JSON 자동 commit/push

`main` 브랜치에 push된 프론트엔드 변경은
`.github/workflows/deploy-frontend.yml`에서 빌드한 뒤 GitHub Pages에
배포됩니다.

## 포트폴리오 관점의 특징

이 프로젝트는 데이터 분석 결과를 노트북이나 단일 스크립트에 머물게 하지
않고, 실제 서비스 형태로 연결한 프로젝트입니다.

- 원본 데이터 변경을 감지하는 재실행 가능한 데이터 파이프라인
- 전처리, 피처 생성, 모델 학습, 확률 보정으로 구성된 예측 과정
- 실제 경기 결과와 확률 시뮬레이션을 함께 제공하는 데이터 구조
- 정적 JSON을 활용한 단순하고 배포 가능한 Vue 대시보드
- 데이터 갱신, 산출물 동기화, 커밋, Pages 배포를 분리한 자동화 구성

이를 통해 데이터 처리, 머신러닝, 프론트엔드 시각화, CI/CD 자동화를 하나의
저장소 안에서 설계하고 운영하는 과정을 구현했습니다.
