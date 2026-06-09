# World Cup Predictor

2026 월드컵 승부 예측 및 조별리그/토너먼트 시뮬레이터 프로젝트입니다.

## 1. 프로젝트 개요

이 프로젝트는 국가대표 경기 데이터를 기반으로 각 팀의 전력을 수치화하고, 머신러닝 모델을 활용해 경기 결과를 예측한 뒤 조별리그와 토너먼트 진행 확률을 시뮬레이션하는 프로젝트입니다.

단순히 특정 경기의 승패만 예측하는 것이 아니라, 조별리그 순위와 토너먼트 진출 확률까지 계산하여 월드컵 전체 흐름을 예측하는 것을 목표로 합니다.

## 2. 개발 목적

- 국가대표 경기 데이터를 활용한 데이터 분석 경험 정리
- scikit-learn 기반 머신러닝 분류 모델 학습
- 경기 결과 예측 로직 구현
- 조별리그 및 토너먼트 시뮬레이션 구현
- Vue 기반 대시보드 시각화로 확장 예정

## 3. 사용 기술

### Data Analysis / Machine Learning

- Python
- pandas
- numpy
- scikit-learn
- LogisticRegression

### Frontend

- Vue.js

### Output

- JSON 기반 예측 결과 저장
- 대시보드 연동용 데이터 생성 예정

## 4. 프로젝트 구조

worldcup-predictor/
├── data/
│   ├── raw/             # 원본 경기 데이터
│   └── processed/       # 전처리 데이터 및 예측 결과
├── models/              # 학습 모델 저장
├── scripts/             # 데이터 로딩, 학습, 예측, 시뮬레이션 스크립트
├── frontend/            # Vue 기반 대시보드
├── requirements.txt
└── README.md

## 5. 현재 구현 내용

### 데이터 처리

- 국제 경기 데이터 로딩
- 전체 경기 수 및 기간 확인
- 참가 국가 목록 생성
- 분석 및 예측에 필요한 데이터 전처리

### 머신러닝 예측

- 승/무/패 다중분류 모델 학습
- scikit-learn LogisticRegression 기반 베이스라인 모델 구현
- 단일 경기 결과 예측
- 예측 확률 보정 로직 적용

### 시뮬레이션

- 조별리그 경기 결과 예측
- 조별 순위 계산
- 토너먼트 진출 확률 계산
- 10,000회 반복 시뮬레이션 수행

## 6. 모델 성능

현재 베이스라인 모델 성능은 다음과 같습니다.

| Metric | Score |
| --- | ---: |
| Accuracy | 0.5633 |
| Log Loss | 0.9624 |

## 7. 주요 결과 예시

대한민국의 토너먼트 시뮬레이션 결과는 다음과 같습니다.

| Stage | Probability |
| --- | ---: |
| 32강 진출 | 85.85% |
| 16강 진출 | 42.14% |
| 8강 진출 | 12.77% |
| 4강 진출 | 5.56% |
| 결승 진출 | 1.32% |
| 우승 | 0.40% |

## 8. 실행 방법

### Python 가상환경 생성

    python -m venv .venv
    source .venv/bin/activate

### 패키지 설치

    pip install -r requirements.txt

### 스크립트 실행 예시

    python scripts/01_load_matches.py
    python scripts/05_train_sklearn_model.py
    python scripts/03_predict_match.py

### 일일 데이터 업데이트 파이프라인

GitHub 원본 `results.csv`를 로컬 파일과 비교하고, SHA-256 해시, 행 수,
최신 경기 날짜 중 하나라도 변경된 경우에만 전체 예측 파이프라인을 실행합니다.

    python scripts/00_update_results_csv.py
    python scripts/99_run_daily_pipeline.py

원본 데이터는 스코어 존재 여부에 따라 다음 두 파일로 분리됩니다.

- `data/processed/completed_matches.csv`: 모델 학습, Elo 계산, 과거 경기 분석
- `data/processed/upcoming_fixtures.csv`: 남은 월드컵 경기 예측 및 일정 표시

분리 결과만 확인하려면 다음 명령을 실행합니다.

    python scripts/01_load_matches.py

실행 결과는 `data/processed/update_status.json`에서 확인할 수 있습니다.
GitHub Actions는 매일 00:00 UTC(한국 시간 09:00)에 자동 실행되며,
Actions 화면에서 수동 실행할 수도 있습니다.

## 9. 주요 산출물

현재 생성된 주요 산출물은 다음과 같습니다.

- data/processed/teams.json
- data/processed/completed_matches.csv
- data/processed/upcoming_fixtures.csv
- data/processed/model_metrics.json
- data/processed/predictions_sklearn.json
- data/processed/predictions_adjusted.json
- data/processed/group_predictions.json
- data/processed/group_simulation.json
- data/processed/tournament_simulation.json

## 10. 향후 개발 계획

- 대시보드용 summary JSON 생성
- Vue 기반 메인 화면 구현
- 국가별 예측 결과 카드 구현
- 조별리그 순위표 시각화
- 토너먼트 브래킷 시각화
- 모델 성능 개선 및 피처 추가

## 11. 프로젝트 진행 단계

현재 프로젝트는 데이터 분석, 모델 학습, 경기 예측, 조별리그 및 토너먼트 시뮬레이션까지 구현된 상태입니다.

이후 단계에서는 생성된 JSON 결과를 Vue 프론트엔드와 연결하여 사용자가 국가별 예측 결과와 토너먼트 진행 확률을 확인할 수 있는 대시보드를 구현할 예정입니다.
