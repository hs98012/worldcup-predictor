import json
from collections import defaultdict, deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_PATH = Path("data/raw/results.csv")
RECENT_PATH = Path("data/manual/recent_matches.csv")
MATCHES_PATH = Path("data/processed/matches.json")

MODEL_PATH = Path("models/sklearn_match_result_model.joblib")
PREDICTIONS_PATH = Path("data/processed/predictions_sklearn.json")
METRICS_PATH = Path("data/processed/model_metrics.json")

INITIAL_ELO = 1500
BASE_K = 30

FEATURE_COLUMNS = [
    "team_a_elo",
    "team_b_elo",
    "elo_diff",
    "team_a_recent_points5",
    "team_b_recent_points5",
    "recent_points_diff",
    "team_a_recent_goal_diff5",
    "team_b_recent_goal_diff5",
    "recent_goal_diff_gap",
    "team_a_avg_goals_for5",
    "team_b_avg_goals_for5",
    "team_a_avg_goals_against5",
    "team_b_avg_goals_against5",
    "is_neutral",
    "team_a_home_advantage",
]

HOST_TEAMS = {"Mexico", "Canada", "United States"}


def load_matches():
    base_df = pd.read_csv(BASE_PATH)

    if RECENT_PATH.exists():
        recent_df = pd.read_csv(RECENT_PATH)
    else:
        recent_df = pd.DataFrame(columns=base_df.columns)

    required_columns = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "city",
        "country",
        "neutral",
    ]

    base_df = base_df[required_columns]
    recent_df = recent_df[required_columns]

    df = pd.concat([base_df, recent_df], ignore_index=True)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_score", "away_score"])
    df = df.sort_values("date").reset_index(drop=True)

    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # 너무 오래된 경기는 현대 축구와 차이가 크므로 학습은 2000년 이후만 사용
    df = df[df["date"].dt.year >= 2000].reset_index(drop=True)

    return df


def to_bool(value):
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.lower() == "true"

    return bool(value)


def get_actual_label(score_a, score_b):
    if score_a > score_b:
        return "A_WIN"
    elif score_a == score_b:
        return "DRAW"
    else:
        return "B_WIN"


def get_actual_score(score_a, score_b):
    if score_a > score_b:
        return 1.0
    elif score_a == score_b:
        return 0.5
    else:
        return 0.0


def get_expected_score(team_elo, opponent_elo):
    return 1 / (1 + 10 ** ((opponent_elo - team_elo) / 400))


def get_time_weight(match_date, current_year=2026):
    years_ago = current_year - match_date.year

    if years_ago <= 0:
        return 1.0
    elif years_ago == 1:
        return 0.9
    elif years_ago == 2:
        return 0.75
    elif years_ago <= 5:
        return 0.55
    elif years_ago <= 10:
        return 0.3
    else:
        return 0.1


def summarize_recent_form(history):
    if not history:
        return {
            "points": 0,
            "goal_diff": 0,
            "avg_goals_for": 0,
            "avg_goals_against": 0,
        }

    points = sum(item["points"] for item in history)
    goals_for = sum(item["goals_for"] for item in history)
    goals_against = sum(item["goals_against"] for item in history)
    match_count = len(history)

    return {
        "points": points,
        "goal_diff": goals_for - goals_against,
        "avg_goals_for": goals_for / match_count,
        "avg_goals_against": goals_against / match_count,
    }


def build_feature_row(
    team_a,
    team_b,
    elo,
    recent_history,
    is_neutral,
    team_a_home_advantage,
):
    team_a_elo = elo[team_a]
    team_b_elo = elo[team_b]

    form_a = summarize_recent_form(recent_history[team_a])
    form_b = summarize_recent_form(recent_history[team_b])

    return {
        "team_a_elo": team_a_elo,
        "team_b_elo": team_b_elo,
        "elo_diff": team_a_elo - team_b_elo,
        "team_a_recent_points5": form_a["points"],
        "team_b_recent_points5": form_b["points"],
        "recent_points_diff": form_a["points"] - form_b["points"],
        "team_a_recent_goal_diff5": form_a["goal_diff"],
        "team_b_recent_goal_diff5": form_b["goal_diff"],
        "recent_goal_diff_gap": form_a["goal_diff"] - form_b["goal_diff"],
        "team_a_avg_goals_for5": form_a["avg_goals_for"],
        "team_b_avg_goals_for5": form_b["avg_goals_for"],
        "team_a_avg_goals_against5": form_a["avg_goals_against"],
        "team_b_avg_goals_against5": form_b["avg_goals_against"],
        "is_neutral": int(is_neutral),
        "team_a_home_advantage": int(team_a_home_advantage),
    }


def update_elo(elo, team_a, team_b, score_a, score_b, match_date):
    team_a_elo = elo[team_a]
    team_b_elo = elo[team_b]

    expected_a = get_expected_score(team_a_elo, team_b_elo)
    actual_a = get_actual_score(score_a, score_b)

    k = BASE_K * get_time_weight(match_date)

    elo[team_a] = team_a_elo + k * (actual_a - expected_a)
    elo[team_b] = team_b_elo + k * ((1 - actual_a) - (1 - expected_a))


def update_recent_history(recent_history, team_a, team_b, score_a, score_b):
    if score_a > score_b:
        points_a, points_b = 3, 0
    elif score_a == score_b:
        points_a, points_b = 1, 1
    else:
        points_a, points_b = 0, 3

    recent_history[team_a].append(
        {
            "points": points_a,
            "goals_for": score_a,
            "goals_against": score_b,
        }
    )

    recent_history[team_b].append(
        {
            "points": points_b,
            "goals_for": score_b,
            "goals_against": score_a,
        }
    )


def build_training_dataset(df):
    elo = defaultdict(lambda: INITIAL_ELO)
    recent_history = defaultdict(lambda: deque(maxlen=5))

    rows = []
    labels = []
    dates = []

    for _, row in df.iterrows():
        team_a = row["home_team"]
        team_b = row["away_team"]
        score_a = row["home_score"]
        score_b = row["away_score"]
        match_date = row["date"]
        is_neutral = to_bool(row["neutral"])

        # 경기 전 기준 피처 생성
        feature_row = build_feature_row(
            team_a=team_a,
            team_b=team_b,
            elo=elo,
            recent_history=recent_history,
            is_neutral=is_neutral,
            team_a_home_advantage=not is_neutral,
        )

        rows.append(feature_row)
        labels.append(get_actual_label(score_a, score_b))
        dates.append(match_date)

        # 경기 후 상태 업데이트
        update_elo(elo, team_a, team_b, score_a, score_b, match_date)
        update_recent_history(recent_history, team_a, team_b, score_a, score_b)

    X = pd.DataFrame(rows)[FEATURE_COLUMNS]
    y = np.array(labels)
    dates = pd.Series(dates)

    return X, y, dates, elo, recent_history


def train_model(X, y, dates):
    # 시간 순서를 지켜서 마지막 20%를 테스트로 사용
    split_index = int(len(X) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    metrics = {
        "trainSize": int(len(X_train)),
        "testSize": int(len(X_test)),
        "testStartDate": str(dates.iloc[split_index].date()),
        "testEndDate": str(dates.iloc[-1].date()),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "logLoss": round(log_loss(y_test, y_proba, labels=model.classes_), 4),
        "classes": model.classes_.tolist(),
        "classificationReport": classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
    }

    return model, metrics


def predict_worldcup_matches(model, elo, recent_history):
    with open(MATCHES_PATH, "r", encoding="utf-8") as f:
        matches = json.load(f)

    predictions = []

    for match in matches:
        team_a = match["teamA"]
        team_b = match["teamB"]

        if team_a not in elo:
            print(f"스킵: {team_a} Elo 정보 없음")
            continue

        if team_b not in elo:
            print(f"스킵: {team_b} Elo 정보 없음")
            continue

        is_neutral = True
        team_a_home_advantage = team_a in HOST_TEAMS

        feature_row = build_feature_row(
            team_a=team_a,
            team_b=team_b,
            elo=elo,
            recent_history=recent_history,
            is_neutral=is_neutral,
            team_a_home_advantage=team_a_home_advantage,
        )

        X_match = pd.DataFrame([feature_row])[FEATURE_COLUMNS]

        proba = model.predict_proba(X_match)[0]
        class_to_prob = dict(zip(model.classes_, proba))

        team_a_win = float(class_to_prob.get("A_WIN", 0))
        draw = float(class_to_prob.get("DRAW", 0))
        team_b_win = float(class_to_prob.get("B_WIN", 0))

        predictions.append(
            {
                "matchId": match["matchId"],
                "stage": match["stage"],
                "group": match["group"],
                "date": match["date"],
                "displayTeamA": match.get("displayTeamA", team_a),
                "displayTeamB": match.get("displayTeamB", team_b),
                "teamA": team_a,
                "teamB": team_b,
                "teamAWinProb": round(team_a_win, 4),
                "drawProb": round(draw, 4),
                "teamBWinProb": round(team_b_win, 4),
                "predictedResult": max(
                    [
                        ("A_WIN", team_a_win),
                        ("DRAW", draw),
                        ("B_WIN", team_b_win),
                    ],
                    key=lambda x: x[1],
                )[0],
                "features": {
                    key: round(value, 4) if isinstance(value, float) else value
                    for key, value in feature_row.items()
                },
            }
        )

    return predictions


def main():
    df = load_matches()

    print("학습 데이터 생성 중...")
    X, y, dates, elo, recent_history = build_training_dataset(df)

    print("모델 학습 중...")
    model, metrics = train_model(X, y, dates)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    predictions = predict_worldcup_matches(model, elo, recent_history)

    with open(PREDICTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print()
    print("학습 완료")
    print(f"모델 저장: {MODEL_PATH}")
    print(f"성능 저장: {METRICS_PATH}")
    print(f"월드컵 예측 저장: {PREDICTIONS_PATH}")
    print()
    print("모델 성능")
    print(f"- Accuracy: {metrics['accuracy']}")
    print(f"- Log Loss: {metrics['logLoss']}")
    print(f"- 테스트 기간: {metrics['testStartDate']} ~ {metrics['testEndDate']}")
    print()
    print("월드컵 예측 샘플")
    for prediction in predictions[:10]:
        print(
            f"{prediction['displayTeamA']} vs {prediction['displayTeamB']} | "
            f"{prediction['teamAWinProb'] * 100:.1f}% / "
            f"{prediction['drawProb'] * 100:.1f}% / "
            f"{prediction['teamBWinProb'] * 100:.1f}%"
        )


if __name__ == "__main__":
    main()
