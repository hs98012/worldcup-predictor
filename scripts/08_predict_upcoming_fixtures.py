import importlib.util
import json
import warnings
from pathlib import Path

import joblib
import pandas as pd

from utils.team_aliases import normalize_team_name


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPCOMING_PATH = PROJECT_ROOT / "data/processed/upcoming_fixtures.csv"
MODEL_PATH = PROJECT_ROOT / "models/sklearn_match_result_model.joblib"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/fixture_predictions.json"
TRAINING_SCRIPT_PATH = PROJECT_ROOT / "scripts/05_train_sklearn_model.py"


def load_training_module():
    spec = importlib.util.spec_from_file_location(
        "train_sklearn_model",
        TRAINING_SCRIPT_PATH,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"학습 스크립트를 불러올 수 없습니다: {TRAINING_SCRIPT_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_model_team_name(normalized_name, original_name, elo):
    candidates = [normalized_name, original_name]

    candidates.extend(
        team_name
        for team_name in elo
        if normalize_team_name(team_name) == normalized_name
    )

    for candidate in candidates:
        if candidate in elo:
            return candidate

    warnings.warn(
        f"{original_name} ({normalized_name})의 학습 이력을 찾지 못해 "
        "초기 Elo와 빈 최근 폼을 사용합니다.",
        stacklevel=2,
    )
    return normalized_name


def get_predicted_result(home_win, draw, away_win):
    return max(
        [
            ("HOME_WIN", home_win),
            ("DRAW", draw),
            ("AWAY_WIN", away_win),
        ],
        key=lambda item: item[1],
    )[0]


def main():
    training = load_training_module()
    fixtures = pd.read_csv(UPCOMING_PATH, dtype={"date": str})
    model = joblib.load(MODEL_PATH)

    completed_matches = training.load_matches()
    _, _, _, elo, recent_history = training.build_training_dataset(
        completed_matches
    )

    predictions = []

    for _, fixture in fixtures.iterrows():
        home_team = fixture["home_team"]
        away_team = fixture["away_team"]
        normalized_home = normalize_team_name(
            fixture.get("normalized_home_team", home_team)
        )
        normalized_away = normalize_team_name(
            fixture.get("normalized_away_team", away_team)
        )

        model_home = resolve_model_team_name(
            normalized_home,
            home_team,
            elo,
        )
        model_away = resolve_model_team_name(
            normalized_away,
            away_team,
            elo,
        )

        feature_row = training.build_feature_row(
            team_a=model_home,
            team_b=model_away,
            elo=elo,
            recent_history=recent_history,
            is_neutral=True,
            team_a_home_advantage=model_home in training.HOST_TEAMS,
        )
        feature_frame = pd.DataFrame(
            [feature_row],
            columns=training.FEATURE_COLUMNS,
        )

        probabilities = model.predict_proba(feature_frame)[0]
        class_to_probability = dict(zip(model.classes_, probabilities))

        home_win = float(class_to_probability.get("A_WIN", 0.0))
        draw = float(class_to_probability.get("DRAW", 0.0))
        away_win = float(class_to_probability.get("B_WIN", 0.0))

        predictions.append(
            {
                "date": fixture["date"],
                "homeTeam": home_team,
                "awayTeam": away_team,
                "normalizedHomeTeam": normalized_home,
                "normalizedAwayTeam": normalized_away,
                "tournament": fixture["tournament"],
                "homeWinProb": round(home_win, 4),
                "drawProb": round(draw, 4),
                "awayWinProb": round(away_win, 4),
                "predictedResult": get_predicted_result(
                    home_win,
                    draw,
                    away_win,
                ),
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(predictions, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"예정 경기 예측 완료: {OUTPUT_PATH}")
    print(f"입력 경기 수: {len(fixtures)}")
    print(f"생성된 예측 수: {len(predictions)}")


if __name__ == "__main__":
    main()
