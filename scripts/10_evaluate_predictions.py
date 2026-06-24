import csv
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATH = PROJECT_ROOT / "data/processed/prediction_history.json"
COMPLETED_MATCHES_PATH = PROJECT_ROOT / "data/processed/completed_matches.csv"
EVALUATION_PATH = PROJECT_ROOT / "data/processed/prediction_evaluation.json"
SUMMARY_PATH = PROJECT_ROOT / "data/processed/prediction_evaluation_summary.json"

RESULTS = ("home_win", "draw", "away_win")

COLUMN_CANDIDATES = {
    "date": ("date", "match_date"),
    "home_team": ("home_team", "homeTeam", "team_a", "teamA"),
    "away_team": ("away_team", "awayTeam", "team_b", "teamB"),
    "home_score": ("home_score", "homeScore", "score_home"),
    "away_score": ("away_score", "awayScore", "score_away"),
    "tournament": ("tournament", "competition"),
}


def load_json(path, default):
    if not path.exists():
        return default

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def get_value(item, *keys):
    for key in keys:
        if key in item and item[key] not in (None, ""):
            return item[key]
    return None


def resolve_columns(fieldnames):
    field_set = set(fieldnames or [])
    resolved = {}

    for name, candidates in COLUMN_CANDIDATES.items():
        for candidate in candidates:
            if candidate in field_set:
                resolved[name] = candidate
                break

    missing = [
        name
        for name in ("date", "home_team", "away_team", "home_score", "away_score")
        if name not in resolved
    ]
    if missing:
        raise ValueError(
            "completed_matches.csv 필수 컬럼을 찾을 수 없습니다: "
            + ", ".join(missing)
        )

    return resolved


def parse_score(value):
    if value in (None, ""):
        return None

    try:
        score = float(value)
    except ValueError:
        return None

    if score.is_integer():
        return int(score)
    return score


def match_key(date, home_team, away_team):
    return (str(date or ""), str(home_team or ""), str(away_team or ""))


def normalize_result(value):
    if value is None:
        return None

    normalized = str(value).strip().lower()
    mapping = {
        "home_win": "home_win",
        "home win": "home_win",
        "home": "home_win",
        "a_win": "home_win",
        "home_win": "home_win",
        "draw": "draw",
        "away_win": "away_win",
        "away win": "away_win",
        "away": "away_win",
        "b_win": "away_win",
    }
    return mapping.get(normalized, normalized)


def actual_result(home_score, away_score):
    if home_score > away_score:
        return "home_win"
    if home_score == away_score:
        return "draw"
    return "away_win"


def build_empty_summary():
    return {
        "evaluated_matches": 0,
        "correct_predictions": 0,
        "accuracy": 0.0,
        "wrong_predictions": 0,
        "latest_evaluated_matches": [],
        "result_breakdown": {
            result: {
                "predictions": 0,
                "correct_predictions": 0,
                "accuracy": 0.0,
            }
            for result in RESULTS
        },
        "last_updated": datetime.now(timezone.utc).isoformat().replace(
            "+00:00",
            "Z",
        ),
    }


def build_summary(evaluations):
    summary = build_empty_summary()
    evaluated_matches = len(evaluations)
    correct_predictions = sum(1 for item in evaluations if item["correct"])

    summary.update(
        {
            "evaluated_matches": evaluated_matches,
            "correct_predictions": correct_predictions,
            "accuracy": (
                round(correct_predictions / evaluated_matches, 4)
                if evaluated_matches
                else 0.0
            ),
            "wrong_predictions": evaluated_matches - correct_predictions,
            "latest_evaluated_matches": sorted(
                evaluations,
                key=lambda item: (
                    item["date"],
                    item["home_team"],
                    item["away_team"],
                ),
                reverse=True,
            )[:10],
        }
    )

    for result in RESULTS:
        result_items = [
            item
            for item in evaluations
            if item["predicted_result"] == result
        ]
        correct_count = sum(1 for item in result_items if item["correct"])
        prediction_count = len(result_items)
        summary["result_breakdown"][result] = {
            "predictions": prediction_count,
            "correct_predictions": correct_count,
            "accuracy": (
                round(correct_count / prediction_count, 4)
                if prediction_count
                else 0.0
            ),
        }

    return summary


def load_completed_matches():
    if not COMPLETED_MATCHES_PATH.exists():
        return {}

    with COMPLETED_MATCHES_PATH.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        columns = resolve_columns(reader.fieldnames)
        matches = {}

        for row in reader:
            home_score = parse_score(row.get(columns["home_score"]))
            away_score = parse_score(row.get(columns["away_score"]))

            if home_score is None or away_score is None:
                continue

            date = row.get(columns["date"])
            home_team = row.get(columns["home_team"])
            away_team = row.get(columns["away_team"])
            key = match_key(date, home_team, away_team)

            matches[key] = {
                "date": date,
                "tournament": row.get(columns.get("tournament", ""), ""),
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "actual_result": actual_result(home_score, away_score),
            }

    return matches


def evaluate_predictions(history, completed_matches):
    evaluations = []

    for prediction in history:
        date = get_value(prediction, "date")
        home_team = get_value(prediction, "home_team", "homeTeam")
        away_team = get_value(prediction, "away_team", "awayTeam")
        match = completed_matches.get(match_key(date, home_team, away_team))

        if not match:
            continue

        predicted_result = normalize_result(
            get_value(prediction, "predicted_result", "predictedResult")
        )
        if predicted_result not in RESULTS:
            continue

        evaluations.append(
            {
                "date": match["date"],
                "tournament": (
                    get_value(prediction, "tournament", "competition")
                    or match["tournament"]
                ),
                "home_team": match["home_team"],
                "away_team": match["away_team"],
                "home_score": match["home_score"],
                "away_score": match["away_score"],
                "predicted_result": predicted_result,
                "actual_result": match["actual_result"],
                "correct": predicted_result == match["actual_result"],
                "predicted_probability": float(
                    get_value(prediction, "predicted_probability") or 0.0
                ),
                "home_win_probability": float(
                    get_value(prediction, "home_win_probability", "homeWinProb")
                    or 0.0
                ),
                "draw_probability": float(
                    get_value(prediction, "draw_probability", "drawProb") or 0.0
                ),
                "away_win_probability": float(
                    get_value(prediction, "away_win_probability", "awayWinProb")
                    or 0.0
                ),
            }
        )

    return sorted(
        evaluations,
        key=lambda item: (item["date"], item["home_team"], item["away_team"]),
    )


def main():
    history = load_json(HISTORY_PATH, [])
    if not isinstance(history, list):
        raise ValueError(f"예측 이력 파일 형식이 리스트가 아닙니다: {HISTORY_PATH}")

    completed_matches = load_completed_matches()
    evaluations = evaluate_predictions(history, completed_matches)
    summary = build_summary(evaluations)

    write_json(EVALUATION_PATH, evaluations)
    write_json(SUMMARY_PATH, summary)

    print(f"예측 검증 상세 저장 완료: {EVALUATION_PATH}")
    print(f"예측 검증 요약 저장 완료: {SUMMARY_PATH}")
    print(f"검증 완료 경기 수: {summary['evaluated_matches']}")
    print(f"누적 정확도: {summary['accuracy']}")


if __name__ == "__main__":
    main()
