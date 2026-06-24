import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = PROJECT_ROOT / "data/processed/fixture_predictions.json"
HISTORY_PATH = PROJECT_ROOT / "data/processed/prediction_history.json"


RESULT_KEYS = {
    "home_win": ("home_win_probability", "homeWinProb"),
    "draw": ("draw_probability", "drawProb"),
    "away_win": ("away_win_probability", "awayWinProb"),
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


def get_probability(item, snake_key, camel_key):
    value = get_value(item, snake_key, camel_key)
    return float(value or 0.0)


def prediction_key(item):
    return (
        str(get_value(item, "date") or ""),
        str(get_value(item, "home_team", "homeTeam") or ""),
        str(get_value(item, "away_team", "awayTeam") or ""),
    )


def get_predicted_result(probabilities):
    return max(probabilities.items(), key=lambda item: item[1])[0]


def archive_prediction(item, archived_at):
    probabilities = {
        result: get_probability(item, *keys)
        for result, keys in RESULT_KEYS.items()
    }
    predicted_result = get_predicted_result(probabilities)

    return {
        "date": get_value(item, "date"),
        "tournament": get_value(item, "tournament", "competition"),
        "home_team": get_value(item, "home_team", "homeTeam"),
        "away_team": get_value(item, "away_team", "awayTeam"),
        "predicted_result": predicted_result,
        "home_win_probability": round(probabilities["home_win"], 4),
        "draw_probability": round(probabilities["draw"], 4),
        "away_win_probability": round(probabilities["away_win"], 4),
        "predicted_probability": round(probabilities[predicted_result], 4),
        "archived_at": archived_at,
    }


def main():
    fixture_predictions = load_json(PREDICTIONS_PATH, [])
    prediction_history = load_json(HISTORY_PATH, [])

    if not isinstance(fixture_predictions, list):
        raise ValueError(f"예측 파일 형식이 리스트가 아닙니다: {PREDICTIONS_PATH}")
    if not isinstance(prediction_history, list):
        raise ValueError(f"예측 이력 파일 형식이 리스트가 아닙니다: {HISTORY_PATH}")

    existing_keys = {prediction_key(item) for item in prediction_history}
    archived_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    added_count = 0

    for item in fixture_predictions:
        key = prediction_key(item)
        if not all(key) or key in existing_keys:
            continue

        prediction_history.append(archive_prediction(item, archived_at))
        existing_keys.add(key)
        added_count += 1

    write_json(HISTORY_PATH, prediction_history)

    print(f"예측 이력 저장 완료: {HISTORY_PATH}")
    print(f"기존 이력 수: {len(prediction_history) - added_count}")
    print(f"신규 보관 수: {added_count}")
    print(f"전체 이력 수: {len(prediction_history)}")


if __name__ == "__main__":
    main()
