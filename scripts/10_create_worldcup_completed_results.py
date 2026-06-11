import json
from pathlib import Path

import pandas as pd

from utils.team_aliases import normalize_team_name


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPLETED_MATCHES_PATH = PROJECT_ROOT / "data/processed/completed_matches.csv"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/worldcup_completed_results.json"
WORLD_CUP_YEAR = 2026


def get_result(home_score, away_score):
    if home_score > away_score:
        return "HOME_WIN"
    if home_score < away_score:
        return "AWAY_WIN"
    return "DRAW"


def create_completed_results(matches):
    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    matches["home_score"] = pd.to_numeric(
        matches["home_score"],
        errors="coerce",
    )
    matches["away_score"] = pd.to_numeric(
        matches["away_score"],
        errors="coerce",
    )

    completed = matches[
        matches["tournament"].eq("FIFA World Cup")
        & matches["date"].dt.year.eq(WORLD_CUP_YEAR)
        & matches["home_score"].notna()
        & matches["away_score"].notna()
    ].copy()
    completed = completed.sort_values("date")

    results = []

    for _, match in completed.iterrows():
        home_team = match["home_team"]
        away_team = match["away_team"]
        home_score = int(match["home_score"])
        away_score = int(match["away_score"])

        results.append(
            {
                "date": match["date"].date().isoformat(),
                "tournament": match["tournament"],
                "homeTeam": home_team,
                "awayTeam": away_team,
                "normalizedHomeTeam": normalize_team_name(home_team),
                "normalizedAwayTeam": normalize_team_name(away_team),
                "homeScore": home_score,
                "awayScore": away_score,
                "result": get_result(home_score, away_score),
            }
        )

    return results


def main():
    matches = pd.read_csv(COMPLETED_MATCHES_PATH)
    results = create_completed_results(matches)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"완료된 2026 월드컵 경기 결과 생성: {OUTPUT_PATH}")
    print(f"완료 경기 수: {len(results)}")


if __name__ == "__main__":
    main()
