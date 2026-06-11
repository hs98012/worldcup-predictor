from pathlib import Path

import pandas as pd

from utils.team_aliases import normalize_team_name


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = PROJECT_ROOT / "data/raw/results.csv"
COMPLETED_PATH = PROJECT_ROOT / "data/processed/completed_matches.csv"
UPCOMING_PATH = PROJECT_ROOT / "data/processed/upcoming_fixtures.csv"
WORLD_CUP_YEAR = 2026


def load_matches():
    df = pd.read_csv(BASE_PATH)
    match_dates = pd.to_datetime(df["date"], errors="coerce")

    completed_matches = df.dropna(
        subset=["home_score", "away_score"]
    ).copy()
    upcoming_fixtures = df[
        df[["home_score", "away_score"]].isna().any(axis=1)
        & df["tournament"].eq("FIFA World Cup")
        & match_dates.dt.year.eq(WORLD_CUP_YEAR)
    ].copy()

    completed_matches["normalized_home_team"] = completed_matches[
        "home_team"
    ].apply(normalize_team_name)
    completed_matches["normalized_away_team"] = completed_matches[
        "away_team"
    ].apply(normalize_team_name)
    upcoming_fixtures["normalized_home_team"] = upcoming_fixtures[
        "home_team"
    ].apply(normalize_team_name)
    upcoming_fixtures["normalized_away_team"] = upcoming_fixtures[
        "away_team"
    ].apply(normalize_team_name)

    COMPLETED_PATH.parent.mkdir(parents=True, exist_ok=True)
    completed_matches.to_csv(COMPLETED_PATH, index=False)
    upcoming_fixtures.to_csv(UPCOMING_PATH, index=False)

    return df, completed_matches, upcoming_fixtures


def main():
    df, completed_matches, upcoming_fixtures = load_matches()

    teams = sorted(
        set(completed_matches["home_team"])
        | set(completed_matches["away_team"])
    )

    print("전체 경기 수:", len(df))
    print("완료 경기 수:", len(completed_matches))
    print("예정 경기 수:", len(upcoming_fixtures))
    print("completed_matches 저장 경로:", COMPLETED_PATH)
    print("upcoming_fixtures 저장 경로:", UPCOMING_PATH)
    print("팀 수:", len(teams))
    print()
    print("최근 완료 경기 5개:")
    print(
        completed_matches.tail(5)[
            [
                "date",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "tournament",
            ]
        ]
    )


if __name__ == "__main__":
    main()
