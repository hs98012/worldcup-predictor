import json
from pathlib import Path

import pandas as pd

from utils.team_aliases import normalize_team_name
from utils.worldcup_groups import WORLD_CUP_FIXTURES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEAMS_PATH = PROJECT_ROOT / "data/processed/teams.json"
UPCOMING_PATH = PROJECT_ROOT / "data/processed/upcoming_fixtures.csv"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/matches.json"


def load_team_names():
    with open(TEAMS_PATH, "r", encoding="utf-8") as f:
        teams = json.load(f)

    return {team["team"] for team in teams}


def fixture_key(date, team_a, team_b):
    return (
        str(date),
        frozenset(
            (
                normalize_team_name(team_a),
                normalize_team_name(team_b),
            )
        ),
    )


def build_group_map():
    return {
        fixture_key(date, team_a, team_b): group
        for date, group, team_a, team_b in WORLD_CUP_FIXTURES
    }


def resolve_team_name(display_name, normalized_name, team_names):
    if normalized_name in team_names:
        return normalized_name

    if display_name in team_names:
        return display_name

    return normalized_name


def main():
    team_names = load_team_names()
    upcoming_fixtures = pd.read_csv(UPCOMING_PATH, dtype={"date": str})
    group_map = build_group_map()

    matches = []
    missing = []

    for idx, fixture in upcoming_fixtures.reset_index(drop=True).iterrows():
        date = fixture["date"]
        display_a = fixture["home_team"]
        display_b = fixture["away_team"]
        normalized_a = normalize_team_name(display_a)
        normalized_b = normalize_team_name(display_b)
        team_a = resolve_team_name(display_a, normalized_a, team_names)
        team_b = resolve_team_name(display_b, normalized_b, team_names)
        group = group_map.get(fixture_key(date, normalized_a, normalized_b))

        if team_a not in team_names:
            missing.append(display_a)

        if team_b not in team_names:
            missing.append(display_b)

        matches.append(
            {
                "matchId": idx + 1,
                "stage": "GROUP",
                "group": group,
                "date": date,
                "displayTeamA": display_a,
                "displayTeamB": display_b,
                "teamA": team_a,
                "teamB": team_b,
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(matches, f, ensure_ascii=False, indent=2)

    print(f"생성 완료: {OUTPUT_PATH}")
    print(f"경기 수: {len(matches)}")

    missing_groups = [
        match
        for match in matches
        if match["group"] is None
    ]

    if missing:
        print()
        print("teams.json에서 못 찾은 팀명:")
        for name in sorted(set(missing)):
            print("-", name)
    else:
        print("모든 팀명이 teams.json과 매칭되었습니다.")

    if missing_groups:
        print()
        print("조 정보를 찾지 못한 예정 경기:")
        for match in missing_groups:
            print(
                f"- {match['date']} "
                f"{match['displayTeamA']} vs {match['displayTeamB']}"
            )


if __name__ == "__main__":
    main()
