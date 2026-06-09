import json
from pathlib import Path

from utils.team_aliases import normalize_team_name
from utils.worldcup_groups import WORLD_CUP_FIXTURES, WORLD_CUP_GROUPS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "data/processed/worldcup_completed_results.json"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/worldcup_group_standings.json"


def init_team(display_name):
    return {
        "rank": 0,
        "team": display_name,
        "played": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "goalsFor": 0,
        "goalsAgainst": 0,
        "goalDifference": 0,
        "points": 0,
    }


def apply_result(home, away, home_score, away_score):
    home["played"] += 1
    away["played"] += 1
    home["goalsFor"] += home_score
    home["goalsAgainst"] += away_score
    away["goalsFor"] += away_score
    away["goalsAgainst"] += home_score

    if home_score > away_score:
        home["wins"] += 1
        away["losses"] += 1
        home["points"] += 3
    elif home_score < away_score:
        away["wins"] += 1
        home["losses"] += 1
        away["points"] += 3
    else:
        home["draws"] += 1
        away["draws"] += 1
        home["points"] += 1
        away["points"] += 1

    home["goalDifference"] = home["goalsFor"] - home["goalsAgainst"]
    away["goalDifference"] = away["goalsFor"] - away["goalsAgainst"]


def create_group_standings(completed_results):
    groups = {}
    fixture_groups = {
        (
            date,
            frozenset(
                (
                    normalize_team_name(home_team),
                    normalize_team_name(away_team),
                )
            ),
        ): group
        for date, group, home_team, away_team in WORLD_CUP_FIXTURES
    }

    for group, team_names in WORLD_CUP_GROUPS.items():
        group_teams = {}

        for display_name in team_names:
            normalized_name = normalize_team_name(display_name)
            group_teams[normalized_name] = init_team(display_name)

        groups[group] = group_teams

    for match in completed_results:
        home_name = match["normalizedHomeTeam"]
        away_name = match["normalizedAwayTeam"]
        fixture_key = (
            match["date"],
            frozenset((home_name, away_name)),
        )
        group = fixture_groups.get(fixture_key)

        if group is None:
            continue

        apply_result(
            groups[group][home_name],
            groups[group][away_name],
            int(match["homeScore"]),
            int(match["awayScore"]),
        )

    output = []

    for group, teams_by_name in groups.items():
        teams = sorted(
            teams_by_name.values(),
            key=lambda team: (
                -team["points"],
                -team["goalDifference"],
                -team["goalsFor"],
                team["team"],
            ),
        )

        for rank, team in enumerate(teams, start=1):
            team["rank"] = rank

        output.append({"group": group, "teams": teams})

    return output


def main():
    with RESULTS_PATH.open("r", encoding="utf-8") as file:
        completed_results = json.load(file)

    standings = create_group_standings(completed_results)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(standings, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"2026 월드컵 조별 순위 생성: {OUTPUT_PATH}")
    print(f"조 수: {len(standings)}")
    print(f"반영된 완료 경기 수: {len(completed_results)}")


if __name__ == "__main__":
    main()
