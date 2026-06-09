import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPLETED_PATH = PROJECT_ROOT / "data/processed/completed_matches.csv"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/teams.json"

INITIAL_ELO = 1500
BASE_K = 30


def load_matches():
    df = pd.read_csv(COMPLETED_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_score", "away_score"])
    df = df.sort_values("date").reset_index(drop=True)

    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    return df


def get_actual_score(home_score, away_score):
    if home_score > away_score:
        return 1.0
    elif home_score == away_score:
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


def calculate_recent_form(team_matches, team_name, n=5):
    recent = team_matches.tail(n)

    form = []
    points = 0
    goals_for = 0
    goals_against = 0

    for _, row in recent.iterrows():
        is_home = row["home_team"] == team_name

        if is_home:
            gf = row["home_score"]
            ga = row["away_score"]
        else:
            gf = row["away_score"]
            ga = row["home_score"]

        goals_for += gf
        goals_against += ga

        if gf > ga:
            form.append("W")
            points += 3
        elif gf == ga:
            form.append("D")
            points += 1
        else:
            form.append("L")

    match_count = len(recent)

    if match_count == 0:
        return {
            "recentForm": "",
            "recentPoints5": 0,
            "recentGoalDiff5": 0,
            "recentAvgGoalsFor5": 0,
            "recentAvgGoalsAgainst5": 0,
        }

    return {
        "recentForm": "-".join(form),
        "recentPoints5": points,
        "recentGoalDiff5": int(goals_for - goals_against),
        "recentAvgGoalsFor5": round(goals_for / match_count, 2),
        "recentAvgGoalsAgainst5": round(goals_against / match_count, 2),
    }


def main():
    df = load_matches()

    elo = {}
    match_counts = {}

    for _, row in df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]

        if home_team not in elo:
            elo[home_team] = INITIAL_ELO
            match_counts[home_team] = 0

        if away_team not in elo:
            elo[away_team] = INITIAL_ELO
            match_counts[away_team] = 0

        home_elo = elo[home_team]
        away_elo = elo[away_team]

        expected_home = get_expected_score(home_elo, away_elo)
        actual_home = get_actual_score(row["home_score"], row["away_score"])

        time_weight = get_time_weight(row["date"])
        k = BASE_K * time_weight

        elo[home_team] = home_elo + k * (actual_home - expected_home)
        elo[away_team] = away_elo + k * ((1 - actual_home) - (1 - expected_home))

        match_counts[home_team] += 1
        match_counts[away_team] += 1

    teams = []

    for team_name, team_elo in elo.items():
        team_matches = df[
            (df["home_team"] == team_name) | (df["away_team"] == team_name)
        ]

        recent_form = calculate_recent_form(team_matches, team_name, n=5)

        teams.append(
            {
                "team": team_name,
                "elo": round(team_elo, 2),
                "matches": match_counts[team_name],
                **recent_form,
            }
        )

    teams = sorted(teams, key=lambda x: x["elo"], reverse=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(teams, f, ensure_ascii=False, indent=2)

    print(f"저장 완료: {OUTPUT_PATH}")
    print()
    print("Elo 상위 20개 팀")
    for idx, team in enumerate(teams[:20], start=1):
        print(
            f"{idx}. {team['team']} | Elo: {team['elo']} | "
            f"최근 5경기: {team['recentForm']} | "
            f"최근 승점: {team['recentPoints5']}"
        )

    korea = next((team for team in teams if team["team"] == "South Korea"), None)
    if korea:
        print()
        print("대한민국")
        print(korea)


if __name__ == "__main__":
    main()
