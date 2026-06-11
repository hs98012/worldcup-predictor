import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPLETED_PATH = PROJECT_ROOT / "data/processed/completed_matches.csv"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/teams.json"

INITIAL_ELO = 1500
BASE_K = 30
OPPONENT_STRENGTH_MIN = 0.75
OPPONENT_STRENGTH_MAX = 1.25
RECENT_FORM_CONTRIBUTION_CAP = 20.0
SIMULATION_STRENGTH_RANGE = 20.0


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


def get_tournament_k_factor(tournament):
    name = str(tournament).lower()
    if "fifa world cup" in name and "qualification" not in name:
        return 1.15
    if any(
        keyword in name
        for keyword in (
            "uefa euro",
            "copa américa",
            "african cup of nations",
            "afc asian cup",
            "gold cup",
        )
    ) and "qualification" not in name:
        return 0.9
    if "qualification" in name or "nations league" in name:
        return 0.7
    if "friendly" in name:
        return 0.35
    return 0.55


def get_goal_difference_multiplier(home_score, away_score):
    goal_difference = abs(home_score - away_score)
    if goal_difference <= 1:
        return 1.0
    return min(1.5, 1.0 + 0.15 * (goal_difference - 1))


def summarize_recent_form(history, n):
    recent = list(history)[-n:]
    match_count = len(recent)
    points = sum(item["points"] for item in recent)
    goals_for = sum(item["goals_for"] for item in recent)
    goals_against = sum(item["goals_against"] for item in recent)
    adjusted_points = sum(item["adjusted_points"] for item in recent)
    adjusted_goal_diff = sum(
        item["adjusted_goal_diff"] for item in recent
    )
    avg_opponent_elo = (
        sum(item["opponent_elo"] for item in recent) / match_count
        if match_count
        else INITIAL_ELO
    )

    return {
        f"recentForm{n}": "-".join(item["result"] for item in recent),
        f"recentPoints{n}": points,
        f"recentWinRate{n}": round(
            sum(item["result"] == "W" for item in recent) / match_count
            if match_count
            else 0,
            4,
        ),
        f"recentGoalDiff{n}": int(goals_for - goals_against),
        f"recentAvgGoalsFor{n}": round(
            goals_for / match_count if match_count else 0,
            2,
        ),
        f"recentAvgGoalsAgainst{n}": round(
            goals_against / match_count if match_count else 0,
            2,
        ),
        f"adjustedPoints{n}": round(adjusted_points, 4),
        f"adjustedGoalDiff{n}": round(adjusted_goal_diff, 4),
        f"avgOpponentElo{n}": round(avg_opponent_elo, 2),
    }


def build_history_item(
    points,
    goals_for,
    goals_against,
    team_elo,
    opponent_elo,
):
    strength_factor = float(
        np.clip(
            opponent_elo / team_elo,
            OPPONENT_STRENGTH_MIN,
            OPPONENT_STRENGTH_MAX,
        )
    )
    result = "W" if points == 3 else "D" if points == 1 else "L"
    return {
        "result": result,
        "points": points,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "opponent_elo": opponent_elo,
        "adjusted_points": points * strength_factor,
        "adjusted_goal_diff": (
            goals_for - goals_against
        ) * strength_factor,
    }


def calculate_simulation_strength(
    team_elo,
    adjusted_points5,
    adjusted_goal_diff5,
    avg_opponent_elo5,
):
    recent_form_contribution = (
        (adjusted_points5 - 7.5) * 4
        + float(np.clip(adjusted_goal_diff5, -8, 8)) * 1.5
    )
    capped_contribution = float(
        np.clip(
            recent_form_contribution,
            -RECENT_FORM_CONTRIBUTION_CAP,
            RECENT_FORM_CONTRIBUTION_CAP,
        )
    )
    schedule_strength_penalty = 0.0

    if capped_contribution > 0 and avg_opponent_elo5 < 1550:
        limited_contribution = capped_contribution * 0.1
    elif capped_contribution > 0 and avg_opponent_elo5 < 1600:
        limited_contribution = capped_contribution * 0.5
    else:
        limited_contribution = capped_contribution

    schedule_strength_penalty = limited_contribution - capped_contribution
    simulation_strength = float(
        np.clip(
            team_elo + limited_contribution,
            team_elo - SIMULATION_STRENGTH_RANGE,
            team_elo + SIMULATION_STRENGTH_RANGE,
        )
    )
    return {
        "recentFormContribution": round(recent_form_contribution, 4),
        "cappedRecentFormContribution": round(limited_contribution, 4),
        "scheduleStrengthPenalty": round(schedule_strength_penalty, 4),
        "simulationStrength": round(simulation_strength, 2),
    }


def main():
    df = load_matches()

    elo = {}
    match_counts = {}
    recent_history = defaultdict(lambda: deque(maxlen=10))

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
        if row["home_score"] > row["away_score"]:
            home_points, away_points = 3, 0
        elif row["home_score"] == row["away_score"]:
            home_points, away_points = 1, 1
        else:
            home_points, away_points = 0, 3

        recent_history[home_team].append(
            build_history_item(
                home_points,
                row["home_score"],
                row["away_score"],
                home_elo,
                away_elo,
            )
        )
        recent_history[away_team].append(
            build_history_item(
                away_points,
                row["away_score"],
                row["home_score"],
                away_elo,
                home_elo,
            )
        )

        time_weight = get_time_weight(row["date"])
        tournament_factor = get_tournament_k_factor(row["tournament"])
        goal_difference_factor = get_goal_difference_multiplier(
            row["home_score"],
            row["away_score"],
        )
        k = (
            BASE_K
            * time_weight
            * tournament_factor
            * goal_difference_factor
        )

        elo[home_team] = home_elo + k * (actual_home - expected_home)
        elo[away_team] = away_elo + k * ((1 - actual_home) - (1 - expected_home))

        match_counts[home_team] += 1
        match_counts[away_team] += 1

    teams = []

    for team_name, team_elo in elo.items():
        recent_form5 = summarize_recent_form(
            recent_history[team_name],
            n=5,
        )
        recent_form10 = summarize_recent_form(
            recent_history[team_name],
            n=10,
        )
        strength_summary = calculate_simulation_strength(
            team_elo,
            recent_form5["adjustedPoints5"],
            recent_form5["adjustedGoalDiff5"],
            recent_form5["avgOpponentElo5"],
        )

        teams.append(
            {
                "team": team_name,
                "elo": round(team_elo, 2),
                "matches": match_counts[team_name],
                "recentForm": recent_form5["recentForm5"],
                **strength_summary,
                **recent_form5,
                **recent_form10,
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
