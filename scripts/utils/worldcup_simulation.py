from datetime import date
from pathlib import Path

import pandas as pd

from utils.team_aliases import normalize_team_name
from utils.worldcup_groups import WORLD_CUP_FIXTURES, WORLD_CUP_GROUPS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPLETED_RESULTS_PATH = (
    PROJECT_ROOT / "data/processed/worldcup_completed_results.json"
)
COMPLETED_MATCHES_PATH = PROJECT_ROOT / "data/processed/completed_matches.csv"
WORLD_CUP_YEAR = 2026


def init_team(display_name, team_name):
    return {
        "displayTeam": display_name,
        "team": team_name,
        "played": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "goalsFor": 0,
        "goalsAgainst": 0,
        "goalDiff": 0,
        "points": 0,
        "tieBreaker": 0,
    }


def apply_match_result(team_a_stats, team_b_stats, score_a, score_b):
    team_a_stats["played"] += 1
    team_b_stats["played"] += 1

    team_a_stats["goalsFor"] += score_a
    team_a_stats["goalsAgainst"] += score_b

    team_b_stats["goalsFor"] += score_b
    team_b_stats["goalsAgainst"] += score_a

    if score_a > score_b:
        team_a_stats["wins"] += 1
        team_b_stats["losses"] += 1
        team_a_stats["points"] += 3
    elif score_a < score_b:
        team_b_stats["wins"] += 1
        team_a_stats["losses"] += 1
        team_b_stats["points"] += 3
    else:
        team_a_stats["draws"] += 1
        team_b_stats["draws"] += 1
        team_a_stats["points"] += 1
        team_b_stats["points"] += 1

    team_a_stats["goalDiff"] = (
        team_a_stats["goalsFor"] - team_a_stats["goalsAgainst"]
    )
    team_b_stats["goalDiff"] = (
        team_b_stats["goalsFor"] - team_b_stats["goalsAgainst"]
    )


def fixture_key(date_value, team_a, team_b):
    return (
        str(date_value),
        frozenset(
            (
                normalize_team_name(team_a),
                normalize_team_name(team_b),
            )
        ),
    )


def fixture_group_map():
    return {
        fixture_key(match_date, home_team, away_team): group
        for match_date, group, home_team, away_team in WORLD_CUP_FIXTURES
    }


def completed_fixture_keys(completed_results):
    return {
        fixture_key(
            match["date"],
            match["normalizedHomeTeam"],
            match["normalizedAwayTeam"],
        )
        for match in completed_results
    }


def is_2026_world_cup_match(match):
    try:
        match_year = date.fromisoformat(str(match["date"])).year
    except (KeyError, TypeError, ValueError):
        return False

    return (
        match.get("tournament") == "FIFA World Cup"
        and match_year == WORLD_CUP_YEAR
    )


def load_completed_results():
    if COMPLETED_RESULTS_PATH.exists():
        import json

        with COMPLETED_RESULTS_PATH.open("r", encoding="utf-8") as file:
            return [
                match
                for match in json.load(file)
                if is_2026_world_cup_match(match)
            ]

    if not COMPLETED_MATCHES_PATH.exists():
        return []

    matches = pd.read_csv(COMPLETED_MATCHES_PATH)
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

    results = []
    for _, match in completed.iterrows():
        home_score = int(match["home_score"])
        away_score = int(match["away_score"])
        if home_score > away_score:
            result = "HOME_WIN"
        elif home_score < away_score:
            result = "AWAY_WIN"
        else:
            result = "DRAW"

        results.append(
            {
                "date": match["date"].date().isoformat(),
                "tournament": match["tournament"],
                "homeTeam": match["home_team"],
                "awayTeam": match["away_team"],
                "normalizedHomeTeam": normalize_team_name(
                    match["home_team"]
                ),
                "normalizedAwayTeam": normalize_team_name(
                    match["away_team"]
                ),
                "homeScore": home_score,
                "awayScore": away_score,
                "result": result,
            }
        )

    return results


def resolve_team_name(display_name, normalized_name, team_names):
    if team_names and normalized_name in team_names:
        return normalized_name
    if team_names and display_name in team_names:
        return display_name
    return normalized_name


def initialize_group_teams(team_names=None):
    groups = {}
    for group, group_team_names in WORLD_CUP_GROUPS.items():
        teams = {}
        for display_name in group_team_names:
            team_name = normalize_team_name(display_name)
            team_key = resolve_team_name(display_name, team_name, team_names)
            teams[team_key] = init_team(display_name, team_key)
        groups[group] = teams
    return groups


def build_initial_group_state(completed_results, team_names=None):
    groups = initialize_group_teams(team_names)
    groups_by_fixture = fixture_group_map()
    applied_results = 0

    for match in completed_results:
        key = fixture_key(
            match["date"],
            match["normalizedHomeTeam"],
            match["normalizedAwayTeam"],
        )
        group = groups_by_fixture.get(key)
        if group is None:
            continue

        home_name = resolve_team_name(
            match["homeTeam"],
            normalize_team_name(match["normalizedHomeTeam"]),
            team_names,
        )
        away_name = resolve_team_name(
            match["awayTeam"],
            normalize_team_name(match["normalizedAwayTeam"]),
            team_names,
        )
        if home_name not in groups[group] or away_name not in groups[group]:
            continue

        apply_match_result(
            groups[group][home_name],
            groups[group][away_name],
            int(match["homeScore"]),
            int(match["awayScore"]),
        )
        applied_results += 1

    return groups, applied_results


def is_completed_prediction(prediction, completed_keys):
    return (
        fixture_key(
            prediction["date"],
            prediction["teamA"],
            prediction["teamB"],
        )
        in completed_keys
    )


def is_valid_group_match(match, groups):
    group = match.get("group")
    if not group or group not in groups:
        return False

    team_a = match.get("teamA")
    team_b = match.get("teamB")
    if not team_a or not team_b:
        return False

    return team_a in groups[group] and team_b in groups[group]


def excluded_group_match_label(match):
    team_a = match.get("displayTeamA") or match.get("teamA") or "?"
    team_b = match.get("displayTeamB") or match.get("teamB") or "?"
    return f"- {team_a} vs {team_b} | group={match.get('group')!r}"
