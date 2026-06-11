import json
import random
from collections import defaultdict
from pathlib import Path

import joblib
import pandas as pd

PREDICTIONS_PATH = Path("data/processed/predictions_adjusted.json")
TEAMS_PATH = Path("data/processed/teams.json")
MODEL_PATH = Path("models/sklearn_match_result_model.joblib")
OUTPUT_PATH = Path("data/processed/tournament_simulation.json")

SIMULATION_COUNT = 10000
RANDOM_SEED = 42

HOST_TEAMS = {"Mexico", "Canada", "United States"}
ADVANCE_PROBABILITY_CACHE = {}

FEATURE_COLUMNS = [
    "team_a_elo",
    "team_b_elo",
    "elo_diff",
    "team_a_recent_points5",
    "team_b_recent_points5",
    "recent_points_diff",
    "team_a_win_rate5",
    "team_b_win_rate5",
    "win_rate_diff5",
    "team_a_avg_goals_for5",
    "team_b_avg_goals_for5",
    "avg_goals_for_diff5",
    "team_a_avg_goals_against5",
    "team_b_avg_goals_against5",
    "avg_goals_against_diff5",
    "team_a_recent_goal_diff5",
    "team_b_recent_goal_diff5",
    "recent_goal_diff_gap",
    "team_a_win_rate10",
    "team_b_win_rate10",
    "win_rate_diff10",
    "team_a_recent_goal_diff10",
    "team_b_recent_goal_diff10",
    "recent_goal_diff_gap10",
    "is_neutral",
    "team_a_home_advantage",
    "tournament_importance",
]


# 실제 2026 월드컵 32강 슬롯
# 3ABCDF = A/B/C/D/F조 3위 중 하나
ROUND_OF_32_MATCHES = {
    73: ("2A", "2B"),
    74: ("1E", "3ABCDF"),
    75: ("1F", "2C"),
    76: ("1C", "2F"),
    77: ("1I", "3CDFGH"),
    78: ("2E", "2I"),
    79: ("1A", "3CEFHI"),
    80: ("1L", "3EHIJK"),
    81: ("1D", "3BEFIJ"),
    82: ("1G", "3AEHIJ"),
    83: ("2K", "2L"),
    84: ("1H", "2J"),
    85: ("1B", "3EFGIJ"),
    86: ("1J", "2H"),
    87: ("1K", "3DEIJL"),
    88: ("2D", "2G"),
}

ROUND_OF_16_MATCHES = {
    89: (74, 77),
    90: (73, 75),
    91: (76, 78),
    92: (79, 80),
    93: (83, 84),
    94: (81, 82),
    95: (86, 88),
    96: (85, 87),
}

QUARTER_FINAL_MATCHES = {
    97: (89, 90),
    98: (93, 94),
    99: (91, 92),
    100: (95, 96),
}

SEMI_FINAL_MATCHES = {
    101: (97, 98),
    102: (99, 100),
}

FINAL_MATCH = (101, 102)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_teams_map():
    teams = load_json(TEAMS_PATH)
    return {team["team"]: team for team in teams}


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
        "tieBreaker": random.random(),
    }


def weighted_choice(items, weights):
    r = random.random()
    cumulative = 0

    for item, weight in zip(items, weights):
        cumulative += weight
        if r <= cumulative:
            return item

    return items[-1]


def sample_group_result(prediction):
    r = random.random()

    a_win = prediction["teamAWinProb"]
    draw = prediction["drawProb"]

    if r < a_win:
        return "A_WIN"
    elif r < a_win + draw:
        return "DRAW"
    else:
        return "B_WIN"


def sample_group_score(result, prediction):
    a_win = prediction["teamAWinProb"]
    b_win = prediction["teamBWinProb"]
    draw = prediction["drawProb"]

    if result == "DRAW":
        return weighted_choice(
            [(0, 0), (1, 1), (2, 2)],
            [0.25, 0.60, 0.15],
        )

    if result == "A_WIN":
        margin_strength = a_win - max(draw, b_win)

        if margin_strength >= 0.35:
            return weighted_choice(
                [(2, 0), (3, 0), (3, 1), (4, 1)],
                [0.35, 0.30, 0.25, 0.10],
            )
        elif margin_strength >= 0.20:
            return weighted_choice(
                [(1, 0), (2, 0), (2, 1), (3, 1)],
                [0.25, 0.25, 0.35, 0.15],
            )
        else:
            return weighted_choice(
                [(1, 0), (2, 1), (3, 2)],
                [0.35, 0.50, 0.15],
            )

    if result == "B_WIN":
        margin_strength = b_win - max(draw, a_win)

        if margin_strength >= 0.35:
            return weighted_choice(
                [(0, 2), (0, 3), (1, 3), (1, 4)],
                [0.35, 0.30, 0.25, 0.10],
            )
        elif margin_strength >= 0.20:
            return weighted_choice(
                [(0, 1), (0, 2), (1, 2), (1, 3)],
                [0.25, 0.25, 0.35, 0.15],
            )
        else:
            return weighted_choice(
                [(0, 1), (1, 2), (2, 3)],
                [0.35, 0.50, 0.15],
            )

    return 1, 1


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

    team_a_stats["goalDiff"] = team_a_stats["goalsFor"] - team_a_stats["goalsAgainst"]
    team_b_stats["goalDiff"] = team_b_stats["goalsFor"] - team_b_stats["goalsAgainst"]


def sort_standings(standings):
    return sorted(
        standings,
        key=lambda x: (
            x["points"],
            x["goalDiff"],
            x["goalsFor"],
            x["wins"],
            x["tieBreaker"],
        ),
        reverse=True,
    )


def run_group_stage(predictions_by_group):
    group_results = {}
    third_place_teams = []

    for group, predictions in predictions_by_group.items():
        teams = {}

        for prediction in predictions:
            team_a = prediction["teamA"]
            team_b = prediction["teamB"]
            display_a = prediction["displayTeamA"]
            display_b = prediction["displayTeamB"]

            if team_a not in teams:
                teams[team_a] = init_team(display_a, team_a)

            if team_b not in teams:
                teams[team_b] = init_team(display_b, team_b)

            result = sample_group_result(prediction)
            score_a, score_b = sample_group_score(result, prediction)

            apply_match_result(
                teams[team_a],
                teams[team_b],
                score_a,
                score_b,
            )

        standings = sort_standings(list(teams.values()))

        for rank, team in enumerate(standings, start=1):
            team["rank"] = rank
            team["group"] = group

        group_results[group] = standings
        third_place_teams.append(standings[2])

    best_third_place_teams = sort_standings(third_place_teams)[:8]

    return group_results, best_third_place_teams


def assign_third_place_slots(best_third_place_teams):
    """
    실제 32강 슬롯의 후보 조합을 기준으로,
    진출한 3위 8팀을 각 3위 슬롯에 배정한다.
    """
    third_by_group = {
        team["group"]: team
        for team in best_third_place_teams
    }

    available_groups = set(third_by_group.keys())

    third_slots = []

    for match_no, (left, right) in ROUND_OF_32_MATCHES.items():
        if left.startswith("3"):
            third_slots.append((match_no, "left", list(left[1:])))
        if right.startswith("3"):
            third_slots.append((match_no, "right", list(right[1:])))

    # 가능한 후보가 적은 슬롯부터 배정
    third_slots = sorted(
        third_slots,
        key=lambda x: len(set(x[2]) & available_groups),
    )

    assignment = {}

    def backtrack(index, used_groups):
        if index == len(third_slots):
            return True

        match_no, side, candidates = third_slots[index]

        for group in candidates:
            if group not in available_groups:
                continue

            if group in used_groups:
                continue

            assignment[match_no] = third_by_group[group]

            if backtrack(index + 1, used_groups | {group}):
                return True

            assignment.pop(match_no, None)

        return False

    success = backtrack(0, set())

    if not success:
        raise ValueError(
            f"3위 팀 슬롯 배정 실패: 진출 3위 조합={sorted(available_groups)}"
        )

    return assignment


def resolve_slot(slot, group_results, third_assignment, match_no):
    if slot.startswith("1"):
        group = slot[1]
        return group_results[group][0]

    if slot.startswith("2"):
        group = slot[1]
        return group_results[group][1]

    if slot.startswith("3"):
        return third_assignment[match_no]

    raise ValueError(f"알 수 없는 슬롯: {slot}")


def build_feature_row(team_a, team_b, teams_map):
    a = teams_map[team_a["team"]]
    b = teams_map[team_b["team"]]

    team_a_elo = a["elo"]
    team_b_elo = b["elo"]

    return {
        "team_a_elo": team_a_elo,
        "team_b_elo": team_b_elo,
        "elo_diff": team_a_elo - team_b_elo,
        "team_a_recent_points5": a["recentPoints5"],
        "team_b_recent_points5": b["recentPoints5"],
        "recent_points_diff": a["recentPoints5"] - b["recentPoints5"],
        "team_a_win_rate5": a["recentWinRate5"],
        "team_b_win_rate5": b["recentWinRate5"],
        "win_rate_diff5": a["recentWinRate5"] - b["recentWinRate5"],
        "team_a_avg_goals_for5": a["recentAvgGoalsFor5"],
        "team_b_avg_goals_for5": b["recentAvgGoalsFor5"],
        "avg_goals_for_diff5": (
            a["recentAvgGoalsFor5"] - b["recentAvgGoalsFor5"]
        ),
        "team_a_avg_goals_against5": a["recentAvgGoalsAgainst5"],
        "team_b_avg_goals_against5": b["recentAvgGoalsAgainst5"],
        "avg_goals_against_diff5": (
            a["recentAvgGoalsAgainst5"] - b["recentAvgGoalsAgainst5"]
        ),
        "team_a_recent_goal_diff5": a["recentGoalDiff5"],
        "team_b_recent_goal_diff5": b["recentGoalDiff5"],
        "recent_goal_diff_gap": a["recentGoalDiff5"] - b["recentGoalDiff5"],
        "team_a_win_rate10": a["recentWinRate10"],
        "team_b_win_rate10": b["recentWinRate10"],
        "win_rate_diff10": a["recentWinRate10"] - b["recentWinRate10"],
        "team_a_recent_goal_diff10": a["recentGoalDiff10"],
        "team_b_recent_goal_diff10": b["recentGoalDiff10"],
        "recent_goal_diff_gap10": (
            a["recentGoalDiff10"] - b["recentGoalDiff10"]
        ),
        "is_neutral": 1,
        "team_a_home_advantage": int(team_a["team"] in HOST_TEAMS),
        "tournament_importance": 1.0,
    }


def get_knockout_advance_prob(model, team_a, team_b, teams_map):
    cache_key = (team_a["team"], team_b["team"])
    if cache_key in ADVANCE_PROBABILITY_CACHE:
        return ADVANCE_PROBABILITY_CACHE[cache_key]

    feature_row = build_feature_row(team_a, team_b, teams_map)
    feature_columns = list(
        getattr(model, "feature_names_in_", FEATURE_COLUMNS)
    )
    X = pd.DataFrame([feature_row], columns=feature_columns)

    proba = model.predict_proba(X)[0]
    class_to_prob = dict(zip(model.classes_, proba))

    a_win = float(class_to_prob.get("A_WIN", 0))
    draw = float(class_to_prob.get("DRAW", 0))
    b_win = float(class_to_prob.get("B_WIN", 0))

    # 토너먼트는 무승부가 없으므로 무승부 확률을 양 팀에 절반씩 배분
    a_advance = a_win + draw * 0.5
    b_advance = b_win + draw * 0.5

    total = a_advance + b_advance

    if total == 0:
        return 0.5

    advance_probability = a_advance / total
    ADVANCE_PROBABILITY_CACHE[cache_key] = advance_probability
    return advance_probability


def play_knockout_match(model, team_a, team_b, teams_map):
    a_advance_prob = get_knockout_advance_prob(
        model,
        team_a,
        team_b,
        teams_map,
    )

    if random.random() < a_advance_prob:
        return team_a
    else:
        return team_b


def add_count(stats, team, key):
    team_key = team["team"]

    stats[team_key]["displayTeam"] = team["displayTeam"]
    stats[team_key]["team"] = team["team"]
    stats[team_key]["group"] = team.get("group", "")
    stats[team_key][key] += 1


def run_tournament_once(predictions_by_group, model, teams_map):
    group_results, best_third_place_teams = run_group_stage(predictions_by_group)
    third_assignment = assign_third_place_slots(best_third_place_teams)

    winners = {}

    # Round of 32
    for match_no, (left_slot, right_slot) in ROUND_OF_32_MATCHES.items():
        team_a = resolve_slot(left_slot, group_results, third_assignment, match_no)
        team_b = resolve_slot(right_slot, group_results, third_assignment, match_no)

        winners[match_no] = play_knockout_match(model, team_a, team_b, teams_map)

    # Round of 16
    for match_no, (left_match, right_match) in ROUND_OF_16_MATCHES.items():
        winners[match_no] = play_knockout_match(
            model,
            winners[left_match],
            winners[right_match],
            teams_map,
        )

    # Quarterfinals
    for match_no, (left_match, right_match) in QUARTER_FINAL_MATCHES.items():
        winners[match_no] = play_knockout_match(
            model,
            winners[left_match],
            winners[right_match],
            teams_map,
        )

    # Semifinals
    for match_no, (left_match, right_match) in SEMI_FINAL_MATCHES.items():
        winners[match_no] = play_knockout_match(
            model,
            winners[left_match],
            winners[right_match],
            teams_map,
        )

    # Final
    finalist_a = winners[FINAL_MATCH[0]]
    finalist_b = winners[FINAL_MATCH[1]]

    champion = play_knockout_match(
        model,
        finalist_a,
        finalist_b,
        teams_map,
    )

    return {
        "groupResults": group_results,
        "roundOf32Winners": [winners[i] for i in range(73, 89)],
        "roundOf16Winners": [winners[i] for i in range(89, 97)],
        "quarterFinalWinners": [winners[i] for i in range(97, 101)],
        "semiFinalWinners": [winners[i] for i in range(101, 103)],
        "finalists": [finalist_a, finalist_b],
        "champion": champion,
    }


def main():
    random.seed(RANDOM_SEED)
    ADVANCE_PROBABILITY_CACHE.clear()

    predictions = load_json(PREDICTIONS_PATH)
    teams_map = load_teams_map()
    model = joblib.load(MODEL_PATH)

    predictions_by_group = defaultdict(list)

    for prediction in predictions:
        predictions_by_group[prediction["group"]].append(prediction)

    stats = defaultdict(
        lambda: {
            "displayTeam": "",
            "team": "",
            "group": "",
            "roundOf32Count": 0,
            "roundOf16Count": 0,
            "quarterFinalCount": 0,
            "semiFinalCount": 0,
            "finalCount": 0,
            "winnerCount": 0,
        }
    )

    for i in range(SIMULATION_COUNT):
        result = run_tournament_once(predictions_by_group, model, teams_map)

        # 조별리그 통과 팀
        for group, standings in result["groupResults"].items():
            # 1, 2위는 무조건 32강
            add_count(stats, standings[0], "roundOf32Count")
            add_count(stats, standings[1], "roundOf32Count")

        # 3위 중 상위 8팀은 roundOf32Winners 이전 단계에서 별도 카운트가 필요하므로
        # Round of 32 출전팀 전체를 역으로 모으기 위해 각 32강 경기의 양 팀을 다시 세는 대신,
        # Round of 32 승자만으로는 출전팀 32개를 알 수 없다.
        # 따라서 간단히 group stage 결과에서 직접 3위 상위 8팀을 다시 계산한다.
        all_thirds = []
        for group, standings in result["groupResults"].items():
            all_thirds.append(standings[2])

        best_thirds = sort_standings(all_thirds)[:8]

        for team in best_thirds:
            add_count(stats, team, "roundOf32Count")

        for team in result["roundOf32Winners"]:
            add_count(stats, team, "roundOf16Count")

        for team in result["roundOf16Winners"]:
            add_count(stats, team, "quarterFinalCount")

        for team in result["quarterFinalWinners"]:
            add_count(stats, team, "semiFinalCount")

        for team in result["semiFinalWinners"]:
            add_count(stats, team, "finalCount")

        add_count(stats, result["champion"], "winnerCount")

        if (i + 1) % 1000 == 0:
            print(f"{i + 1}회 시뮬레이션 완료")

    output = []

    for team_key, item in stats.items():
        output.append(
            {
                "displayTeam": item["displayTeam"],
                "team": item["team"],
                "group": item["group"],
                "roundOf32Prob": round(item["roundOf32Count"] / SIMULATION_COUNT, 4),
                "roundOf16Prob": round(item["roundOf16Count"] / SIMULATION_COUNT, 4),
                "quarterFinalProb": round(item["quarterFinalCount"] / SIMULATION_COUNT, 4),
                "semiFinalProb": round(item["semiFinalCount"] / SIMULATION_COUNT, 4),
                "finalProb": round(item["finalCount"] / SIMULATION_COUNT, 4),
                "winnerProb": round(item["winnerCount"] / SIMULATION_COUNT, 4),
            }
        )

    output = sorted(
        output,
        key=lambda x: (
            x["winnerProb"],
            x["finalProb"],
            x["semiFinalProb"],
            x["quarterFinalProb"],
        ),
        reverse=True,
    )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print()
    print(f"토너먼트 시뮬레이션 완료: {OUTPUT_PATH}")
    print(f"반복 횟수: {SIMULATION_COUNT}")
    print()
    print("우승 확률 상위 20팀")

    for idx, team in enumerate(output[:20], start=1):
        print(
            f"{idx}. {team['displayTeam']} | "
            f"우승 {team['winnerProb'] * 100:.2f}% | "
            f"결승 {team['finalProb'] * 100:.2f}% | "
            f"4강 {team['semiFinalProb'] * 100:.2f}% | "
            f"8강 {team['quarterFinalProb'] * 100:.2f}% | "
            f"16강 {team['roundOf16Prob'] * 100:.2f}% | "
            f"32강 {team['roundOf32Prob'] * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
