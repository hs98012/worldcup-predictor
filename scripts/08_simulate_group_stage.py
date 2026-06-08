import json
import random
from collections import defaultdict
from pathlib import Path

INPUT_PATH = Path("data/processed/predictions_adjusted.json")
OUTPUT_PATH = Path("data/processed/group_simulation.json")

SIMULATION_COUNT = 10000
RANDOM_SEED = 42


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


def sample_result(prediction):
    """
    경기별 승/무/패 확률에 따라 결과를 랜덤 샘플링한다.
    """
    r = random.random()

    a_win = prediction["teamAWinProb"]
    draw = prediction["drawProb"]
    b_win = prediction["teamBWinProb"]

    if r < a_win:
        return "A_WIN"
    elif r < a_win + draw:
        return "DRAW"
    else:
        return "B_WIN"


def sample_score(result, prediction):
    """
    MVP용 예상 스코어 샘플링.
    실제 득점 모델은 아니고, 조별 순위 계산을 위한 단순 점수 생성 로직이다.
    """
    a_win = prediction["teamAWinProb"]
    b_win = prediction["teamBWinProb"]
    draw = prediction["drawProb"]

    if result == "DRAW":
        draw_scores = [(0, 0), (1, 1), (2, 2)]
        weights = [0.25, 0.60, 0.15]
        return weighted_choice(draw_scores, weights)

    if result == "A_WIN":
        margin_strength = a_win - max(draw, b_win)

        if margin_strength >= 0.35:
            scores = [(2, 0), (3, 0), (3, 1), (4, 1)]
            weights = [0.35, 0.30, 0.25, 0.10]
        elif margin_strength >= 0.20:
            scores = [(1, 0), (2, 0), (2, 1), (3, 1)]
            weights = [0.25, 0.25, 0.35, 0.15]
        else:
            scores = [(1, 0), (2, 1), (3, 2)]
            weights = [0.35, 0.50, 0.15]

        return weighted_choice(scores, weights)

    if result == "B_WIN":
        margin_strength = b_win - max(draw, a_win)

        if margin_strength >= 0.35:
            scores = [(0, 2), (0, 3), (1, 3), (1, 4)]
            weights = [0.35, 0.30, 0.25, 0.10]
        elif margin_strength >= 0.20:
            scores = [(0, 1), (0, 2), (1, 2), (1, 3)]
            weights = [0.25, 0.25, 0.35, 0.15]
        else:
            scores = [(0, 1), (1, 2), (2, 3)]
            weights = [0.35, 0.50, 0.15]

        return weighted_choice(scores, weights)

    return 1, 1


def weighted_choice(items, weights):
    r = random.random()
    cumulative = 0

    for item, weight in zip(items, weights):
        cumulative += weight
        if r <= cumulative:
            return item

    return items[-1]


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
    """
    FIFA 실제 타이브레이커 전체를 모두 구현하지는 않고,
    MVP에서는 승점 → 득실차 → 다득점 → 승수 → 랜덤 타이브레이커 순으로 정렬한다.
    """
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


def run_one_simulation(predictions_by_group):
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

            sampled_result = sample_result(prediction)
            score_a, score_b = sample_score(sampled_result, prediction)

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

    # 12개 조 3위 중 상위 8팀이 32강 진출
    best_third_place_teams = sort_standings(third_place_teams)[:8]
    best_third_team_keys = {team["team"] for team in best_third_place_teams}

    return group_results, best_third_team_keys


def main():
    random.seed(RANDOM_SEED)

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    predictions_by_group = defaultdict(list)

    for prediction in predictions:
        predictions_by_group[prediction["group"]].append(prediction)

    stats = defaultdict(
        lambda: {
            "displayTeam": "",
            "team": "",
            "group": "",
            "rank1Count": 0,
            "rank2Count": 0,
            "rank3Count": 0,
            "rank4Count": 0,
            "directAdvanceCount": 0,
            "thirdPlaceAdvanceCount": 0,
            "roundOf32Count": 0,
            "totalRank": 0,
        }
    )

    for _ in range(SIMULATION_COUNT):
        group_results, best_third_team_keys = run_one_simulation(predictions_by_group)

        for group, standings in group_results.items():
            for team in standings:
                key = team["team"]

                stats[key]["displayTeam"] = team["displayTeam"]
                stats[key]["team"] = team["team"]
                stats[key]["group"] = group
                stats[key]["totalRank"] += team["rank"]

                if team["rank"] == 1:
                    stats[key]["rank1Count"] += 1
                    stats[key]["directAdvanceCount"] += 1
                    stats[key]["roundOf32Count"] += 1

                elif team["rank"] == 2:
                    stats[key]["rank2Count"] += 1
                    stats[key]["directAdvanceCount"] += 1
                    stats[key]["roundOf32Count"] += 1

                elif team["rank"] == 3:
                    stats[key]["rank3Count"] += 1

                    if key in best_third_team_keys:
                        stats[key]["thirdPlaceAdvanceCount"] += 1
                        stats[key]["roundOf32Count"] += 1

                elif team["rank"] == 4:
                    stats[key]["rank4Count"] += 1

    output_by_group = defaultdict(list)

    for team_key, item in stats.items():
        result = {
            "displayTeam": item["displayTeam"],
            "team": item["team"],
            "group": item["group"],
            "rank1Prob": round(item["rank1Count"] / SIMULATION_COUNT, 4),
            "rank2Prob": round(item["rank2Count"] / SIMULATION_COUNT, 4),
            "rank3Prob": round(item["rank3Count"] / SIMULATION_COUNT, 4),
            "rank4Prob": round(item["rank4Count"] / SIMULATION_COUNT, 4),
            "directAdvanceProb": round(item["directAdvanceCount"] / SIMULATION_COUNT, 4),
            "thirdPlaceAdvanceProb": round(item["thirdPlaceAdvanceCount"] / SIMULATION_COUNT, 4),
            "roundOf32Prob": round(item["roundOf32Count"] / SIMULATION_COUNT, 4),
            "averageRank": round(item["totalRank"] / SIMULATION_COUNT, 2),
        }

        output_by_group[item["group"]].append(result)

    final_output = []

    for group in sorted(output_by_group.keys()):
        teams = sorted(
            output_by_group[group],
            key=lambda x: (
                x["roundOf32Prob"],
                x["directAdvanceProb"],
                x["rank1Prob"],
                -x["averageRank"],
            ),
            reverse=True,
        )

        final_output.append(
            {
                "group": group,
                "simulationCount": SIMULATION_COUNT,
                "teams": teams,
            }
        )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    print(f"조별리그 시뮬레이션 완료: {OUTPUT_PATH}")
    print(f"반복 횟수: {SIMULATION_COUNT}")
    print()

    for group_data in final_output:
        print(f"Group {group_data['group']}")
        for team in group_data["teams"]:
            print(
                f"{team['displayTeam']} | "
                f"1위 {team['rank1Prob'] * 100:.1f}% | "
                f"2위 {team['rank2Prob'] * 100:.1f}% | "
                f"3위 {team['rank3Prob'] * 100:.1f}% | "
                f"32강 {team['roundOf32Prob'] * 100:.1f}% | "
                f"평균순위 {team['averageRank']}"
            )
        print()


if __name__ == "__main__":
    main()
