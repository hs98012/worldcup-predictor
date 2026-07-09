import json
import random
from copy import deepcopy
from collections import defaultdict
from pathlib import Path

from utils.worldcup_simulation import (
    apply_match_result,
    build_initial_group_state,
    completed_fixture_keys,
    excluded_group_match_label,
    is_valid_group_match,
    is_completed_prediction,
    load_completed_results,
)

INPUT_PATH = Path("data/processed/predictions_adjusted.json")
OUTPUT_PATH = Path("data/processed/group_simulation.json")
ACTUAL_STANDINGS_PATH = Path("data/processed/worldcup_group_standings.json")

SIMULATION_COUNT = 10000
RANDOM_SEED = 42


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


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"필수 입력 파일을 찾을 수 없습니다: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"JSON 파일을 읽을 수 없습니다: {path} "
            f"(line {error.lineno}, column {error.colno})"
        ) from error


def log_excluded_group_matches(excluded_matches):
    if not excluded_matches:
        return

    print("조별리그 시뮬레이션에서 제외된 경기:")
    for match in excluded_matches:
        print(excluded_group_match_label(match))


def create_final_group_simulation(actual_standings):
    third_place_teams = [
        group_data["teams"][2]
        for group_data in actual_standings
        if len(group_data.get("teams", [])) >= 3
    ]
    best_third_teams = sorted(
        third_place_teams,
        key=lambda team: (
            team["points"],
            team["goalDifference"],
            team["goalsFor"],
            team["wins"],
        ),
        reverse=True,
    )[:8]
    best_third_team_names = {team["team"] for team in best_third_teams}

    output = []
    for group_data in actual_standings:
        teams = []
        for team in group_data["teams"]:
            rank = team["rank"]
            is_direct = rank <= 2
            is_best_third = rank == 3 and team["team"] in best_third_team_names
            teams.append(
                {
                    "displayTeam": team["team"],
                    "team": team["team"],
                    "group": group_data["group"],
                    "rank1Prob": 1.0 if rank == 1 else 0.0,
                    "rank2Prob": 1.0 if rank == 2 else 0.0,
                    "rank3Prob": 1.0 if rank == 3 else 0.0,
                    "rank4Prob": 1.0 if rank == 4 else 0.0,
                    "directAdvanceProb": 1.0 if is_direct else 0.0,
                    "thirdPlaceAdvanceProb": 1.0 if is_best_third else 0.0,
                    "roundOf32Prob": 1.0 if is_direct or is_best_third else 0.0,
                    "averageRank": float(rank),
                }
            )

        output.append(
            {
                "group": group_data["group"],
                "simulationCount": 0,
                "remainingGroupMatches": 0,
                "teams": teams,
            }
        )

    return output


def run_one_simulation(predictions_by_group, initial_groups):
    group_results = {}
    third_place_teams = []
    simulated_groups = deepcopy(initial_groups)

    for group, teams in simulated_groups.items():
        for team in teams.values():
            team["tieBreaker"] = random.random()

        predictions = predictions_by_group.get(group, [])
        for prediction in predictions:
            team_a = prediction["teamA"]
            team_b = prediction["teamB"]

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

    predictions = load_json(INPUT_PATH)

    completed_results = load_completed_results()
    completed_keys = completed_fixture_keys(completed_results)
    team_names = {
        prediction["teamA"]
        for prediction in predictions
    } | {
        prediction["teamB"]
        for prediction in predictions
    }
    initial_groups, applied_completed_count = build_initial_group_state(
        completed_results,
        team_names,
    )

    predictions_by_group = defaultdict(list)
    excluded_predictions = []

    for prediction in predictions:
        if is_completed_prediction(prediction, completed_keys):
            continue
        if not is_valid_group_match(prediction, initial_groups):
            excluded_predictions.append(prediction)
            continue
        predictions_by_group[prediction["group"]].append(prediction)

    log_excluded_group_matches(excluded_predictions)

    remaining_group_matches = sum(
        len(items) for items in predictions_by_group.values()
    )

    if remaining_group_matches == 0:
        final_output = create_final_group_simulation(
            load_json(ACTUAL_STANDINGS_PATH)
        )

        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)
            f.write("\n")

        print(f"조별리그 시뮬레이션 완료: {OUTPUT_PATH}")
        print("반복 횟수: 0")
        print(f"초기 standings에 반영한 완료 경기 수: {applied_completed_count}")
        print("시뮬레이션 대상 남은 조별리그 경기 수: 0")
        print("확정 조별리그 순위를 기반으로 출력했습니다.")
        return

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
        group_results, best_third_team_keys = run_one_simulation(
            predictions_by_group,
            initial_groups,
        )

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
    print(f"초기 standings에 반영한 완료 경기 수: {applied_completed_count}")
    print(
        "시뮬레이션 대상 남은 경기 수: "
        f"{remaining_group_matches}"
    )
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
