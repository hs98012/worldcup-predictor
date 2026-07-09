import json
from collections import defaultdict
from pathlib import Path

from utils.worldcup_simulation import (
    build_initial_group_state,
    completed_fixture_keys,
    excluded_group_match_label,
    fixture_group_map,
    is_valid_group_match,
    is_completed_prediction,
    load_completed_results,
)

INPUT_PATH = Path("data/processed/predictions_adjusted.json")
OUTPUT_PATH = Path("data/processed/group_predictions.json")
ACTUAL_STANDINGS_PATH = Path("data/processed/worldcup_group_standings.json")


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
    }


def predict_score(prediction):
    """
    확률 기반으로 예상 스코어를 단순 생성한다.
    현재는 확률 차이를 이용한 MVP용 스코어 추정이다.
    """
    a_win = prediction["teamAWinProb"]
    draw = prediction["drawProb"]
    b_win = prediction["teamBWinProb"]

    result = prediction["predictedResult"]

    if result == "DRAW":
        if draw >= 0.36:
            return 1, 1
        return 0, 0

    if result == "A_WIN":
        diff = a_win - max(draw, b_win)

        if diff >= 0.35:
            return 3, 0
        elif diff >= 0.20:
            return 2, 0
        else:
            return 2, 1

    if result == "B_WIN":
        diff = b_win - max(draw, a_win)

        if diff >= 0.35:
            return 0, 3
        elif diff >= 0.20:
            return 0, 2
        else:
            return 1, 2

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


def convert_actual_standings(actual_standings, match_results_by_group):
    output = []

    for group_data in actual_standings:
        group = group_data["group"]
        standings = []

        for team in group_data["teams"]:
            rank = team["rank"]
            standings.append(
                {
                    "displayTeam": team["team"],
                    "team": team["team"],
                    "played": team["played"],
                    "wins": team["wins"],
                    "draws": team["draws"],
                    "losses": team["losses"],
                    "goalsFor": team["goalsFor"],
                    "goalsAgainst": team["goalsAgainst"],
                    "goalDiff": team["goalDifference"],
                    "points": team["points"],
                    "rank": rank,
                    "qualifiedStatus": get_qualified_status(rank),
                }
            )

        output.append(
            {
                "group": group,
                "standings": standings,
                "matches": match_results_by_group[group],
                "remainingGroupMatches": 0,
            }
        )

    return output


def log_excluded_group_matches(excluded_matches):
    if not excluded_matches:
        return

    print("조별리그 순위 계산에서 제외된 경기:")
    for match in excluded_matches:
        print(excluded_group_match_label(match))


def main():
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
    groups, applied_completed_count = build_initial_group_state(
        completed_results,
        team_names,
    )
    groups_by_fixture = fixture_group_map()
    match_results_by_group = defaultdict(list)
    valid_pending_predictions = []
    excluded_predictions = []

    for match in completed_results:
        group = groups_by_fixture.get(
            (
                match["date"],
                frozenset(
                    (
                        match["normalizedHomeTeam"],
                        match["normalizedAwayTeam"],
                    )
                ),
            )
        )
        if group is None:
            continue

        match_results_by_group[group].append(
            {
                "date": match["date"],
                "displayTeamA": match["homeTeam"],
                "displayTeamB": match["awayTeam"],
                "predictedScoreA": match["homeScore"],
                "predictedScoreB": match["awayScore"],
                "predictedResult": match["result"],
                "isCompleted": True,
            }
        )

    for prediction in predictions:
        if is_completed_prediction(prediction, completed_keys):
            continue

        if not is_valid_group_match(prediction, groups):
            excluded_predictions.append(prediction)
            continue

        valid_pending_predictions.append(prediction)

    log_excluded_group_matches(excluded_predictions)

    if not valid_pending_predictions:
        actual_standings = load_json(ACTUAL_STANDINGS_PATH)
        output = convert_actual_standings(
            actual_standings,
            match_results_by_group,
        )

        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
            f.write("\n")

        print(f"조별리그 예상 순위 생성 완료: {OUTPUT_PATH}")
        print(f"초기 standings에 반영한 완료 경기 수: {applied_completed_count}")
        print("시뮬레이션 대상 남은 조별리그 경기 수: 0")
        print("확정 조별리그 순위를 기반으로 출력했습니다.")
        return

    for prediction in valid_pending_predictions:
        group = prediction["group"]

        display_a = prediction["displayTeamA"]
        display_b = prediction["displayTeamB"]
        team_a = prediction["teamA"]
        team_b = prediction["teamB"]

        score_a, score_b = predict_score(prediction)

        apply_match_result(
            groups[group][team_a],
            groups[group][team_b],
            score_a,
            score_b,
        )

        match_results_by_group[group].append(
            {
                "matchId": prediction["matchId"],
                "date": prediction["date"],
                "displayTeamA": display_a,
                "displayTeamB": display_b,
                "predictedScoreA": score_a,
                "predictedScoreB": score_b,
                "teamAWinProb": prediction["teamAWinProb"],
                "drawProb": prediction["drawProb"],
                "teamBWinProb": prediction["teamBWinProb"],
                "predictedResult": prediction["predictedResult"],
                "isCompleted": False,
            }
        )

    output = []

    for group in sorted(groups.keys()):
        standings = list(groups[group].values())

        standings = sorted(
            standings,
            key=lambda x: (
                x["points"],
                x["goalDiff"],
                x["goalsFor"],
                x["wins"],
            ),
            reverse=True,
        )

        for rank, team in enumerate(standings, start=1):
            team["rank"] = rank
            team["qualifiedStatus"] = get_qualified_status(rank)

        output.append(
            {
                "group": group,
                "standings": standings,
                "matches": match_results_by_group[group],
                "remainingGroupMatches": len(match_results_by_group[group])
                - sum(
                    1
                    for match in match_results_by_group[group]
                    if match.get("isCompleted")
                ),
            }
        )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"조별리그 예상 순위 생성 완료: {OUTPUT_PATH}")
    print(f"초기 standings에 반영한 완료 경기 수: {applied_completed_count}")
    print(
        "시뮬레이션 대상 남은 조별리그 경기 수: "
        f"{len(valid_pending_predictions)}"
    )
    print()

    for group_data in output:
        print(f"Group {group_data['group']}")
        for team in group_data["standings"]:
            print(
                f"{team['rank']}. {team['displayTeam']} "
                f"{team['points']}점 "
                f"{team['wins']}승 {team['draws']}무 {team['losses']}패 "
                f"득실 {team['goalDiff']} "
                f"({team['qualifiedStatus']})"
            )
        print()


def get_qualified_status(rank):
    if rank <= 2:
        return "DIRECT_ADVANCE"
    elif rank == 3:
        return "THIRD_PLACE_CANDIDATE"
    else:
        return "ELIMINATED"


if __name__ == "__main__":
    main()
