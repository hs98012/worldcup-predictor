import json
from collections import defaultdict
from pathlib import Path

INPUT_PATH = Path("data/processed/predictions_adjusted.json")
OUTPUT_PATH = Path("data/processed/group_predictions.json")


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


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    groups = defaultdict(dict)
    match_results_by_group = defaultdict(list)

    for prediction in predictions:
        group = prediction["group"]

        display_a = prediction["displayTeamA"]
        display_b = prediction["displayTeamB"]
        team_a = prediction["teamA"]
        team_b = prediction["teamB"]

        if team_a not in groups[group]:
            groups[group][team_a] = init_team(display_a, team_a)

        if team_b not in groups[group]:
            groups[group][team_b] = init_team(display_b, team_b)

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
            }
        )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"조별리그 예상 순위 생성 완료: {OUTPUT_PATH}")
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
