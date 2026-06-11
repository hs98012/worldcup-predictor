import json
from pathlib import Path

TOURNAMENT_PATH = Path("data/processed/tournament_simulation.json")
GROUP_PREDICTIONS_PATH = Path("data/processed/group_predictions.json")
GROUP_SIMULATION_PATH = Path("data/processed/group_simulation.json")
PREDICTIONS_PATH = Path("data/processed/predictions_adjusted.json")
METRICS_PATH = Path("data/processed/model_metrics.json")
OUTPUT_PATH = Path("data/processed/dashboard_summary.json")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def percent(value):
    return round(value * 100, 2)


def main():
    tournament = load_json(TOURNAMENT_PATH)
    group_predictions = load_json(GROUP_PREDICTIONS_PATH)
    group_simulation = load_json(GROUP_SIMULATION_PATH)
    predictions = load_json(PREDICTIONS_PATH)
    metrics = load_json(METRICS_PATH)

    winner_top10 = [
        {
            "rank": idx + 1,
            "displayTeam": team["displayTeam"],
            "team": team["team"],
            "group": team["group"],
            "winnerProb": percent(team["winnerProb"]),
            "finalProb": percent(team["finalProb"]),
            "semiFinalProb": percent(team["semiFinalProb"]),
            "quarterFinalProb": percent(team["quarterFinalProb"]),
            "roundOf16Prob": percent(team["roundOf16Prob"]),
            "roundOf32Prob": percent(team["roundOf32Prob"]),
        }
        for idx, team in enumerate(tournament[:10])
    ]

    korea = next(
        (team for team in tournament if team["displayTeam"] == "South Korea"),
        None,
    )

    korea_summary = None

    if korea:
        korea_summary = {
            "displayTeam": korea["displayTeam"],
            "team": korea["team"],
            "group": korea["group"],
            "roundOf32Prob": percent(korea["roundOf32Prob"]),
            "roundOf16Prob": percent(korea["roundOf16Prob"]),
            "quarterFinalProb": percent(korea["quarterFinalProb"]),
            "semiFinalProb": percent(korea["semiFinalProb"]),
            "finalProb": percent(korea["finalProb"]),
            "winnerProb": percent(korea["winnerProb"]),
        }

    model_summary = {
        "modelName": f"scikit-learn {metrics['selectedModel']}",
        "task": "A_WIN / DRAW / B_WIN multiclass classification",
        "accuracy": percent(metrics["accuracy"]),
        "logLoss": metrics["logLoss"],
        "testStartDate": metrics["testStartDate"],
        "testEndDate": metrics["testEndDate"],
        "trainSize": metrics["trainSize"],
        "testSize": metrics["testSize"],
        "drawRecall": round(
            metrics["classificationReport"]["DRAW"]["recall"],
            4,
        ),
        "note": metrics["selectionReason"],
    }

    group_round32_summary = []

    for group_data in group_simulation:
        teams = sorted(
            group_data["teams"],
            key=lambda x: x["roundOf32Prob"],
            reverse=True,
        )

        group_round32_summary.append(
            {
                "group": group_data["group"],
                "teams": [
                    {
                        "displayTeam": team["displayTeam"],
                        "team": team["team"],
                        "roundOf32Prob": percent(team["roundOf32Prob"]),
                        "directAdvanceProb": percent(team["directAdvanceProb"]),
                        "rank1Prob": percent(team["rank1Prob"]),
                        "rank2Prob": percent(team["rank2Prob"]),
                        "rank3Prob": percent(team["rank3Prob"]),
                        "averageRank": team["averageRank"],
                    }
                    for team in teams
                ],
            }
        )

    upcoming_match_cards = [
        {
            "matchId": match["matchId"],
            "group": match["group"],
            "date": match["date"],
            "displayTeamA": match["displayTeamA"],
            "displayTeamB": match["displayTeamB"],
            "teamAWinProb": percent(match["teamAWinProb"]),
            "drawProb": percent(match["drawProb"]),
            "teamBWinProb": percent(match["teamBWinProb"]),
            "predictedResult": match["predictedResult"],
        }
        for match in predictions[:12]
    ]

    summary = {
        "projectTitle": "2026 World Cup Predictor",
        "description": "국가대표 경기 데이터를 기반으로 경기별 승/무/패 확률과 월드컵 토너먼트 진출 확률을 시뮬레이션한 대시보드 데이터입니다.",
        "simulationCount": 10000,
        "modelSummary": model_summary,
        "winnerTop10": winner_top10,
        "southKorea": korea_summary,
        "groupStandings": group_predictions,
        "groupRound32Summary": group_round32_summary,
        "upcomingMatchCards": upcoming_match_cards,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"대시보드 요약 생성 완료: {OUTPUT_PATH}")
    print()
    print("우승 확률 TOP 10")
    for team in winner_top10:
        print(
            f"{team['rank']}. {team['displayTeam']} "
            f"우승 {team['winnerProb']}% / 결승 {team['finalProb']}%"
        )

    if korea_summary:
        print()
        print("대한민국 요약")
        print(korea_summary)


if __name__ == "__main__":
    main()
