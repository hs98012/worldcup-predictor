import json
import math
from pathlib import Path

TEAMS_PATH = Path("data/processed/teams.json")
OUTPUT_PATH = Path("data/processed/sample_prediction.json")


def load_teams():
    with open(TEAMS_PATH, "r", encoding="utf-8") as f:
        teams = json.load(f)

    return {team["team"]: team for team in teams}


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def calculate_match_prediction(team_a, team_b):
    elo_a = team_a["elo"]
    elo_b = team_b["elo"]

    elo_diff = elo_a - elo_b

    # Elo 차이를 승리 확률의 기본값으로 변환
    base_win_prob_a = 1 / (1 + 10 ** (-elo_diff / 400))
    base_win_prob_b = 1 - base_win_prob_a

    # 무승부 확률은 팀 전력 차이가 작을수록 높게 설정
    # 전력 차이가 크면 무승부 확률을 낮춤
    draw_prob = 0.28 * math.exp(-abs(elo_diff) / 600)

    # 남은 확률을 양 팀 승리 확률로 나눔
    remain = 1 - draw_prob

    win_prob_a = remain * base_win_prob_a
    win_prob_b = remain * base_win_prob_b

    # 최근 폼 보정
    form_diff = team_a["recentPoints5"] - team_b["recentPoints5"]
    form_adjust = form_diff * 0.01

    win_prob_a += form_adjust
    win_prob_b -= form_adjust

    # 확률이 음수/1 초과로 튀지 않게 보정
    win_prob_a = max(0.01, min(0.90, win_prob_a))
    win_prob_b = max(0.01, min(0.90, win_prob_b))

    # 다시 합이 1이 되도록 정규화
    total = win_prob_a + draw_prob + win_prob_b

    win_prob_a /= total
    draw_prob /= total
    win_prob_b /= total

    return {
        "teamA": team_a["team"],
        "teamB": team_b["team"],
        "teamAElo": team_a["elo"],
        "teamBElo": team_b["elo"],
        "teamARecentForm": team_a["recentForm"],
        "teamBRecentForm": team_b["recentForm"],
        "teamARecentPoints5": team_a["recentPoints5"],
        "teamBRecentPoints5": team_b["recentPoints5"],
        "teamAWinProb": round(win_prob_a, 4),
        "drawProb": round(draw_prob, 4),
        "teamBWinProb": round(win_prob_b, 4),
    }


def main():
    teams = load_teams()

    team_a_name = "South Korea"
    team_b_name = "Portugal"

    if team_a_name not in teams:
        raise ValueError(f"{team_a_name} 팀을 찾을 수 없습니다.")

    if team_b_name not in teams:
        raise ValueError(f"{team_b_name} 팀을 찾을 수 없습니다.")

    prediction = calculate_match_prediction(
        teams[team_a_name],
        teams[team_b_name],
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(prediction, f, ensure_ascii=False, indent=2)

    print("예측 완료")
    print(f"저장 위치: {OUTPUT_PATH}")
    print()
    print(f"{prediction['teamA']} vs {prediction['teamB']}")
    print(f"{prediction['teamA']} 승: {prediction['teamAWinProb'] * 100:.1f}%")
    print(f"무승부: {prediction['drawProb'] * 100:.1f}%")
    print(f"{prediction['teamB']} 승: {prediction['teamBWinProb'] * 100:.1f}%")
    print()
    print("근거")
    print(f"- {prediction['teamA']} Elo: {prediction['teamAElo']}")
    print(f"- {prediction['teamB']} Elo: {prediction['teamBElo']}")
    print(f"- {prediction['teamA']} 최근 5경기: {prediction['teamARecentForm']}")
    print(f"- {prediction['teamB']} 최근 5경기: {prediction['teamBRecentForm']}")


if __name__ == "__main__":
    main()
