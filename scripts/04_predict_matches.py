import json
import math
from pathlib import Path

TEAMS_PATH = Path("data/processed/teams.json")
MATCHES_PATH = Path("data/processed/matches.json")
OUTPUT_PATH = Path("data/processed/predictions.json")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_teams_map():
    teams = load_json(TEAMS_PATH)
    return {team["team"]: team for team in teams}


def calculate_match_prediction(team_a, team_b):
    elo_a = team_a["elo"]
    elo_b = team_b["elo"]

    elo_diff = elo_a - elo_b

    # Elo 기반 기본 승률
    base_win_prob_a = 1 / (1 + 10 ** (-elo_diff / 400))
    base_win_prob_b = 1 - base_win_prob_a

    # 전력 차이가 작을수록 무승부 확률 증가
    draw_prob = 0.28 * math.exp(-abs(elo_diff) / 600)

    remain = 1 - draw_prob

    win_prob_a = remain * base_win_prob_a
    win_prob_b = remain * base_win_prob_b

    # 최근 5경기 승점 차이 보정
    form_diff = team_a["recentPoints5"] - team_b["recentPoints5"]
    form_adjust = form_diff * 0.01

    win_prob_a += form_adjust
    win_prob_b -= form_adjust

    # 확률 범위 보정
    win_prob_a = max(0.01, min(0.90, win_prob_a))
    win_prob_b = max(0.01, min(0.90, win_prob_b))

    # 합이 1이 되도록 정규화
    total = win_prob_a + draw_prob + win_prob_b

    win_prob_a /= total
    draw_prob /= total
    win_prob_b /= total

    # 예상 득점은 임시 버전
    expected_goals_a = 1.25 + (elo_diff / 400) * 0.35 + form_diff * 0.03
    expected_goals_b = 1.25 - (elo_diff / 400) * 0.35 - form_diff * 0.03

    expected_goals_a = max(0.3, expected_goals_a)
    expected_goals_b = max(0.3, expected_goals_b)

    return {
        "teamAWinProb": round(win_prob_a, 4),
        "drawProb": round(draw_prob, 4),
        "teamBWinProb": round(win_prob_b, 4),
        "expectedGoalsA": round(expected_goals_a, 2),
        "expectedGoalsB": round(expected_goals_b, 2),
        "confidence": get_confidence(abs(elo_diff), abs(form_diff)),
        "reasons": build_reasons(team_a, team_b, elo_diff, form_diff),
    }


def get_confidence(abs_elo_diff, abs_form_diff):
    if abs_elo_diff >= 180 or abs_form_diff >= 8:
        return "HIGH"
    elif abs_elo_diff >= 80 or abs_form_diff >= 4:
        return "MEDIUM"
    else:
        return "LOW"


def build_reasons(team_a, team_b, elo_diff, form_diff):
    reasons = []

    if elo_diff > 0:
        reasons.append(f"{team_a['team']}의 Elo 점수가 더 높음")
    elif elo_diff < 0:
        reasons.append(f"{team_b['team']}의 Elo 점수가 더 높음")
    else:
        reasons.append("두 팀의 Elo 점수가 동일함")

    if form_diff > 0:
        reasons.append(f"{team_a['team']}의 최근 5경기 승점이 더 높음")
    elif form_diff < 0:
        reasons.append(f"{team_b['team']}의 최근 5경기 승점이 더 높음")
    else:
        reasons.append("두 팀의 최근 5경기 승점이 동일함")

    reasons.append(
        f"{team_a['team']} 최근 5경기: {team_a['recentForm']}"
    )
    reasons.append(
        f"{team_b['team']} 최근 5경기: {team_b['recentForm']}"
    )

    return reasons


def main():
    teams = get_teams_map()
    matches = load_json(MATCHES_PATH)

    predictions = []

    for match in matches:
        team_a_name = match["teamA"]
        team_b_name = match["teamB"]

        if team_a_name not in teams:
            print(f"스킵: {team_a_name} 팀을 teams.json에서 찾을 수 없음")
            continue

        if team_b_name not in teams:
            print(f"스킵: {team_b_name} 팀을 teams.json에서 찾을 수 없음")
            continue

        team_a = teams[team_a_name]
        team_b = teams[team_b_name]

        prediction = calculate_match_prediction(team_a, team_b)

        predictions.append(
            {
                "matchId": match["matchId"],
                "stage": match["stage"],
                "group": match["group"],
                "date": match["date"],
                "teamA": team_a_name,
                "teamB": team_b_name,
                "teamAElo": team_a["elo"],
                "teamBElo": team_b["elo"],
                "teamARecentForm": team_a["recentForm"],
                "teamBRecentForm": team_b["recentForm"],
                **prediction,
            }
        )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"예측 완료: {OUTPUT_PATH}")
    print(f"생성된 예측 수: {len(predictions)}")
    print()

    for p in predictions:
        print(f"{p['teamA']} vs {p['teamB']}")
        print(f"- {p['teamA']} 승: {p['teamAWinProb'] * 100:.1f}%")
        print(f"- 무승부: {p['drawProb'] * 100:.1f}%")
        print(f"- {p['teamB']} 승: {p['teamBWinProb'] * 100:.1f}%")
        print(f"- 예상 스코어: {p['expectedGoalsA']} : {p['expectedGoalsB']}")
        print(f"- 신뢰도: {p['confidence']}")
        print()


if __name__ == "__main__":
    main()
