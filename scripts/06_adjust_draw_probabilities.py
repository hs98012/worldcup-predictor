import json
from pathlib import Path

INPUT_PATH = Path("data/processed/predictions_sklearn.json")
OUTPUT_PATH = Path("data/processed/predictions_adjusted.json")

MAX_DRAW_PROB = 0.38


def normalize(a_win, draw, b_win):
    total = a_win + draw + b_win

    if total == 0:
        return 1 / 3, 1 / 3, 1 / 3

    return a_win / total, draw / total, b_win / total


def cap_draw_probability(a_win, draw, b_win):
    """
    최종 무승부 확률이 너무 높아지는 것을 방지한다.
    무승부 확률이 상한을 넘으면 초과분을 양 팀 승률에 비례해서 되돌린다.
    """
    if draw <= MAX_DRAW_PROB:
        return normalize(a_win, draw, b_win)

    excess = draw - MAX_DRAW_PROB
    draw = MAX_DRAW_PROB

    win_total = a_win + b_win

    if win_total == 0:
        a_win += excess / 2
        b_win += excess / 2
    else:
        a_win += excess * (a_win / win_total)
        b_win += excess * (b_win / win_total)

    return normalize(a_win, draw, b_win)


def adjust_draw_probability(prediction):
    a_win = prediction["teamAWinProb"]
    draw = prediction["drawProb"]
    b_win = prediction["teamBWinProb"]

    features = prediction.get("features", {})

    elo_diff = abs(features.get("elo_diff", 0))
    recent_points_diff = abs(features.get("recent_points_diff", 0))
    recent_goal_diff_gap = abs(features.get("recent_goal_diff_gap", 0))

    win_prob_gap = abs(a_win - b_win)

    adjustment = 0.0
    reasons = []

    # 이미 무승부 확률이 충분히 높으면 추가 보정하지 않음
    if draw >= 0.35:
        adjusted = prediction.copy()
        a_win, draw, b_win = cap_draw_probability(a_win, draw, b_win)

        adjusted["teamAWinProbBeforeAdjust"] = prediction["teamAWinProb"]
        adjusted["drawProbBeforeAdjust"] = prediction["drawProb"]
        adjusted["teamBWinProbBeforeAdjust"] = prediction["teamBWinProb"]

        adjusted["teamAWinProb"] = round(a_win, 4)
        adjusted["drawProb"] = round(draw, 4)
        adjusted["teamBWinProb"] = round(b_win, 4)

        adjusted["drawAdjustment"] = 0.0
        adjusted["drawAdjustmentReasons"] = ["기존 무승부 확률이 높아 추가 보정하지 않음"]

        adjusted["predictedResult"] = max(
            [
                ("A_WIN", adjusted["teamAWinProb"]),
                ("DRAW", adjusted["drawProb"]),
                ("B_WIN", adjusted["teamBWinProb"]),
            ],
            key=lambda x: x[1],
        )[0]

        return adjusted

    # 전력 차이가 매우 작은 경기
    if elo_diff < 60:
        adjustment += 0.03
        reasons.append("Elo 차이가 작아 박빙 경기로 판단")

    # 최근 5경기 승점 차이가 작은 경기
    if recent_points_diff <= 2:
        adjustment += 0.02
        reasons.append("최근 5경기 승점 차이가 작음")

    # 최근 득실차 흐름도 비슷한 경기
    if recent_goal_diff_gap <= 2:
        adjustment += 0.01
        reasons.append("최근 득실차 흐름이 비슷함")

    # 모델의 양 팀 승률 차이가 작음
    if win_prob_gap < 0.08:
        adjustment += 0.02
        reasons.append("양 팀 승률 차이가 작아 무승부 가능성 보정")

    # 너무 과하게 올리지 않도록 제한
    adjustment = min(adjustment, 0.06)

    if adjustment > 0:
        draw += adjustment
        a_win -= adjustment / 2
        b_win -= adjustment / 2

    # 음수 방지
    a_win = max(a_win, 0.01)
    draw = max(draw, 0.01)
    b_win = max(b_win, 0.01)

    # 무승부 상한 적용
    a_win, draw, b_win = cap_draw_probability(a_win, draw, b_win)

    adjusted = prediction.copy()
    adjusted["teamAWinProbBeforeAdjust"] = prediction["teamAWinProb"]
    adjusted["drawProbBeforeAdjust"] = prediction["drawProb"]
    adjusted["teamBWinProbBeforeAdjust"] = prediction["teamBWinProb"]

    adjusted["teamAWinProb"] = round(a_win, 4)
    adjusted["drawProb"] = round(draw, 4)
    adjusted["teamBWinProb"] = round(b_win, 4)

    adjusted["drawAdjustment"] = round(adjustment, 4)
    adjusted["drawAdjustmentReasons"] = reasons

    adjusted["predictedResult"] = max(
        [
            ("A_WIN", adjusted["teamAWinProb"]),
            ("DRAW", adjusted["drawProb"]),
            ("B_WIN", adjusted["teamBWinProb"]),
        ],
        key=lambda x: x[1],
    )[0]

    return adjusted


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    adjusted_predictions = [
        adjust_draw_probability(prediction)
        for prediction in predictions
    ]

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(adjusted_predictions, f, ensure_ascii=False, indent=2)

    print(f"보정 완료: {OUTPUT_PATH}")
    print(f"경기 수: {len(adjusted_predictions)}")
    print()

    print("보정 샘플")
    for prediction in adjusted_predictions[:10]:
        print(
            f"{prediction['displayTeamA']} vs {prediction['displayTeamB']} | "
            f"{prediction['teamAWinProb'] * 100:.1f}% / "
            f"{prediction['drawProb'] * 100:.1f}% / "
            f"{prediction['teamBWinProb'] * 100:.1f}% | "
            f"보정: +{prediction['drawAdjustment'] * 100:.1f}%"
        )


if __name__ == "__main__":
    main()
