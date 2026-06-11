import json
from collections import Counter, defaultdict, deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPLETED_PATH = PROJECT_ROOT / "data/processed/completed_matches.csv"
MATCHES_PATH = PROJECT_ROOT / "data/processed/matches.json"
TEAM_STRENGTH_PATH = (
    PROJECT_ROOT / "data/external/team_strength_2026.csv"
)

MODEL_PATH = PROJECT_ROOT / "models/sklearn_match_result_model.joblib"
PREDICTIONS_PATH = PROJECT_ROOT / "data/processed/predictions_sklearn.json"
METRICS_PATH = PROJECT_ROOT / "data/processed/model_metrics.json"
FEATURE_COLUMNS_PATH = (
    PROJECT_ROOT / "data/processed/model_feature_columns.json"
)
DRAW_CALIBRATION_PATH = (
    PROJECT_ROOT / "data/processed/draw_calibration.json"
)

INITIAL_ELO = 1500
BASE_K = 30
TRAIN_RATIO = 0.8
RANDOM_STATE = 42
CLASS_LABELS = ["A_WIN", "B_WIN", "DRAW"]
# SQUAD_ADJUSTMENT_ALPHA = 0.35
SQUAD_ADJUSTMENT_ALPHA = 0.55

BASE_FEATURE_COLUMNS = [
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

OPPONENT_ADJUSTED_FEATURE_COLUMNS = [
    "team_a_adj_points5",
    "team_b_adj_points5",
    "adj_points_diff5",
    "team_a_adj_goal_diff5",
    "team_b_adj_goal_diff5",
    "adj_goal_diff_gap5",
    "team_a_avg_opponent_elo5",
    "team_b_avg_opponent_elo5",
    "avg_opponent_elo_diff5",
    "team_a_adj_points10",
    "team_b_adj_points10",
    "adj_points_diff10",
]

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + OPPONENT_ADJUSTED_FEATURE_COLUMNS

HOST_TEAMS = {"Mexico", "Canada", "United States"}
OPPONENT_STRENGTH_MIN = 0.75
OPPONENT_STRENGTH_MAX = 1.25
ADJ_POINTS_DIFF5_CAP = 8.0
ADJ_GOAL_DIFF5_CAP = 10.0
ADJ_POINTS_DIFF10_CAP = 12.0


def load_matches():
    df = pd.read_csv(COMPLETED_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["date", "home_score", "away_score"])
    df = df.sort_values("date", kind="stable").reset_index(drop=True)

    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # Modern matches are more representative of the 2026 prediction target.
    return df[df["date"].dt.year >= 2000].reset_index(drop=True)


def to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def get_actual_label(score_a, score_b):
    if score_a > score_b:
        return "A_WIN"
    if score_a == score_b:
        return "DRAW"
    return "B_WIN"


def get_actual_score(score_a, score_b):
    if score_a > score_b:
        return 1.0
    if score_a == score_b:
        return 0.5
    return 0.0


def load_team_strength():
    if not TEAM_STRENGTH_PATH.exists():
        print(f"선수단 전력 CSV 없음: {TEAM_STRENGTH_PATH}")
        return {}

    df = pd.read_csv(TEAM_STRENGTH_PATH)
    required_columns = {"team", "team_strength_score"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "team_strength_2026.csv에 필요한 컬럼이 없습니다: "
            f"{missing_columns}"
        )

    valid_rows = df.dropna(subset=["team", "team_strength_score"])
    return {
        str(row["team"]): float(row["team_strength_score"])
        for _, row in valid_rows.iterrows()
    }


def apply_squad_strength_adjustment(
    team_a_win,
    draw,
    team_b_win,
    team_a,
    team_b,
    team_strength,
):
    strength_a = team_strength.get(team_a)
    strength_b = team_strength.get(team_b)

    if strength_a is None or strength_b is None:
        return team_a_win, draw, team_b_win, 0.0

    strength_diff = strength_a - strength_b
    team_a_win *= np.exp(SQUAD_ADJUSTMENT_ALPHA * strength_diff)
    team_b_win *= np.exp(-SQUAD_ADJUSTMENT_ALPHA * strength_diff)
    total = team_a_win + draw + team_b_win

    if total == 0:
        return 1 / 3, 1 / 3, 1 / 3, strength_diff

    return (
        team_a_win / total,
        draw / total,
        team_b_win / total,
        strength_diff,
    )


def get_expected_score(team_elo, opponent_elo):
    return 1 / (1 + 10 ** ((opponent_elo - team_elo) / 400))


def get_time_weight(match_date, current_year=2026):
    years_ago = current_year - match_date.year
    if years_ago <= 0:
        return 1.0
    if years_ago == 1:
        return 0.9
    if years_ago == 2:
        return 0.75
    if years_ago <= 5:
        return 0.55
    if years_ago <= 10:
        return 0.3
    return 0.1


def get_tournament_importance(tournament):
    name = str(tournament).lower()
    if "fifa world cup" in name and "qualification" not in name:
        return 1.0
    if any(
        keyword in name
        for keyword in (
            "uefa euro",
            "copa américa",
            "african cup of nations",
            "afc asian cup",
            "concacaf gold cup",
        )
    ) and "qualification" not in name:
        return 0.85
    if "qualification" in name or "nations league" in name:
        return 0.65
    if "friendly" in name:
        return 0.25
    return 0.5


def get_tournament_k_factor(tournament):
    importance = get_tournament_importance(tournament)
    if importance >= 1.0:
        return 1.15
    if importance >= 0.85:
        return 0.9
    if importance >= 0.65:
        return 0.7
    if importance <= 0.25:
        return 0.35
    return 0.55


def get_goal_difference_multiplier(score_a, score_b):
    goal_difference = abs(score_a - score_b)
    if goal_difference <= 1:
        return 1.0
    return min(1.5, 1.0 + 0.15 * (goal_difference - 1))


def summarize_recent_form(history, window):
    recent = list(history)[-window:]
    if not recent:
        return {
            "points": 0,
            "win_rate": 0,
            "goal_diff": 0,
            "avg_goals_for": 0,
            "avg_goals_against": 0,
            "adjusted_points": 0,
            "adjusted_goal_diff": 0,
            "avg_opponent_elo": INITIAL_ELO,
        }

    match_count = len(recent)
    points = sum(item["points"] for item in recent)
    wins = sum(item["points"] == 3 for item in recent)
    goals_for = sum(item["goals_for"] for item in recent)
    goals_against = sum(item["goals_against"] for item in recent)
    adjusted_points = sum(item["adjusted_points"] for item in recent)
    adjusted_goal_diff = sum(
        item["adjusted_goal_diff"] for item in recent
    )
    avg_opponent_elo = sum(
        item["opponent_elo"] for item in recent
    ) / match_count
    return {
        "points": points,
        "win_rate": wins / match_count,
        "goal_diff": goals_for - goals_against,
        "avg_goals_for": goals_for / match_count,
        "avg_goals_against": goals_against / match_count,
        "adjusted_points": adjusted_points,
        "adjusted_goal_diff": adjusted_goal_diff,
        "avg_opponent_elo": avg_opponent_elo,
    }


def build_feature_row(
    team_a,
    team_b,
    elo,
    recent_history,
    is_neutral,
    team_a_home_advantage,
    tournament="Friendly",
):
    team_a_elo = elo[team_a]
    team_b_elo = elo[team_b]
    form_a5 = summarize_recent_form(recent_history[team_a], 5)
    form_b5 = summarize_recent_form(recent_history[team_b], 5)
    form_a10 = summarize_recent_form(recent_history[team_a], 10)
    form_b10 = summarize_recent_form(recent_history[team_b], 10)

    return {
        "team_a_elo": team_a_elo,
        "team_b_elo": team_b_elo,
        "elo_diff": team_a_elo - team_b_elo,
        "team_a_recent_points5": form_a5["points"],
        "team_b_recent_points5": form_b5["points"],
        "recent_points_diff": form_a5["points"] - form_b5["points"],
        "team_a_win_rate5": form_a5["win_rate"],
        "team_b_win_rate5": form_b5["win_rate"],
        "win_rate_diff5": form_a5["win_rate"] - form_b5["win_rate"],
        "team_a_avg_goals_for5": form_a5["avg_goals_for"],
        "team_b_avg_goals_for5": form_b5["avg_goals_for"],
        "avg_goals_for_diff5": (
            form_a5["avg_goals_for"] - form_b5["avg_goals_for"]
        ),
        "team_a_avg_goals_against5": form_a5["avg_goals_against"],
        "team_b_avg_goals_against5": form_b5["avg_goals_against"],
        "avg_goals_against_diff5": (
            form_a5["avg_goals_against"] - form_b5["avg_goals_against"]
        ),
        "team_a_recent_goal_diff5": form_a5["goal_diff"],
        "team_b_recent_goal_diff5": form_b5["goal_diff"],
        "recent_goal_diff_gap": (
            form_a5["goal_diff"] - form_b5["goal_diff"]
        ),
        "team_a_win_rate10": form_a10["win_rate"],
        "team_b_win_rate10": form_b10["win_rate"],
        "win_rate_diff10": form_a10["win_rate"] - form_b10["win_rate"],
        "team_a_recent_goal_diff10": form_a10["goal_diff"],
        "team_b_recent_goal_diff10": form_b10["goal_diff"],
        "recent_goal_diff_gap10": (
            form_a10["goal_diff"] - form_b10["goal_diff"]
        ),
        "team_a_adj_points5": form_a5["adjusted_points"],
        "team_b_adj_points5": form_b5["adjusted_points"],
        "adj_points_diff5": float(
            np.clip(
                form_a5["adjusted_points"]
                - form_b5["adjusted_points"],
                -ADJ_POINTS_DIFF5_CAP,
                ADJ_POINTS_DIFF5_CAP,
            )
        ),
        "team_a_adj_goal_diff5": form_a5["adjusted_goal_diff"],
        "team_b_adj_goal_diff5": form_b5["adjusted_goal_diff"],
        "adj_goal_diff_gap5": float(
            np.clip(
                form_a5["adjusted_goal_diff"]
                - form_b5["adjusted_goal_diff"],
                -ADJ_GOAL_DIFF5_CAP,
                ADJ_GOAL_DIFF5_CAP,
            )
        ),
        "team_a_avg_opponent_elo5": form_a5["avg_opponent_elo"],
        "team_b_avg_opponent_elo5": form_b5["avg_opponent_elo"],
        "avg_opponent_elo_diff5": (
            form_a5["avg_opponent_elo"]
            - form_b5["avg_opponent_elo"]
        ),
        "team_a_adj_points10": form_a10["adjusted_points"],
        "team_b_adj_points10": form_b10["adjusted_points"],
        "adj_points_diff10": float(
            np.clip(
                form_a10["adjusted_points"]
                - form_b10["adjusted_points"],
                -ADJ_POINTS_DIFF10_CAP,
                ADJ_POINTS_DIFF10_CAP,
            )
        ),
        "is_neutral": int(is_neutral),
        "team_a_home_advantage": int(team_a_home_advantage),
        "tournament_importance": get_tournament_importance(tournament),
    }


def update_elo(
    elo,
    team_a,
    team_b,
    score_a,
    score_b,
    match_date,
    tournament,
):
    team_a_elo = elo[team_a]
    team_b_elo = elo[team_b]
    expected_a = get_expected_score(team_a_elo, team_b_elo)
    actual_a = get_actual_score(score_a, score_b)
    k = (
        BASE_K
        * get_time_weight(match_date)
        * get_tournament_k_factor(tournament)
        * get_goal_difference_multiplier(score_a, score_b)
    )
    elo[team_a] = team_a_elo + k * (actual_a - expected_a)
    elo[team_b] = team_b_elo + k * ((1 - actual_a) - (1 - expected_a))


def update_recent_history(
    recent_history,
    team_a,
    team_b,
    score_a,
    score_b,
    team_a_elo,
    team_b_elo,
):
    if score_a > score_b:
        points_a, points_b = 3, 0
    elif score_a == score_b:
        points_a, points_b = 1, 1
    else:
        points_a, points_b = 0, 3

    strength_a = float(
        np.clip(
            team_b_elo / team_a_elo,
            OPPONENT_STRENGTH_MIN,
            OPPONENT_STRENGTH_MAX,
        )
    )
    strength_b = float(
        np.clip(
            team_a_elo / team_b_elo,
            OPPONENT_STRENGTH_MIN,
            OPPONENT_STRENGTH_MAX,
        )
    )
    recent_history[team_a].append(
        {
            "points": points_a,
            "goals_for": score_a,
            "goals_against": score_b,
            "opponent_elo": team_b_elo,
            "adjusted_points": points_a * strength_a,
            "adjusted_goal_diff": (score_a - score_b) * strength_a,
        }
    )
    recent_history[team_b].append(
        {
            "points": points_b,
            "goals_for": score_b,
            "goals_against": score_a,
            "opponent_elo": team_a_elo,
            "adjusted_points": points_b * strength_b,
            "adjusted_goal_diff": (score_b - score_a) * strength_b,
        }
    )


def build_training_dataset(df):
    elo = defaultdict(lambda: INITIAL_ELO)
    recent_history = defaultdict(lambda: deque(maxlen=10))
    rows = []
    labels = []
    dates = []

    for _, row in df.iterrows():
        team_a = row["home_team"]
        team_b = row["away_team"]
        score_a = row["home_score"]
        score_b = row["away_score"]
        match_date = row["date"]
        is_neutral = to_bool(row["neutral"])

        rows.append(
            build_feature_row(
                team_a=team_a,
                team_b=team_b,
                elo=elo,
                recent_history=recent_history,
                is_neutral=is_neutral,
                team_a_home_advantage=not is_neutral,
                tournament=row["tournament"],
            )
        )
        labels.append(get_actual_label(score_a, score_b))
        dates.append(match_date)

        team_a_elo = elo[team_a]
        team_b_elo = elo[team_b]
        update_elo(
            elo,
            team_a,
            team_b,
            score_a,
            score_b,
            match_date,
            row["tournament"],
        )
        update_recent_history(
            recent_history,
            team_a,
            team_b,
            score_a,
            score_b,
            team_a_elo,
            team_b_elo,
        )

    return (
        pd.DataFrame(rows, columns=FEATURE_COLUMNS),
        np.asarray(labels),
        pd.Series(dates),
        elo,
        recent_history,
    )


def class_distribution(labels):
    counts = Counter(labels)
    total = len(labels)
    return {
        label: round(counts.get(label, 0) / total, 4)
        for label in CLASS_LABELS
    }


def align_probabilities(probabilities, model_classes):
    aligned = np.full((len(probabilities), len(CLASS_LABELS)), 1e-15)
    class_indexes = {label: index for index, label in enumerate(model_classes)}
    for output_index, label in enumerate(CLASS_LABELS):
        if label in class_indexes:
            aligned[:, output_index] = probabilities[
                :, class_indexes[label]
            ]
    return aligned / aligned.sum(axis=1, keepdims=True)


def evaluate_predictions(name, y_true, y_pred, y_proba):
    draw_precision, draw_recall, _, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=["DRAW"],
        zero_division=0,
    )
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "logLoss": round(
            log_loss(y_true, y_proba, labels=CLASS_LABELS),
            4,
        ),
        "confusionMatrix": confusion_matrix(
            y_true,
            y_pred,
            labels=CLASS_LABELS,
        ).tolist(),
        "drawRecall": round(float(draw_recall[0]), 4),
        "drawPrecision": round(float(draw_precision[0]), 4),
        "macroF1": round(
            f1_score(
                y_true,
                y_pred,
                labels=CLASS_LABELS,
                average="macro",
                zero_division=0,
            ),
            4,
        ),
    }


def get_draw_candidate_mask(features, params):
    mask = (
        features["elo_diff"].abs().le(params["eloThreshold"])
        & features["win_rate_diff5"].abs().le(
            params["winRateThreshold"]
        )
        & features["avg_goals_for_diff5"].abs().le(
            params["avgGoalsForThreshold"]
        )
        & features["avg_goals_against_diff5"].abs().le(
            params["avgGoalsAgainstThreshold"]
        )
        & features["recent_goal_diff_gap"].abs().le(
            params["recentGoalDiffThreshold"]
        )
    )
    if params.get("requireNeutral", True):
        mask &= features["is_neutral"].eq(1)
    return mask.to_numpy()


def apply_draw_calibration(probabilities, features, params):
    adjusted = np.asarray(probabilities, dtype=float).copy()
    candidate_mask = get_draw_candidate_mask(features, params)
    draw_index = CLASS_LABELS.index("DRAW")
    adjusted[candidate_mask, draw_index] += params["drawBoost"]
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    return adjusted, candidate_mask


def evaluate_draw_calibration(y_true, probabilities, name):
    predictions = np.asarray(CLASS_LABELS)[probabilities.argmax(axis=1)]
    return evaluate_predictions(
        name,
        y_true,
        predictions,
        probabilities,
    )


def run_draw_calibration_experiment(X_test, y_test, base_probabilities):
    base_metrics = evaluate_draw_calibration(
        y_test,
        base_probabilities,
        "draw_calibration_base",
    )
    tested_candidates = []

    for elo_threshold in (50, 75, 100, 125):
        for draw_boost in (0.02, 0.04, 0.06, 0.08):
            for win_rate_threshold in (0.15, 0.2, 0.25):
                for goal_threshold in (0.35, 0.5):
                    params = {
                        "eloThreshold": elo_threshold,
                        "drawBoost": draw_boost,
                        "winRateThreshold": win_rate_threshold,
                        "avgGoalsForThreshold": goal_threshold,
                        "avgGoalsAgainstThreshold": goal_threshold,
                        "recentGoalDiffThreshold": 3,
                        "requireNeutral": True,
                    }
                    adjusted, candidate_mask = apply_draw_calibration(
                        base_probabilities,
                        X_test,
                        params,
                    )
                    metrics = evaluate_draw_calibration(
                        y_test,
                        adjusted,
                        "draw_calibration_candidate",
                    )
                    accuracy_drop = (
                        base_metrics["accuracy"] - metrics["accuracy"]
                    )
                    log_loss_increase = (
                        metrics["logLoss"] - base_metrics["logLoss"]
                    )
                    eligible = (
                        metrics["drawRecall"] > base_metrics["drawRecall"]
                        and accuracy_drop <= 0.01
                        and log_loss_increase <= 0.02
                    )
                    tested_candidates.append(
                        {
                            "params": params,
                            "adjustedMatchCount": int(
                                candidate_mask.sum()
                            ),
                            "accuracy": metrics["accuracy"],
                            "logLoss": metrics["logLoss"],
                            "drawRecall": metrics["drawRecall"],
                            "drawPrecision": metrics["drawPrecision"],
                            "macroF1": metrics["macroF1"],
                            "confusionMatrix": metrics[
                                "confusionMatrix"
                            ],
                            "accuracyDrop": round(accuracy_drop, 4),
                            "logLossIncrease": round(
                                log_loss_increase,
                                4,
                            ),
                            "eligible": eligible,
                        }
                    )

    eligible_candidates = [
        candidate
        for candidate in tested_candidates
        if candidate["eligible"]
    ]
    if not eligible_candidates:
        return {
            "enabled": False,
            "selectionReason": (
                "DRAW recall 개선, Accuracy 하락 0.01 이하, "
                "Log Loss 상승 0.02 이하 조건을 모두 만족한 후보가 없음"
            ),
            "baseMetrics": base_metrics,
            "calibratedMetrics": base_metrics,
            "selectedParams": None,
            "testedCandidates": tested_candidates,
        }

    selected = min(
        eligible_candidates,
        key=lambda candidate: (
            -candidate["drawRecall"],
            candidate["logLoss"],
            -candidate["drawPrecision"],
            -candidate["accuracy"],
        ),
    )
    calibrated_metrics = {
        key: selected[key]
        for key in (
            "accuracy",
            "logLoss",
            "drawRecall",
            "drawPrecision",
            "macroF1",
            "confusionMatrix",
        )
    }
    return {
        "enabled": True,
        "selectionReason": (
            f"DRAW recall이 {base_metrics['drawRecall']:.4f}에서 "
            f"{selected['drawRecall']:.4f}로 개선되었고, Accuracy 하락 "
            f"{selected['accuracyDrop']:.4f}, Log Loss 변화 "
            f"{selected['logLossIncrease']:+.4f}로 허용 범위 이내여서 적용"
        ),
        "baseMetrics": base_metrics,
        "calibratedMetrics": calibrated_metrics,
        "selectedParams": selected["params"],
        "testedCandidates": tested_candidates,
    }


def evaluate_baselines(X_train, y_train, X_test, y_test):
    counts = Counter(y_train)
    majority_class = counts.most_common(1)[0][0]
    train_distribution = np.array(
        [counts.get(label, 0) / len(y_train) for label in CLASS_LABELS]
    )
    majority_predictions = np.full(len(y_test), majority_class)
    majority_probabilities = np.tile(train_distribution, (len(y_test), 1))

    elo_diff = X_test["elo_diff"].to_numpy()
    draw_probability = 0.28 * np.exp(-np.abs(elo_diff) / 600)
    team_a_strength = 1 / (1 + np.power(10, -elo_diff / 400))
    remaining = 1 - draw_probability
    elo_probabilities = np.column_stack(
        (
            remaining * team_a_strength,
            remaining * (1 - team_a_strength),
            draw_probability,
        )
    )
    elo_predictions = np.where(
        np.abs(elo_diff) < 50,
        "DRAW",
        np.where(elo_diff > 0, "A_WIN", "B_WIN"),
    )

    return [
        evaluate_predictions(
            "majority_class_baseline",
            y_test,
            majority_predictions,
            majority_probabilities,
        ),
        evaluate_predictions(
            "elo_rule_baseline",
            y_test,
            elo_predictions,
            elo_probabilities,
        ),
    ]


def build_model_factories():
    base_models = {
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=RANDOM_STATE,
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=200,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=RANDOM_STATE,
        ),
    }
    factories = dict(base_models)
    calibration_cv = TimeSeriesSplit(n_splits=3)
    for name in (
        "RandomForestClassifier",
        "GradientBoostingClassifier",
        "HistGradientBoostingClassifier",
    ):
        factories[f"{name}_calibrated"] = CalibratedClassifierCV(
            estimator=clone(base_models[name]),
            method="sigmoid",
            cv=calibration_cv,
            n_jobs=1,
        )
    return factories


def select_model(model_metrics, baseline_metrics):
    baseline_accuracy = max(item["accuracy"] for item in baseline_metrics)
    eligible = [
        item
        for item in model_metrics
        if item["accuracy"] >= baseline_accuracy
    ]
    if not eligible:
        eligible = model_metrics

    best_accuracy = max(item["accuracy"] for item in eligible)
    comparable = [
        item
        for item in eligible
        if best_accuracy - item["accuracy"] <= 0.01
    ]
    selected = min(
        comparable,
        key=lambda item: (item["logLoss"], -item["accuracy"]),
    )
    reason = (
        f"최고 모델 accuracy {best_accuracy:.4f}와 0.01 이내인 후보 중 "
        f"logLoss가 가장 낮고, 최고 baseline accuracy "
        f"{baseline_accuracy:.4f} 이상이어서 선택"
    )
    return selected["model"], reason


def compare_model_candidates(
    X_train,
    y_train,
    X_test,
    y_test,
    training_distribution,
    experiment_label,
):
    factories = build_model_factories()
    model_comparisons = []
    test_predictions = {}
    test_probabilities = {}

    for name, estimator in factories.items():
        print(f"- {experiment_label} / {name} 학습 및 평가")
        candidate = clone(estimator)
        candidate.fit(X_train, y_train)
        y_pred = candidate.predict(X_test)
        test_predictions[name] = y_pred
        y_proba = align_probabilities(
            candidate.predict_proba(X_test),
            candidate.classes_,
        )
        test_probabilities[name] = y_proba
        comparison = evaluate_predictions(name, y_test, y_pred, y_proba)
        comparison.update(
            {
                "trainSize": int(len(X_train)),
                "testSize": int(len(X_test)),
                "classDistribution": training_distribution,
            }
        )
        model_comparisons.append(comparison)

    return (
        factories,
        model_comparisons,
        test_predictions,
        test_probabilities,
    )


def train_model(X, y, dates):
    split_index = int(len(X) * TRAIN_RATIO)
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]

    baseline_metrics = evaluate_baselines(
        X_train,
        y_train,
        X_test,
        y_test,
    )
    training_distribution = class_distribution(y_train)

    (
        base_factories,
        base_comparisons,
        base_predictions,
        base_probabilities,
    ) = compare_model_candidates(
        X_train[BASE_FEATURE_COLUMNS],
        y_train,
        X_test[BASE_FEATURE_COLUMNS],
        y_test,
        training_distribution,
        "기존 피처",
    )
    base_selected_name, base_selection_reason = select_model(
        base_comparisons,
        baseline_metrics,
    )
    base_selected_metrics = next(
        item
        for item in base_comparisons
        if item["model"] == base_selected_name
    )

    (
        adjusted_factories,
        adjusted_comparisons,
        adjusted_predictions,
        adjusted_probabilities,
    ) = compare_model_candidates(
        X_train[FEATURE_COLUMNS],
        y_train,
        X_test[FEATURE_COLUMNS],
        y_test,
        training_distribution,
        "상대 강도 보정 피처",
    )
    adjusted_selected_name, adjusted_selection_reason = select_model(
        adjusted_comparisons,
        baseline_metrics,
    )
    adjusted_selected_metrics = next(
        item
        for item in adjusted_comparisons
        if item["model"] == adjusted_selected_name
    )

    accuracy_drop = (
        base_selected_metrics["accuracy"]
        - adjusted_selected_metrics["accuracy"]
    )
    log_loss_increase = (
        adjusted_selected_metrics["logLoss"]
        - base_selected_metrics["logLoss"]
    )
    adjusted_features_enabled = (
        accuracy_drop < 0.01 and log_loss_increase < 0.02
    )

    if adjusted_features_enabled:
        selected_name = adjusted_selected_name
        selected_metrics = adjusted_selected_metrics
        selected_columns = FEATURE_COLUMNS
        factories = adjusted_factories
        test_predictions = adjusted_predictions
        test_probabilities = adjusted_probabilities
        selection_reason = (
            f"{adjusted_selection_reason}. 상대 강도 보정 피처가 기존 대비 "
            f"Accuracy 변화 {-accuracy_drop:+.4f}, Log Loss 변화 "
            f"{log_loss_increase:+.4f}로 보호 기준 이내여서 사용"
        )
    else:
        selected_name = base_selected_name
        selected_metrics = base_selected_metrics
        selected_columns = BASE_FEATURE_COLUMNS
        factories = base_factories
        test_predictions = base_predictions
        test_probabilities = base_probabilities
        selection_reason = (
            f"{base_selection_reason}. 상대 강도 보정 피처의 Accuracy 하락 "
            f"{accuracy_drop:.4f} 또는 Log Loss 상승 "
            f"{log_loss_increase:.4f}이 보호 기준을 넘어 기존 피처 유지"
        )

    draw_calibration = run_draw_calibration_experiment(
        X_test[selected_columns],
        y_test,
        test_probabilities[selected_name],
    )

    # Evaluation uses the held-out future block; production model then learns
    # from every completed match before predicting the 2026 fixtures.
    selected_model = clone(factories[selected_name])
    selected_model.fit(X[selected_columns], y)

    metrics = {
        "selectedModel": selected_name,
        "selectionReason": selection_reason,
        "trainSize": int(len(X_train)),
        "testSize": int(len(X_test)),
        "testStartDate": str(dates.iloc[split_index].date()),
        "testEndDate": str(dates.iloc[-1].date()),
        "classDistribution": training_distribution,
        "baselineMetrics": baseline_metrics,
        "modelComparisons": (
            adjusted_comparisons
            if adjusted_features_enabled
            else base_comparisons
        ),
        "accuracy": selected_metrics["accuracy"],
        "logLoss": selected_metrics["logLoss"],
        "confusionMatrix": selected_metrics["confusionMatrix"],
        "classes": CLASS_LABELS,
        "featureColumns": selected_columns,
        "opponentAdjustedFeaturesEnabled": adjusted_features_enabled,
        "featureExperiment": {
            "guardrails": {
                "maxAccuracyDrop": 0.01,
                "maxLogLossIncrease": 0.02,
            },
            "base": {
                "selectedModel": base_selected_name,
                "selectionReason": base_selection_reason,
                "accuracy": base_selected_metrics["accuracy"],
                "logLoss": base_selected_metrics["logLoss"],
                "featureColumns": BASE_FEATURE_COLUMNS,
                "modelComparisons": base_comparisons,
            },
            "opponentAdjusted": {
                "selectedModel": adjusted_selected_name,
                "selectionReason": adjusted_selection_reason,
                "accuracy": adjusted_selected_metrics["accuracy"],
                "logLoss": adjusted_selected_metrics["logLoss"],
                "accuracyDropVsBase": round(accuracy_drop, 4),
                "logLossIncreaseVsBase": round(
                    log_loss_increase,
                    4,
                ),
                "featureColumns": FEATURE_COLUMNS,
                "modelComparisons": adjusted_comparisons,
            },
        },
        "classificationReport": classification_report(
            y_test,
            test_predictions[selected_name],
            labels=CLASS_LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "drawCalibrationEnabled": draw_calibration["enabled"],
        "drawCalibrationSummary": {
            "selectionReason": draw_calibration["selectionReason"],
            "baseMetrics": draw_calibration["baseMetrics"],
            "calibratedMetrics": draw_calibration[
                "calibratedMetrics"
            ],
            "selectedParams": draw_calibration["selectedParams"],
        },
    }
    return selected_model, metrics, draw_calibration


def predict_worldcup_matches(model, elo, recent_history):
    with open(MATCHES_PATH, "r", encoding="utf-8") as file:
        matches = json.load(file)

    predictions = []
    team_strength = load_team_strength()
    feature_columns = list(
        getattr(model, "feature_names_in_", FEATURE_COLUMNS)
    )
    for match in matches:
        team_a = match["teamA"]
        team_b = match["teamB"]
        feature_row = build_feature_row(
            team_a=team_a,
            team_b=team_b,
            elo=elo,
            recent_history=recent_history,
            is_neutral=True,
            team_a_home_advantage=team_a in HOST_TEAMS,
            tournament="FIFA World Cup",
        )
        X_match = pd.DataFrame([feature_row], columns=feature_columns)
        proba = model.predict_proba(X_match)[0]
        class_to_prob = dict(zip(model.classes_, proba))
        team_a_win = float(class_to_prob.get("A_WIN", 0))
        draw = float(class_to_prob.get("DRAW", 0))
        team_b_win = float(class_to_prob.get("B_WIN", 0))
        (
            team_a_win,
            draw,
            team_b_win,
            squad_strength_diff,
        ) = apply_squad_strength_adjustment(
            team_a_win,
            draw,
            team_b_win,
            team_a,
            team_b,
            team_strength,
        )
        predictions.append(
            {
                "matchId": match["matchId"],
                "stage": match["stage"],
                "group": match["group"],
                "date": match["date"],
                "displayTeamA": match.get("displayTeamA", team_a),
                "displayTeamB": match.get("displayTeamB", team_b),
                "teamA": team_a,
                "teamB": team_b,
                "teamAWinProb": round(team_a_win, 4),
                "drawProb": round(draw, 4),
                "teamBWinProb": round(team_b_win, 4),
                "squadStrengthDiff": round(squad_strength_diff, 4),
                "predictedResult": max(
                    [
                        ("A_WIN", team_a_win),
                        ("DRAW", draw),
                        ("B_WIN", team_b_win),
                    ],
                    key=lambda item: item[1],
                )[0],
                "features": {
                    key: round(value, 4)
                    if isinstance(value, float)
                    else value
                    for key, value in feature_row.items()
                },
            }
        )
    return predictions


def main():
    df = load_matches()
    print(f"완료 경기 학습 데이터 생성 중: {len(df)}경기")
    X, y, dates, elo, recent_history = build_training_dataset(df)

    print("baseline 및 모델 후보 비교 중...")
    model, metrics, draw_calibration = train_model(X, y, dates)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
        file.write("\n")
    with FEATURE_COLUMNS_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            metrics["featureColumns"],
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")
    with DRAW_CALIBRATION_PATH.open("w", encoding="utf-8") as file:
        json.dump(draw_calibration, file, ensure_ascii=False, indent=2)
        file.write("\n")

    predictions = predict_worldcup_matches(model, elo, recent_history)
    with PREDICTIONS_PATH.open("w", encoding="utf-8") as file:
        json.dump(predictions, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print()
    print(f"선택 모델: {metrics['selectedModel']}")
    print(f"선택 이유: {metrics['selectionReason']}")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Log Loss: {metrics['logLoss']}")
    print(f"모델 저장: {MODEL_PATH}")
    print(f"성능 저장: {METRICS_PATH}")
    print(f"피처 목록 저장: {FEATURE_COLUMNS_PATH}")
    print(f"무승부 보정 설정 저장: {DRAW_CALIBRATION_PATH}")
    print(
        "무승부 보정: "
        f"{'적용' if draw_calibration['enabled'] else '미적용'}"
    )


if __name__ == "__main__":
    main()
