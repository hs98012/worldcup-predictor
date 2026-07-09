import importlib.util
import json
import sys
import unittest
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


def load_script_module(script_name, module_name):
    spec = importlib.util.spec_from_file_location(
        module_name,
        SCRIPTS_DIR / script_name,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tournament = load_script_module(
    "09_simulate_tournament.py",
    "simulate_tournament",
)


def load_json(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


class TournamentProgressionTest(unittest.TestCase):
    def test_group_stage_finished_starts_round_of_32(self):
        self.assertEqual(tournament.infer_current_tournament_stage(0, 16), "ROUND_OF_32")

    def test_round_of_32_partial_completion_stays_round_of_32(self):
        self.assertEqual(tournament.infer_current_tournament_stage(5, 11), "ROUND_OF_32")

    def test_round_of_16_complete_moves_to_quarter_final(self):
        self.assertEqual(tournament.infer_current_tournament_stage(24, 4), "QUARTER_FINAL")

    def test_penalty_winner_is_inferred_from_next_round(self):
        assigned = {
            96: {
                "date": "2026-07-07",
                "teamA": "Switzerland",
                "teamB": "Colombia",
                "scoreA": 0,
                "scoreB": 0,
            },
            100: {
                "date": "2026-07-11",
                "teamA": "Argentina",
                "teamB": "Switzerland",
            },
        }
        winner, decision = tournament.completed_record_winner(
            assigned[96],
            96,
            assigned,
            {},
            [],
        )

        self.assertEqual(winner, "Switzerland")
        self.assertEqual(decision, "NEXT_ROUND_INFERENCE")

    def test_unresolved_draw_does_not_pick_random_winner(self):
        warnings = []
        record = {
            "date": "2026-07-19",
            "teamA": "Team A",
            "teamB": "Team B",
            "scoreA": 1,
            "scoreB": 1,
        }

        winner, decision = tournament.completed_record_winner(
            record,
            103,
            {103: record},
            {},
            warnings,
        )

        self.assertIsNone(winner)
        self.assertEqual(decision, "UNRESOLVED_DRAW")
        self.assertTrue(warnings)

    def test_eliminated_team_has_zero_future_probabilities(self):
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
        stats["Brazil"].update(
            {
                "displayTeam": "Brazil",
                "team": "Brazil",
                "group": "C",
                "roundOf32Count": 10,
                "roundOf16Count": 10,
            }
        )

        output = tournament.create_progressive_output(
            stats,
            10,
            "QUARTER_FINAL",
            7,
            {"France"},
            {"Brazil": "ROUND_OF_16"},
            10,
        )
        brazil = next(team for team in output if team["team"] == "Brazil")

        self.assertEqual(brazil["winnerProb"], 0.0)
        self.assertEqual(brazil["finalProb"], 0.0)
        self.assertTrue(brazil["isEliminated"])

    def test_only_active_teams_contribute_to_winner_sum(self):
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
        for team, wins in {"France": 6, "Morocco": 4, "Brazil": 0}.items():
            stats[team].update(
                {
                    "displayTeam": team,
                    "team": team,
                    "group": "X",
                    "winnerCount": wins,
                }
            )

        output = tournament.create_progressive_output(
            stats,
            10,
            "QUARTER_FINAL",
            7,
            {"France", "Morocco"},
            {"Brazil": "ROUND_OF_16"},
            10,
        )
        active_winner_sum = sum(
            team["winnerProb"] for team in output if team["isActive"]
        )

        self.assertAlmostEqual(active_winner_sum, 1.0, places=4)

    def test_final_completed_sets_champion_to_100_percent(self):
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
        stats["France"].update(
            {
                "displayTeam": "France",
                "team": "France",
                "group": "I",
                "winnerCount": 1,
            }
        )

        output = tournament.create_progressive_output(
            stats,
            1,
            "COMPLETED",
            0,
            set(),
            {},
            0,
        )

        self.assertEqual(output[0]["winnerProb"], 1.0)
        self.assertEqual(output[0]["currentTournamentStage"], "COMPLETED")
        self.assertEqual(output[0]["remainingTournamentMatches"], 0)

    def test_current_data_assigns_completed_and_pending_records(self):
        teams = load_json(PROJECT_ROOT / "data/processed/teams.json")
        teams_map = {team["team"]: team for team in teams}
        completed = load_json(
            PROJECT_ROOT / "data/processed/worldcup_completed_results.json"
        )
        predictions = load_json(
            PROJECT_ROOT / "data/processed/predictions_adjusted.json"
        )
        group_results, group_by_team = tournament.load_actual_group_results(teams_map)
        completed_records = tournament.completed_knockout_records(
            completed,
            teams_map,
            group_by_team,
        )
        pending_records = tournament.pending_knockout_records(predictions, teams_map)
        assigned, warnings = tournament.assign_knockout_records_to_bracket(
            completed_records,
            pending_records,
            group_results,
        )

        self.assertFalse(warnings)
        self.assertEqual(len([record for record in assigned.values() if record["isCompleted"]]), 24)
        self.assertEqual(len([record for record in assigned.values() if not record["isCompleted"]]), 4)

    def test_current_quarter_final_bracket_paths_are_preserved(self):
        teams = load_json(PROJECT_ROOT / "data/processed/teams.json")
        teams_map = {team["team"]: team for team in teams}
        completed = load_json(
            PROJECT_ROOT / "data/processed/worldcup_completed_results.json"
        )
        predictions = load_json(
            PROJECT_ROOT / "data/processed/predictions_adjusted.json"
        )
        group_results, group_by_team = tournament.load_actual_group_results(teams_map)
        assigned, _ = tournament.assign_knockout_records_to_bracket(
            tournament.completed_knockout_records(completed, teams_map, group_by_team),
            tournament.pending_knockout_records(predictions, teams_map),
            group_results,
        )

        self.assertEqual(
            {assigned[97]["teamA"], assigned[97]["teamB"]},
            {"France", "Morocco"},
        )
        self.assertEqual(tournament.MATCH_CHILDREN[101], (97, 98))


if __name__ == "__main__":
    unittest.main()
