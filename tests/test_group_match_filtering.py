import importlib.util
import sys
import unittest
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from utils.worldcup_simulation import initialize_group_teams, is_valid_group_match


def load_script_module(script_name, module_name):
    spec = importlib.util.spec_from_file_location(
        module_name,
        SCRIPTS_DIR / script_name,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


group_standings = load_script_module(
    "07_calculate_group_standings.py",
    "calculate_group_standings",
)
group_simulation = load_script_module(
    "08_simulate_group_stage.py",
    "simulate_group_stage",
)
worldcup_matches = load_script_module(
    "04_create_worldcup_matches.py",
    "create_worldcup_matches",
)


class GroupMatchFilteringTest(unittest.TestCase):
    def setUp(self):
        self.groups = initialize_group_teams()

    def test_valid_group_match(self):
        match = {
            "group": "A",
            "teamA": "Mexico",
            "teamB": "South Africa",
        }

        self.assertTrue(is_valid_group_match(match, self.groups))

    def test_tournament_match_without_group_is_invalid(self):
        match = {
            "group": None,
            "teamA": "France",
            "teamB": "Morocco",
        }

        self.assertFalse(is_valid_group_match(match, self.groups))

    def test_unknown_group_is_invalid(self):
        match = {
            "group": "Z",
            "teamA": "Mexico",
            "teamB": "South Africa",
        }

        self.assertFalse(is_valid_group_match(match, self.groups))

    def test_no_remaining_group_matches_uses_actual_standings_shape(self):
        actual_standings = [
            {
                "group": "A",
                "teams": [
                    {
                        "rank": 1,
                        "team": "Mexico",
                        "played": 3,
                        "wins": 3,
                        "draws": 0,
                        "losses": 0,
                        "goalsFor": 5,
                        "goalsAgainst": 1,
                        "goalDifference": 4,
                        "points": 9,
                    },
                    {
                        "rank": 2,
                        "team": "South Africa",
                        "played": 3,
                        "wins": 2,
                        "draws": 0,
                        "losses": 1,
                        "goalsFor": 4,
                        "goalsAgainst": 3,
                        "goalDifference": 1,
                        "points": 6,
                    },
                    {
                        "rank": 3,
                        "team": "South Korea",
                        "played": 3,
                        "wins": 1,
                        "draws": 0,
                        "losses": 2,
                        "goalsFor": 3,
                        "goalsAgainst": 4,
                        "goalDifference": -1,
                        "points": 3,
                    },
                    {
                        "rank": 4,
                        "team": "Czech Republic",
                        "played": 3,
                        "wins": 0,
                        "draws": 0,
                        "losses": 3,
                        "goalsFor": 1,
                        "goalsAgainst": 5,
                        "goalDifference": -4,
                        "points": 0,
                    },
                ],
            }
        ]

        output = group_standings.convert_actual_standings(
            actual_standings,
            defaultdict(list),
        )

        self.assertEqual(output[0]["remainingGroupMatches"], 0)
        self.assertEqual(output[0]["standings"][0]["displayTeam"], "Mexico")
        self.assertEqual(output[0]["standings"][0]["goalDiff"], 4)
        self.assertEqual(
            output[0]["standings"][0]["qualifiedStatus"],
            "DIRECT_ADVANCE",
        )

    def test_mixed_group_and_tournament_matches_only_keeps_group_matches(self):
        matches = [
            {
                "group": "A",
                "teamA": "Mexico",
                "teamB": "South Africa",
            },
            {
                "group": None,
                "teamA": "France",
                "teamB": "Morocco",
            },
        ]

        valid_matches = [
            match for match in matches if is_valid_group_match(match, self.groups)
        ]

        self.assertEqual(valid_matches, [matches[0]])

    def test_mixed_group_and_tournament_matches_get_correct_stages(self):
        group_fixture = {
            "tournament": "FIFA World Cup",
            "date": "2026-06-11",
            "home_team": "Mexico",
            "away_team": "South Africa",
        }
        knockout_fixture = {
            "tournament": "FIFA World Cup",
            "date": "2026-07-09",
            "home_team": "France",
            "away_team": "Morocco",
        }

        self.assertEqual(
            worldcup_matches.determine_stage(group_fixture, "A"),
            "GROUP",
        )
        self.assertEqual(
            worldcup_matches.determine_stage(knockout_fixture, None),
            "KNOCKOUT",
        )

    def test_final_group_simulation_marks_zero_remaining_matches(self):
        actual_standings = [
            {
                "group": "A",
                "teams": [
                    {
                        "rank": rank,
                        "team": team,
                        "points": 4 - rank,
                        "goalDifference": 0,
                        "goalsFor": 0,
                        "wins": 0,
                    }
                    for rank, team in enumerate(
                        [
                            "Mexico",
                            "South Africa",
                            "South Korea",
                            "Czech Republic",
                        ],
                        start=1,
                    )
                ],
            }
        ]

        output = group_simulation.create_final_group_simulation(actual_standings)

        self.assertEqual(output[0]["simulationCount"], 0)
        self.assertEqual(output[0]["remainingGroupMatches"], 0)
        self.assertEqual(output[0]["teams"][0]["rank1Prob"], 1.0)
        self.assertEqual(output[0]["teams"][3]["rank4Prob"], 1.0)


if __name__ == "__main__":
    unittest.main()
