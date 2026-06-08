import json
from pathlib import Path

TEAMS_PATH = Path("data/processed/teams.json")
OUTPUT_PATH = Path("data/processed/matches.json")


# 화면에 보여줄 팀명과 results.csv/teams.json에 들어있는 팀명이 다를 수 있어서 보정
ALIASES = {
    "USA": ["United States", "USA"],
    "Türkiye": ["Turkey", "Türkiye"],
    "Czechia": ["Czech Republic", "Czechia"],
    "Congo DR": ["DR Congo", "Congo DR", "Democratic Republic of the Congo"],
    "Ivory Coast": ["Ivory Coast", "Côte d'Ivoire"],
    "Curacao": ["Curaçao", "Curacao"],
    "Cape Verde": ["Cape Verde", "Cabo Verde"],
    "South Korea": ["South Korea", "Korea Republic"],
    "Bosnia and Herzegovina": ["Bosnia and Herzegovina", "Bosnia-Herzegovina"],
}


FIXTURES = [
    # Group A
    ("2026-06-11", "A", "Mexico", "South Africa"),
    ("2026-06-11", "A", "South Korea", "Czechia"),
    ("2026-06-18", "A", "Czechia", "South Africa"),
    ("2026-06-18", "A", "Mexico", "South Korea"),
    ("2026-06-24", "A", "Mexico", "Czechia"),
    ("2026-06-24", "A", "South Korea", "South Africa"),

    # Group B
    ("2026-06-12", "B", "Canada", "Bosnia and Herzegovina"),
    ("2026-06-13", "B", "Qatar", "Switzerland"),
    ("2026-06-18", "B", "Switzerland", "Bosnia and Herzegovina"),
    ("2026-06-18", "B", "Canada", "Qatar"),
    ("2026-06-24", "B", "Switzerland", "Canada"),
    ("2026-06-24", "B", "Bosnia and Herzegovina", "Qatar"),

    # Group C
    ("2026-06-13", "C", "Brazil", "Morocco"),
    ("2026-06-13", "C", "Haiti", "Scotland"),
    ("2026-06-19", "C", "Scotland", "Morocco"),
    ("2026-06-19", "C", "Brazil", "Haiti"),
    ("2026-06-24", "C", "Brazil", "Scotland"),
    ("2026-06-24", "C", "Morocco", "Haiti"),

    # Group D
    ("2026-06-12", "D", "USA", "Paraguay"),
    ("2026-06-13", "D", "Australia", "Türkiye"),
    ("2026-06-19", "D", "USA", "Australia"),
    ("2026-06-19", "D", "Türkiye", "Paraguay"),
    ("2026-06-25", "D", "USA", "Türkiye"),
    ("2026-06-25", "D", "Paraguay", "Australia"),

    # Group E
    ("2026-06-14", "E", "Germany", "Curacao"),
    ("2026-06-14", "E", "Ivory Coast", "Ecuador"),
    ("2026-06-20", "E", "Germany", "Ivory Coast"),
    ("2026-06-20", "E", "Ecuador", "Curacao"),
    ("2026-06-25", "E", "Ecuador", "Germany"),
    ("2026-06-25", "E", "Curacao", "Ivory Coast"),

    # Group F
    ("2026-06-14", "F", "Netherlands", "Japan"),
    ("2026-06-14", "F", "Tunisia", "Sweden"),
    ("2026-06-20", "F", "Netherlands", "Sweden"),
    ("2026-06-20", "F", "Tunisia", "Japan"),
    ("2026-06-25", "F", "Tunisia", "Netherlands"),
    ("2026-06-25", "F", "Japan", "Sweden"),

    # Group G
    ("2026-06-15", "G", "Belgium", "Egypt"),
    ("2026-06-15", "G", "Iran", "New Zealand"),
    ("2026-06-21", "G", "Belgium", "Iran"),
    ("2026-06-21", "G", "New Zealand", "Egypt"),
    ("2026-06-26", "G", "New Zealand", "Belgium"),
    ("2026-06-26", "G", "Egypt", "Iran"),

    # Group H
    ("2026-06-15", "H", "Spain", "Cape Verde"),
    ("2026-06-15", "H", "Saudi Arabia", "Uruguay"),
    ("2026-06-21", "H", "Spain", "Saudi Arabia"),
    ("2026-06-21", "H", "Uruguay", "Cape Verde"),
    ("2026-06-26", "H", "Uruguay", "Spain"),
    ("2026-06-26", "H", "Cape Verde", "Saudi Arabia"),

    # Group I
    ("2026-06-16", "I", "France", "Senegal"),
    ("2026-06-16", "I", "Iraq", "Norway"),
    ("2026-06-22", "I", "France", "Iraq"),
    ("2026-06-22", "I", "Norway", "Senegal"),
    ("2026-06-26", "I", "Norway", "France"),
    ("2026-06-26", "I", "Senegal", "Iraq"),

    # Group J
    ("2026-06-16", "J", "Argentina", "Algeria"),
    ("2026-06-16", "J", "Austria", "Jordan"),
    ("2026-06-22", "J", "Argentina", "Austria"),
    ("2026-06-22", "J", "Jordan", "Algeria"),
    ("2026-06-27", "J", "Argentina", "Jordan"),
    ("2026-06-27", "J", "Algeria", "Austria"),

    # Group K
    ("2026-06-17", "K", "Portugal", "Congo DR"),
    ("2026-06-17", "K", "Uzbekistan", "Colombia"),
    ("2026-06-23", "K", "Portugal", "Uzbekistan"),
    ("2026-06-23", "K", "Colombia", "Congo DR"),
    ("2026-06-27", "K", "Colombia", "Portugal"),
    ("2026-06-27", "K", "Congo DR", "Uzbekistan"),

    # Group L
    ("2026-06-17", "L", "England", "Croatia"),
    ("2026-06-17", "L", "Ghana", "Panama"),
    ("2026-06-23", "L", "England", "Ghana"),
    ("2026-06-23", "L", "Panama", "Croatia"),
    ("2026-06-27", "L", "Panama", "England"),
    ("2026-06-27", "L", "Croatia", "Ghana"),
]


def load_team_names():
    with open(TEAMS_PATH, "r", encoding="utf-8") as f:
        teams = json.load(f)

    return {team["team"] for team in teams}


def resolve_team_name(display_name, team_names):
    candidates = ALIASES.get(display_name, [display_name])

    for candidate in candidates:
        if candidate in team_names:
            return candidate

    return None


def main():
    team_names = load_team_names()

    matches = []
    missing = []

    for idx, (date, group, display_a, display_b) in enumerate(FIXTURES, start=1):
        team_a = resolve_team_name(display_a, team_names)
        team_b = resolve_team_name(display_b, team_names)

        if team_a is None:
            missing.append(display_a)

        if team_b is None:
            missing.append(display_b)

        matches.append(
            {
                "matchId": idx,
                "stage": "GROUP",
                "group": group,
                "date": date,
                "displayTeamA": display_a,
                "displayTeamB": display_b,
                "teamA": team_a if team_a else display_a,
                "teamB": team_b if team_b else display_b,
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(matches, f, ensure_ascii=False, indent=2)

    print(f"생성 완료: {OUTPUT_PATH}")
    print(f"경기 수: {len(matches)}")

    if missing:
        print()
        print("teams.json에서 못 찾은 팀명:")
        for name in sorted(set(missing)):
            print("-", name)
        print()
        print("위 팀들은 ALIASES에 후보 이름을 추가해야 합니다.")
    else:
        print("모든 팀명이 teams.json과 매칭되었습니다.")


if __name__ == "__main__":
    main()
