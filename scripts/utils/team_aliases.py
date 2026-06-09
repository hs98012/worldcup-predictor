TEAM_ALIASES = {
    "South Korea": "Korea Republic",
    "Korea Republic": "Korea Republic",
    "USA": "United States",
    "United States": "United States",
    "Czech Republic": "Czech Republic",
    "Czechia": "Czech Republic",
    "Ivory Coast": "Ivory Coast",
    "Côte d'Ivoire": "Ivory Coast",
    "DR Congo": "DR Congo",
    "Congo DR": "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    "Türkiye": "Turkey",
    "Curacao": "Curaçao",
    "Cabo Verde": "Cape Verde",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
}


def normalize_team_name(team_name: str) -> str:
    if team_name is None:
        return team_name

    name = str(team_name).strip()
    return TEAM_ALIASES.get(name, name)
