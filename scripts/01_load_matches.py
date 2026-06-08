import pandas as pd
from pathlib import Path

BASE_PATH = Path("data/raw/results.csv")
RECENT_PATH = Path("data/manual/recent_matches.csv")


def load_matches():
    base_df = pd.read_csv(BASE_PATH)

    if RECENT_PATH.exists():
        recent_df = pd.read_csv(RECENT_PATH)
    else:
        recent_df = pd.DataFrame(columns=base_df.columns)

    required_columns = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "city",
        "country",
        "neutral",
    ]

    base_df = base_df[required_columns]
    recent_df = recent_df[required_columns]

    df = pd.concat([base_df, recent_df], ignore_index=True)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_score", "away_score"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


def main():
    df = load_matches()

    teams = sorted(set(df["home_team"]) | set(df["away_team"]))

    print("전체 경기 수:", len(df))
    print("기간:", df["date"].min().date(), "~", df["date"].max().date())
    print("팀 수:", len(teams))
    print()
    print("최근 5경기:")
    print(df.tail(5)[["date", "home_team", "away_team", "home_score", "away_score", "tournament"]])


if __name__ == "__main__":
    main()
