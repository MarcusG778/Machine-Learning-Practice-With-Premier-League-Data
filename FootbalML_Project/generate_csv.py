import pandas as pd
import numpy as np

df = pd.read_csv("fixtures.csv")

df = df.drop_duplicates(subset=["match_id"], keep="first")

df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

df = df.dropna(subset=["date", "home_team", "away_team"])

df = df.sort_values(["date", "match_id"]).reset_index(drop=True)

team_last_match = {}

home_rest = []
away_rest = []

for _, row in df.iterrows():
    date = row["date"]
    home = row["home_team"]
    away = row["away_team"]

    if home in team_last_match:
        home_days = (date - team_last_match[home]).days
    else:
        home_days = np.nan

    if away in team_last_match:
        away_days = (date - team_last_match[away]).days
    else:
        away_days = np.nan

    home_rest.append(home_days)
    away_rest.append(away_days)

    team_last_match[home] = date
    team_last_match[away] = date

df["home_days_rest"] = home_rest
df["away_days_rest"] = away_rest
df["rest_diff"] = df["home_days_rest"] - df["away_days_rest"]

df["home_days_rest"] = df["home_days_rest"].fillna(df["home_days_rest"].median())
df["away_days_rest"] = df["away_days_rest"].fillna(df["away_days_rest"].median())
df["rest_diff"] = df["rest_diff"].fillna(0)

K = 20
initial_rating = 1500
teams = {}

def get_rating(team):
    if team not in teams:
        teams[team] = initial_rating
    return teams[team]

def expected(r_a, r_b):
    return 1 / (1 + 10 ** ((r_b - r_a) / 400))

home_elo_list = []
away_elo_list = []

for _, row in df.iterrows():
    home = row["home_team"]
    away = row["away_team"]

    r_home = get_rating(home)
    r_away = get_rating(away)

    home_elo_list.append(r_home)
    away_elo_list.append(r_away)

    if row["home_goals"] > row["away_goals"]:
        score_home, score_away = 1, 0
    elif row["home_goals"] < row["away_goals"]:
        score_home, score_away = 0, 1
    else:
        score_home = score_away = 0.5

    exp_home = expected(r_home, r_away)
    exp_away = 1 - exp_home

    teams[home] = r_home + K * (score_home - exp_home)
    teams[away] = r_away + K * (score_away - exp_away)

df["home_elo_rating"] = home_elo_list
df["away_elo_rating"] = away_elo_list
df["elo_diff"] = df["home_elo_rating"] - df["away_elo_rating"]

df = df.drop_duplicates(subset=["match_id"], keep="first")

df = df.sort_values(["date", "match_id"]).reset_index(drop=True)

df.to_csv("fixtures_with_features.csv", index=False)

print("Done! Clean dataset saved as fixtures_with_features.csv")