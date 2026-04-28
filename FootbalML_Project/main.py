import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


df = pd.read_csv("fixtures_with_features.csv")

tottenham_stats = df[(df['home_team'] == 'Tottenham Hotspur') | (df['away_team'] == 'Tottenham Hotspur')]

is_home = tottenham_stats['home_team'] == 'Tottenham Hotspur'

cols = {
    'spurs_xg':           ('home_avg_xg_last5',           'away_avg_xg_last5'),
    'spurs_xg_against':   ('home_avg_xg_against_last5',   'away_avg_xg_against_last5'),
    'spurs_goals_scored': ('home_avg_goals_scored_last5',  'away_avg_goals_scored_last5'),
    'spurs_winrate':      ('home_winrate_last5',           'away_winrate_last5'),
    'spurs_possession':   ('home_avg_possession_last5',    'away_avg_possession_last5'),
    'spurs_elo':          ('home_elo_rating',              'away_elo_rating'),
    'opp_xg':             ('away_avg_xg_last5',            'home_avg_xg_last5'),
    'opp_xg_against':     ('away_avg_xg_against_last5',   'home_avg_xg_against_last5'),
    'opp_goals_scored':   ('away_avg_goals_scored_last5',  'home_avg_goals_scored_last5'),
    'opp_winrate':        ('away_winrate_last5',           'home_winrate_last5'),
    'opp_possession':     ('away_avg_possession_last5',    'home_avg_possession_last5'),
    'opp_elo':            ('away_elo_rating',              'home_elo_rating'),
    'elo_diff':           ('elo_diff',                     'elo_diff'),
}

spurs_df = tottenham_stats.copy()

for new_col, (home_col, away_col) in cols.items():
    spurs_df[new_col] = np.where(is_home, spurs_df[home_col], spurs_df[away_col])

feature_cols = ['spurs_xg', 'spurs_xg_against', 'spurs_goals_scored', 'spurs_winrate', 'spurs_possession', 'spurs_elo', 'opp_xg', 'opp_xg_against', 'opp_goals_scored', 'opp_winrate', 'opp_possession','opp_elo', 'elo_diff']
x = spurs_df[feature_cols]

spurs_df['spurs_result'] = np.where(
    is_home,
    spurs_df['result'],
    spurs_df['result'] * -1
)

y = spurs_df['spurs_result']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

last_spurs = spurs_df.iloc[-1]

def get_team_stats(team):

    if team not in df['home_team'].values and team not in df['away_team'].values:
        print("Team not in dataset")
        print("Ensure Team is formatted correctly")
        return None
    else:
        team_stats = df[(df['home_team'] == team) | (df['away_team'] == team)]
        

    is_home = team_stats['home_team'] == team

    cols = {
        'team_xg':           ('home_avg_xg_last5',           'away_avg_xg_last5'),
        'team_xg_against':   ('home_avg_xg_against_last5',   'away_avg_xg_against_last5'),
        'team_goals_scored': ('home_avg_goals_scored_last5',  'away_avg_goals_scored_last5'),
        'team_winrate':      ('home_winrate_last5',           'away_winrate_last5'),
        'team_possession':   ('home_avg_possession_last5',    'away_avg_possession_last5'),
        'team_elo':          ('home_elo_rating',              'away_elo_rating'),
        'opp_xg':             ('away_avg_xg_last5',            'home_avg_xg_last5'),
        'opp_xg_against':     ('away_avg_xg_against_last5',   'home_avg_xg_against_last5'),
        'opp_goals_scored':   ('away_avg_goals_scored_last5',  'home_avg_goals_scored_last5'),
        'opp_winrate':        ('away_winrate_last5',           'home_winrate_last5'),
        'opp_possession':     ('away_avg_possession_last5',    'home_avg_possession_last5'),
        'opp_elo':            ('away_elo_rating',              'home_elo_rating'),
        'elo_diff':           ('elo_diff',                     'elo_diff'),
    }

    team_df = team_stats.copy()

    for new_col, (home_col, away_col) in cols.items():
        team_df[new_col] = np.where(is_home, team_df[home_col], team_df[away_col])

    feature_cols = ['team_xg', 'team_xg_against', 'team_goals_scored', 'team_winrate', 'team_possession', 'team_elo', 'opp_xg', 'opp_xg_against', 'opp_goals_scored', 'opp_winrate', 'opp_possession', 'opp_elo', 'elo_diff']
    x = team_df[feature_cols]

    team_df['team_result'] = np.where(
        is_home,
        team_df['result'],
        team_df['result'] * -1
    )

    y = team_df['team_result']

    last_team = team_df.iloc[-1]

    team_fixture = pd.DataFrame([{
        'spurs_xg': last_spurs['spurs_xg'],
        'spurs_xg_against': last_spurs['spurs_xg_against'],
        'spurs_goals_scored': last_spurs['spurs_goals_scored'],
        'spurs_winrate': last_spurs['spurs_winrate'],
        'spurs_possession': last_spurs['spurs_possession'],
        'spurs_elo': last_spurs['spurs_elo'],
        'opp_xg': last_team['team_xg'],
        'opp_xg_against': last_team['team_xg_against'],
        'opp_goals_scored': last_team['team_goals_scored'],
        'opp_winrate': last_team['team_winrate'],
        'opp_possession': last_team['team_possession'],
        'opp_elo': last_team['team_elo'],
        'elo_diff': last_spurs['elo_diff'],
    }]) 
    return team_fixture 

def predict_fixture(team, spurs_home): 

    fixture = get_team_stats(team)
    if fixture is None:
        return None

    fixture_scaled = scaler.transform(fixture)
    proba = model.predict_proba(fixture_scaled)[0]

    if spurs_home == True:
        print(f"Tottenham Hotspur vs {team}")
    else:
        print(f"{team} vs Tottenham Hotspur")
    outcomes = [f'{team} win', 'Draw', 'Spurs win']
    for outcome, prob in zip(outcomes, proba):
        print(f"{outcome}: {prob*100:.1f}%")

predict_fixture('Wolves', False)
predict_fixture('Aston Villa', False)
predict_fixture('Leeds United', True)
predict_fixture('Chelsea', False)
predict_fixture('Everton', True)