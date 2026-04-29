# Machine-Learning-Practice-With-Premier-League-Data

A practice machine learning project built to predict Premier League match outcomes and generate win/draw/loss probabilities — similar to the live win percentages shown on broadcast TV.

Built as a first venture into ML by a student and aspiring data engineer using Python and scikit-learn to apply familiar concepts (feature engineering, scaling, classification) in a new context.

---

## Project Goal

Predict the outcome of Tottenham Hotspur's remaining fixtures in the 2025/26 Premier League season, outputting probabilities for a home win, draw, or away win based on recent team form and performance statistics.

---

## Dataset

- **Source:** Manually compiled CSV of all 2025/26 Premier League fixtures using data from Fbref.com
- **Coverage:** 319 games across 33 gameweeks
- **Features per match:**
  - Expected goals (xG) averaged over last 5 games
  - Goals scored and conceded averages
  - Possession averages
  - Win rate over last 5 games
  - Fouls and card averages
  - ELO ratings and ELO differential
  - Rest days between fixtures
  
The `fixtures_with_features.csv` used in this project is included in the repo. 
Feel free to use it for your own experiments — if you do something interesting 
with it I'd love to see it.

---

## How It Works

### Concepts Used

| scikit-learn | Spark Equivalent (familiar) |
|---|---|
| `StandardScaler` | `StandardScaler` |
| `LogisticRegression` | `LogisticRegression` |
| Column selection | `VectorAssembler` |
| `LabelEncoder` | `StringIndexer` |

### Pipeline

1. **Load** all 319 Premier League fixtures
2. **Train/test split** — 80% training, 20% testing
3. **Scale features** with StandardScaler so large values (ELO ~1500) don't dominate small ones (xG ~1.5)
4. **Train** a `LogisticRegression` classifier on all games — not just Spurs — so it learns general football patterns
5. **Build fixture rows** for upcoming games using each team's most recent home/away stats
6. **Predict** using predict_proba() to get three probabilities: opponent win, draw, Spurs win

### ELO Implementation
 
ELO ratings are calculated with three improvements over a naive implementation:
 
- **Starting ratings based on previous season finish** — rather than everyone starting at 1500, teams are seeded by league position (Liverpool 1800 → promoted sides 1420)
- **Goal difference multiplier** — a 3-0 win shifts ELO more than a 1-0 win (multipliers: 1.0, 1.5, 1.75)
- **Home advantage** — a 50 point bonus applied to the home team's expected score during ELO calculation
### Feature Importance (learned from the model)
 
| Feature | Importance |
|---|---|
| home_avg_xg_last5 | 0.430 |
| home_avg_yellow_cards_last5 | 0.300 |
| home_elo_rating | 0.288 |
| away_avg_goals_scored_last5 | 0.264 |
| home_avg_possession_last5 | 0.249 |
| elo_diff | 0.064 |
 
xG is the strongest predictor. ELO differential is surprisingly low — reflecting that upsets are common enough in football that the raw quality gap doesn't dominate predictions.
 
### Example Output
 
```
Wolves vs Tottenham Hotspur
Wolves win: 48.0%
Draw: 20.6%
Spurs win: 31.4%
 
Aston Villa vs Tottenham Hotspur
Aston Villa win: 17.9%
Draw: 22.7%
Spurs win: 59.4%
 
Tottenham Hotspur vs Leeds United
Leeds United win: 35.1%
Draw: 58.5%
Spurs win: 6.3%
 
Chelsea vs Tottenham Hotspur
Chelsea win: 28.2%
Draw: 26.4%
Spurs win: 45.5%
 
Tottenham Hotspur vs Everton
Everton win: 43.3%
Draw: 48.8%
Spurs win: 8.0%
```
 
**Model accuracy: 42.2%** (vs 33.3% random baseline for 3-outcome classification)
 
---
## Completed Improvements

### v1 → v2
- **Expanded training data** — moved from 32 Spurs-only games to all 319 league fixtures
- **Fixed ELO** — replaced flat 1500 starting point with previous season finish seeding, added goal difference multiplier and home advantage bonus
- **Fixed ELO diff calculation** — was incorrectly pulling from a cached row rather than calculating fresh per fixture

### Accuracy progression
| Version | Training data | Accuracy |
|---|---|---|
| v1 | 32 Spurs games | 45.3% |
| v2 | 319 all games | 42.2% |
---
## Limitations

### Data Quality
The model design is sound but data quality is the primary limiting factor. Key issues:
 
- **Single season only** — 319 games is a small training set. The model can't reliably distinguish genuine team quality from short-term form fluctuations
- **xG averages over last 5 games** — small sample, noisy. A team that had one exceptional game skews the average significantly
- **ELO low feature importance** — despite proper implementation, `elo_diff` ranks second to last. The model has learned that upsets are common enough that ELO gap is a weak signal at this data volume
- **Yellow cards as second most important feature** — almost certainly spurious correlation in a small dataset rather than a genuine causal relationship. More data would reduce this
### Model Design
- **No sequential updating** — when predicting 5 consecutive fixtures, Spurs' stats don't update between predictions. A win vs Wolves should improve their ELO and form before predicting Villa, but currently all fixtures use the same snapshot of Spurs' last recorded stats
- **No recency weighting** — all 319 games are treated equally in training. A game from August is weighted the same as a game from April
- **Venue-specific patterns not captured** — e.g. Spurs' historically poor record at specific grounds like Stamford Bridge
### The Ceiling Problem
Even professional models with vastly more data typically achieve 50-55% accuracy on football prediction. Genuine randomness — injuries in the warmup, a moment of individual brilliance, controversial refereeing — cannot be modelled. This project is a structural foundation, not a betting system.
 
---
 
## Potential Improvements
 
### Short Term
- [ ] **Sequential fixture prediction** — after predicting each game, update Spurs' ELO and form stats based on the most likely outcome before predicting the next fixture
- [ ] **Sample weighting** — use `model.fit(X_train_scaled, y_train, sample_weight=weights)` with exponential decay so recent games matter more than older ones
### Medium Term
- [ ] **Historical data from previous seasons** — more rows = better generalisation, combined with recency weighting to avoid stale data dominating
- [ ] **Other competitions** — Champions League, FA Cup, EFL Cup data to capture fixture congestion and rotation effects
- [ ] **Venue-specific features** — each team's win rate at specific grounds rather than just generic home/away
- [ ] **Cross-validation** — replace single train/test split with k-fold cross-validation for more reliable accuracy estimates
### Long Term
- [ ] **Player-level data** — squad availability, individual form, key player absence flags
- [ ] **Injury data** — binary flags for key player availability (hard to source reliably but high signal)
- [ ] **Referee data** — some referees statistically favour home teams or card more heavily, a genuine signal used in professional models
- [ ] **Automated data pipeline** — replace manual CSV with a football data API
- [ ] **More powerful models** — Random Forest, XGBoost — compare accuracy against LogisticRegression baseline
- [ ] **Dashboard** — visualise predictions and track model accuracy across the season in real time
---
 
## Key Learnings
 
**On data vs model design:** At this scale, data quality matters more than model sophistication. A better dataset fed into LogisticRegression will outperform a complex neural network fed poor data.
 
**On historical data:** Adding previous seasons without recency weighting could hurt accuracy — a Spurs team with different players and a different manager three seasons ago is a different entity. The solution is exponential decay sample weighting, not excluding historical data entirely.
 
**On feature engineering:** The most impactful improvements are better xG data (larger rolling window, opposition-adjusted), proper venue encoding, and sequential ELO updates between fixtures — not adding more exotic features.
 
**On the ceiling:** Football has genuine irreducible randomness. 55% accuracy over a large sample is considered excellent. The goal of this project was always to understand the pipeline, not to beat the bookmakers.
 
**On "too much data":** More rows (games) is almost always beneficial. More features (columns) can hurt if there isn't enough data to support them — this is the curse of dimensionality. The right approach is to grow features and rows together as data becomes available.
 
---
 
## Tech Stack
 
- Python 3
- pandas
- NumPy
- scikit-learn
---
 
## Project Structure
 
```
FootballML_Project/
│
├── fixtures.csv                 # raw match data
├── elo_generator.py             # ELO calculation with goal multiplier and home advantage
├── fixtures_with_features.csv   # processed data with all engineered features
├── main.py                      # full pipeline: training + prediction
└── README.md
```
 
---
 
## Background
 
This project was built as a hands-on way to apply ML concepts for the first time outside of a formal course setting. Coming from a data engineering background with exposure to Apache Spark's MLlib (VectorAssembler, StringIndexer, StandardScaler, LinearRegression), the goal was to translate those concepts into a real-world project using scikit-learn — something with a genuine outcome to evaluate rather than a toy dataset, choosing spurs as a spurs fan, working with this data was a constant reminder how shocking bad we are.
 
The approach taken during development was deliberately Socratic — writing each piece of code independently before seeing a solution — so that every line of the pipeline was understood rather than copy-pasted. The result is a model with known limitations and a clear roadmap for improvement, which is a more honest and useful outcome than a black box that produces confident but unexplained predictions.
