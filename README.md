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

---

## 🧠 How It Works

### Concepts Used

| scikit-learn | Spark Equivalent (familiar) |
|---|---|
| `StandardScaler` | `StandardScaler` |
| `LogisticRegression` | `LogisticRegression` |
| Column selection | `VectorAssembler` |
| `LabelEncoder` | `StringIndexer` |

### Pipeline

1. **Filter** all Tottenham fixtures from the dataset
2. **Engineer features** from a home/away perspective into Spurs-perspective columns (e.g. `spurs_xg`, `opp_xg`) using `np.where`
3. **Flip result labels** so that a Spurs win is always `1` regardless of home/away
4. **Train/test split** — 80% training, 20% testing
5. **Scale features** with `StandardScaler` so large values (ELO ~1500) don't dominate small ones (xG ~1.5)
6. **Train** a `LogisticRegression` classifier
7. **Predict** using `predict_proba()` to get three probabilities: opponent win, draw, Spurs win

### Example Output

```
Aston Villa vs Tottenham Hotspur
Aston Villa win: 16.8%
Draw: 82.0%
Spurs win: 1.1%

Tottenham Hotspur vs Leeds United
Leeds United win: 59.5%
Draw: 37.4%
Spurs win: 3.1%
```

---

## ⚠️ Limitations

This is a first ML project and comes with known limitations:

- **Small training set** — only 32 Spurs games used for training, which is far too few for a reliable model
- **Broken ELO** — all teams start at ELO 1500 and aren't properly updated after each game, making `elo_diff` largely meaningless early in the season
- **Spurs-only training** — the model only learns from Spurs games rather than all 319 fixtures, missing patterns from the wider league
- **No home advantage feature** — whether Spurs are home or away isn't explicitly encoded as a feature
- **Single season** — no historical context from previous seasons

These limitations explain why the model heavily favours draws and rarely predicts a Spurs win — it reflects the data honestly, but the data isn't rich enough yet.

---

## 🚀 Planned Improvements

### Short Term
- [ ] Train on all 319 games across all teams rather than just Spurs fixtures — more data means better learned patterns
- [ ] Fix ELO ratings to properly update after each game using the standard ELO formula
- [ ] Add `is_home` as an explicit binary feature so the model accounts for home advantage

### Medium Term
- [ ] Add historical data from previous Premier League seasons to massively increase the training set
- [ ] Include other competitions (Champions League, FA Cup, EFL Cup) to capture fixture congestion and rotation effects
- [ ] Evaluate model accuracy properly using cross-validation rather than a single train/test split

### Long Term
- [ ] Automate data collection via a football data API rather than maintaining a manual CSV
- [ ] Experiment with more powerful models (Random Forest, XGBoost) and compare accuracy
- [ ] Build a simple dashboard to visualise predictions across all fixtures

---

## 🛠️ Tech Stack

- Python 3
- pandas
- NumPy
- scikit-learn

---

## 📁 Project Structure

```
FootballML_Project/
│
├── fixtures.csv                 # basic version of the csv without elo ratings and rest days
├── generate_csv.py              # Add elo ratings and rest days to the csv and saves as fixtures_with_features.csv
├── fixtures_with_features.csv   # match data with engineered features
├── main.py                      # full pipeline: training + prediction
└── README.md
```

---

## 💡 Background

This project was built as a hands-on way to apply ML concepts for the first time outside of a formal course setting. Coming from a data engineering background with exposure to Apache Spark's MLlib (VectorAssembler, StringIndexer, StandardScaler, LinearRegression), the goal was to translate those concepts into a real-world project using scikit-learn — something with a genuine outcome to evaluate rather than a toy dataset.

The Socratic approach taken during development (writing each piece of code before seeing the solution) meant the pipeline was understood step by step rather than copy-pasted, which was the whole point.
