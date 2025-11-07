
# NBA Value Bets — Extended Pipeline

## CSV Inputs

### team_games.csv  (one row per team per game)
Required:
- `game_id`, `date`, `team`, `opponent`, `home_away`

Recommended numeric columns (any subset):
- `ortg`, `drtg`, `pace`, `efg`, `ts`, `tov_pct`, `orb_pct`, `drb_pct`, `ftr`, `threepar`, `ast`, `reb`, `pts`, `poss`

### market.csv  (one row per game)
Required:
- `game_id`, `date`, `home_team`, `away_team`, `home_moneyline`, `away_moneyline`

Optional (helps evaluation):
- `spread_open`, `spread_close`, `total_open`, `total_close`
- `home_pts`, `away_pts`  -> creates labels `result_home_win`, `result_point_diff`

## What the pipeline does
1. Computes **rolling means/std** (5,10) and **trend slopes** (10) per team, shifted to avoid leakage.
2. Builds **defensive allowed** rolling means per team (e.g., `pts_allowed`, `efg_allowed`).
3. Merges to **game-level** rows with `home_*` and `away_*` features.
4. Trains **LogisticRegression** (home win probability) & **Ridge** (point diff) with **TimeSeriesSplit**.
5. Scores, selects bets by **edge (bp)** and **capped Kelly**, and exports **picks**.
6. Computes **ROI** and **CLV (bp)** if closing odds present. Saves a **calibration plot**.

## Run
```bash
python nba_value_bets_pipeline_plus.py \
  --team_games team_games.csv \
  --market market.csv \
  --out picks.csv \
  --min_edge_bp 2.0 \
  --kelly_cap 0.03 \
  --folds 5
```

Outputs:
- `picks.csv`, `picks.roi.csv`, `picks.summary.csv`
- `picks.calibration.png` (if labels available)
- `picks.fold_metrics.json`
