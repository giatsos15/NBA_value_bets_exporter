
# Player Props Pipeline (Data Prep + Modeling)

## 1) Data Prep
Creates:
- `player_props_tidy_allbooks.csv` — long/tidy player props for all books with data
- `boxscores_player_features.csv` — player rolling means/std & trend features
- `player_prop_model_dataset.csv` — merged props + features + labels (actual stat)

Run:
```bash
python data_prep_player_props.py \
  --boxscores nba_boxscores_2025-26.csv \
  --odds rotowire_odds_wide_mgm_20251106_110605.csv \
  --props_out player_props_tidy_allbooks.csv \
  --features_out boxscores_player_features.csv \
  --dataset_out player_prop_model_dataset.csv
```

## 2) Modeling
Trains per-prop Ridge regressions (time-series CV), converts prediction to P(Over) via a normal approximation, de-vigs book probabilities, computes edges, and uses capped Kelly sizing.

Run:
```bash
python prop_modeling.py \
  --dataset player_prop_model_dataset.csv \
  --out prop_picks.csv \
  --min_edge_bp 2.0 \
  --kelly_cap 0.02 \
  --folds 5
```

Outputs:
- `prop_picks.csv` — picks with `pick_side`, `pick_edge_bp`, `pick_prob`, `kelly_frac`
- `prop_picks.report.json` — per-prop CV metrics & residual SD used

### Notes
- Labels are added only when the actual stat is available in boxscores on `game_date`. Composite props (e.g., PRA) are computed from components where possible.
- You can expand label mapping in `attach_targets_for_props` to include `STL`, `BLK` for stocks.
- For better calibration, consider per-player residual SDs or quantile models.
