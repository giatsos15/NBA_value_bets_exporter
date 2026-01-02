# prop_modeling.py — trains from props if labels exist; else from boxscore training set
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer

def american_to_prob(odds):
    if pd.isna(odds): return np.nan
    odds=float(odds); return 100/(odds+100) if odds>0 else (-odds)/((-odds)+100)

def no_vig_probs(odds_over,odds_under):
    p_over,p_under=american_to_prob(odds_over),american_to_prob(odds_under)
    if pd.isna(p_over) or pd.isna(p_under) or p_over+p_under==0: return np.nan,np.nan
    z=p_over+p_under; return p_over/z,p_under/z

def kelly_fraction(p,odds,cap=0.02):
    if pd.isna(p) or pd.isna(odds): return 0.0
    b=odds/100 if odds>0 else 100/(-odds)
    f=(b*p-(1-p))/b if b!=0 else 0.0
    return float(np.clip(f,0.0,cap))

def select_feature_columns(df, min_nonnull_frac=0.30):
    """
    Pick numeric rolling/trend features and categorical context.
    Drop identifier/target/market columns.
    Keep only numeric columns with at least `min_nonnull_frac` non-null in TRAIN.
    """
    exclude = {
        "book","name","team","opponent","game_date","prop_type",
        "line","odds_over","odds_under","actual",
        "mkt_p_over","mkt_p_under","model_p_over","pred_mean","resid_sd_used",
        "pick_side","pick_edge_bp","pick_prob","pick_odds","bet_flag","kelly_frac"
    }
    # initial numeric candidates (dtype != 'O')
    num_all = [c for c in df.columns if c not in exclude and df[c].dtype != "O"]
    # filter by non-null fraction (computed on TRAIN later)
    cat_cols = [c for c in ["team","opponent"] if c in df.columns]
    return num_all, cat_cols, min_nonnull_frac


def fit_regression_time_series(train_df, num_all, cat_cols, date_col="game_date", n_splits=5, min_nonnull_frac=0.30):
    """
    Build a preprocessing pipeline with imputers:
      - Numeric: SimpleImputer(median) -> StandardScaler
      - Categorical: SimpleImputer(most_frequent) -> OneHot
    Also prunes numeric columns that are too sparse on TRAIN.
    """
    train_df = train_df.sort_values(date_col).copy()

    # prune numeric cols by non-null fraction on TRAIN
    nnfrac = train_df[num_all].notna().mean()
    num_cols = sorted(list(nnfrac[nnfrac >= min_nonnull_frac].index))

    # if everything is sparse, fall back to whatever exists
    if not num_cols:
        num_cols = num_all

    X = train_df[num_cols + cat_cols].copy()
    y = train_df["actual"].astype(float).values

    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh",  OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ], remainder="drop")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    models, fold_stats = [], []
    for fold, (tr, te) in enumerate(tscv.split(X)):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y[tr], y[te]
        pipe = Pipeline([("pre", pre), ("reg", Ridge(alpha=3.0))])
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        rmse = float(np.sqrt(mean_squared_error(yte, pred)))
        mae  = float(mean_absolute_error(yte, pred))
        resid = ytr - pipe.predict(Xtr)
        sd = float(np.std(resid, ddof=1)) if len(resid) > 2 else np.nan
        fold_stats.append({"fold": fold, "rmse": rmse, "mae": mae, "resid_sd": sd})
        models.append(pipe)

    # return the *kept* numeric cols so scoring uses the same set
    return models, fold_stats, num_cols


def normal_over_prob(mu,sd,line):
    if pd.isna(mu) or pd.isna(sd) or sd<=1e-6 or pd.isna(line): return np.nan
    from math import erf, sqrt
    z=(line-mu)/(sd*np.sqrt(2)); return 0.5*(1-erf(z))

def train_models_for_prop(prop_type, props_df, train_df_box, n_splits=5, min_train_rows=20):
    """
    Trains a per-prop regression model.
    Prefers labeled rows from props; if not enough, falls back to the boxscore training set.
    Uses select_feature_columns() to get candidate numeric/categorical features,
    and fit_regression_time_series() to prune sparse features + impute NaNs.
    Returns (model, kept_num_cols, cat_cols, sd_pred), fold_stats, error_or_None.
    """
    # 1) Build TRAIN set (prefer props with labels, else boxscore training)
    train = props_df[(props_df["prop_type"].str.lower()==prop_type) & (props_df["actual"].notna())].copy()
    if len(train) < min_train_rows:
        train = train_df_box[(train_df_box["prop_type"].str.lower()==prop_type) & (train_df_box["actual"].notna())].copy()
    if len(train) < min_train_rows:
        return None, [], f"insufficient training rows for {prop_type} (found {len(train)})"

    # 2) Feature selection (candidates) and time-series CV fit (with imputers)
    num_all, cat_cols, min_frac = select_feature_columns(train)
    if not num_all and not cat_cols:
        return None, [], f"no features for {prop_type}"

    models, stats, kept_num_cols = fit_regression_time_series(
        train, num_all, cat_cols, n_splits=n_splits, min_nonnull_frac=min_frac
    )
    model = models[-1]

    # 3) Residual SD across folds (for normal approximation later)
    resid_sds = [s["resid_sd"] for s in stats if not pd.isna(s["resid_sd"])]
    sd_pred = float(np.nanmean(resid_sds)) if resid_sds else np.nan

    return (model, kept_num_cols, cat_cols, sd_pred), stats, None


def score_today_for_prop(prop_type, model_tuple, score_df):
    """
    Scores all rows with odds+line for the given prop_type using the trained model.
    model_tuple: (model, kept_num_cols, cat_cols, sd_pred)
    Returns a dataframe with predictions and market no-vig probabilities merged in.
    """
    model, num_cols, cat_cols, sd_pred = model_tuple

    df = score_df[
        (score_df["prop_type"].str.lower() == prop_type) &
        score_df["line"].notna() &
        score_df["odds_over"].notna() &
        score_df["odds_under"].notna()
    ].copy()
    if df.empty:
        return pd.DataFrame()

    # --- NEW: take an explicit feature slice and keep its index for alignment
    X_in = df[num_cols + cat_cols].copy()
    mu = model.predict(X_in)

    # Force df to the exact same row order/length as X_in before attaching preds
    df = df.loc[X_in.index].reset_index(drop=True)

    # Market no-vig probabilities
    mkt = df.apply(
        lambda r: pd.Series(
            no_vig_probs(r["odds_over"], r["odds_under"]),
            index=["mkt_p_over", "mkt_p_under"]
        ),
        axis=1
    )
    df = pd.concat([df, mkt], axis=1)

    # Model probability P(Over)
    from math import isfinite
    df["model_p_over"] = [
        normal_over_prob(m, sd_pred, l) if isfinite(m) else np.nan
        for m, l in zip(mu, df["line"])
    ]
    df["model_p_over"] = df["model_p_over"].clip(1e-6, 1 - 1e-6)
    df["pred_mean"] = mu
    df["resid_sd_used"] = sd_pred
    return df



def main():
    import argparse, json
    ap=argparse.ArgumentParser(description="Train player-prop models and export value bets.")
    ap.add_argument("--dataset", required=True, help="player_prop_model_dataset.csv (props+features+labels when available)")
    ap.add_argument("--train_from_boxscores", default="player_prop_training_from_boxscores.csv", help="fallback training set from boxscores (long format)")
    ap.add_argument("--out", default="prop_picks.csv")
    ap.add_argument("--min_edge_bp", type=float, default=2.0)
    ap.add_argument("--kelly_cap", type=float, default=0.02)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--book", type=str, default=None)
    ap.add_argument("--min_train_rows", type=int, default=20)
    args=ap.parse_args()

    props_df=pd.read_csv(args.dataset, parse_dates=["game_date"])
    train_df_box=pd.read_csv(args.train_from_boxscores, parse_dates=["game_date"])

    if args.book:
        props_df=props_df[props_df["book"].str.lower()==args.book.lower()]

    # What prop types do we want?
    prop_types = sorted(list(set(props_df["prop_type"].dropna().str.lower()).union(set(train_df_box["prop_type"].dropna().str.lower()))))

    all_scored=[]; report={}
    for pt in prop_types:
        model_tuple, stats, err = train_models_for_prop(pt, props_df, train_df_box, n_splits=args.folds, min_train_rows=args.min_train_rows)
        if err:
            report[pt]={"error":err}
            continue
        scored = score_today_for_prop(pt, model_tuple, props_df)
        if scored.empty:
            report[pt]={"warning":"no rows with odds/line to score"}
            continue

        # choose side & size
        def choose(row):
            if pd.isna(row.get("mkt_p_over")) or pd.isna(row.get("mkt_p_under")):
                return pd.Series({"pick_side":np.nan,"pick_edge_bp":np.nan,"pick_prob":np.nan,"pick_odds":np.nan})
            edge_over  = (row["model_p_over"] - row["mkt_p_over"])*100
            edge_under = ((1-row["model_p_over"]) - row["mkt_p_under"])*100
            if edge_over>=edge_under:
                return pd.Series({"pick_side":"Over","pick_edge_bp":edge_over,"pick_prob":row["model_p_over"],"pick_odds":row["odds_over"]})
            return pd.Series({"pick_side":"Under","pick_edge_bp":edge_under,"pick_prob":1-row["model_p_over"],"pick_odds":row["odds_under"]})

        picks=pd.concat([scored.reset_index(drop=True), scored.apply(choose,axis=1)], axis=1)
        picks["bet_flag"]=picks["pick_edge_bp"]>=args.min_edge_bp
        picks["kelly_frac"]=picks.apply(lambda r: kelly_fraction(r["pick_prob"], r["pick_odds"], cap=args.kelly_cap) if r["bet_flag"] else 0.0, axis=1)

        cols=["book","name","team","opponent","game_date","prop_type","line","odds_over","odds_under",
              "mkt_p_over","mkt_p_under","model_p_over","pick_side","pick_edge_bp","pick_prob","pick_odds",
              "bet_flag","kelly_frac","pred_mean","resid_sd_used"]
        all_scored.append(picks[cols].copy())
        report[pt]={"fold_stats":stats}

    out=Path(args.out)
    if all_scored:
        out_df=pd.concat(all_scored, ignore_index=True).sort_values(["game_date","prop_type","name"])
        out_df.to_csv(out, index=False)
        with open(out.with_suffix(".report.json"),"w") as f: json.dump(report,f,indent=2)
        print(f"Saved picks: {out}")
        print(f"Saved report: {out.with_suffix('.report.json')}")
    else:
        print("No picks produced (no rows with odds to score or no trainable props).")
        with open(out.with_suffix(".report.json"),"w") as f: json.dump(report,f,indent=2)

if __name__=="__main__":
    main()
