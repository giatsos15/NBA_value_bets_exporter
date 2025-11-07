
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score, log_loss, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

from feature_engineering import add_team_rolling_trend, build_allowed_defense, merge_home_away_rows

def american_to_prob(odds: float) -> float:
    if pd.isna(odds):
        return np.nan
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return (-odds) / ((-odds) + 100.0)

def kelly_fraction(p: float, odds_american: float, cap: float=0.03) -> float:
    if pd.isna(p) or pd.isna(odds_american):
        return 0.0
    if odds_american > 0:
        b = odds_american / 100.0
    else:
        b = 100.0 / (-odds_american)
    f = (b*p - (1-p)) / b if b != 0 else 0.0
    return float(np.clip(f, 0.0, cap))

def build_preprocessor(df: pd.DataFrame):
    cat_cols = [c for c in df.columns if c in ["home_team","away_team"]]
    exclude = set(["date","game_id","home_team","away_team","home_moneyline","away_moneyline",
                   "spread_open","spread_close","total_open","total_close",
                   "home_pts","away_pts","result_home_win","result_point_diff"])
    num_cols = [c for c in df.columns if c not in exclude and df[c].dtype != "O"]
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols))
    return ColumnTransformer(transformers=transformers, remainder="drop")

def fit_models(df: pd.DataFrame, n_splits=5):
    df = df.sort_values("date")
    X = df.copy()
    y_bin = df["result_home_win"] if "result_home_win" in df.columns else None
    y_reg = df["result_point_diff"] if "result_point_diff" in df.columns else None

    pre = build_preprocessor(X)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_metrics = []
    clf_models, reg_models = [], []

    for fold, (tr, te) in enumerate(tscv.split(X)):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        metrics = {"fold": fold}

        if y_bin is not None:
            clf = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000, C=1.0))])
            clf.fit(X_tr, y_bin.iloc[tr])
            proba = clf.predict_proba(X_te)[:,1]
            yt = y_bin.iloc[te].values
            metrics.update({
                "acc": float(accuracy_score(yt, (proba>=0.5).astype(int))),
                "brier": float(brier_score_loss(yt, proba)),
                "logloss": float(log_loss(yt, proba)),
                "auc": float(roc_auc_score(yt, proba)) if len(np.unique(yt))==2 else np.nan
            })
            clf_models.append(clf)

        if y_reg is not None:
            reg = Pipeline([("pre", pre), ("reg", Ridge(alpha=3.0))])
            reg.fit(X_tr, y_reg.iloc[tr])
            preds = reg.predict(X_te)
            yt = y_reg.iloc[te].values
            metrics.update({
                "mae_pd": float(mean_absolute_error(yt, preds)),
                "rmse_pd": float(mean_squared_error(yt, preds, squared=False))
            })
            reg_models.append(reg)

        fold_metrics.append(metrics)

    return clf_models, reg_models, fold_metrics

def select_bets(scored: pd.DataFrame, min_edge_bp=2.0, kelly_cap=0.03):
    out = scored.copy()
    out["mkt_home_prob"] = out["home_moneyline"].apply(american_to_prob)
    out["model_home_win_prob"] = out["model_home_win_prob"].clip(1e-6, 1-1e-6)
    out["edge_bp"] = (out["model_home_win_prob"] - out["mkt_home_prob"]) * 100.0
    out["bet_flag"] = out["edge_bp"] >= min_edge_bp
    out["kelly_frac"] = out.apply(lambda r: kelly_fraction(r["model_home_win_prob"], r["home_moneyline"], cap=kelly_cap) if r["bet_flag"] else 0.0, axis=1)
    return out

def compute_roi_clv(picks: pd.DataFrame):
    df = picks.copy()
    if "home_moneyline_close" in df.columns:
        df["entry_prob"] = df["home_moneyline"].apply(american_to_prob)
        df["close_prob"] = df["home_moneyline_close"].apply(american_to_prob)
        df["clv_bp"] = (df["close_prob"] - df["entry_prob"]) * 100.0

    if "result_home_win" in df.columns:
        def to_decimal(odds):
            return (odds/100.0)+1.0 if odds > 0 else (100.0/(-odds))+1.0
        df["decimal"] = df["home_moneyline"].apply(to_decimal)
        df["profit_units"] = df.apply(lambda r: r["kelly_frac"]*(r["decimal"]-1.0) if (r.get("bet_flag",False) and r["result_home_win"]==1) else (-r["kelly_frac"] if r.get("bet_flag",False) else 0.0), axis=1)
        staked = df.loc[df.get("bet_flag",False),"kelly_frac"].sum()
        roi = (df["profit_units"].sum()/staked*100.0) if staked>0 else np.nan
        summary = {"n_bets": int(df.get("bet_flag",False).sum()), "units": float(df["profit_units"].sum()), "roi_%": float(roi) if roi==roi else np.nan}
    else:
        summary = {"n_bets": int(df.get("bet_flag",False).sum())}
    return df, pd.DataFrame([summary])

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--team_games", required=True, help="CSV: one row per team per game. Must include game_id,date,team,opponent,home_away.")
    ap.add_argument("--market", required=True, help="CSV: one row per game. Must include game_id,date,home_team,away_team,home_moneyline,away_moneyline; optional: spread/total open/close; optional: home_pts,away_pts.")
    ap.add_argument("--out", default="picks.csv")
    ap.add_argument("--min_edge_bp", type=float, default=2.0)
    ap.add_argument("--kelly_cap", type=float, default=0.03)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    tg = pd.read_csv(args.team_games, parse_dates=["date"])
    mk = pd.read_csv(args.market, parse_dates=["date"])

    tg_enh = add_team_rolling_trend(tg, team_col="team", date_col="date")

    try:
        allowed = build_allowed_defense(tg, team_col="team", opp_col="opponent", date_col="date")
        tg_enh = tg_enh.merge(allowed, left_on=["team","date"], right_on=["team_allowed_key","date"], how="left")
        tg_enh = tg_enh.drop(columns=["team_allowed_key"])
    except Exception as e:
        print("Skipping allowed-defense features:", e)

    df = merge_home_away_rows(tg_enh, mk, key_cols=("game_id","date","home_team","away_team"))

    clf_models, reg_models, fold_metrics = fit_models(df, n_splits=args.folds)

    if not clf_models:
        raise ValueError("Classification target 'result_home_win' not found or insufficient data.")
    clf = clf_models[-1]
    df = df.sort_values("date")
    df["model_home_win_prob"] = clf.predict_proba(df)[:,1]

    picks = select_bets(df, min_edge_bp=args.min_edge_bp, kelly_cap=args.kelly_cap)

    out_path = Path(args.out)
    picks.to_csv(out_path, index=False)

    roi_df, summary = compute_roi_clv(picks)
    roi_df.to_csv(out_path.with_suffix(".roi.csv"), index=False)
    summary.to_csv(out_path.with_suffix(".summary.csv"), index=False)

    if "result_home_win" in df.columns:
        from sklearn.calibration import calibration_curve
        prob_true, prob_pred = calibration_curve(df["result_home_win"], df["model_home_win_prob"], n_bins=10, strategy="quantile")
        plt.figure()
        plt.plot(prob_pred, prob_true, marker="o")
        plt.plot([0,1],[0,1], linestyle="--")
        plt.title("Calibration curve (home win)")
        plt.xlabel("Predicted probability")
        plt.ylabel("Empirical frequency")
        plt.savefig(out_path.with_suffix(".calibration.png"), dpi=150)

    with open(out_path.with_suffix(".fold_metrics.json"), "w") as f:
        import json
        json.dump(fold_metrics, f, indent=2)

    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
