
import numpy as np
import pandas as pd

ROLL_NUM_COLS = [
    "ortg","drg","drtg","netrtg","pace","efg","ts","tov_pct","orb_pct","drb_pct",
    "ftr","threepar","ast","reb","pts","poss"
]

def _rolling_stats(s: pd.Series, windows=(5,10)):
    out = {}
    for w in windows:
        r = s.rolling(w, min_periods=max(1, min(3,w)))
        out[f"roll{w}_mean"] = r.mean()
        out[f"roll{w}_std"]  = r.std()
    return pd.DataFrame(out)

def _trend_slope(s: pd.Series, lookback=10):
    vals = s.tail(lookback).values
    n = len(vals)
    if n < 3:
        return np.nan
    x = np.arange(n)
    if np.isnan(vals).any():
        return np.nan
    slope = np.polyfit(x, vals, 1)[0]
    return slope

def add_team_rolling_trend(team_games: pd.DataFrame,
                           team_col="team", date_col="date",
                           num_cols=None,
                           windows=(5,10), trend_lb=10) -> pd.DataFrame:
    df = team_games.sort_values([team_col, date_col]).copy()
    if num_cols is None:
        num_cols = [c for c in ROLL_NUM_COLS if c in df.columns]

    parts = []
    for team, g in df.groupby(team_col, sort=False):
        g = g.copy()
        for col in num_cols:
            rs = _rolling_stats(g[col], windows=windows)
            for name, series in rs.items():
                g[f"{col}_{name}"] = series.shift(1)
            g[f"{col}_trend"] = g[col].rolling(trend_lb, min_periods=3).apply(
                lambda _: _trend_slope(pd.Series(_), lookback=trend_lb), raw=False
            ).shift(1)
        parts.append(g)
    out = pd.concat(parts, axis=0).sort_values([team_col, date_col])
    return out

def build_allowed_defense(team_games: pd.DataFrame,
                          team_col="team", opp_col="opponent",
                          date_col="date", cols=None) -> pd.DataFrame:
    df = team_games.sort_values([date_col]).copy()
    if cols is None:
        cols = ["pts","ortg","efg","ts","threepar","tov_pct","ftr","reb","ast","pace","poss"]
    cols = [c for c in cols if c in df.columns]

    records = []
    for _, r in df.iterrows():
        rec = { "def_team": r[opp_col], "date": r[date_col] }
        for c in cols:
            rec[f"{c}_vs_def"] = r.get(c, np.nan)
        records.append(rec)
    vs_def = pd.DataFrame(records)

    vs_def = vs_def.sort_values(["def_team","date"])
    parts = []
    for t, g in vs_def.groupby("def_team", sort=False):
        g = g.copy()
        for c in [c for c in g.columns if c.endswith("_vs_def")]:
            g[f"{c}_roll5_allowed"]  = g[c].rolling(5, min_periods=3).mean().shift(1)
            g[f"{c}_roll10_allowed"] = g[c].rolling(10, min_periods=3).mean().shift(1)
        parts.append(g)
    allowed = pd.concat(parts, axis=0)
    keep = ["def_team","date"] + [c for c in allowed.columns if c.endswith("_allowed")]
    allowed = allowed[keep].rename(columns={"def_team":"team_allowed_key"})
    return allowed

def merge_home_away_rows(team_games_enh: pd.DataFrame,
                         market_df: pd.DataFrame,
                         key_cols=("game_id","date","home_team","away_team"),
                         team_col="team", opp_col="opponent") -> pd.DataFrame:
    home = team_games_enh.rename(columns={team_col:"home_team"}).set_index(["game_id","home_team"])
    away = team_games_enh.rename(columns={team_col:"away_team"}).set_index(["game_id","away_team"])

    feat_cols = [c for c in team_games_enh.columns if c not in ["team","opponent","game_id","date","home_away"]]
    home_feats = home[feat_cols].add_prefix("home_").reset_index()
    away_feats = away[feat_cols].add_prefix("away_").reset_index()

    out = market_df.copy()
    out = out.merge(home_feats, on=["game_id","home_team"], how="left")
    out = out.merge(away_feats, on=["game_id","away_team"], how="left")

    if "home_pts" in out.columns and "away_pts" in out.columns:
        out["result_home_win"] = (out["home_pts"] > out["away_pts"]).astype(int)
        out["result_point_diff"] = (out["home_pts"] - out["away_pts"]).astype(float)

    return out
