# data_prep_player_props.py
import pandas as pd
import numpy as np
import re
from pathlib import Path

PROP_LONG_LIST = ["pts","reb","ast","3pm","turnovers","pra","ptsreb","ptsast"]

def parse_dates(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def tidy_props_all_books(odds_wide: pd.DataFrame) -> pd.DataFrame:
    prefixes = ["mgm_","draftkings_","fanduel_","caesars_","betrivers_","espnbet_","hardrock_"]
    frames = []
    for prefix in prefixes:
        cols = [c for c in odds_wide.columns if c.startswith(prefix)]
        if not cols: 
            continue
        base_map = {}
        for c in cols:
            m = re.match(rf"^({re.escape(prefix)}.+?)(Over|Under)?$", c)
            if not m: 
                continue
            base = m.group(1)
            suf = m.group(2)
            base_map.setdefault(base, {"line": None, "over": None, "under": None})
            if suf == "Over":
                base_map[base]["over"] = c
            elif suf == "Under":
                base_map[base]["under"] = c
            else:
                base_map[base]["line"] = c

        id_cols = [c for c in ["name","playerID","firstName","lastName","team","opponent","gameID","asof_date","game_date"] if c in odds_wide.columns]
        rows = []
        for _, r in odds_wide.iterrows():
            ident = {k: r.get(k, np.nan) for k in id_cols}
            for base, parts in base_map.items():
                line = r.get(parts["line"], np.nan) if parts["line"] else np.nan
                over = r.get(parts["over"], np.nan) if parts["over"] else np.nan
                under = r.get(parts["under"], np.nan) if parts["under"] else np.nan
                if pd.isna(line) and pd.isna(over) and pd.isna(under):
                    continue
                prop_type = base.replace(prefix,"").lower()
                rows.append({
                    **ident, "book": prefix[:-1], "prop_type": prop_type,
                    "line": pd.to_numeric(line, errors="coerce"),
                    "odds_over": pd.to_numeric(over, errors="coerce"),
                    "odds_under": pd.to_numeric(under, errors="coerce")
                })
        if rows:
            frames.append(pd.DataFrame(rows))

    if not frames:
        return pd.DataFrame(columns=["book","name","team","opponent","game_date","prop_type","line","odds_over","odds_under"])

    tidy = pd.concat(frames, ignore_index=True)
    tidy["prop_type"] = tidy["prop_type"].replace({"threes":"3pm","threepm":"3pm","3p":"3pm","ptsrebast":"pra","stlblk":"stocks"})
    if "opponent" in tidy.columns:
        tidy["opponent"] = tidy["opponent"].astype(str).str.replace("@","", regex=False).str.replace("vs","", regex=False).str.replace("vs.","", regex=False).str.strip()
    return tidy

def add_player_rolling_features(box: pd.DataFrame) -> pd.DataFrame:
    def parse_matchup(row):
        parts = str(row).split()
        if len(parts) >= 3:
            team = parts[0]
            ha = "H" if parts[1] in ("vs","vs.") else "A"
            opp = parts[-1]
            return team, opp, ha
        return np.nan, np.nan, np.nan

    team_parsed, opp_parsed, ha_parsed = zip(*box["MATCHUP"].map(parse_matchup))
    box["TEAM_ABBR_PARSED"] = team_parsed
    box["OPP_ABBR"] = opp_parsed
    box["HOME_AWAY"] = ha_parsed

    num_cols = ["PTS","REB","AST","STL","BLK","TOV","MIN","FGM","FGA","FG_PCT","FG3M","FG3A","FTM","FTA","FT_PCT","PLUS_MINUS"]
    num_cols = [c for c in num_cols if c in box.columns]
    for c in num_cols:
        box[c] = pd.to_numeric(box[c], errors="coerce")

    box = box.sort_values(["PLAYER_NAME","GAME_DATE"])

    def add_rolling(df, group, cols, windows=(5,10)):
        out = df.copy()
        g = out.groupby(group, group_keys=False)
        for w in windows:
            for c in cols:
                out[f"{c}_roll{w}_mean"] = g[c].apply(lambda s: s.shift(1).rolling(w, min_periods=max(1, min(3, w))).mean())
                out[f"{c}_roll{w}_std"]  = g[c].apply(lambda s: s.shift(1).rolling(w, min_periods=max(1, min(3, w))).std())
        def trend10(s):
            s = s.dropna().tail(10).values
            if len(s) < 3: 
                return np.nan
            x = np.arange(len(s))
            return np.polyfit(x, s, 1)[0]
        for c in cols:
            out[f"{c}_trend10"] = g[c].apply(lambda s: s.shift(1).rolling(10, min_periods=3).apply(lambda arr: trend10(pd.Series(arr)), raw=False))
        return out

    feat = add_rolling(box, "PLAYER_NAME", num_cols, windows=(5,10))
    keep_merge = ["PLAYER_NAME","TEAM_ABBREVIATION","OPP_ABBR","GAME_DATE","GAME_ID"]
    feat_cols = [c for c in feat.columns if any(t in c for t in ["_roll5_","_roll10_","_trend10"])]
    player_features = feat[keep_merge + feat_cols].rename(columns={
        "PLAYER_NAME":"name","TEAM_ABBREVIATION":"team","OPP_ABBR":"opponent","GAME_DATE":"game_date","GAME_ID":"game_id"
    })
    return player_features

def attach_targets_for_props(merged_df: pd.DataFrame, box_raw: pd.DataFrame) -> pd.DataFrame:
    target_map = {"pts":"PTS","reb":"REB","ast":"AST","3pm":"FG3M","turnovers":"TOV", "pra":None,"ptsreb":None,"ptsast":None}
    labels = box_raw[["PLAYER_NAME","TEAM_ABBREVIATION","GAME_DATE","PTS","REB","AST","FG3M","TOV"]].copy()
    labels = labels.rename(columns={"PLAYER_NAME":"name","TEAM_ABBREVIATION":"team","GAME_DATE":"game_date"})
    merged = merged_df.merge(labels, on=["name","team","game_date"], how="left")

    def pick_actual(row):
        pt = str(row.get("prop_type","")).lower()
        col = target_map.get(pt)
        if col is None:
            if pt == "pra":
                return (row.get("PTS",np.nan) or 0) + (row.get("REB",np.nan) or 0) + (row.get("AST",np.nan) or 0)
            if pt == "ptsreb":
                return (row.get("PTS",np.nan) or 0) + (row.get("REB",np.nan) or 0)
            if pt == "ptsast":
                return (row.get("PTS",np.nan) or 0) + (row.get("AST",np.nan) or 0)
            return np.nan
        return row.get(col, np.nan)

    merged["actual"] = merged.apply(pick_actual, axis=1)
    return merged

def build_boxscore_training_long(player_features: pd.DataFrame, box_raw: pd.DataFrame) -> pd.DataFrame:
    base = box_raw.rename(columns={"PLAYER_NAME":"name","TEAM_ABBREVIATION":"team","GAME_DATE":"game_date"})
    # props actuals by stat
    pieces = []
    if set(["PTS","REB","AST","FG3M","TOV"]).issubset(base.columns):
        mapping = {
            "pts":"PTS","reb":"REB","ast":"AST","3pm":"FG3M","turnovers":"TOV",
            "pra":None,"ptsreb":None,"ptsast":None
        }
        for prop, col in mapping.items():
            dfp = base[["name","team","game_date","MATCHUP"]].copy()
            dfp["prop_type"] = prop
            if col is not None:
                dfp["actual"] = base[col]
            else:
                if prop == "pra":
                    dfp["actual"] = base["PTS"] + base["REB"] + base["AST"]
                elif prop == "ptsreb":
                    dfp["actual"] = base["PTS"] + base["REB"]
                elif prop == "ptsast":
                    dfp["actual"] = base["PTS"] + base["AST"]
            pieces.append(dfp)
    train_long = pd.concat(pieces, ignore_index=True)
    # derive opponent from MATCHUP in same way
    def parse_matchup(row):
        parts = str(row).split()
        if len(parts) >= 3:
            team = parts[0]
            opp = parts[-1]
            return team, opp
        return np.nan, np.nan
    team_parsed, opp_parsed = zip(*train_long["MATCHUP"].map(parse_matchup))
    train_long["team_abbr_from_matchup"] = team_parsed
    train_long["opponent"] = opp_parsed
    train_long.drop(columns=["MATCHUP"], inplace=True)
    # join features
    feat = player_features.copy()
    # norm keys
    train_long["name"] = train_long["name"].astype(str).str.strip()
    train_long["team"] = train_long["team"].astype(str).str.strip()
    train_long["opponent"] = train_long["opponent"].astype(str).str.replace("@","", regex=False).str.replace("vs","", regex=False).str.replace("vs.","", regex=False).str.strip()
    feat["name"] = feat["name"].astype(str).str.strip()
    feat["team"] = feat["team"].astype(str).str.strip()
    feat["opponent"] = feat["opponent"].astype(str).str.strip()
    train = train_long.merge(feat, on=["name","team","opponent","game_date"], how="left")
    return train

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Prepare tidy props, engineered features, merged dataset, and boxscore training set.")
    ap.add_argument("--boxscores", required=True, help="nba_boxscores_YYYY-YY.csv")
    ap.add_argument("--odds", required=True, help="rotowire_odds_wide_*.csv")
    ap.add_argument("--props_out", default="player_props_tidy_allbooks.csv")
    ap.add_argument("--features_out", default="boxscores_player_features.csv")
    ap.add_argument("--dataset_out", default="player_prop_model_dataset.csv")
    ap.add_argument("--train_out", default="player_prop_training_from_boxscores.csv")
    args = ap.parse_args()

    box = pd.read_csv(args.boxscores)
    odds = pd.read_csv(args.odds)

    box = parse_dates(box, ["GAME_DATE"])
    odds = parse_dates(odds, ["game_date","asof_date"])

    tidy = tidy_props_all_books(odds)
    tidy.to_csv(args.props_out, index=False)

    player_features = add_player_rolling_features(box)
    player_features.to_csv(args.features_out, index=False)

    # normalize keys and merge props+features
    for df in (tidy, player_features):
        for c in ["name","team","opponent"]:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()

    merged = tidy.merge(player_features, on=["name","team","opponent","game_date"], how="left")
    labeled = attach_targets_for_props(merged, box)
    labeled.to_csv(args.dataset_out, index=False)

    # NEW: training dataset from boxscores (long)
    train_long = build_boxscore_training_long(player_features, box)
    train_long.to_csv(args.train_out, index=False)

    print(f"Saved: {args.props_out}")
    print(f"Saved: {args.features_out}")
    print(f"Saved: {args.dataset_out}")
    print(f"Saved: {args.train_out}")

if __name__ == "__main__":
    main()
