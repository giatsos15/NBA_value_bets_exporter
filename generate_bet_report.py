import argparse
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Optional charts
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


# ---------- helpers ----------

def _round_cols(df, cols, ndigits=4):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(ndigits)
    return df

def _dict_to_ul(d):
    if not d:
        return "<li><i>None</i></li>"
    return "".join(f"<li><b>{k}</b>: {v}</li>" for k, v in d.items())

def american_to_decimal(odds):
    if pd.isna(odds):
        return np.nan
    odds = float(odds)
    return 1.0 + (odds / 100.0) if odds > 0 else 1.0 + (100.0 / -odds)

def _settle_row(row):
    """
    Returns (won_bool, pnl_flat_units, pnl_kelly_units, chosen_dec_odds)
    - flat stake = 1 unit per bet
    - kelly stake = row['kelly_frac'] units (capped earlier)
    Outcome uses (pick_side, actual, line).
    """
    actual = row.get("actual", np.nan)
    line = row.get("line", np.nan)
    side = str(row.get("pick_side", "")).lower()

    if pd.isna(actual) or pd.isna(line) or side not in ("over", "under"):
        return (np.nan, np.nan, np.nan, np.nan)

    if side == "over":
        won = bool(actual > line)
        dec = american_to_decimal(row.get("odds_over", np.nan))
    else:
        won = bool(actual < line)
        dec = american_to_decimal(row.get("odds_under", np.nan))

    if pd.isna(dec):
        return (np.nan, np.nan, np.nan, np.nan)

    # Flat 1u
    pnl_flat = (dec - 1.0) if won else -1.0
    # Kelly
    stake = float(row.get("kelly_frac", 0.0) or 0.0)
    pnl_kelly = stake * ((dec - 1.0) if won else -1.0)
    return (won, pnl_flat, pnl_kelly, dec)

def _build_actuals_long_from_boxscores(path):
    """
    Expects a 'long' CSV with at least: name, team, game_date, prop_type, actual
    If your file is the one we created earlier (player_prop_training_from_boxscores.csv),
    it already matches this schema.
    """
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["game_date"], dayfirst=False)
    # minimal schema check
    need = {"name", "team", "game_date", "prop_type", "actual"}
    if not need.issubset(df.columns):
        return None
    # normalize
    for c in ["name", "team", "prop_type"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df[["name", "team", "game_date", "prop_type", "actual"]].copy()

def _calibration_table(df):
    """
    Calibration uses pick-specific probability:
      - If pick_side == 'Over': use model_p_over (or pick_prob if present)
      - If 'Under': use 1 - model_p_over (or pick_prob if present)
    Bins by deciles and compares mean predicted vs realized win rate.
    """
    if "won" not in df.columns:
        return pd.DataFrame()

    # prefer pick_prob if available; else derive from model_p_over
    if "pick_prob" in df.columns and df["pick_prob"].notna().any():
        pred = df["pick_prob"].astype(float)
    else:
        mpo = pd.to_numeric(df.get("model_p_over", np.nan), errors="coerce")
        side = df.get("pick_side", "").astype(str).str.lower()
        pred = np.where(side == "over", mpo, 1 - mpo)

    ok = pd.Series(~pd.isna(pred)) & df["won"].isin([0, 1])
    d = pd.DataFrame({"pred": pred[ok], "won": df.loc[ok, "won"].astype(int)})
    if d.empty:
        return pd.DataFrame()

    d["bin"] = pd.qcut(d["pred"], q=10, duplicates="drop")
    tab = d.groupby("bin").agg(pred_avg=("pred", "mean"),
                               win_rate=("won", "mean"),
                               n=("won", "size")).reset_index()
    return tab


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Generate a deduplicated bet report from prop_picks.csv (with performance tracking)")
    ap.add_argument("--input", required=True, help="Path to prop_picks.csv")
    ap.add_argument("--out_csv", default="bets_to_place.csv", help="Output CSV path for deduped bets")
    ap.add_argument("--out_json", default="bets_to_place_report.json", help="Output JSON path for evaluation summary")
    ap.add_argument("--out_html", default="bets_to_place.html", help="Output HTML report path")
    ap.add_argument("--min_edge_bp", type=float, default=None, help="Optional minimum edge (basis points) filter")
    ap.add_argument("--only_bet_flag", action="store_true", help="If set, keep only rows where bet_flag==True")
    ap.add_argument("--no_charts", action="store_true", help="Skip chart generation in the HTML report")
    ap.add_argument("--dedup_keys", nargs="*", default=["name","prop_type","line","pick_side"],
                    help="Columns to deduplicate on (default: name prop_type line pick_side)")
    # Performance tracking inputs/outputs
    ap.add_argument("--actuals_long", default="player_prop_training_from_boxscores.csv",
                    help="CSV with columns [name, team, game_date, prop_type, actual] for settling results")
    ap.add_argument("--history_csv", default="bets_history.csv",
                    help="CSV log to append settled bets over time")
    args = ap.parse_args()

    # Load picks
    df = pd.read_csv(args.input, parse_dates=["game_date"], dayfirst=False)
    if not len(df):
        print("Input CSV is empty.")
        return

    # ---------- Filters ----------
    work = df.copy()
    if args.only_bet_flag and "bet_flag" in work.columns:
        work = work[work["bet_flag"] == True].copy()
    if args.min_edge_bp is not None and "pick_edge_bp" in work.columns:
        work = work[work["pick_edge_bp"] >= args.min_edge_bp].copy()

    # ---------- Sort ----------
    priority_cols = []
    if "pick_edge_bp" in work.columns: priority_cols.append(("pick_edge_bp", False))
    if "pick_prob" in work.columns:    priority_cols.append(("pick_prob", False))
    if "kelly_frac" in work.columns:   priority_cols.append(("kelly_frac", False))

    if priority_cols:
        sort_by = [c for c,_ in priority_cols]
        ascending = [asc for _,asc in priority_cols]
        work = work.sort_values(sort_by, ascending=ascending)
    else:
        fallback_cols = [c for c in ["book","name","prop_type","line","pick_side","game_date"] if c in work.columns]
        work = work.sort_values(fallback_cols, ascending=True) if fallback_cols else work

    # ---------- Deduplicate ----------
    dedup_keys = [k for k in args.dedup_keys if k in work.columns]
    deduped = work.drop_duplicates(subset=dedup_keys, keep="first") if dedup_keys else work.copy()

    # ---------- Output CSV ----------
    cols = [c for c in [
        "book","name","team","opponent","game_date","prop_type","line","pick_side",
        "pick_edge_bp","pick_prob","odds_over","odds_under","mkt_p_over","mkt_p_under",
        "model_p_over","pred_mean","resid_sd_used","kelly_frac"
    ] if c in deduped.columns]
    out_df = deduped[cols].copy()

    # Clean formatting
    if "game_date" in out_df.columns:
        out_df["game_date"] = pd.to_datetime(out_df["game_date"], errors="coerce")
    out_df = _round_cols(out_df, ["pick_edge_bp","pick_prob","model_p_over","mkt_p_over","mkt_p_under","kelly_frac","pred_mean","resid_sd_used"], 4)
    if "line" in out_df.columns:
        out_df["line"] = pd.to_numeric(out_df["line"], errors="coerce").round(2)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    # ---------- Summary JSON (pre-performance) ----------
    summary = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "input_rows": int(len(df)),
        "after_filters_rows": int(len(work)),
        "deduped_rows": int(len(out_df)),
        "books": out_df["book"].value_counts().to_dict() if "book" in out_df.columns else {},
        "prop_types": out_df["prop_type"].value_counts().to_dict() if "prop_type" in out_df.columns else {},
        "avg_edge_bp": float(out_df["pick_edge_bp"].mean()) if "pick_edge_bp" in out_df.columns and len(out_df) else None,
        "median_edge_bp": float(out_df["pick_edge_bp"].median()) if "pick_edge_bp" in out_df.columns and len(out_df) else None,
        "avg_kelly_frac": float(out_df["kelly_frac"].mean()) if "kelly_frac" in out_df.columns and len(out_df) else None,
    }

    # ---------- PERFORMANCE: merge actuals & settle ----------
    perf = {}
    actuals_long_df = _build_actuals_long_from_boxscores(args.actuals_long)
    settled_df = pd.DataFrame()

    if actuals_long_df is not None and not out_df.empty:
        # Merge on (name, team, game_date, prop_type)
        merge_keys = [k for k in ["name","team","game_date","prop_type"] if k in out_df.columns]
        if "game_date" in actuals_long_df.columns:
            actuals_long_df["game_date"] = pd.to_datetime(actuals_long_df["game_date"], errors="coerce")
        if set(["name","team","game_date","prop_type"]).issubset(set(merge_keys)):
            merged = out_df.merge(actuals_long_df, on=["name","team","game_date","prop_type"], how="left", suffixes=("","_y"))
        else:
            # fallback: looser merge on name, prop_type, game_date
            merged = out_df.merge(actuals_long_df, on=[k for k in ["name","game_date","prop_type"] if k in out_df.columns], how="left", suffixes=("","_y"))

        # Settle bets where actual is present & game_date already played
        today = pd.Timestamp.utcnow().normalize()
        merged["won"], merged["pnl_flat"], merged["pnl_kelly"], merged["dec_odds"] = zip(*merged.apply(_settle_row, axis=1))
        settled_df = merged[merged["won"].isin([True, False])].copy()
        if not settled_df.empty:
            settled_df["won"] = settled_df["won"].astype(int)
            perf["n_settled"] = int(len(settled_df))
            perf["hit_rate"] = float(settled_df["won"].mean())

            # ROI (flat 1u and Kelly)
            perf["roi_flat"] = float(settled_df["pnl_flat"].sum() / max(1, len(settled_df)))
            total_kelly_staked = float(np.nansum(settled_df["kelly_frac"]))
            perf["roi_kelly"] = float(settled_df["pnl_kelly"].sum() / total_kelly_staked) if total_kelly_staked > 1e-9 else None

            # CLV (closing line value) placeholder if you later add closing odds:
            # perf["avg_clv_bp"] = ...

            # Equity curves by date
            if "game_date" in settled_df.columns:
                sd = settled_df.copy()
                sd["date"] = pd.to_datetime(sd["game_date"]).dt.date
                ec = sd.groupby("date")[["pnl_flat","pnl_kelly"]].sum().cumsum().reset_index()
            else:
                ec = pd.DataFrame()

            # Calibration
            cal = _calibration_table(settled_df)

        else:
            perf["n_settled"] = 0
            perf["hit_rate"] = None
            perf["roi_flat"] = None
            perf["roi_kelly"] = None
            ec, cal = pd.DataFrame(), pd.DataFrame()
    else:
        perf["n_settled"] = 0
        perf["hit_rate"] = None
        perf["roi_flat"] = None
        perf["roi_kelly"] = None
        ec, cal = pd.DataFrame(), pd.DataFrame()

    # Append to history log
    if not settled_df.empty and args.history_csv:
        hist_path = Path(args.history_csv)
        append_cols = list(out_df.columns) + ["actual","won","pnl_flat","pnl_kelly","dec_odds"]
        to_append = settled_df[[c for c in append_cols if c in settled_df.columns]].copy()
        to_append["logged_at_utc"] = summary["generated_at"]
        if hist_path.exists():
            prev = pd.read_csv(hist_path, parse_dates=["game_date"], dayfirst=False)
            all_hist = pd.concat([prev, to_append], ignore_index=True)
            all_hist.to_csv(hist_path, index=False)
        else:
            to_append.to_csv(hist_path, index=False)

    # ---------- Charts ----------
    charts = []
    can_chart = (not args.no_charts) and _HAS_MPL

    out_dir = Path(args.out_html).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Edge distribution
    if can_chart and "pick_edge_bp" in out_df and len(out_df):
        plt.figure()
        out_df["pick_edge_bp"].hist(bins=30)
        plt.title("Distribution of Model Edge (basis points)")
        plt.xlabel("Edge (bps)")
        plt.ylabel("Count")
        plt.tight_layout()
        edge_png = out_dir / f"{Path(args.out_html).stem}_edge_distribution.png"
        plt.savefig(edge_png)
        plt.close()
        charts.append(edge_png.name)

    # Kelly distribution
    if can_chart and "kelly_frac" in out_df and len(out_df):
        plt.figure()
        out_df["kelly_frac"].hist(bins=30)
        plt.title("Distribution of Kelly Fraction")
        plt.xlabel("Kelly fraction")
        plt.ylabel("Count")
        plt.tight_layout()
        kelly_png = out_dir / f"{Path(args.out_html).stem}_kelly_distribution.png"
        plt.savefig(kelly_png)
        plt.close()
        charts.append(kelly_png.name)

    # Equity curve
    ec_png = None
    if can_chart and not ec.empty:
        plt.figure()
        plt.plot(ec["date"], ec["pnl_flat"], label="Flat 1u")
        if "pnl_kelly" in ec.columns and not ec["pnl_kelly"].isna().all():
            plt.plot(ec["date"], ec["pnl_kelly"], label="Kelly")
        plt.title("Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Units")
        plt.legend()
        plt.tight_layout()
        ec_png = out_dir / f"{Path(args.out_html).stem}_equity_curve.png"
        plt.savefig(ec_png)
        plt.close()
        charts.append(ec_png.name)

    # Calibration plot
    cal_png = None
    if can_chart and not cal.empty:
        plt.figure()
        plt.scatter(cal["pred_avg"], cal["win_rate"])
        # 45-degree line
        xs = np.linspace(0, 1, 100)
        plt.plot(xs, xs)
        plt.title("Calibration (Predicted vs Realized)")
        plt.xlabel("Predicted Win Probability")
        plt.ylabel("Realized Win Rate")
        plt.tight_layout()
        cal_png = out_dir / f"{Path(args.out_html).stem}_calibration.png"
        plt.savefig(cal_png)
        plt.close()
        charts.append(cal_png.name)

    # ---------- HTML ----------
    fmt = out_df.copy()
    if "game_date" in fmt.columns:
        fmt["game_date"] = pd.to_datetime(fmt["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    html_table = fmt.to_html(index=False, escape=False)

    perf_ul = "".join([
        f"<li><b>Settled bets:</b> {perf.get('n_settled', 0)}</li>",
        f"<li><b>Hit rate:</b> {perf['hit_rate']:.2%}</li>" if perf.get("hit_rate") is not None else "<li><b>Hit rate:</b> —</li>",
        f"<li><b>ROI (flat 1u):</b> {perf['roi_flat']:.3f} u/bet</li>" if perf.get("roi_flat") is not None else "<li><b>ROI (flat 1u):</b> —</li>",
        f"<li><b>ROI (Kelly):</b> {perf['roi_kelly']:.3f} u per 1u stake</li>" if perf.get("roi_kelly") is not None else "<li><b>ROI (Kelly):</b> —</li>",
        f"<li><b>History log:</b> {args.history_csv}</li>" if args.history_csv else "",
    ])

    html = f"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NBA Prop Picks — Betting Report</title>
<style>
 body {{ font-family: Arial, sans-serif; margin: 24px; }}
 h1, h2 {{ margin: 0.2rem 0; }}
 .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
 .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 16px; }}
 table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
 th, td {{ border: 1px solid #eee; padding: 6px; text-align: left; }}
 th {{ background: #f7f7f7; position: sticky; top: 0; }}
 img {{ max-width: 100%; height: auto; border-radius: 6px; border: 1px solid #eee; }}
 .small {{ color: #666; font-size: 12px; }}
</style>
</head>
<body>
<h1>NBA Prop Picks — Automated Report</h1>
<p class="small">Generated at {summary['generated_at']}</p>

<div class="grid">
  <div class="card">
    <h2>Summary</h2>
    <ul>
      <li><b>Input rows:</b> {summary['input_rows']}</li>
      <li><b>Rows after filters:</b> {summary['after_filters_rows']}</li>
      <li><b>Deduplicated bets:</b> {summary['deduped_rows']}</li>
      <li><b>Avg edge (bps):</b> {summary['avg_edge_bp'] if summary['avg_edge_bp'] is not None else '—'}</li>
      <li><b>Median edge (bps):</b> {summary['median_edge_bp'] if summary['median_edge_bp'] is not None else '—'}</li>
      <li><b>Avg Kelly fraction:</b> {summary['avg_kelly_frac'] if summary['avg_kelly_frac'] is not None else '—'}</li>
    </ul>
  </div>
  <div class="card">
    <h2>Breakdowns</h2>
    <p><b>Books:</b></p>
    <ul>{_dict_to_ul(summary['books'])}</ul>
    <p><b>Prop types:</b></p>
    <ul>{_dict_to_ul(summary['prop_types'])}</ul>
  </div>
</div>

<div class="card" style="margin-top: 16px;">
  <h2>Performance</h2>
  <ul>
    {perf_ul}
  </ul>
</div>

<div class="grid" style="margin-top: 16px;">
  {"".join(f'<div class="card"><img src="{c}" alt="{c}"></div>' for c in charts)}
</div>

<div class="card" style="margin-top: 16px;">
  <h2>Bets to Place (deduplicated)</h2>
  {html_table}
</div>
</body>
</html>
"""
    Path(args.out_html).write_text(html, encoding="utf-8")

    # ---------- JSON write ----------
    with open(args.out_json, "w") as f:
        json.dump({**summary, **{"performance": perf}}, f, indent=2)

    # ---------- Console summary ----------
    print("\n=== Bet Report Summary ===")
    print(f"Input rows: {summary['input_rows']}")
    print(f"Rows after filters: {summary['after_filters_rows']}")
    print(f"Deduplicated bets: {summary['deduped_rows']}")
    print(f"Avg edge (bp): {summary['avg_edge_bp']} | Median edge (bp): {summary['median_edge_bp']}")
    print(f"Avg Kelly fraction: {summary['avg_kelly_frac']}")
    print("\nPerformance:")
    print(f"  Settled: {perf.get('n_settled')}")
    print(f"  Hit rate: {perf.get('hit_rate')}")
    print(f"  ROI flat: {perf.get('roi_flat')}")
    print(f"  ROI Kelly: {perf.get('roi_kelly')}")
    print("\nSaved CSV:", Path(args.out_csv).resolve())
    print("Saved JSON:", Path(args.out_json).resolve())
    print("Saved HTML:", Path(args.out_html).resolve())
    if args.history_csv: print("Updated history log:", Path(args.history_csv).resolve())
    if not _HAS_MPL and not args.no_charts:
        print("Note: matplotlib not available; charts skipped. Install matplotlib to enable charts.")


if __name__ == "__main__":
    main()
