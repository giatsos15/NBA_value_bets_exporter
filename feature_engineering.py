#--cell 1--#
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# For data analysis
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# -- Cell 2 -- #
import json
import re
import requests
import pandas as pd
from datetime import datetime

class NBAOddsScraper:
    def __init__(self):
        self.session = requests.Session()
        self.setup_headers()
    
    def setup_headers(self):
        self.headers = {
            'accept': '*/*',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130.0.0.0 Safari/537.36',
            'referer': 'https://www.rotowire.com/',
        }

    def get_player_props_odds_wide_raw(self, book='mgm'):
        url = f"https://www.rotowire.com/betting/nba/player-props.php?book={book}"
        try:
            r = self.session.get(url, headers=self.headers)
            r.raise_for_status()
        except Exception as e:
            print(f"❌ Failed to GET odds page: {e}")
            return pd.DataFrame()

        matches = re.findall(r"data:\s*(\[\{.*?\}\])", r.text, flags=re.DOTALL)
        frames = []
        for m in matches:
            try:
                rows = json.loads(m)
                if isinstance(rows, list) and rows:
                    frames.append(pd.DataFrame(rows))
            except Exception:
                continue

        if not frames:
            print("⚠️ No odds JSON blocks found.")
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        base_cols = [c for c in ["name","gameID","playerID","firstName","lastName","team","opp","logo","playerLink"] if c in df.columns]
        other_cols = [c for c in df.columns if c not in base_cols]
        df = df[base_cols + other_cols]
        if "opp" in df.columns and "opponent" not in df.columns:
            df = df.rename(columns={"opp": "opponent"})
        df["asof_date"] = datetime.utcnow().strftime("%Y-%m-%d")
        if "game_date" not in df.columns:
            df["game_date"] = df["asof_date"]
        print(f"✅ Fetched {len(df)} odds rows | {len(df.columns)} columns")
        return df

def extract_betting_lines(df, markets=("pts", "reb", "ast")):
    def pick_line(row, market):
        m = market.lower()
        line = over = under = None
        for col in row.index:
            c = col.lower()
            if c.endswith(f"_{m}"):
                line = row[col]
            elif c.endswith(f"_{m}over"):
                over = row[col]
            elif c.endswith(f"_{m}under"):
                under = row[col]
        try:
            line = float(line) if line and str(line).strip().lower() not in ("", "none", "nan") else None
        except Exception:
            line = None
        return line, over, under

    lines = []
    for _, r in df.iterrows():
        name = r.get('name', '')
        for mk in markets:
            line, over, under = pick_line(r, mk)
            if line is not None:
                lines.append({
                    "player": name,
                    "stat": {"pts":"points","reb":"rebounds","ast":"assists"}[mk],
                    "line": line,
                    "over_odds": over,
                    "under_odds": under
                })
    return pd.DataFrame(lines)

# Usage:
scraper = NBAOddsScraper()
odds_df = scraper.get_player_props_odds_wide_raw(book="mgm")
betting_lines_df = extract_betting_lines(odds_df)

print(betting_lines_df.head(10))
print(odds_df.head(10))
odds_df.to_csv("rotowire_nba_player_props_odds_mgm.csv", index=False)

# pip install selenium webdriver-manager bs4 pandas lxml

import os, re, time, pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# ---------------- helpers ----------------

def _clean_list(xs):
    return [re.sub(r"\s+\(.*?\)\s*$", "", x) for x in xs]

def _try_click_consent(driver, timeout=6):
    XPATHS = [
        "//button[contains(.,'Accept')]",
        "//button[contains(.,'I Agree')]",
        "//button[contains(.,'Agree')]",
        "//button[contains(.,'Αποδοχή')]",
        "//button[contains(.,'Συμφωνώ')]",
    ]
    end = time.time() + timeout
    for xp in XPATHS:
        try:
            btn = WebDriverWait(driver, 2).until(EC.element_to_be_clickable((By.XPATH, xp)))
            btn.click()
            return True
        except Exception:
            if time.time() > end: break
    return False

def _progress_scroll(driver, steps=10, pause=0.8):
    h = driver.execute_script("return document.body.scrollHeight || document.documentElement.scrollHeight;")
    for i in range(1, steps + 1):
        y = int(h * i / steps)
        driver.execute_script(f"window.scrollTo(0, {y});")
        time.sleep(pause)

def _extract_team(side):
    team_el = side.select_one(".lineup__abbr, .lineup__team-name, .lineup__name")
    if team_el:
        return team_el.get_text(strip=True)
    logo = side.select_one("img[alt]")
    return (logo.get("alt") or "").strip() if logo else ""

def _extract_status(side):
    status_el = side.select_one(".lineup__status")
    txt = (status_el.get_text(" ", strip=True) if status_el else "").upper()
    if "CONFIRM" in txt:  return "CONFIRMED"
    if "EXPECT" in txt or "PROBABLE" in txt: return "EXPECTED"
    return "UNKNOWN"

def _extract_starters(side):
    # Try several variants for starters content
    containers = side.select(".lineup__list--starters, .lineup__list, .lineup__players")
    if not containers:
        containers = [side]

    names = []
    for blk in containers:
        for a in blk.select("a.lineup__player-link, .lineup__player a"):
            t = a.get_text(" ", strip=True)
            if t: names.append(t)
        if not names:
            for row in blk.select(".lineup__player"):
                t = row.get_text(" ", strip=True)
                if re.match(r"^(PG|SG|SF|PF|C)\b", t): names.append(t)
        if not names:
            for li in blk.select("li"):
                t = li.get_text(" ", strip=True)
                if re.match(r"^(PG|SG|SF|PF|C)\b", t): names.append(t)

    if not names:
        txt = side.get_text("\n", strip=True)
        names = re.findall(r"(?:^|\n)(?:PG|SG|SF|PF|C)\s+[^\n]+", txt)

    return _clean_list(names)[:5]

# ---------------- main ----------------

def fetch_rotowire_lineups_selenium(date: str | None = None,
                                    wait_sec: float = 14.0,
                                    headless: bool = False) -> pd.DataFrame:
    """
    Render Rotowire lineups & parse BOTH sides per game (global side selectors).
    Returns:
      game_time, team, side (AWAY/HOME), lineup_status, starters,
      starter_1..starter_5, lineup_confirmed (0/1)
    """
    base = "https://www.rotowire.com/basketball/nba-lineups.php"
    url = base if not date else f"{base}?date={date}"

    opts = Options()
    if headless: opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1400,1000")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--lang=en-US,en;q=0.9")
    opts.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
    driver.get(url)

    _try_click_consent(driver, timeout=6)
    time.sleep(1.2)
    try:
        WebDriverWait(driver, int(wait_sec)).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".lineup, .lineup.is-nba"))
        )
    except Exception:
        pass

    _progress_scroll(driver, steps=10, pause=0.8)
    time.sleep(1.0)

    # quick diagnostics
    blocks = driver.find_elements(By.CSS_SELECTOR, ".lineup.is-nba, .lineup")
    players = driver.find_elements(By.CSS_SELECTOR, ".lineup__player, a.lineup__player-link")
    print(f"diagnostics: lineup blocks={len(blocks)}, player nodes={len(players)}")

    html = driver.page_source
    os.makedirs("_rotowire_debug", exist_ok=True)
    with open("_rotowire_debug/last_lineups.html", "w", encoding="utf-8") as f:
        f.write(html)
    try:
        driver.save_screenshot("_rotowire_debug/last_lineups.png")
    except Exception:
        pass
    driver.quit()

    # -------- parse globally by side classes ----------
    soup = BeautifulSoup(html, "lxml")

    # game time map: find each game container time
    game_time_map = {}
    for gi, g in enumerate(soup.select(".lineup__main, .lineup.is-nba, .lineup")):
        t = g.select_one(".lineup__time, .game-time")
        game_time_map[id(g)] = t.get_text(strip=True) if t else ""

    # Select **visit/away** & **home** side boxes explicitly
    visit_sel = (
        '[class*="lineup__box"][class*="is-visit"], '
        '[class*="lineup__team"][class*="is-visit"], '
        '[class*="lineup__side"][class*="is-visit"], '
        '[class*="visit"]'
    )
    home_sel = (
        '[class*="lineup__box"][class*="is-home"], '
        '[class*="lineup__team"][class*="is-home"], '
        '[class*="lineup__side"][class*="is-home"], '
        '[class*="home"]'
    )

    visit_boxes = soup.select(visit_sel)
    home_boxes  = soup.select(home_sel)

    rows = []

    def add_rows(boxes, side_label):
        for box in boxes:
            # nearest parent game container for time
            parent = box.find_parent(lambda tag: tag.has_attr("class") and any(
                c in {"lineup__main","lineup","lineup is-nba"} for c in tag.get("class", [])
            ))
            game_time = game_time_map.get(id(parent), "") if parent else ""
            team = _extract_team(box)
            starters = _extract_starters(box)
            status = _extract_status(box)
            if starters or team:
                rows.append({
                    "game_time": game_time,
                    "team": team,
                    "side": side_label,
                    "lineup_status": status,
                    "starters": starters,
                    "starter_1": starters[0] if len(starters)>0 else None,
                    "starter_2": starters[1] if len(starters)>1 else None,
                    "starter_3": starters[2] if len(starters)>2 else None,
                    "starter_4": starters[3] if len(starters)>3 else None,
                    "starter_5": starters[4] if len(starters)>4 else None,
                    "lineup_confirmed": int(status == "CONFIRMED"),
                })

    add_rows(visit_boxes, "AWAY")
    add_rows(home_boxes,  "HOME")

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.drop_duplicates(
            subset=["game_time","team","side","starter_1","starter_2","starter_3","starter_4","starter_5"]
        )
        all_na = df[["starter_1","starter_2","starter_3","starter_4","starter_5"]].isna().all(axis=1)
        df = df[~all_na].reset_index(drop=True)
    else:
        print("⚠️ Parsed zero rows. Check _rotowire_debug/last_lineups.html & .png")

    return df


# ---------- run it ----------
df_lineups = fetch_rotowire_lineups_selenium(wait_sec=14.0, headless=False)
print("✅ Shape:", df_lineups.shape)
print(df_lineups.sort_values(["game_time","side"]).head(12).to_string(index=False))

# pip install bs4 lxml pandas
import re, os, pandas as pd
from bs4 import BeautifulSoup

def _txt(x):
    return re.sub(r"\s+", " ", x.get_text(" ", strip=True)) if x else ""

def _clean_player(n):
    if not n: return n
    n = re.sub(r"\s+\(.*?\)\s*$", "", n).strip()
    n = re.sub(r"^(PG|SG|SF|PF|C)\s+", "", n, flags=re.I)
    return n

def _get_mnp_from_ul(ul):
    """Extract 'May Not Play' entries from a team UL."""
    mnp = []
    # Strategy 1: find the title li inside this UL, then collect following player lis until next title
    title = ul.find("li", class_=lambda c: c and "lineup__title" in c and re.search(r"may\s+not\s+play", _txt(ul.find("li", class_=c)) if ul.find("li", class_=c) else "", re.I))
    if title:
        for li in title.find_all_next("li"):
            # stop if next section title
            if "lineup__title" in (li.get("class") or []):
                break
            if "lineup__player" in (li.get("class") or []):
                a = li.select_one("a")
                tag = li.select_one(".lineup__inj")
                nm = _txt(a) if a else ""
                if nm:
                    mnp.append(f"{nm} ({_txt(tag)})" if tag else nm)
        # normalize
        return [_clean_player(x) for x in mnp if x and x.lower() != "none"]

    # Strategy 2: common MNP containers inside UL
    for li in ul.select(".lineup__notplay li, .lineup__status--out, .lineup__inj-list li"):
        nm = _txt(li)
        if nm: mnp.append(_clean_player(nm))
    return [x for x in mnp if x and x.lower() != "none"]

def _extract_starters_from_ul(ul):
    """Try multiple ways to get five starters out of a team UL."""
    names = []
    # Most reliable: 100% rows
    for li in ul.select("li.lineup__player.is-pct-play-100 a"):
        nm = _txt(li)
        if nm: names.append(nm)
    # Fallback: any lineup__player anchors in first list group
    if len(names) < 5:
        for li in ul.select("li.lineup__player a"):
            nm = _txt(li)
            if nm: names.append(nm)
            if len(names) >= 5: break
    # Final cleanup + trim
    names = [_clean_player(n) for n in names]
    return names[:5]

def _lineup_status(ul):
    st = _txt(ul.select_one(".lineup__status"))
    stU = st.upper()
    if "CONFIRM" in stU: return "CONFIRMED"
    if "EXPECT" in stU or "PROBABLE" in stU: return "EXPECTED"
    return "UNKNOWN"

def parse_rotowire_lineups_flexible(html_path: str) -> pd.DataFrame:
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()
    soup = BeautifulSoup(html, "lxml")

    # --- Diagnostics to understand the DOM you have ---
    diag = {
        "lineup__teams": len(soup.select("div.lineup__teams")),
        "ul.lineup__list": len(soup.select("ul.lineup__list")),
        "ul.is-visit": len(soup.select("ul.lineup__list.is-visit")),
        "ul.is-home": len(soup.select("ul.lineup__list.is-home")),
        "see-proj-minutes buttons": len(soup.select("button.see-proj-minutes")),
        "header abbr": len(soup.select(".lineup__hdr .lineup__abbr")),
        "header team": len(soup.select(".lineup__hdr .lineup__team")),
        "player anchors": len(soup.select("a.lineup__player-link, .lineup__player a")),
        "MNP titles": len(soup.find_all(string=re.compile(r"^\s*may\s+not\s+play\s*$", re.I))),
    }
    print("DOM diagnostics:", diag)

    rows = []

    # ========== STRATEGY A: by matchup blocks ==========
    for teams_div in soup.select("div.lineup__teams"):
        # game time near this block (looks upward for a sibling header)
        time_el = teams_div.find_previous("div", class_="lineup__time")
        game_time = _txt(time_el)

        # find both team ULs inside this matchup
        uls = teams_div.select("ul.lineup__list")
        if len(uls) < 1:
            continue

        # Try to pair AWAY then HOME by class flags; else preserve order
        away_ul = None
        home_ul = None
        for ul in uls:
            classes = " ".join(ul.get("class", [])).lower()
            if "is-visit" in classes or "visit" in classes or "away" in classes:
                away_ul = ul
            if "is-home" in classes or "home" in classes:
                home_ul = home_ul or ul  # keep the first

        if away_ul is None and home_ul is None and len(uls) >= 2:
            away_ul, home_ul = uls[0], uls[1]
        elif away_ul is None and len(uls) >= 1:
            away_ul = uls[0]
        elif home_ul is None and len(uls) >= 2:
            # pick the other UL as home
            home_ul = next((u for u in uls if u is not away_ul), None)

        pairs = [("AWAY", away_ul), ("HOME", home_ul)]
        # Extract team code (prefer button data-team; else header abbrs in the same matchup)
        header_abbrs = [ _txt(el) for el in teams_div.select(".lineup__abbr") if _txt(el) ]
        # If header not inside teams_div, try its parent block
        if not header_abbrs:
            parent_main = teams_div.find_parent(["div","section"])
            if parent_main:
                header_abbrs = [ _txt(el) for el in parent_main.select(".lineup__abbr") if _txt(el) ]

        for idx, (side, ul) in enumerate(pairs):
            if not ul: continue
            btn = ul.select_one("button.see-proj-minutes")
            team = btn["data-team"].strip().upper() if btn and btn.has_attr("data-team") else None
            if not team and header_abbrs and idx < len(header_abbrs):
                team = header_abbrs[idx].upper()

            starters = _extract_starters_from_ul(ul)
            mnp = _get_mnp_from_ul(ul)
            status = _lineup_status(ul)

            # Only add if we have at least a team or any player info
            if team or starters or mnp:
                rows.append({
                    "game_time": game_time,
                    "team": team,
                    "side": side,
                    "lineup_status": status,
                    "starters": starters,
                    "may_not_play": mnp,
                    "may_not_play_count": len(mnp),
                    "lineup_confirmed": int(status == "CONFIRMED"),
                })

    # ========== STRATEGY B: fall back to any lineup ULs globally ==========
    if not rows:
        print("Fallback B: scanning all ul.lineup__list globally...")
        for ul in soup.select("ul.lineup__list"):
            # Guess side by class or position among siblings
            side = "AWAY" if "is-visit" in (ul.get("class") or []) else ("HOME" if "is-home" in (ul.get("class") or []) else None)
            # Team from button
            btn = ul.select_one("button.see-proj-minutes")
            team = btn["data-team"].strip().upper() if btn and btn.has_attr("data-team") else None
            starters = _extract_starters_from_ul(ul)
            mnp = _get_mnp_from_ul(ul)
            status = _lineup_status(ul)

            if side and (team or starters or mnp):
                rows.append({
                    "game_time": "",  # unknown at this scope
                    "team": team,
                    "side": side,
                    "lineup_status": status,
                    "starters": starters,
                    "may_not_play": mnp,
                    "may_not_play_count": len(mnp),
                    "lineup_confirmed": int(status == "CONFIRMED"),
                })

    # ========== STRATEGY C: header-driven pairing (very defensive) ==========
    if not rows:
        print("Fallback C: pairing by header labels and nearest lists...")
        for block in soup.select(".lineup, .lineup__main"):
            hdr = block.select(".lineup__hdr .lineup__abbr, .lineup__hdr .lineup__team")
            labels = [ _txt(x) for x in hdr if _txt(x) ]
            if len(labels) < 2:
                continue
            away_label, home_label = labels[:2]
            lists = block.select("ul.lineup__list")
            if len(lists) < 2:
                continue
            for side, lab, ul in [("AWAY", away_label, lists[0]), ("HOME", home_label, lists[1])]:
                starters = _extract_starters_from_ul(ul)
                mnp = _get_mnp_from_ul(ul)
                status = _lineup_status(ul)
                rows.append({
                    "game_time": _txt(block.select_one(".lineup__time, .game-time")),
                    "team": lab.upper(),
                    "side": side,
                    "lineup_status": status,
                    "starters": starters,
                    "may_not_play": mnp,
                    "may_not_play_count": len(mnp),
                    "lineup_confirmed": int(status == "CONFIRMED"),
                })

    df = pd.DataFrame(rows)
    # Expand starters to columns for easier merging
    for i in range(5):
        col = f"starter_{i+1}"
        df[col] = df["starters"].apply(lambda xs: xs[i] if isinstance(xs, list) and len(xs) > i else None)

    print(f"→ Parsed rows: {len(df)}")
    return df

# ---- RUN IT (point to your saved file) ----
HTML_PATH = "_rotowire_debug/last_lineups.html"  # change if needed
if not os.path.exists(HTML_PATH):
    # if you uploaded as 'last_lineups.html' in current directory
    if os.path.exists("last_lineups.html"):
        HTML_PATH = "last_lineups.html"

df_lineups = parse_rotowire_lineups_flexible(HTML_PATH)

# Safe display
if df_lineups.empty:
    print("\n⚠️ Still empty. Please share the values printed in 'DOM diagnostics' (above).")
else:
    cols = ["game_time","team","side","lineup_status","may_not_play_count",
            "starter_1","starter_2","starter_3","starter_4","starter_5"]
    print("\n✅ Preview:")
    print(df_lineups[cols].sort_values(["game_time","side","team"], na_position="last").to_string(index=False))


# pip install bs4 lxml pandas
import os, re, pandas as pd
from bs4 import BeautifulSoup

HTML_PATH = "_rotowire_debug/last_lineups.html" if os.path.exists("_rotowire_debug/last_lineups.html") else "last_lineups.html"

LIKELIHOOD_MAP = {
    "is-pct-play-100": 100, "is-pct-play-90": 90, "is-pct-play-75": 75,
    "is-pct-play-60": 60, "is-pct-play-50": 50, "is-pct-play-40": 40,
    "is-pct-play-25": 25, "is-pct-play-10": 10, "is-pct-play-0": 0
}

def _txt(node): return re.sub(r"\s+", " ", node.get_text(" ", strip=True)) if node else ""
def _likelihood(classes): 
    for c in classes: 
        if c in LIKELIHOOD_MAP: 
            return LIKELIHOOD_MAP[c]
    return None

def parse_rotowire_mnp_final(html_path: str) -> pd.DataFrame:
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "lxml")

    rows = []
    games = soup.select("div.lineup.is-nba[data-lnum]")
    print(f"Found {len(games)} games in HTML.")

    for game in games:
        game_time = _txt(game.select_one(".lineup__time"))
        team_blocks = game.select(".lineup__team")
        teams = []
        for tb in team_blocks:
            abbr = _txt(tb.select_one(".lineup__abbr"))
            side = "AWAY" if "is-visit" in tb.get("class", []) else "HOME" if "is-home" in tb.get("class", []) else None
            teams.append((abbr, side))

        ul_lists = game.select("ul.lineup__list")
        for idx, ul in enumerate(ul_lists):
            if idx >= len(teams):  # mismatch safety
                continue
            team, side = teams[idx]
            mnp_title = ul.find("li", class_="lineup__title", string=lambda s: s and "MAY NOT PLAY" in s.upper())
            if not mnp_title:
                continue

            for li in mnp_title.find_next_siblings("li"):
                classes = li.get("class") or []
                if "lineup__title" in classes:
                    break
                if "lineup__player" not in classes:
                    continue

                pos = _txt(li.select_one(".lineup__pos"))
                a = li.select_one("a")
                player = _txt(a)
                if not player:
                    continue

                status = _txt(li.select_one(".lineup__inj"))
                title_text = (li.get("title") or "").strip()
                likelihood_pct = _likelihood(classes)

                rows.append({
                    "game_time": game_time,
                    "team": team,
                    "side": side,
                    "position": pos,
                    "player": player,
                    "status": status,
                    "title_text": title_text,
                    "likelihood_pct": likelihood_pct
                })

    df = pd.DataFrame(rows)
    if df.empty:
        print("⚠️ No 'May Not Play' players found. Check if Rotowire changed markup.")
    else:
        df = df.sort_values(["game_time","side","team","player"]).reset_index(drop=True)
        print(f"✅ Parsed {len(df)} 'May Not Play' players across {df['team'].nunique()} teams.")
    return df


# ---- RUN ----
mnp_df = parse_rotowire_mnp_final(HTML_PATH)
if not mnp_df.empty:
    print(mnp_df.head(30).to_string(index=False))
    mnp_df.to_csv("may_not_play_players.csv", index=False)
    print("\nSaved: may_not_play_players.csv")


def get_daily_matchups(date=None):
    """Get NBA games for a specific date"""
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    # Placeholder demo; replace with a real schedule API if desired
    sample_matchups = [
        {'home_team': 'GSW', 'away_team': 'LAL', 'time': '7:30 PM ET'},
        {'home_team': 'BOS', 'away_team': 'MIA', 'time': '8:00 PM ET'},
        {'home_team': 'DEN', 'away_team': 'DAL', 'time': '9:00 PM ET'},
    ]
    return sample_matchups

def calculate_player_correlations(player_a_logs, player_b_logs):
    """Calculate correlation between two players' performances"""
    merged = pd.merge(player_a_logs, player_b_logs, on='GAME_DATE', suffixes=('_a', '_b'))
    correlations = {}
    for stat in ['PTS', 'REB', 'AST']:
        if f'{stat}_a' in merged.columns and f'{stat}_b' in merged.columns:
            corr = merged[f'{stat}_a'].corr(merged[f'{stat}_b'])
            correlations[stat] = corr
    return correlations

# Export results to Excel
def export_analysis(results, filename='nba_betting_analysis.xlsx'):
    """Export analysis results to Excel"""
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        if 'value_bets' in results:
            pd.DataFrame(results['value_bets']).to_excel(writer, sheet_name='Value_Bets', index=False)
        if 'predictions' in results:
            predictions_df = pd.DataFrame.from_dict(results['predictions'], orient='index')
            predictions_df.to_excel(writer, sheet_name='Player_Predictions')
    print(f"Analysis exported to {filename}")


from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pytz

# --- helper: time-aware probability based on hours until tipoff --- #
def compute_time_based_prob(game_time_str: str, lineup_status: str) -> float:
    try:
        # Parse game_time (e.g. '7:30 PM ET')
        # Convert to 24h + assume Eastern time (Rotowire always ET)
        if not game_time_str:
            return 0.7  # fallback
        
        game_time_clean = game_time_str.replace("ET", "").strip()
        game_dt = datetime.strptime(game_time_clean, "%I:%M %p")
        today = datetime.now(pytz.timezone("US/Eastern"))
        game_dt = today.replace(hour=game_dt.hour, minute=game_dt.minute, second=0, microsecond=0)
        
        hours_to_tip = (game_dt - today).total_seconds() / 3600.0
        
        # If game already started or passed midnight, assume next day
        if hours_to_tip < -3:
            game_dt += timedelta(days=1)
            hours_to_tip = (game_dt - today).total_seconds() / 3600.0
    except Exception:
        hours_to_tip = 6.0  # fallback if parsing fails

    # Map to base probability range
    if lineup_status == "CONFIRMED":
        return 1.0
    elif lineup_status == "EXPECTED":
        if hours_to_tip > 6: return 0.7
        elif 2 < hours_to_tip <= 6: return 0.85
        else: return 0.95
    else:
        return 0.6 if hours_to_tip > 4 else 0.8


# --- build starter + time-aware probability flags --- #
def build_starter_flags_timeaware(df_lineups: pd.DataFrame, mnp_df: pd.DataFrame) -> pd.DataFrame:
    mnp_players = set(mnp_df["player"].str.strip()) if not mnp_df.empty else set()
    starter_rows = []

    for _, row in df_lineups.iterrows():
        team = row['team']
        lineup_status = row['lineup_status']
        game_time = row.get('game_time', '')

        for p in row['starters']:
            p_clean = p.strip()
            prob = compute_time_based_prob(game_time, lineup_status)
            if p_clean in mnp_players:
                prob *= 0.6  # penalize if on injury/MNP list
            starter_rows.append({
                "player": p_clean,
                "team": team,
                "is_starter": 1,
                "start_prob": round(prob, 2)
            })

    df_out = pd.DataFrame(starter_rows).drop_duplicates(subset=["player"])
    print(f"✅ Created {len(df_out)} starter probability rows.")
    return df_out


# --- build injury flags (unchanged) --- #
def build_injury_flags(mnp_df: pd.DataFrame) -> pd.DataFrame:
    if mnp_df.empty:
        return pd.DataFrame(columns=["player", "may_not_play", "injury_prob"])
    return (
        mnp_df
        .dropna(subset=["player"])
        .assign(
            player=lambda d: d["player"].str.strip(),
            injury_prob=lambda d: d["likelihood_pct"].fillna(40) / 100.0,
            may_not_play=1
        )
        [["player", "may_not_play", "injury_prob"]]
        .drop_duplicates(subset=["player"])
    )


# --- RUN --- #
starter_flags_df = build_starter_flags_timeaware(df_lineups, mnp_df)
injury_flags_df = build_injury_flags(mnp_df)

print("✅ Starter flags sample:")
print(starter_flags_df.head(10).to_string(index=False))

print("\n✅ Injury flags sample:")
print(injury_flags_df.head(10).to_string(index=False))


#--cell 4--#
import requests
import pandas as pd
import time
import random

url = "https://stats.nba.com/stats/leaguedashplayerstats"

base_params = {
    "College": "",
    "Conference": "",
    "Country": "",
    "DateFrom": "",
    "DateTo": "",
    "Division": "",
    "DraftPick": "",
    "DraftYear": "",
    "GameScope": "",
    "GameSegment": "",
    "Height": "",
    "ISTRound": "",
    "LastNGames": "0",
    "LeagueID": "00",
    "Location": "",
    "MeasureType": "Base",
    "Month": "0",
    "OpponentTeamID": "0",
    "Outcome": "",
    "PORound": "0",
    "PaceAdjust": "N",
    "PerMode": "PerGame",
    "Period": "0",
    "PlayerExperience": "",
    "PlayerPosition": "",
    "PlusMinus": "N",
    "Rank": "N",
    "SeasonSegment": "",
    "SeasonType": "Regular Season",
    "ShotClockRange": "",
    "StarterBench": "",
    "TeamID": "0",
    "VsConference": "",
    "VsDivision": "",
    "Weight": ""
}

headers = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/141.0.0.0 Safari/537.36",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true"
}

seasons = ["2023-24", "2024-25"]

def fetch_season_data(season, retries=3):
    """Fetch one season’s player stats, retrying if timeout or network error."""
    params = base_params.copy()
    params["Season"] = season

    for attempt in range(1, retries + 1):
        try:
            print(f"→ Attempt {attempt} fetching {season} data...")
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            print(f"⚠️ Timeout on attempt {attempt}/{retries} for {season}. Retrying...")
            time.sleep(3 * attempt)
        except requests.exceptions.RequestException as e:
            print(f"❌ Error on attempt {attempt}/{retries}: {e}")
            time.sleep(3 * attempt)
    raise RuntimeError(f"Failed to fetch {season} data after {retries} attempts.")

# Main loop
for season in seasons:
    print(f"\n🏀 Fetching NBA stats for {season}...")
    data = fetch_season_data(season)

    headers_list = data["resultSets"][0]["headers"]
    rows = data["resultSets"][0]["rowSet"]

    df = pd.DataFrame(rows, columns=headers_list)
    filename = f"nba_player_stats_{season.replace('-', '_')}.csv"
    df.to_csv(filename, index=False)

    print(f"✅ {season}: saved {len(df)} player records to '{filename}'")

    # Wait 3–6 seconds before next season to avoid throttling
    time.sleep(random.uniform(3, 6))

print("\n🎉 Done! Both 2023-24 and 2024-25 seasons downloaded.")


#--cell 5--#
import requests
import pandas as pd
import time

def get_box_scores(season, season_type="Regular Season"):
    url = "https://stats.nba.com/stats/leaguegamelog"
    params = {
        "Counter": 1000,
        "DateFrom": "",
        "DateTo": "",
        "Direction": "DESC",
        "ISTRound": "",
        "LeagueID": "00",
        "PlayerOrTeam": "P",
        "Season": season,
        "SeasonType": season_type,
        "Sorter": "DATE"
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
        "Referer": "https://www.nba.com/",
        "Origin": "https://www.nba.com",
        "Accept": "application/json, text/plain, */*"
    }

    response = requests.get(url, params=params, headers=headers, timeout=30)
    response.raise_for_status()

    data = response.json()["resultSets"][0]
    df = pd.DataFrame(data["rowSet"], columns=data["headers"])
    return df

# Get all three seasons
seasons = ["2023-24", "2024-25", "2025-26"]
for season in seasons:
    print(f"Fetching {season}...")
    df = get_box_scores(season)
    df.to_csv(f"nba_boxscores_{season}.csv", index=False)
    print(f"✅ Saved {len(df)} records for {season}")
    time.sleep(2)  # polite delay


#--cell 6--#
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import requests
import io
import unicodedata

# ---- Keep/Map settings -------------------------------------------------------

ADV_COLS_KEEP = [
    "Player", "Pos", "Age", "Tm", "G", "MP",
    "PER", "TS%", "3PAr", "FTr",
    "ORB%", "DRB%", "TRB%",
    "AST%", "STL%", "BLK%",
    "TOV%", "USG%",
    "ORtg", "DRtg",
    "OWS", "DWS", "WS", "WS/48",
    "OBPM", "DBPM", "BPM", "VORP"
]

# Basketball-Reference -> NBA/your dataset codes
TEAM_ABBR_MAP = {
    "BRK": "BKN",
    "PHO": "PHX",
    "CHO": "CHA",
    "UTH": "UTA",   # rare alias safety
    "NJN": "BKN",   # historical
    "SEA": "OKC",   # historical
    "VAN": "MEM",   # historical
}

# ---- Helpers -----------------------------------------------------------------

def normalize_name(s):
    """Normalize player names for consistent joining (lowercase, no accents/punct)."""
    if pd.isna(s):
        return s
    s = s.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    for ch in [".", "'", "`", "’", "“", "”", ","]:
        s = s.replace(ch, "")
    s = " ".join(s.split())
    return s

# ---- Fetch advanced table from Basketball-Reference --------------------------

def fetch_advanced_table(season=2026):
    """
    Fetch and clean Basketball-Reference advanced stats table for a given season.
    Example: season=2025 -> https://www.basketball-reference.com/leagues/NBA_2025_advanced.html
    """
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_advanced.html"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    tables = pd.read_html(io.StringIO(resp.text), header=0)
    if not tables:
        raise RuntimeError("No tables found on Basketball-Reference page.")

    df = tables[0].copy()

    # Remove duplicate header rows
    if "Rk" in df.columns:
        df = df[df["Rk"] != "Rk"].copy()
        df.drop(columns=["Rk"], inplace=True, errors="ignore")

    # Normalize column names (strip and upper-case for easy access)
    df.columns = [c.strip() for c in df.columns]

    # Basketball Reference sometimes labels the team column differently — make sure it exists
    team_col = None
    for c in df.columns:
        if c.lower() in ["tm", "team", "team_name"]:
            team_col = c
            break
    if not team_col:
        raise KeyError(f"Could not find a team column in advanced table. Found: {df.columns.tolist()}")
    df.rename(columns={team_col: "Tm"}, inplace=True)

    # Keep relevant columns if present
    keep = [c for c in ADV_COLS_KEEP if c in df.columns]
    df = df[keep].copy()

    # Convert numeric columns
    non_numeric = {"Player", "Pos", "Tm"}
    for c in [c for c in df.columns if c not in non_numeric]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Map team abbreviations to match your dataset
    df["Tm"] = df["Tm"].replace(TEAM_ABBR_MAP)

    # Add join keys
    df["player_key"] = df["Player"].map(normalize_name)
    df["team_key"] = df["Tm"].astype(str).str.strip().str.upper()

    return df

# ---- Load your averages CSV and align columns --------------------------------

def load_averages_csv(path):
    """
    Load your NBA averages CSV (with headers like PLAYER_NAME, TEAM_ABBREVIATION).
    Renames to canonical 'Player' and 'Team' and adds join keys.
    """
    df = pd.read_csv(path)

    # Auto-map your headers to canonical names
    col_map = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl == "player_name":
            col_map[c] = "Player"
        elif cl in ("team_abbreviation", "tm", "team"):
            col_map[c] = "Team"
        # keep other columns as-is

    df = df.rename(columns=col_map)

    if "Player" not in df.columns or "Team" not in df.columns:
        raise ValueError(
            "Couldn't find columns for 'Player' and 'Team'. "
            f"Available columns: {list(df.columns)}"
        )

    # Join keys
    df["player_key"] = df["Player"].map(normalize_name)
    df["team_key"] = df["Team"].astype(str).str.strip().str.upper()

    return df

# ---- Merge logic (with TOT fallback for traded players) ----------------------

def merge_advanced_into_averages(df_avg, df_adv):
    """
    Merge advanced metrics into averages.
    1) Exact Player+Team match (ignore TOT).
    2) For remaining NaNs, fill from TOT row by Player.
    """
    adv_team = df_adv[df_adv["Tm"] != "TOT"].copy()
    adv_tot  = df_adv[df_adv["Tm"] == "TOT"].copy()

    adv_cols_to_add = [c for c in df_adv.columns if c not in {"Player", "Pos", "Age", "Tm", "player_key", "team_key"}]
    meta_cols = [c for c in ["Pos", "Age"] if c in df_adv.columns]
    join_cols_full = meta_cols + adv_cols_to_add

    merged = df_avg.merge(
        adv_team[["player_key", "team_key"] + join_cols_full],
        on=["player_key", "team_key"],
        how="left",
        suffixes=("", "_adv"),
    )

    # Determine "missing" based on a representative advanced column
    probe_col = "PER" if "PER" in merged.columns else ("WS/48" if "WS/48" in merged.columns else None)
    missing_mask = merged[probe_col].isna() if probe_col else merged.isna().any(axis=1)

    if missing_mask.any() and not adv_tot.empty:
        fallback = merged[missing_mask].merge(
            adv_tot[["player_key"] + join_cols_full],
            on="player_key",
            how="left",
            suffixes=("", "_tot"),
        )
        for col in join_cols_full:
            if col in merged.columns and col in fallback.columns:
                merged.loc[missing_mask, col] = merged.loc[missing_mask, col].fillna(fallback[col])

    return merged

# ==============================================================================
# Example usage for your two files
# ==============================================================================

# ---- 2023–24 (Basketball-Reference season code = 2024) -----------------------
df_avg_2024 = load_averages_csv("nba_player_stats_2023_24.csv")
df_adv_2024 = fetch_advanced_table(season=2024)
df_enriched_2024 = merge_advanced_into_averages(df_avg_2024, df_adv_2024)
df_enriched_2024.to_csv("nba_player_stats_2023_24_enriched.csv", index=False)
print("✅ Saved: nba_player_stats_2023_24_enriched.csv")

# ---- 2024–25 (Basketball-Reference season code = 2025) -----------------------
df_avg_2025 = load_averages_csv("nba_player_stats_2024_25.csv")
df_adv_2025 = fetch_advanced_table(season=2025)
df_enriched_2025 = merge_advanced_into_averages(df_avg_2025, df_adv_2025)
df_enriched_2025.to_csv("nba_player_stats_2024_25_enriched.csv", index=False)
print("✅ Saved: nba_player_stats_2024_25_enriched.csv")

# ---- (Optional) Combine both seasons into one file ---------------------------
df_combined = pd.concat([df_enriched_2024, df_enriched_2025], ignore_index=True)
df_combined.to_csv("nba_player_stats_2023_25_combined.csv", index=False)
print("🏀 Combined: nba_player_stats_2023_25_combined.csv")


#--cell 7--#
import pandas as pd
import numpy as np

# -----------------------------
# Input file assumptions:
# - You have player game logs with at least:
#   ['GAME_DATE', 'PLAYER_NAME', 'TEAM_ABBREVIATION', 'OPPONENT_ABBREVIATION',
#    'MIN', 'PTS', 'REB', 'AST', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'TOV', 'BLK', 'STL', 'PLUS_MINUS', 'START_POSITION']
#   Column names can be auto-mapped below if they differ slightly (e.g., 'TEAM_ID' not needed).
# -----------------------------

def standardize_logs_cols(df_logs: pd.DataFrame) -> pd.DataFrame:
    colmap = {}
    for c in df_logs.columns:
        cl = c.strip().lower()
        if cl in ["game_date", "game_date_est", "date"]:
            colmap[c] = "GAME_DATE"
        elif cl in ["player", "player_name"]:
            colmap[c] = "PLAYER_NAME"
        elif cl in ["team", "team_abbreviation", "tm"]:
            colmap[c] = "TEAM_ABBREVIATION"
        elif cl in ["opp", "opponent", "opponent_abbreviation"]:
            colmap[c] = "OPPONENT_ABBREVIATION"
        elif cl in ["min", "minutes"]:
            colmap[c] = "MIN"
    df = df_logs.rename(columns=colmap).copy()
    # types
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["PLAYER_NAME", "GAME_DATE"])
    return df

def add_shooting_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    # Compute TS% from game logs (per game)
    # TS% = PTS / (2*(FGA + 0.44*FTA))
    for col in ["FGA", "FTA", "PTS"]:
        if col not in df.columns:
            df[col] = 0.0
    denom = 2 * (df["FGA"].astype(float) + 0.44 * df["FTA"].astype(float))
    df["TS_game"] = np.where(denom > 0, df["PTS"].astype(float) / denom, np.nan)
    return df

def rolling_player_form(df: pd.DataFrame, windows=(3,5,10,20)) -> pd.DataFrame:
    # Rolling stats per player before each game
    df = df.sort_values(["PLAYER_NAME", "GAME_DATE"]).copy()
    group = df.groupby("PLAYER_NAME", group_keys=False)
    for w in windows:
        for stat in ["PTS", "REB", "AST", "MIN", "TS_game"]:
            if stat not in df.columns:
                df[stat] = np.nan
            col = f"{stat}_roll{w}"
            df[col] = group[stat].shift(1).rolling(w, min_periods=1).mean()
    # recent usage proxy: last-5 share of team FGA
    if {"FGA","TEAM_ABBREVIATION"}.issubset(df.columns):
        df["teamFGA_game"] = df.groupby(["TEAM_ABBREVIATION","GAME_DATE"])["FGA"].transform("sum")
        df["usage_share"] = np.where(df["teamFGA_game"]>0, df["FGA"]/df["teamFGA_game"], np.nan)
        df["usage_share_roll5"] = group["usage_share"].shift(1).rolling(5, min_periods=1).mean()
    return df

def team_daily_ratings(df: pd.DataFrame, windows=(5,10)):
    # Build team-level ORtg/DRtg/Pace rolling using box score approximations
    # Possessions ≈ FGA + 0.44*FTA - OREB + TOV (OREB optional if present)
    need_cols = ["TEAM_ABBREVIATION","OPPONENT_ABBREVIATION","GAME_DATE","PTS","FGA","FTA","TOV","OREB"]
    for c in need_cols:
        if c not in df.columns:
            df[c] = 0.0
    # aggregate team totals per game
    g = df.groupby(["GAME_DATE","TEAM_ABBREVIATION"], as_index=False).agg(
        PTS_team=("PTS","sum"), FGA=("FGA","sum"), FTA=("FTA","sum"),
        TOV=("TOV","sum"), OREB=("OREB","sum")
    )
    g["poss"] = g["FGA"] + 0.44*g["FTA"] - g["OREB"] + g["TOV"]
    # opponent join to get DRtg inputs
    opp = g.rename(columns={
        "TEAM_ABBREVIATION":"OPPONENT_ABBREVIATION",
        "PTS_team":"PTS_opp",
        "poss":"poss_opp"
    })[["GAME_DATE","OPPONENT_ABBREVIATION","PTS_opp","poss_opp"]]
    g2 = g.merge(opp, on=["GAME_DATE"], how="left")
    # approximate per-team DRtg from opponent scoring
    g2["ORtg_g"] = np.where(g2["poss"]>0, 100*g2["PTS_team"]/g2["poss"], np.nan)
    g2["DRtg_g"] = np.where(g2["poss_opp"]>0, 100*g2["PTS_opp"]/g2["poss_opp"], np.nan)
    g2["Pace_g"] = (g2["poss"] + g2["poss_opp"]) / 2.0
    g2 = g2.sort_values(["TEAM_ABBREVIATION","GAME_DATE"])
    # rolling
    out = g2.copy()
    for w in windows:
        for stat in ["ORtg_g","DRtg_g","Pace_g"]:
            out[f"{stat}_roll{w}"] = out.groupby("TEAM_ABBREVIATION")[stat].shift(1).rolling(w, min_periods=1).mean()
    return out[["GAME_DATE","TEAM_ABBREVIATION","ORtg_g_roll5","DRtg_g_roll5","Pace_g_roll5",
                "ORtg_g_roll10","DRtg_g_roll10","Pace_g_roll10"]]

def opponent_position_allowances(df: pd.DataFrame, window=10):
    # How many points/assists/rebounds a team allows per opponent position (rolling)
    if "START_POSITION" not in df.columns:
        df["START_POSITION"] = np.nan  # if not available, this will be sparse
    base = df.groupby(["GAME_DATE","OPPONENT_ABBREVIATION","START_POSITION"], as_index=False)\
             .agg(PTS_allowed=("PTS","sum"), AST_allowed=("AST","sum"), REB_allowed=("REB","sum"))
    base = base.sort_values(["OPPONENT_ABBREVIATION","START_POSITION","GAME_DATE"])
    for w in [window]:
        for stat in ["PTS_allowed","AST_allowed","REB_allowed"]:
            base[f"{stat}_roll{w}"] = base.groupby(["OPPONENT_ABBREVIATION","START_POSITION"])[stat]\
                                            .shift(1).rolling(w, min_periods=3).mean()
    # pivot to wide per opponent (columns per position)
    wide = base.pivot_table(index=["GAME_DATE","OPPONENT_ABBREVIATION"],
                            columns="START_POSITION",
                            values=[f"PTS_allowed_roll{window}",f"AST_allowed_roll{window}",f"REB_allowed_roll{window}"])
    wide.columns = [f"{a}_{b}" for a,b in wide.columns.to_flat_index()]
    wide = wide.reset_index()
    return wide

def assemble_player_game_features(df_logs: pd.DataFrame, df_enriched_season: pd.DataFrame) -> pd.DataFrame:
    df = standardize_logs_cols(df_logs)
    df = add_shooting_efficiency(df)
    df = rolling_player_form(df)

    # Team rolling ratings
    tr = team_daily_ratings(df)
    df = df.merge(tr, on=["GAME_DATE","TEAM_ABBREVIATION"], how="left")

    # Opponent allowances by position
    oppw = opponent_position_allowances(df)
    df = df.merge(oppw, left_on=["GAME_DATE","OPPONENT_ABBREVIATION"], right_on=["GAME_DATE","OPPONENT_ABBREVIATION"], how="left")

    # Merge season-enriched averages (PER/TS%/USG%/ORtg/DRtg etc.)
    tmp = df_enriched_season.copy()
    # normalize keys like before
    def _norm(s):
        import unicodedata
        s = str(s).strip().lower()
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
        for ch in [".","'","`","’","“","”",","]:
            s = s.replace(ch,"")
        return " ".join(s.split())
    df["player_key"] = df["PLAYER_NAME"].map(_norm)
    df["team_key"] = df["TEAM_ABBREVIATION"].str.upper()

    tmp["player_key"] = tmp["Player"].map(_norm)
    tmp["team_key"] = tmp["Team"].astype(str).str.upper()

    keep_adv = [c for c in ["PER","TS%","USG%","ORtg","DRtg","WS/48","BPM","VORP","Pos","Age"] if c in tmp.columns]
    df = df.merge(tmp[["player_key","team_key"] + keep_adv], on=["player_key","team_key"], how="left")

    # simple situational flags
    if "MATCHUP" in df.columns:
        df["HOME"] = df["MATCHUP"].str.contains(" vs. ", regex=False).astype(int)
    else:
        df["HOME"] = np.nan  # placeholder

    # Days rest
    df["prev_date"] = df.groupby("PLAYER_NAME")["GAME_DATE"].shift(1)
    df["days_rest"] = (df["GAME_DATE"] - df["prev_date"]).dt.days
    df["is_b2b"] = (df["days_rest"] == 0).astype(int)

    # Targets: next-game minutes as well (you already have PTS/REB/AST)
    df = df.sort_values(["PLAYER_NAME","GAME_DATE"])
    for target, src in [("PTS_next","PTS"), ("REB_next","REB"), ("AST_next","AST"), ("MIN_next","MIN")]:
        if src not in df.columns:
            df[src] = np.nan
        df[target] = df.groupby("PLAYER_NAME")[src].shift(-1)

    return df


# -- Cell 0_data: load logs + build features_all ------------------------------
from xgboost import XGBRegressor  # ok to import here or later
import numpy as np
import pandas as pd

# 1) Load your season files
logs_2324     = pd.read_csv("nba_boxscores_2023-24.csv")
logs_2425     = pd.read_csv("nba_boxscores_2024-25.csv")
enriched_2324 = pd.read_csv("nba_player_stats_2023_24_enriched.csv")
enriched_2425 = pd.read_csv("nba_player_stats_2024_25_enriched.csv")

# 2) Assemble per-game features and concatenate
feat_2324 = assemble_player_game_features(logs_2324, enriched_2324)
feat_2425 = assemble_player_game_features(logs_2425, enriched_2425)
features_all = pd.concat([feat_2324, feat_2425], ignore_index=True)

# (Optional but recommended) enforce date dtype
if "GAME_DATE" in features_all.columns:
    features_all["GAME_DATE"] = pd.to_datetime(features_all["GAME_DATE"])

print("✅ features_all shape:", features_all.shape)


# -- Cell 5: Build TODAY minutes features ------------------------------------
import re
import numpy as np
import pandas as pd
from datetime import datetime

# Safety checks
assert 'features_all' in globals() and isinstance(features_all, pd.DataFrame) and not features_all.empty, \
    "features_all must exist (built from your game logs + enrichment)."
assert 'starter_flags_df' in globals(), "starter_flags_df missing. Run the previous cell that builds it."
assert 'injury_flags_df' in globals(), "injury_flags_df missing. Run the previous cell that builds it."

# --- light name normalizer (same idea you used elsewhere) ---
def _norm_player(name: str) -> str:
    if not isinstance(name, str): return ""
    s = re.sub(r"[.\-`'’]", "", name).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

# --- current roster snapshot (latest row per player) ---
# keeps the columns your minutes model expects + team + player id
keep_cols = [
    "PLAYER_ID","PLAYER_NAME","TEAM_ABBREVIATION","OPPONENT_ABBREVIATION","GAME_DATE",
    "MIN_roll5","MIN_roll10","days_rest","is_b2b","HOME",
    "USG%","TS%","PER","BPM"
]
keep_cols = [c for c in keep_cols if c in features_all.columns]

latest = (
    features_all.sort_values(["PLAYER_NAME","GAME_DATE"])
                .groupby("PLAYER_NAME", as_index=False)
                .tail(1)[keep_cols]
                .copy()
)

# keys for joining lineup/injury
latest["player_key"] = latest["PLAYER_NAME"].map(_norm_player)
latest = latest.rename(columns={
    "TEAM_ABBREVIATION": "team",
    "OPPONENT_ABBREVIATION": "opponent"
})

# --- starter flags (from lineup scrape) ---
sf = starter_flags_df.copy()
sf["player_key"] = sf["player"].map(_norm_player)
sf["team"] = sf["team"].str.upper().str.strip()

# --- injury flags (from MNP scrape) ---
inj = injury_flags_df.copy()
inj["player_key"] = inj["player"].map(_norm_player)

# --- merge player-level flags into today's roster snapshot ---
today = latest.merge(
    sf[["player_key","team","is_starter","start_prob"]],
    on=["player_key","team"], how="left"
).merge(
    inj[["player_key","may_not_play","injury_prob"]],
    on="player_key", how="left"
)

# defaults if missing
today["is_starter"]   = today["is_starter"].fillna(0).astype(int)
today["start_prob"]   = today["start_prob"].fillna(0.70)
today["may_not_play"] = today["may_not_play"].fillna(0).astype(int)
today["injury_prob"]  = today["injury_prob"].fillna(0.0)

# --- team-level absences (count + missing usage share) ---
# need each player's team + USG%
roster_for_abs = today[["player_key","team","USG%","injury_prob"]].copy()
roster_for_abs["usg_missing_expected"] = roster_for_abs["USG%"].fillna(0) * roster_for_abs["injury_prob"].clip(0,1)

by_team_abs = roster_for_abs.groupby("team", as_index=False).agg(
    teammates_out=("injury_prob", lambda s: float((s > 0).sum())),
    missing_usage_share=("usg_missing_expected","sum")
)

today = today.merge(by_team_abs, on="team", how="left")
today["teammates_out"] = today["teammates_out"].fillna(0.0)
today["missing_usage_share"] = today["missing_usage_share"].fillna(0.0)

# --- final TODAY minutes feature frame ---
features_today_minutes = today.copy()

print("✅ features_today_minutes shape:", features_today_minutes.shape)
print(features_today_minutes[[
    "PLAYER_NAME","team","opponent","is_starter","start_prob",
    "may_not_play","injury_prob","teammates_out","missing_usage_share",
    "MIN_roll5","MIN_roll10","days_rest","is_b2b","HOME","USG%","TS%","PER","BPM"
]].head(12).to_string(index=False))


# -- Cell 7.5: bridge/standardize minutes_today -------------------------------
import pandas as pd
import numpy as np
import re

def _find_minutes_df():
    # Search for likely minutes tables already created
    candidates = []
    for name, obj in globals().items():
        if not isinstance(obj, pd.DataFrame):
            continue
        lname = name.lower()
        if re.search(r"(minutes|mins)", lname) and re.search(r"(today|pred|proj|features)", lname):
            candidates.append((name, obj))
    # heuristic: prefer names containing "minutes_today" or "features_today_minutes"
    prio = ["minutes_today", "features_today_minutes", "minutes_pred", "minutes_predictions", "df_minutes_today"]
    for p in prio:
        for name, df in candidates:
            if name == p:
                return name, df
    return candidates[0] if candidates else (None, None)

name, df_src = _find_minutes_df()
print(f"Detected minutes source: {name or 'None'}")

if df_src is None or df_src.empty:
    # Build a conservative fallback from latest features (MIN_roll5 etc.)
    latest = (features_all.sort_values(["PLAYER_NAME","GAME_DATE"])
                        .groupby("PLAYER_NAME", as_index=False).tail(1).copy())

    # Fallback minutes = clipped MIN_roll5; higher if starter flags are available
    base_min = latest.get("MIN_roll5", pd.Series(24, index=latest.index)).fillna(24)
    base_min = base_min.clip(lower=10, upper=38)

    minutes_today = latest[["PLAYER_ID","PLAYER_NAME","TEAM_ABBREVIATION","OPPONENT_ABBREVIATION"]].copy()
    minutes_today["pred_minutes"] = base_min

    # If you created starter/injury flags earlier, merge them in
    for flags_name in ["starter_flags_df", "injury_flags_df"]:
        if flags_name in globals() and isinstance(globals()[flags_name], pd.DataFrame):
            flags = globals()[flags_name]
            key_cols = [c for c in ["PLAYER_ID","PLAYER_NAME"] if c in flags.columns]
            if key_cols:
                minutes_today = minutes_today.merge(
                    flags.drop_duplicates(subset=key_cols),
                    on=key_cols, how="left"
                )

    # Reasonable defaults
    minutes_today["start_prob"]  = minutes_today.get("start_prob", 0.75)
    minutes_today["is_starter"]  = minutes_today.get("is_starter", 0).fillna(0).astype(int)
    minutes_today["may_not_play"]= minutes_today.get("may_not_play", 0).fillna(0).astype(int)
    minutes_today["injury_prob"] = minutes_today.get("injury_prob", 0.0).fillna(0.0)

else:
    # Standardize from detected df
    df = df_src.copy()

    # Column mappers
    cmap = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ["player_id","id"]: cmap[c] = "PLAYER_ID"
        elif cl in ["player_name","player","name"]: cmap[c] = "PLAYER_NAME"
        elif cl in ["team","team_abbreviation","team_abbr","tm"]: cmap[c] = "TEAM_ABBREVIATION"
        elif cl in ["opponent","opp","opponent_abbreviation","opp_abbr"]: cmap[c] = "OPPONENT_ABBREVIATION"
        elif cl in ["pred_minutes","projected_minutes","minutes","mins","min_pred"]: cmap[c] = "pred_minutes"
        elif cl in ["start_prob","starter_prob","prob_start"]: cmap[c] = "start_prob"
        elif cl in ["is_starter","starter_flag","starter"]: cmap[c] = "is_starter"
        elif cl in ["may_not_play","dnp_flag","out_flag","likely_out"]: cmap[c] = "may_not_play"
        elif cl in ["injury_prob","inj_prob","p_injury"]: cmap[c] = "injury_prob"

    df = df.rename(columns=cmap)

    # If PLAYER_ID missing, merge from features_all by name
    if "PLAYER_ID" not in df.columns or df["PLAYER_ID"].isna().all():
        latest = (features_all.sort_values(["PLAYER_NAME","GAME_DATE"])
                            .groupby("PLAYER_NAME", as_index=False).tail(1)[
                                ["PLAYER_ID","PLAYER_NAME","TEAM_ABBREVIATION","OPPONENT_ABBREVIATION"]
                            ])
        on_cols = [c for c in ["PLAYER_NAME"] if c in df.columns]
        df = df.merge(latest, on=on_cols, how="left")

    # Ensure required columns exist with defaults
    req = ["PLAYER_ID","PLAYER_NAME","TEAM_ABBREVIATION","OPPONENT_ABBREVIATION",
           "pred_minutes","start_prob","is_starter","may_not_play","injury_prob"]
    for r in req:
        if r not in df.columns:
            if r == "pred_minutes":
                # fallback from features_all MIN_roll5
                base = (features_all.sort_values(["PLAYER_NAME","GAME_DATE"])
                                 .groupby("PLAYER_NAME", as_index=False).tail(1))
                base = base[["PLAYER_NAME","MIN_roll5"]].rename(columns={"MIN_roll5":"pred_minutes"})
                df = df.merge(base, on="PLAYER_NAME", how="left")
                df["pred_minutes"] = df["pred_minutes"].fillna(24).clip(10, 38)
            elif r in ["start_prob","injury_prob"]:
                df[r] = 0.75 if r=="start_prob" else 0.0
            elif r in ["is_starter","may_not_play"]:
                df[r] = 0
            else:
                df[r] = np.nan

    # Clean types
    df["pred_minutes"] = pd.to_numeric(df["pred_minutes"], errors="coerce").fillna(24).clip(0,48)
    df["is_starter"]   = df["is_starter"].fillna(0).astype(int)
    df["may_not_play"] = df["may_not_play"].fillna(0).astype(int)
    df["start_prob"]   = df["start_prob"].fillna(0.75)
    df["injury_prob"]  = df["injury_prob"].fillna(0.0)

    minutes_today = df[["PLAYER_ID","PLAYER_NAME","TEAM_ABBREVIATION","OPPONENT_ABBREVIATION",
                        "pred_minutes","start_prob","is_starter","may_not_play","injury_prob"]].copy()

print("minutes_today rows:", len(minutes_today))
print(minutes_today.head(8).to_string(index=False))


# -- Cell 8: per-minute rate models + today projections -----------------------
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

# ---------------------------------------------------------------------------
# 0) Safety checks / inputs from previous cells
# ---------------------------------------------------------------------------
assert 'features_all' in globals() and isinstance(features_all, pd.DataFrame) and not features_all.empty, \
    "features_all missing. Run Cell 7 + Cell 0_data first."
assert 'minutes_today' in globals() and isinstance(minutes_today, pd.DataFrame) and not minutes_today.empty, \
    "minutes_today missing. Run Cells 5–7 first."

# ---------------------------------------------------------------------------
# 1) Build leakage-safe rate targets for training
#    We predict per-minute rates (not raw totals); later multiply by pred_minutes.
# ---------------------------------------------------------------------------
train = features_all.copy()

# Ensure numeric MIN
if "MIN" not in train.columns:
    raise RuntimeError("MIN column not found in features_all.")
train["MIN"] = pd.to_numeric(train["MIN"], errors="coerce")

# Filter out super-low minute games (noisy rate)
train = train[train["MIN"].fillna(0) >= 6].copy()

# Targets as same-day rates (no look-ahead)
for stat in ["PTS","REB","AST"]:
    if stat not in train.columns:
        train[stat] = np.nan
    train[f"{stat}_per_min"] = train[stat] / train["MIN"].replace(0, np.nan)

# Drop impossible rows
for stat in ["PTS_per_min","REB_per_min","AST_per_min"]:
    train = train[~np.isinf(train[stat])].copy()

# ---------------------------------------------------------------------------
# 2) Feature set for rate models
#    Avoid using future minutes; keep context/skill/opponent/usage style features.
# ---------------------------------------------------------------------------
CANDIDATE_FEATURES = [
    # form / efficiency (all must be shift(1) upstream in features_all)
    "TS_game_roll5","TS_game_roll10",
    "MIN_roll5","MIN_roll10",            # ok: proxy for role, but target is rate not minutes
    "PTS_roll5","PTS_roll10",
    "REB_roll5","REB_roll10",
    "AST_roll5","AST_roll10",
    "usage_share_roll5",

    # season labels
    "PER","TS%","USG%","ORtg","DRtg","WS/48","BPM","VORP",

    # team/matchup context (shifted rolling at team level)
    "ORtg_g_roll5","DRtg_g_roll5","Pace_g_roll5",

    # optional opponent allowances by position (if present from Cell 7)
    # Common column names look like: PTS_allowed_roll10_PG, AST_allowed_roll10_C, etc.
    # We'll auto-include any *_allowed_roll10_* columns if present:
] + [c for c in features_all.columns if "_allowed_roll10_" in c]

# Situational flags (can influence rate a bit)
SITUATIONAL = ["HOME","days_rest","is_b2b"]
CANDIDATE_FEATURES += [c for c in SITUATIONAL if c in features_all.columns]

# Robust final feature list (present in the dataframe)
RATE_FEATURES = [c for c in CANDIDATE_FEATURES if c in train.columns]

print(f"Using {len(RATE_FEATURES)} features for rate models.")

# ---------------------------------------------------------------------------
# 3) Train one model per stat rate with GroupKFold by player
# ---------------------------------------------------------------------------
models_rate = {}
cv_scores_rate = {}
gkf = GroupKFold(n_splits=5)

def train_rate_model(df: pd.DataFrame, target_col: str, feat_cols: list[str]):
    df_ = df.dropna(subset=feat_cols + [target_col, "PLAYER_ID"]).copy()
    X = df_[feat_cols]
    y = df_[target_col]
    groups = df_["PLAYER_ID"]

    fold_mae = []
    for tr, te in gkf.split(X, y, groups):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]

        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            reg_alpha=0.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        fold_mae.append(mean_absolute_error(yte, pred))

    model.fit(X, y)  # final fit
    return model, float(np.mean(fold_mae)), float(np.std(fold_mae))

for stat in ["PTS","REB","AST"]:
    target = f"{stat}_per_min"
    model, m, s = train_rate_model(train, target, RATE_FEATURES)
    models_rate[stat] = model
    cv_scores_rate[stat] = (m, s)
    print(f"📏 {stat}_per_min MAE: {m:.4f} ± {s:.4f}")

# ---------------------------------------------------------------------------
# 4) Project TODAY's rates and totals = rate * predicted minutes
# ---------------------------------------------------------------------------
# Build today's feature frame: take latest per player and align with RATE_FEATURES.
latest_today = (
    features_all.sort_values(["PLAYER_NAME","GAME_DATE"])
                .groupby("PLAYER_NAME", as_index=False)
                .tail(1)
                .copy()
)

# Ensure all feature columns exist; if not, fill with neutral values
for c in RATE_FEATURES:
    if c not in latest_today.columns:
        latest_today[c] = 0.0

X_today = latest_today[RATE_FEATURES].copy()

# Join minutes prediction
mt = minutes_today.rename(columns={"PLAYER_NAME":"PLAYER_NAME_mins"})
proj_base = latest_today.merge(
    mt[["PLAYER_ID","pred_minutes","start_prob","is_starter","may_not_play","injury_prob"]],
    on="PLAYER_ID", how="left"
)

# If a player lacks a minutes prediction, give a conservative fallback
proj_base["pred_minutes"] = proj_base["pred_minutes"].fillna(proj_base.get("MIN_roll5", 24)).clip(0, 48)

# Predict rates
out_frames = []
for stat in ["PTS","REB","AST"]:
    mu_rate = models_rate[stat].predict(X_today)

    # Simple uncertainty model:
    #   - empirical residual SD in rate space per stat (global)
    #   - then combine with minutes variance (approx) for total SD
    # Build residual SD once from training data
    df_t = train.dropna(subset=RATE_FEATURES + [f"{stat}_per_min","PLAYER_ID"]).copy()
    pred_rate_t = models_rate[stat].predict(df_t[RATE_FEATURES])
    resid = (df_t[f"{stat}_per_min"] - pred_rate_t).values
    sd_rate = np.nanstd(resid, ddof=1) if len(resid) > 8 else 0.0

    # Minutes uncertainty proxy from minutes model signals:
    #   baseline 3.0 min SD, boosted if not confirmed starter or has injury_prob
    min_sd = 3.0 \
             + 4.0*(1.0 - proj_base["start_prob"].fillna(0.7).values) \
             + 4.0*(proj_base["injury_prob"].fillna(0.0).values)

    # Combine:
    #   Var(total) ≈ Var(rate*min) ≈ E[min]^2 * Var(rate) + E[rate]^2 * Var(min)
    pred_min = proj_base["pred_minutes"].values
    var_total = (pred_min**2) * (sd_rate**2) + (mu_rate**2) * (min_sd**2)
    sd_total = np.sqrt(np.maximum(var_total, 1e-6))

    totals = mu_rate * pred_min

    df_out = pd.DataFrame({
        "PLAYER_ID": proj_base["PLAYER_ID"],
        "PLAYER_NAME": proj_base["PLAYER_NAME"],
        "TEAM_ABBREVIATION": proj_base["TEAM_ABBREVIATION"],
        "OPPONENT_ABBREVIATION": proj_base["OPPONENT_ABBREVIATION"],
        "market": stat,
        "projection_mean": totals,
        "projection_sd": sd_total,
        "pred_minutes": pred_min,
        "pred_rate": mu_rate,
        "start_prob": proj_base["start_prob"].round(2),
        "is_starter": proj_base["is_starter"].fillna(0).astype(int),
        "may_not_play": proj_base["may_not_play"].fillna(0).astype(int),
        "injury_prob": proj_base["injury_prob"].fillna(0.0).round(2)
    })
    out_frames.append(df_out)

df_projections_all = pd.concat(out_frames, ignore_index=True)

# Normalize export columns like your Cell 16 pipeline expects
df_projections_all = df_projections_all.rename(columns={
    "PLAYER_NAME": "player",
    "TEAM_ABBREVIATION": "team",
    "OPPONENT_ABBREVIATION": "opponent",
    "pred_minutes": "projected_minutes"
})
df_projections_all["game_date"] = pd.Timestamp.utcnow().strftime("%Y-%m-%d")

print("✅ Projections ready:")
print(df_projections_all.groupby("market")["player"].count().to_dict())
print(df_projections_all.head(9).to_string(index=False))


import pandas as pd
import numpy as np
from datetime import datetime

def american_to_prob(odds):
    if pd.isna(odds): return np.nan
    o = float(odds)
    return 100.0/(o+100.0) if o>0 else (-o)/(-o+100.0)

def devig_pair(p_over, p_under):
    if pd.isna(p_over) or pd.isna(p_under): return (np.nan, np.nan)
    s = p_over + p_under
    if s <= 0: return (np.nan, np.nan)
    return (p_over/s, p_under/s)

def kelly_fraction(p, american_odds, cap=0.25):
    if pd.isna(p) or pd.isna(american_odds): return 0.0
    o = float(american_odds)
    b = o/100.0 if o>0 else 100.0/(-o)
    f = (p*(b+1)-1)/b
    return float(max(0.0, min(f, cap)))

def ev_flat_over(p, american_odds):
    if pd.isna(p) or pd.isna(american_odds): return np.nan
    o = float(american_odds)
    win = o/100.0 if o>0 else 100.0/(-o)
    lose = 1.0
    return p*win - (1-p)*lose

# Normal CDF helper (if SciPy available) to turn mean/sd into p_over
try:
    from scipy.stats import norm
    def p_over_from_normal(mu, sd, line):
        if pd.isna(mu) or pd.isna(sd) or pd.isna(line) or sd <= 0: return np.nan
        return 1.0 - norm.cdf((line - mu)/sd)
except Exception:
    def p_over_from_normal(mu, sd, line): return np.nan

def build_value_bets_excel(
    df_projections, df_odds, outfile_path=None,
    join_keys=("player","team","opponent","market","line","book","game_date"),
    cap_kelly=0.25
):
    def _norm(x): return None if pd.isna(x) else str(x).strip()
    proj, odds = df_projections.copy(), df_odds.copy()
    for k in join_keys:
        if k in proj: proj[k] = proj[k].map(_norm)
        if k in odds: odds[k] = odds[k].map(_norm)

    merged = proj.merge(odds, on=list(join_keys), how="inner", suffixes=("", "_odds"))

    if "p_over_model" not in merged.columns or merged["p_over_model"].isna().all():
        merged["p_over_model"] = merged.apply(
            lambda r: p_over_from_normal(r.get("projection_mean"), r.get("projection_sd"), r.get("line")), axis=1
        )

    merged["p_over_imp"]  = merged["over_odds"].map(american_to_prob)
    merged["p_under_imp"] = merged["under_odds"].map(american_to_prob)
    merged[["p_over_fair","p_under_fair"]] = merged.apply(
        lambda r: pd.Series(devig_pair(r["p_over_imp"], r["p_under_imp"])), axis=1
    )

    merged["edge_over"]       = merged["p_over_model"] - merged["p_over_fair"]
    merged["kelly_frac_over"] = merged.apply(lambda r: kelly_fraction(r["p_over_model"], r["over_odds"], cap=cap_kelly), axis=1)
    merged["EV_over_1u"]      = merged.apply(lambda r: ev_flat_over(r["p_over_model"], r["over_odds"]), axis=1)
    merged["asof_date"]       = merged.get("asof_date") if "asof_date" in merged else datetime.utcnow().strftime("%Y-%m-%d")

    preferred = [
        "asof_date","game_date","book","player","team","opponent","market","line","lineup_status",
        "over_odds","under_odds","p_over_imp","p_under_imp","p_over_fair","p_under_fair","p_over_model",
        "edge_over","kelly_frac_over","EV_over_1u",
        "projected_minutes","projection_mean","projection_sd","start_prob",
        "opponent_allowance_idx","team_orating","opp_drating",
    ]
    cols = [c for c in preferred if c in merged.columns] + [c for c in merged.columns if c not in preferred]
    bets = merged[cols].sort_values(["edge_over","EV_over_1u"], ascending=False).reset_index(drop=True)

    summary = pd.DataFrame({
        "n_bets":[len(bets)],
        "avg_edge_pp":[bets["edge_over"].mean()*100.0 if len(bets) else np.nan],
        "avg_kelly_pct":[bets["kelly_frac_over"].mean()*100.0 if len(bets) else np.nan],
        "avg_ev_1u":[bets["EV_over_1u"].mean() if len(bets) else np.nan],
    })
    by_market = bets.groupby("market", dropna=False).agg(
        n=("player","count"),
        avg_edge_pp=("edge_over", lambda x: 100.0*x.mean()),
        avg_kelly_pct=("kelly_frac_over", lambda x: 100.0*x.mean()),
        avg_ev_1u=("EV_over_1u","mean")
    ).reset_index()
    by_book = bets.groupby("book", dropna=False).agg(
        n=("player","count"),
        avg_edge_pp=("edge_over", lambda x: 100.0*x.mean()),
        avg_ev_1u=("EV_over_1u","mean")
    ).reset_index()

    if outfile_path is None:
        outfile_path = f"nba_value_bets_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.xlsx"
    with pd.ExcelWriter(outfile_path, engine="openpyxl") as w:
        bets.to_excel(w, sheet_name="Bets", index=False)
        summary.to_excel(w, sheet_name="Summary", index=False, startrow=0)
        by_market.to_excel(w, sheet_name="Summary", index=False, startrow=5)
        by_book.to_excel(w, sheet_name="Summary", index=False, startrow=5+len(by_market)+3)

        dd = pd.DataFrame([
            ("asof_date","UTC run date"), ("game_date","Game date"),
            ("player","Player"), ("team","Team abbr"), ("opponent","Opponent abbr"),
            ("market","PTS/REB/AST/3PM/PRA etc."), ("line","Book line"), ("book","Sportsbook id"),
            ("lineup_status","EXPECTED/CONFIRMED/UNKNOWN"),
            ("over_odds","American odds Over"), ("under_odds","American odds Under"),
            ("p_over_imp","Implied prob Over (pre-vig)"), ("p_under_imp","Implied prob Under (pre-vig)"),
            ("p_over_fair","De-vigged prob Over"), ("p_under_fair","De-vigged prob Under"),
            ("p_over_model","Model prob Over"), ("edge_over","p_model − p_fair"),
            ("kelly_frac_over","Kelly fraction (cap)"), ("EV_over_1u","EV if staking 1u"),
            ("projected_minutes","Projected minutes"), ("projection_mean","Projected mean"),
            ("projection_sd","Projected stdev"), ("start_prob","Start probability"),
            ("opponent_allowance_idx","Opponent allowance index"),
            ("team_orating","Team ORtg"), ("opp_drating","Opponent DRtg"),
        ], columns=["column","description"])
        dd.to_excel(w, sheet_name="Data_Dictionary", index=False)

    return bets, outfile_path


# === 16: raw wide odds + resilient numeric parsing ===
import re, json, pandas as pd
from datetime import datetime

def _first_numeric_float(x):
    """Return the first decimal number in x (e.g., '23.5, 24.5' -> 23.5)."""
    if x is None: return None
    s = str(x)
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else None

def _first_numeric_int(x):
    """Return the first integer in x (e.g., '+110, +105' -> 110)."""
    if x is None: return None
    s = str(x)
    m = re.search(r"[-+]?\d+", s)
    return int(m.group()) if m else None

# override the helpers used by 16d converter (if defined)
def _to_float_or_none(x):  # noqa: F811
    return _first_numeric_float(x)

def _to_int_or_none(x):    # noqa: F811
    return _first_numeric_int(x)

def get_player_props_odds_wide_raw(self, book="mgm"):
    """
    Returns the raw 'wide' odds table from Rotowire (no grouping, no aggregation).
    Contains columns like mgm_pts, mgm_ptsOver, mgm_ptsUnder, etc.
    """
    url = f"https://www.rotowire.com/betting/nba/player-props.php?book={book}"
    r = self.session.get(url, headers=self.headers, timeout=30)
    r.raise_for_status()
    matches = re.findall(r'data:\s*(\[\{.*?\}\])', r.text, flags=re.DOTALL)
    frames = []
    for blob in matches:
        try:
            frames.append(pd.DataFrame(json.loads(blob)))
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    # concat all blocks without grouping to preserve raw book columns
    wide_raw = pd.concat(frames, ignore_index=True)
    return wide_raw

# attach to your scraper class
NBAOddsScraper.get_player_props_odds_wide_raw = get_player_props_odds_wide_raw

# --- Helper: turn wide props (per-book columns) into a long, tidy table ---
import re
import numpy as np
import pandas as pd

def odds_wide_to_long_from_columns(
    wide_df: pd.DataFrame,
    *,
    books: tuple[str, ...] = ("mgm","draftkings","fanduel","caesars","betrivers","espnbet","hardrock"),
    markets: tuple[str, ...] = ("PTS","REB","AST"),
    player_cols=("name","player","PLAYER_NAME"),
    team_cols=("team","TEAM","team_name","TEAM_ABBREVIATION"),
    opp_cols=("opponent","opp","OPPONENT","OPPONENT_ABBREVIATION"),
    date_cols=("game_date","GAME_DATE","date")
) -> pd.DataFrame:
    """
    Convert a 'wide' props frame into a tidy long format:
    one row per (player, market, book), with numeric line and American odds.

    Expected column patterns (flexible by regex):
      <book>_<suffix>                 -> the line (e.g., mgm_pts, fanduel_ast)
      <book>_<suffix>_over_odds       -> over odds (American)
      <book>_<suffix>_under_odds      -> under odds (American)

    Suffixes recognized per market:
      PTS:  'pts','points'
      REB:  'reb','rebounds'
      AST:  'ast','assists'
    """
    df = wide_df.copy()

    # Identify reference columns
    def _first_col(cands):
        for c in cands:
            if c in df.columns: return c
        return None

    player_col = _first_col(player_cols)
    team_col   = _first_col(team_cols)
    opp_col    = _first_col(opp_cols)
    date_col   = _first_col(date_cols)

    # Fallbacks if totally missing
    if player_col is None:
        raise ValueError("Could not find a player name column in wide_df. "
                         f"Tried {player_cols}. Got columns: {list(df.columns)[:20]}...")

    # Normalize helpers
    def _num_float(x):
        if pd.isna(x): return np.nan
        m = re.search(r"[-+]?\d+(?:\.\d+)?", str(x))
        return float(m.group()) if m else np.nan

    def _num_int(x):
        if pd.isna(x): return np.nan
        m = re.search(r"[-+]?\d+", str(x))
        return int(m.group()) if m else np.nan

    # Market suffix map (flex)
    market_suffixes = {
        "PTS": ("pts","points"),
        "REB": ("reb","rebounds"),
        "AST": ("ast","assists"),
    }

    # Build long rows
    long_rows = []
    # Iterate rows once; pull columns per book/market dynamically
    for _, row in df.iterrows():
        base = {
            "player": row[player_col],
            "team": row[team_col] if team_col else np.nan,
            "opponent": row[opp_col] if opp_col else np.nan,
            "game_date": row[date_col] if date_col else np.nan,
        }
        for mkt in markets:
            suffixes = market_suffixes.get(mkt, ())
            for b in books:
                # Find the *line* column by trying allowed suffixes
                line_val = np.nan
                over_val = np.nan
                under_val = np.nan
                line_col_used = None

                for suf in suffixes:
                    # exact line column (most common)
                    c_line = f"{b}_{suf}"
                    if c_line in df.columns and pd.notna(row[c_line]):
                        line_val = row[c_line]
                        line_col_used = c_line
                        # odds columns (several sites use these names)
                        for over_name in (f"{b}_{suf}_over_odds", f"{b}_{suf}_o_odds", f"{b}_{suf}_over"):
                            if over_name in df.columns:
                                over_val = row[over_name]
                                break
                        for under_name in (f"{b}_{suf}_under_odds", f"{b}_{suf}_u_odds", f"{b}_{suf}_under"):
                            if under_name in df.columns:
                                under_val = row[under_name]
                                break
                        break  # found a suffix match

                # If not found, try a looser search (e.g., 'mgm_pts_line')
                if (isinstance(line_val, float) and np.isnan(line_val)) or line_col_used is None:
                    pat = re.compile(rf"^{re.escape(b)}_({ '|'.join(map(re.escape, suffixes)) })(_line)?$", re.I)
                    for c in df.columns:
                        if pat.match(str(c)) and pd.notna(row[c]):
                            line_val = row[c]
                            line_col_used = c
                            # odds columns with same base
                            base_prefix = re.sub(r"(_line)?$", "", c)
                            for over_name in (f"{base_prefix}_over_odds", f"{base_prefix}_o_odds", f"{base_prefix}_over"):
                                if over_name in df.columns:
                                    over_val = row[over_name]
                                    break
                            for under_name in (f"{base_prefix}_under_odds", f"{base_prefix}_u_odds", f"{base_prefix}_under"):
                                if under_name in df.columns:
                                    under_val = row[under_name]
                                    break
                            break

                # Only emit a row if we actually found a line
                if pd.notna(line_val):
                    long_rows.append({
                        **base,
                        "market": mkt,
                        "book": b,
                        "line": _num_float(line_val),
                        "over_odds": _num_int(over_val),
                        "under_odds": _num_int(under_val),
                    })

    out = pd.DataFrame(long_rows)

    # Clean up: drop obviously invalid lines
    if not out.empty:
        out = out[pd.notna(out["line"])]
        # remove zero/negative lines that can't be real for these markets (optional)
        out = out[out["line"] > 0]

        # De-duplicate best-effort (sometimes the page contains duplicates per book)
        out = (out.sort_values(["player","market","book","line"])
                  .drop_duplicates(subset=["player","market","book"], keep="last")
                  .reset_index(drop=True))

    return out

# --- Cell 16a: robust wide->long adapter for Rotowire props ---
import re
import numpy as np
import pandas as pd

def odds_wide_to_long_rotowire(
    wide_df: pd.DataFrame,
    *,
    books=("mgm","draftkings","fanduel","caesars","betrivers","espnbet","hardrock"),
    markets=("PTS","REB","AST"),
    player_cols=("name","player","PLAYER_NAME"),
    team_cols=("team","TEAM_ABBREVIATION"),
    opp_cols=("opponent","opp","OPPONENT_ABBREVIATION"),
    date_cols=("game_date","GAME_DATE")
) -> pd.DataFrame:
    df = wide_df.copy()

    def _first_col(cols):
        for c in cols:
            if c in df.columns: return c
        return None

    ply = _first_col(player_cols)
    tm  = _first_col(team_cols)
    opp = _first_col(opp_cols)
    dt  = _first_col(date_cols)
    if ply is None:
        raise ValueError(f"No player column found. Tried {player_cols}. Got sample: {list(df.columns)[:25]}")

    # market suffixes we’ll search (order matters)
    suffixes = {"PTS": ("pts","p","points"),
                "REB": ("reb","rebounds"),
                "AST": ("ast","assists")}

    # helpers
    def _num_float(x):
        if pd.isna(x): return np.nan
        m = re.search(r"[-+]?\d+(?:\.\d+)?", str(x))
        return float(m.group()) if m else np.nan

    def _num_int(x):
        if pd.isna(x): return np.nan
        m = re.search(r"[-+]?\d+", str(x))
        return int(m.group()) if m else np.nan

    cols_lc = {c.lower(): c for c in df.columns}  # lower->actual

    def _find(name_like: str):
        return cols_lc.get(name_like.lower())

    rows = []
    for _, r in df.iterrows():
        base = {
            "player": r[ply],
            "team": r[tm] if tm else np.nan,
            "opponent": r[opp] if opp else np.nan,
            "game_date": r[dt] if dt else np.nan,
        }
        for mkt in markets:
            for book in books:
                ln = np.nan; ov = np.nan; un = np.nan; used = None
                # find the line column (e.g. mgm_pts / fanduel_p / caesars_ast)
                for suf in suffixes[mkt]:
                    for cand in (f"{book}_{suf}", f"{book}_{suf}_line"):
                        real = _find(cand)
                        if real and pd.notna(r.get(real)):
                            ln = r[real]; used = real
                            break
                    if used: break

                if used:
                    # odds columns around that base; support camel & underscore
                    base_prefix = re.sub(r"_line$", "", used, flags=re.I)
                    over_cands  = [f"{base_prefix}Over", f"{base_prefix}_over",
                                   f"{base_prefix}_o", f"{base_prefix}_over_odds"]
                    under_cands = [f"{base_prefix}Under", f"{base_prefix}_under",
                                   f"{base_prefix}_u", f"{base_prefix}_under_odds"]
                    for oc in over_cands:
                        c = _find(oc)
                        if c and pd.notna(r.get(c)): ov = r[c]; break
                    for uc in under_cands:
                        c = _find(uc)
                        if c and pd.notna(r.get(c)): un = r[c]; break

                    rows.append({
                        **base,
                        "market": mkt,
                        "book": book,
                        "line": _num_float(ln),
                        "over_odds": _num_int(ov),
                        "under_odds": _num_int(un),
                    })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out[pd.notna(out["line"]) & (out["line"] > 0)]
        out = (out.sort_values(["player","market","book","line"])
                 .drop_duplicates(subset=["player","market","book"], keep="last")
                 .reset_index(drop=True))
        if "game_date" in out and out["game_date"].isna().all():
            out["game_date"] = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    return out


# === Cell 16 (final v2): wide→long with robust Over/Under detection ===
import re, numpy as np, pandas as pd

def odds_wide_to_long_rotowire_final_v2(
    wide_df: pd.DataFrame,
    *,
    books=("mgm","draftkings","fanduel","caesars","betrivers","espnbet","hardrock"),
    markets=("PTS","REB","AST"),
    player_cols=("name","player","PLAYER_NAME"),
    team_cols=("team","TEAM_ABBREVIATION"),
    opp_cols=("opponent","opp","OPPONENT_ABBREVIATION"),
    date_cols=("game_date","GAME_DATE")
) -> pd.DataFrame:
    df = wide_df.copy()

    def _first_col(cands):
        for c in cands:
            if c in df.columns: return c
        return None

    ply=_first_col(player_cols)
    tm=_first_col(team_cols)
    opp=_first_col(opp_cols)
    dt=_first_col(date_cols)
    if ply is None:
        raise ValueError("No player column found.")

    suf_map = {"PTS":("pts","p","points"),"REB":("reb","rebounds"),"AST":("ast","assists")}

    def _num_float(x):
        if pd.isna(x): return np.nan
        m = re.search(r"[-+]?\d+(?:\.\d+)?", str(x))
        return float(m.group()) if m else np.nan

    def _num_int(x):
        if pd.isna(x): return np.nan
        m = re.search(r"[-+]?\d+", str(x))
        return int(m.group()) if m else np.nan

    rows=[]
    for _,r in df.iterrows():
        base={"player":r[ply],
              "team":r[tm] if tm else np.nan,
              "opponent":r[opp] if opp else np.nan,
              "game_date":r[dt] if dt else np.nan}

        for mkt in markets:
            for book in books:
                suffixes=suf_map[mkt]
                line=over=under=np.nan

                # ---- find base line column ----
                for suf in suffixes:
                    for cand in (f"{book}_{suf}", f"{book}_{suf}_line"):
                        if cand in df.columns and pd.notna(r[cand]):
                            line=r[cand]; break
                    if pd.notna(line): break

                # ---- find over/under columns (CamelCase or underscored) ----
                for c in df.columns:
                    cl=c.lower()
                    if re.match(fr"^{book}_.+({mkt.lower()}|{suf_map[mkt][0]}).*(over)$", cl):
                        if pd.notna(r[c]): over=r[c]
                    if re.match(fr"^{book}_.+({mkt.lower()}|{suf_map[mkt][0]}).*(under)$", cl):
                        if pd.notna(r[c]): under=r[c]

                if pd.isna(line) and pd.isna(over) and pd.isna(under):
                    continue

                rows.append({
                    **base,
                    "market":mkt,
                    "book":book,
                    "line":_num_float(line),
                    "over_odds":_num_int(over),
                    "under_odds":_num_int(under)
                })

    out=pd.DataFrame(rows)
    if not out.empty:
        out=out[out["line"].isna()|(out["line"]>0)]
        out=(out.sort_values(["player","market","book","line"])
               .drop_duplicates(subset=["player","market","book"],keep="last")
               .reset_index(drop=True))
        if out["game_date"].isna().all():
            out["game_date"]=pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    return out

# === Cell 17: fetch odds → long → join with model projections ===
from datetime import datetime
import re, unicodedata, numpy as np, pandas as pd
from statistics import NormalDist

scraper = NBAOddsScraper()
wide_raw = scraper.get_player_props_odds_wide_raw(book="mgm")
print(f"✅ Raw odds rows: {len(wide_raw)} | columns: {len(wide_raw.columns)}")

odds_long = odds_wide_to_long_rotowire_final_v2(
    wide_raw,
    books=("mgm","draftkings","fanduel","caesars","betrivers","espnbet","hardrock"),
    markets=("PTS","REB","AST")
)
print(f"Converted → long: {len(odds_long)} rows")
print("Non-null odds:",
      "\n  over_odds:", odds_long["over_odds"].notna().sum(),
      "\n  under_odds:", odds_long["under_odds"].notna().sum())
print(odds_long.head(10))

# --- Join with projections ---
if "df_projections_all" not in globals() or df_projections_all.empty:
    raise RuntimeError("df_projections_all missing – run projection cell first.")

def _norm_player(s):
    if not isinstance(s,str): return ""
    s=unicodedata.normalize("NFKD",s)
    s="".join(ch for ch in s if not unicodedata.combining(ch))
    s=re.sub(r"[.\-`'’]","",s).strip().lower()
    s=re.sub(r"\s+"," ",s)
    return s

odds_long["player_key"]=odds_long["player"].map(_norm_player)
df_projections_all["player_key"]=df_projections_all["player"].map(_norm_player)

def p_over_from_normal(mu,sd,line):
    if pd.isna(mu) or pd.isna(sd) or pd.isna(line) or sd<=0: return np.nan
    z=(line-mu)/sd
    return 1.0-NormalDist().cdf(z)
def implied_prob(o):
    if pd.isna(o): return np.nan
    o=int(o)
    return (-o)/(-o+100.0) if o<0 else 100.0/(o+100.0)

joined=[]
for mkt in ["PTS","REB","AST"]:
    proj=df_projections_all.query("market==@mkt")
    odds=odds_long.query("market==@mkt")
    if proj.empty or odds.empty:
        print(f"⚠️ Skipping {mkt}"); continue
    dfj=proj.merge(odds,on=["player_key","market"],how="inner",suffixes=("_proj","_odds"))
    if dfj.empty:
        print(f"⚠️ No matches for {mkt}"); continue
    dfj["p_over_model"]=dfj.apply(lambda r:p_over_from_normal(r.projection_mean,r.projection_sd,r.line),axis=1)
    dfj["p_over_imp"]=dfj["over_odds"].apply(implied_prob)
    dfj["p_under_imp"]=dfj["under_odds"].apply(implied_prob)
    dfj["edge_over"]=dfj["p_over_model"]-dfj["p_over_imp"]
    joined.append(dfj)

df_proj_join_all=pd.concat(joined,ignore_index=True) if joined else pd.DataFrame()
# print(f"Joined frame size: {len(df_proj_join_all)}")
# print(df_proj_join_all.head(10))


def peek_cols(df, book, key="pts"):
    cols = [c for c in df.columns if re.match(fr"^{book}_.{{0,12}}{key}", c, re.I) or c.lower().startswith(f"{book}_{key}")]
    print(book, key, "->", cols[:20])

peek_cols(wide_raw, "mgm", "pts")
peek_cols(wide_raw, "fanduel", "p")     # note the short 'p'
peek_cols(wide_raw, "caesars", "ast")

print(df_proj_join_all['under_odds'].unique())


# === Cell 17B: from df_proj_join_all -> priced slate with edge/EV/Kelly ===
import os, re
import numpy as np, pandas as pd
from datetime import datetime
from statistics import NormalDist

assert "df_proj_join_all" in globals() and not df_proj_join_all.empty, "Run Cell 17 first."

df = df_proj_join_all.copy()

# ---------------------------------------------------------------------
# 1️⃣  Canonicalize entity columns (player/team/opponent/game_date)
# ---------------------------------------------------------------------
def _coalesce(df_, target, candidates):
    s = pd.Series([np.nan] * len(df_), index=df_.index, dtype=object)
    for c in candidates:
        if c in df_:
            s = s.fillna(df_[c])
    df_[target] = s

_coalesce(df, "player",   ["player_odds","player_proj","player"])
_coalesce(df, "team",     ["team_odds","team_proj","team"])
_coalesce(df, "opponent", ["opponent_odds","opponent_proj","opponent"])
_coalesce(df, "game_date",["game_date_odds","game_date_proj","game_date"])

to_drop = [c for c in [
    "player_odds","player_proj",
    "team_odds","team_proj",
    "opponent_odds","opponent_proj",
    "game_date_odds","game_date_proj"
] if c in df.columns]
df.drop(columns=to_drop, inplace=True, errors="ignore")

# Ensure unique column labels
df = df.loc[:, ~df.columns.duplicated()].copy()

# ---------------------------------------------------------------------
# 2️⃣  Filter to priced rows (has any odds)
# ---------------------------------------------------------------------
if not {"over_odds","under_odds"}.issubset(df.columns):
    raise RuntimeError("Expected columns `over_odds` and `under_odds` missing. Re-run Cell 17 join.")
priced = df.dropna(subset=["over_odds","under_odds"], how="all").copy()

print("Priced rows:", len(priced))
print("Columns unique? ->", not priced.columns.duplicated().any())

# ---------------------------------------------------------------------
# 3️⃣  Numeric coercers
# ---------------------------------------------------------------------
def _num_int(x):
    if pd.isna(x): return np.nan
    m = re.search(r"[-+]?\d+", str(x))
    return int(m.group()) if m else np.nan

def _num_float(x):
    if pd.isna(x): return np.nan
    m = re.search(r"[-+]?\d+(?:\.\d+)?", str(x))
    return float(m.group()) if m else np.nan

for c in ["line","projection_mean","projection_sd","over_odds","under_odds"]:
    if c in priced.columns:
        priced[c] = priced[c].apply(_num_float if c not in ("over_odds","under_odds") else _num_int)

# ---------------------------------------------------------------------
# 4️⃣  Handle missing SD (fallback 15% of mean, min 1.0)
# ---------------------------------------------------------------------
if ("projection_sd" not in priced.columns) or priced["projection_sd"].fillna(0).eq(0).all():
    priced["projection_sd"] = (priced["projection_mean"].abs() * 0.15).clip(lower=1.0)

# ---------------------------------------------------------------------
# 5️⃣  Model P(over) from Normal
# ---------------------------------------------------------------------
def p_over_from_normal(mu, sd, line):
    if pd.isna(mu) or pd.isna(sd) or pd.isna(line) or float(sd) <= 0: return np.nan
    z = (float(line) - float(mu)) / float(sd)
    return 1.0 - NormalDist().cdf(z)

priced["p_over_model"] = priced.apply(
    lambda r: p_over_from_normal(r["projection_mean"], r["projection_sd"], r["line"]), axis=1
)

# ---------------------------------------------------------------------
# 6️⃣  Implied probs, de-vig, edge
# ---------------------------------------------------------------------
def implied_prob(a):
    if pd.isna(a): return np.nan
    a = float(a)
    return (-a)/(-a+100.0) if a < 0 else 100.0/(a+100.0)

priced["p_over_imp"]  = priced["over_odds"].map(implied_prob)
priced["p_under_imp"] = priced["under_odds"].map(implied_prob)

def devig_pair(p_o, p_u):
    if pd.isna(p_o) or pd.isna(p_u): return (np.nan, np.nan)
    s = p_o + p_u
    if s <= 0: return (np.nan, np.nan)
    return (p_o/s, p_u/s)

fair = priced.apply(
    lambda r: pd.Series(devig_pair(r["p_over_imp"], r["p_under_imp"]),
                        index=["p_over_fair","p_under_fair"]),
    axis=1
)
priced = pd.concat([priced, fair], axis=1)

priced["edge_over"] = np.where(
    priced["p_over_fair"].notna(),
    priced["p_over_model"] - priced["p_over_fair"],
    priced["p_over_model"] - priced["p_over_imp"]
)

# ---------------------------------------------------------------------
# 7️⃣  EV and Kelly (cap Kelly ≤25%)
# ---------------------------------------------------------------------
def kelly_fraction(p, american, cap=0.25):
    if pd.isna(p) or pd.isna(american): return 0.0
    a = float(american)
    b = (a/100.0) if a > 0 else (100.0/abs(a))
    f = (p*(b+1)-1)/b
    return float(max(0.0, min(f, cap)))

def ev_flat_over(p, american):
    if pd.isna(p) or pd.isna(american): return np.nan
    a = float(american)
    win = (a/100.0) if a > 0 else (100.0/abs(a))
    lose = 1.0
    return p*win - (1-p)*lose

priced["kelly_frac_over"] = priced.apply(lambda r: kelly_fraction(r["p_over_model"], r["over_odds"]), axis=1)
priced["EV_over_1u"]      = priced.apply(lambda r: ev_flat_over(r["p_over_model"], r["over_odds"]), axis=1)

# ---------------------------------------------------------------------
# 8️⃣  Final sanity fix — unique columns before sort
# ---------------------------------------------------------------------
if priced.columns.duplicated().any():
    print("⚠️ Duplicate column names detected, cleaning up…")
    priced = priced.loc[:, ~priced.columns.duplicated()].copy()

dupes = [c for c in priced.columns if c == "player"]
if len(dupes) > 1:
    keep_first = priced.loc[:, ["player"]].iloc[:, 0]
    priced = priced.drop(columns=["player"], errors="ignore")
    priced["player"] = keep_first
    print("✅ Fixed multiple 'player' columns by keeping the first one.")

assert priced.columns.is_unique, "Still duplicate columns after cleanup!"

# ---------------------------------------------------------------------
# 9️⃣  Sort & build slate
# ---------------------------------------------------------------------
cols_keep = [
    "game_date","book","player","team","opponent","market","line","lineup_status",
    "over_odds","under_odds","p_over_imp","p_under_imp","p_over_fair","p_under_fair",
    "p_over_model","edge_over","EV_over_1u","kelly_frac_over",
    "projected_minutes","projection_mean","projection_sd","start_prob"
]
# Only keep those that exist
cols_keep = [c for c in cols_keep if c in priced.columns]

priced_sorted = priced.sort_values(["player","market","edge_over"], ascending=[True,True,False])
best_per_player = priced_sorted.drop_duplicates(subset=["player","market"], keep="first")[cols_keep].reset_index(drop=True)

# ---------------------------------------------------------------------
# 🔟  Reports & export
# ---------------------------------------------------------------------
print("\nCoverage after pricing:")
print("  rows with over_odds:", best_per_player["over_odds"].notna().sum())
print("  rows with under_odds:", best_per_player["under_odds"].notna().sum())
print("\nEdge quantiles:")
print(best_per_player["edge_over"].quantile([0.1,0.25,0.5,0.75,0.9]))

print("\nTop by edge (first 20):")

EDGE_MIN   = 0.02
EV_MIN     = 0.00
KELLY_MIN  = 0.01
MIN_MINUTES= 14
START_PROB = 0.50

slate = best_per_player[
    (best_per_player["edge_over"] >= EDGE_MIN) &
    (best_per_player["EV_over_1u"] >= EV_MIN) &
    (best_per_player["kelly_frac_over"] >= KELLY_MIN) &
    (best_per_player["projected_minutes"].fillna(0) >= MIN_MINUTES) &
    (best_per_player["start_prob"].fillna(1.0) >= START_PROB)
].sort_values(["edge_over","EV_over_1u"], ascending=False).reset_index(drop=True)

print(f"\nFinal slate size: {len(slate)}")

stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
os.makedirs("data/bets", exist_ok=True)
csv_all   = f"data/bets/nba_priced_candidates_{stamp}.csv"
csv_slate = f"data/bets/nba_priced_slate_{stamp}.csv"
xlsx_path = f"data/bets/nba_priced_{stamp}.xlsx"

best_per_player.to_csv(csv_all, index=False)
slate.to_csv(csv_slate, index=False)
with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
    best_per_player.to_excel(w, sheet_name="Candidates", index=False)
    slate.to_excel(w, sheet_name="Slate", index=False)

print("\nSaved:")
print("  •", csv_all)
print("  •", csv_slate)
print("  •", xlsx_path)


# === Diagnostics + more-forgiving slate builder ===
import numpy as np, pandas as pd
from statistics import NormalDist

assert 'df_best' in globals() and isinstance(df_best, pd.DataFrame), "Run the priced-slate cell first."

ND = NormalDist()

def implied_from_decimal(d):
    return np.nan if (pd.isna(d) or d<=1) else 1.0/d

def kelly_fraction_decimal(p, dec_odds, cap=0.25):
    if pd.isna(p) or pd.isna(dec_odds) or dec_odds<=1: return 0.0
    b = dec_odds - 1.0
    f = (p*(b+1) - 1)/b
    return float(max(0.0, min(f, cap)))

def ev_flat_decimal(p, dec_odds):
    if pd.isna(p) or pd.isna(dec_odds) or dec_odds<=1: return np.nan
    return p*(dec_odds-1) - (1-p)*1.0

def p_over_from_normal(mu, sd, line):
    if pd.isna(mu) or pd.isna(sd) or pd.isna(line) or sd<=0: return np.nan
    z = (line - mu)/sd
    return 1.0 - ND.cdf(z)

dfD = df_best.copy()

# --- Quick coverage checks
have_over = dfD['over_dec'].notna() & (dfD['over_dec'] > 1.0)
have_under = dfD['under_dec'].notna() & (dfD['under_dec'] > 1.0)
print("Coverage:")
print(f"  with over_dec:  {have_over.sum()} / {len(dfD)}")
print(f"  with under_dec: {have_under.sum()} / {len(dfD)}")
print(f"  both prices:    {(have_over & have_under).sum()} / {len(dfD)}")

# --- Edge distribution snapshot
q = dfD['edge_over'].dropna().quantile([0.1,0.25,0.5,0.75,0.9]) if 'edge_over' in dfD else pd.Series(dtype=float)
print("\nEdge quantiles (model - fair/imp):")
print(q.to_string())

# --- Top 20 by edge (even if below your threshold)
top_by_edge = dfD.sort_values('edge_over', ascending=False).head(20)
cols_preview = [c for c in [
    "player","team","opponent","market","line","book",
    "over_dec","under_dec","p_over_model","p_over_fair","p_over_imp","edge_over","EV_over_1u","kelly_frac_over"
] if c in dfD.columns]
print("\nTop by edge (first 20):")
print(top_by_edge[cols_preview].to_string(index=False))

# =========================
# Alternative slates
# =========================

# 1) EV-positive slate (uses your model p_over and the posted OVER decimal price)
MIN_EV  = 0.01   # > 0.01u per 1u stake
MIN_DEC = 1.01   # must have a real price
slate_ev = dfD[
    (dfD["EV_over_1u"] > MIN_EV) &
    (dfD["over_dec"].fillna(0) > MIN_DEC)
].copy()

# 2) Lower-edge slate (relax edge threshold)
MIN_EDGE_RELAXED = 0.005   # 0.5%
slate_edge_relaxed = dfD[
    (dfD["edge_over"] >= MIN_EDGE_RELAXED) &
    (dfD["over_dec"].fillna(0) > MIN_DEC)
].copy()

# 3) Price-only slate (ignore de-vig; compare model vs break-even p from OVER decimal)
dfD["p_over_price"] = dfD["over_dec"].map(implied_from_decimal)
dfD["edge_vs_price"] = dfD["p_over_model"] - dfD["p_over_price"]
MIN_EDGE_PRICE = 0.01  # 1% vs break-even
slate_price_only = dfD[
    (dfD["edge_vs_price"] >= MIN_EDGE_PRICE) &
    (dfD["over_dec"].fillna(0) > MIN_DEC)
].copy()

# 4) Sensitivity slate: try a slightly tighter SD (10% of mean) when SD came from fallback
#    This often reveals edges when your fallback SD (15%) was too conservative.
need_sd_tighten = dfD["projection_sd"].isna() | (dfD["projection_sd"] <= 0)
sd_tight = (dfD["projection_mean"].abs()*0.10).clip(lower=0.75)  # tighter and lower min sd
p_model_tight = []
for mu, sd, line, tight_sd in zip(dfD["projection_mean"], dfD["projection_sd"], dfD["line"], sd_tight):
    use_sd = sd if pd.notna(sd) and sd > 0 else tight_sd
    p_model_tight.append(p_over_from_normal(mu, use_sd, line))
dfD["p_over_model_tight"] = p_model_tight
dfD["EV_over_1u_tight"] = dfD.apply(lambda r: ev_flat_decimal(r["p_over_model_tight"], r["over_dec"]), axis=1)
dfD["edge_over_tight"] = np.where(dfD["p_over_fair"].notna(),
                                  dfD["p_over_model_tight"] - dfD["p_over_fair"],
                                  dfD["p_over_model_tight"] - dfD["p_over_imp"])
slate_tight = dfD[
    (dfD["EV_over_1u_tight"] > MIN_EV) &
    (dfD["over_dec"].fillna(0) > MIN_DEC)
].copy()

def _keep_cols(d):
    keep = [c for c in [
        "asof_date","game_date","book","player","team","opponent","market","line","lineup_status",
        "over_dec","under_dec",
        "p_over_imp","p_under_imp","p_over_fair","p_under_fair",
        "p_over_model","edge_over","EV_over_1u","kelly_frac_over",
        "p_over_price","edge_vs_price",
        "p_over_model_tight","edge_over_tight","EV_over_1u_tight",
        "projected_minutes","projection_mean","projection_sd","start_prob"
    ] if c in d.columns]
    return d[keep].sort_values(["market","edge_over"], ascending=[True, False])

print("\nSlate sizes:")
print(f"  EV-positive (>{MIN_EV:.2f}u):     {len(slate_ev)}")
print(f"  Relaxed edge (≥{MIN_EDGE_RELAXED*100:.1f}%): {len(slate_edge_relaxed)}")
print(f"  Price-only edge (≥{MIN_EDGE_PRICE*100:.1f}%): {len(slate_price_only)}")
print(f"  Tight-SD EV-positive:              {len(slate_tight)}")

# Preview a few from each
for name, slate_df in [
    ("EV-positive", slate_ev),
    ("Relaxed-edge", slate_edge_relaxed),
    ("Price-only", slate_price_only),
    ("Tight-SD EV+", slate_tight),
]:
    if not slate_df.empty:
        print(f"\n{name} — top 10")
        print(_keep_cols(slate_df).head(10).to_string(index=False))

# Save all variants
ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
_keep_cols(slate_ev).to_csv(f"nba_slate_evpos_{ts}.csv", index=False)
_keep_cols(slate_edge_relaxed).to_csv(f"nba_slate_edge_relaxed_{ts}.csv", index=False)
_keep_cols(slate_price_only).to_csv(f"nba_slate_price_only_{ts}.csv", index=False)
_keep_cols(slate_tight).to_csv(f"nba_slate_tightsd_{ts}.csv", index=False)
print(f"\nSaved CSVs with the four slate variants (timestamp {ts}).")


merged_df = best_per_player.merge(dfD, on=["player", "market"], how="inner", suffixes=("_best", "_dfD"))
display(merged_df['p_over_model_dfD'])
# Select the columns you want to keep
selected_columns = [
    'player', 'team_best', 'opponent_best', 'market', 'line_dfD', 
    'lineup_status_best', 'over_odds_best', 'under_odds_best', 
    'p_over_imp_best', 'p_under_imp_best', 'p_over_fair_best', 
    'p_under_fair_best', 'projected_minutes_best', 'projection_mean_best', 
    'projection_sd_best', 'p_over_model_dfD'
]

# Create clean column name mapping
column_rename = {
    'player': 'Player',
    'team_best': 'Team',
    'opponent_best': 'Opponent',
    'market': 'Market',
    'line_best': 'Line',
    'lineup_status_best': 'Lineup',
    'over_odds_best': 'OverOdds',
    'under_odds_best': 'UnderOdds',
    'p_over_imp_best': 'POverImp',
    'p_under_imp_best': 'PUnderImp',
    'p_over_fair_best': 'POverFair',
    'p_under_fair_best': 'PUnderFair',
    'projected_minutes_best': 'ProjMins',
    'projection_mean_best': 'ProjMean',
    'projection_sd_best': 'ProjSD',
    'p_over_model_dfD': 'POverModel'
}

# Select and rename columns
merged_df_clean = merged_df[selected_columns].rename(columns=column_rename)

# Export to CSV
csv_path = os.path.join("data/bets", f"value_bets_top100_{datetime.now().strftime('%Y%m%d')}.csv")
merged_df_clean.to_csv(csv_path, index=False)

print(f"✅ Saved clean value bets to: {csv_path}")
print(f"Columns: {list(merged_df_clean.columns)}")
print(f"Rows: {len(merged_df_clean)}")

# Preview the data
print("\nFirst few rows:")
print(merged_df_clean.head())

import pandas as pd

# --- Load the CSV ---
df = pd.read_csv("data/bets/value_bets_top100_20251104.csv")

# --- Define conversion function ---
def american_to_decimal(american):
    """Convert American odds to decimal odds."""
    if pd.isna(american):
        return None
    try:
        american = float(american)
        if american > 0:
            return 1 + (american / 100)
        elif american < 0:
            return 1 + (100 / abs(american))
        else:
            return None
    except Exception:
        return None

# --- Apply conversion ---
df["OverDecimal"] = df["OverOdds"].apply(american_to_decimal)
df["UnderDecimal"] = df["UnderOdds"].apply(american_to_decimal)
df['OverOdds'] = df['OverOdds']
df['UnderOdds'] = df['UnderOdds']
df.drop(columns=['OverOdds', 'UnderOdds'], inplace=True)
# --- Save updated CSV ---
output_path = "data/bets/value_bets_top100_20251104_decimal.csv"
df.to_csv(output_path, index=False)

print(f"✅ Converted odds and saved to: {output_path}")

