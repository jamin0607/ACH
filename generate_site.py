#!/usr/bin/env python3
"""
Fantasy Baseball league website generator.

Builds a self-contained index.html (overview page + clickable team deep dives)
from either of two data sources:

  ESPN (recommended, live):
      python generate_site.py espn
    Configure via environment variables (see README):
      ESPN_LEAGUE_ID   your league id (required)
      ESPN_YEAR        season, e.g. 2026 (default: current year)
      ESPN_S2, ESPN_SWID   the two browser cookies (required for PRIVATE leagues)

  Excel (fallback, manual):
      python generate_site.py excel [FantasyWeeklyStats.xlsx]

Requires: pandas, openpyxl, scikit-learn, and (for ESPN) requests.
    pip install pandas openpyxl scikit-learn requests
"""

import datetime as _dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

OUT = "index.html"
DEFAULT_LEAGUE_ID = "877696965"   # public league; override with ESPN_LEAGUE_ID

# ---- category config -------------------------------------------------------
# (key, label, lower_is_better, decimals)
CATS = [
    ("R", "Runs", False, 0),
    ("HR", "Home Runs", False, 0),
    ("RBI", "RBI", False, 0),
    ("SB", "Stolen Bases", False, 0),
    ("OBP", "OBP", False, 3),
    ("K", "Strikeouts", False, 0),
    ("QS", "Quality Starts", False, 0),
    ("ERA", "ERA", True, 2),
    ("WHIP", "WHIP", True, 2),
    ("SVHD", "Saves + Holds", False, 1),
]
CAT_KEYS = [c[0] for c in CATS]
LOWER = {c[0]: c[2] for c in CATS}
DECIMALS = {c[0]: c[3] for c in CATS}
LABEL = {c[0]: c[1] for c in CATS}
BAT = ["R", "HR", "RBI", "SB", "OBP"]
PIT = ["K", "QS", "ERA", "WHIP", "SVHD"]

# Fallback display names for the Excel source (ESPN supplies its own).
TEAM_NAMES = {
    "JANS": "David DeJesus Take The Wheel", "DING": "Milwaukee Dinger Society",
    "KORO": "Ex-Beliebers", "WERT": "Haders Gonna Hade", "Bs": "Durham Bullish",
    "COLE": "Skene on da Pene", "ICM": "Indianapolis Clergymen", "IOWA": "Corn Boys",
    "CATS": "Cheesetown Rat-Bashers", "MAZ": "Big Al's Dingers",
    "FAIN": "From Ragans to Riches", "CMH": "Uecker's Uephemisms",
}

# ESPN fantasy-baseball stat IDs for each of our 10 scoring categories.
STAT_ID = {"R": 20, "HR": 5, "RBI": 21, "SB": 23, "OBP": 17,
           "K": 48, "QS": 63, "ERA": 47, "WHIP": 41, "SVHD": 83}
ESPN_HOST = "https://lm-api-reads.fantasy.espn.com"


def rnd(x, d):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return round(float(x), d)


# ===========================================================================
#  DATA SOURCES  -> each returns a "base" DataFrame with columns:
#  Week, M.ID, Team, <10 cats>, R.<10 cats>   (one row per team per matchup)
#  plus a names dict {team-key: display name}
# ===========================================================================
def base_from_excel(path):
    df = pd.read_excel(path, sheet_name="Data")
    df = df.rename(columns={"OBS": "OBP", "R.OBS": "R.OBP"})
    df = df.dropna(subset=["Team", "Week"])
    df["Week"] = df["Week"].astype(int)
    keep = ["Week", "M.ID", "Team"] + CAT_KEYS + ["R." + c for c in CAT_KEYS]
    base = df[keep].copy()
    names = {t: TEAM_NAMES.get(t, t) for t in base["Team"].unique()}
    logos = {}   # no logos available from the spreadsheet source
    return base, names, "Atlantic City Hambinos", logos


def base_from_espn(league_id, year, s2, swid):
    """Pull completed weeks straight from ESPN's raw v3 API.

    The espn_api library can't read H2H_MOST_CATEGORIES leagues, so we call the
    API directly. Each completed matchup period is fetched with the mMatchup
    view, which returns per-team cumulative category totals in
    home/away.cumulativeScore.scoreByStat keyed by ESPN stat id.
    """
    import requests

    base_url = (f"{ESPN_HOST}/apis/v3/games/flb/seasons/{year}"
                f"/segments/0/leagues/{league_id}")
    cookies = {"espn_s2": s2, "SWID": swid} if (s2 and swid) else None
    headers = {"User-Agent": "Mozilla/5.0"}

    def get(params):
        r = requests.get(base_url, params=params, headers=headers,
                         cookies=cookies, timeout=30)
        r.raise_for_status()
        return r.json()

    meta = get({"view": ["mTeam", "mStatus", "mSettings"]})
    league_name = ((meta.get("settings") or {}).get("name") or "").strip() or None
    names, abbr_of, logos, bye_ids = {}, {}, {}, set()
    for t in meta.get("teams", []):
        tid = t["id"]
        ab = (t.get("abbrev") or f"T{tid}").strip()
        raw_name = t.get("name") or f"{t.get('location', '')} {t.get('nickname', '')}"
        nm = " ".join(str(raw_name).split()) or ab      # collapse stray whitespace
        abbr_of[tid] = ab
        names[ab] = nm
        logos[ab] = (t.get("logo") or "").strip() or None
        if ab.upper() == "BYE" or nm.upper().startswith("BYE"):
            bye_ids.add(tid)

    cur_mp = meta.get("status", {}).get("currentMatchupPeriod")
    if not cur_mp:
        sched = get({"view": "mMatchupScore"}).get("schedule", [])
        decided = [m["matchupPeriodId"] for m in sched
                   if m.get("winner", "UNDECIDED") != "UNDECIDED"]
        cur_mp = (max(decided) + 1) if decided else 1

    def score(sb, cat):
        cell = sb.get(str(STAT_ID[cat]))
        return float(cell["score"]) if cell and cell.get("score") is not None else np.nan

    def win_flag(mine, theirs, cat):
        if np.isnan(mine) or np.isnan(theirs):
            return np.nan
        if mine == theirs:
            return np.nan                      # category tie -> credited to neither
        better = mine < theirs if LOWER[cat] else mine > theirs
        return 1.0 if better else 0.0

    rows = []
    for mp in range(1, int(cur_mp)):           # completed weeks only (skip live week)
        data = get({"view": "mMatchup", "matchupPeriodId": mp})
        games = [m for m in data.get("schedule", []) if m.get("matchupPeriodId") == mp]
        idx = 0
        for m in games:
            home, away = m.get("home"), m.get("away")
            if not home or not away:
                continue
            ht, at = home.get("teamId"), away.get("teamId")
            if ht in bye_ids or at in bye_ids:
                continue
            hsb = (home.get("cumulativeScore") or {}).get("scoreByStat") or {}
            asb = (away.get("cumulativeScore") or {}).get("scoreByStat") or {}
            if not hsb or not asb:
                continue                        # not played
            idx += 1
            mid = f"{mp}.{idx}"
            hv = {c: score(hsb, c) for c in CAT_KEYS}
            av = {c: score(asb, c) for c in CAT_KEYS}
            for mine, theirs, tid in ((hv, av, ht), (av, hv, at)):
                row = {"Week": mp, "M.ID": mid, "Team": abbr_of[tid]}
                for c in CAT_KEYS:
                    row[c] = mine[c]
                    row["R." + c] = win_flag(mine[c], theirs[c], c)
                rows.append(row)

    if not rows:
        raise SystemExit("No completed matchups returned from ESPN — check the league "
                         "id/year, and that at least one week has finished.")
    base = pd.DataFrame(rows)
    base["Week"] = base["Week"].astype(int)
    print(f"ESPN: pulled {base['Week'].nunique()} completed weeks, "
          f"{len(names) - len(bye_ids)} teams.")
    missing = [c for c in CAT_KEYS if base[c].isna().all()]
    if missing:
        print(f"WARNING: no values for {missing} — a stat id in STAT_ID may be wrong.")
    return base, {ab: names[ab] for ab in base["Team"].unique()}, league_name, \
           {ab: logos.get(ab) for ab in base["Team"].unique()}


def add_opponents(base):
    """Self-join each matchup's two rows to attach opponent + O.<cat> columns."""
    opp = base[["Week", "M.ID", "Team"] + CAT_KEYS].rename(
        columns={"Team": "OppTeam", **{c: "O." + c for c in CAT_KEYS}})
    merged = base.merge(opp, on=["Week", "M.ID"])
    merged = merged[merged["Team"] != merged["OppTeam"]].copy()
    return merged


# ===========================================================================
#  ANALYTICS  (source-independent)
# ===========================================================================
def cat_wins_row(row):
    return sum(float(row["R." + c]) for c in CAT_KEYS if pd.notna(row["R." + c]))


def fmtcw(v):
    """Format a category-win count: 7 -> '7', 6.5 -> '6.5'."""
    v = float(v)
    return str(int(v)) if v == int(v) else f"{v:g}"


def build(merged, names, league_name=None, logos=None):
    logos = logos or {}
    df = merged
    teams = sorted(df["Team"].unique(), key=lambda t: names.get(t, t))
    weeks = sorted(int(w) for w in df["Week"].unique())
    reg_weeks = [w for w in weeks if w != min(weeks)]  # drop opening week for models

    df["catwins"] = df.apply(cat_wins_row, axis=1)
    cw_map = df.set_index(["Week", "M.ID", "Team"])["catwins"].to_dict()
    df["oppCatwins"] = df.apply(
        lambda r: cw_map.get((r["Week"], r["M.ID"], r["OppTeam"]), np.nan), axis=1)
    df["margin"] = df["catwins"] - df["oppCatwins"]

    def outcome(r):
        if np.isnan(r["oppCatwins"]):
            return np.nan
        return "W" if r["margin"] > 0 else ("L" if r["margin"] < 0 else "T")
    df["result"] = df.apply(outcome, axis=1)

    # standings
    standings = []
    for t in teams:
        sub = df[df["Team"] == t]
        w = int((sub["result"] == "W").sum()); l = int((sub["result"] == "L").sum())
        tie = int((sub["result"] == "T").sum()); gp = w + l + tie
        standings.append({"team": t, "name": names.get(t, t), "logo": logos.get(t),
                          "w": w, "l": l, "t": tie,
                          "gp": gp, "pct": rnd((w + tie / 2) / gp if gp else 0, 3),
                          "catWins": int(sub["catwins"].sum())})
    standings.sort(key=lambda s: (-s["pct"], -s["catWins"]))
    for i, s in enumerate(standings):
        s["rank"] = i + 1

    league_week_avg = {w: {c: float(df[df["Week"] == w][c].mean()) for c in CAT_KEYS}
                       for w in weeks}
    league = {"avgPerTeamWeek": {c: rnd(df[c].mean(), max(DECIMALS[c], 1)) for c in CAT_KEYS}}

    # wins over an average team
    avg_team = []
    for t in teams:
        wins = 0
        for _, r in df[df["Team"] == t].iterrows():
            la = league_week_avg[r["Week"]]
            beat = sum((r[c] < la[c]) if LOWER[c] else (r[c] > la[c]) for c in CAT_KEYS)
            wins += 1 if beat > 5 else 0
        avg_team.append({"team": t, "name": names.get(t, t), "wins": wins})
    avg_team.sort(key=lambda x: -x["wins"])

    # opponent averages faced (luck) + SOS rank
    luck = {t: {c: rnd(df[df["Team"] == t]["O." + c].mean(),
                       DECIMALS[c] if DECIMALS[c] >= 2 else 1) for c in CAT_KEYS}
            for t in teams}
    sos_rank = {t: {} for t in teams}
    for c in CAT_KEYS:
        order = sorted(teams, key=lambda t: luck[t][c], reverse=not LOWER[c])
        for i, t in enumerate(order):
            sos_rank[t][c] = i + 1
    sos = sorted([{"team": t, "name": names.get(t, t),
                   "avgRank": rnd(np.mean([sos_rank[t][c] for c in CAT_KEYS]), 1),
                   "ranks": sos_rank[t]} for t in teams], key=lambda x: x["avgRank"])

    # category ranks for radar (12 best .. 1 worst)
    season_val = {t: {c: (df[df["Team"] == t][c].mean() if c in ("OBP", "ERA", "WHIP")
                          else df[df["Team"] == t][c].sum()) for c in CAT_KEYS}
                  for t in teams}
    cat_rank = {t: {} for t in teams}
    for c in CAT_KEYS:
        order = sorted(teams, key=lambda t: season_val[t][c], reverse=not LOWER[c])
        for i, t in enumerate(order):
            cat_rank[t][c] = len(teams) - i

    # batting vs pitching category-win differential
    batpitch = []
    for t in teams:
        sub = df[df["Team"] == t]
        r_bat = np.nansum([sub["R." + c].sum() for c in BAT])
        r_pit = np.nansum([sub["R." + c].sum() for c in PIT])
        o_bat = 0.0; o_pit = 0.0
        for _, r in sub.iterrows():
            opp = df[(df["Week"] == r["Week"]) & (df["M.ID"] == r["M.ID"]) &
                     (df["Team"] == r["OppTeam"])]
            if len(opp):
                o = opp.iloc[0]
                o_bat += np.nansum([o["R." + c] for c in BAT])
                o_pit += np.nansum([o["R." + c] for c in PIT])
        batpitch.append({"team": t, "name": names.get(t, t),
                         "batting": rnd(r_bat - o_bat, 0), "pitching": rnd(r_pit - o_pit, 0)})

    # category win-probability curves (logistic regression)
    reg_df = df[df["Week"].isin(reg_weeks)] if reg_weeks else df
    ranges = {"R": (1, 55, 1), "HR": (1, 25, 1), "RBI": (1, 60, 1), "SB": (0, 16, 1),
              "OBP": (0.200, 0.450, 0.01), "K": (0, 100, 2), "QS": (0, 8, 1),
              "ERA": (0.5, 8, 0.25), "WHIP": (0.5, 2, 0.05), "SVHD": (0, 6, 0.5)}
    prob_curves = {}
    models = {}
    for c in CAT_KEYS:
        y = reg_df["R." + c].fillna(0).astype(int).values
        X = reg_df[c].values.reshape(-1, 1)
        if len(np.unique(y)) < 2 or np.isnan(X).any():
            continue
        m = LogisticRegression().fit(X, y)
        models[c] = m
        lo, hi, step = ranges[c]
        xs = np.arange(lo, hi + step / 2, step)
        ps = m.predict_proba(xs.reshape(-1, 1))[:, 1]
        prob_curves[c] = {"x": [rnd(v, 3) for v in xs], "p": [rnd(v, 4) for v in ps]}

    # weekly category highs: best single-week performance for each category,
    # across every team/week played (favorable direction per category)
    weekly_highs = []
    for c in CAT_KEYS:
        idx = df[c].idxmin() if LOWER[c] else df[c].idxmax()
        if pd.isna(idx):
            continue
        row = df.loc[idx]
        weekly_highs.append({
            "cat": c, "label": LABEL[c], "lower": LOWER[c],
            "value": rnd(row[c], DECIMALS[c]), "team": row["Team"],
            "name": names.get(row["Team"], row["Team"]), "week": int(row["Week"]),
        })

    # ---- season race: cumulative match points (W=1, T=0.5) per league week ----
    race = {}
    for t in teams:
        pts = {int(r["Week"]): (1.0 if r["result"] == "W" else
                                0.5 if r["result"] == "T" else 0.0)
               for _, r in df[df["Team"] == t].iterrows() if pd.notna(r["result"])}
        cum, series = 0.0, []
        for w in weeks:
            cum += pts.get(w, 0.0)
            series.append(rnd(cum, 1))
        race[t] = series

    # ---- most recent week recap ----
    lw = max(weeks)
    seen, last_week = set(), []
    for _, r in df[df["Week"] == lw].sort_values("M.ID").iterrows():
        if r["M.ID"] in seen:
            continue
        seen.add(r["M.ID"])
        last_week.append({"a": r["Team"], "b": r["OppTeam"],
                          "aw": rnd(r["catwins"], 1), "bw": rnd(r["oppCatwins"], 1)})

    # per-team: best single week (most category wins that week) + current streak
    best_week = {}
    streaks = {}
    for t in teams:
        sub = df[df["Team"] == t].sort_values("Week")
        if len(sub):
            bi = sub["catwins"].idxmax()
            br = sub.loc[bi]
            best_week[t] = {"week": int(br["Week"]), "catWins": rnd(br["catwins"], 1),
                            "opp": names.get(br["OppTeam"], br["OppTeam"]),
                            "oppAbbr": br["OppTeam"], "result": br["result"]}
        else:
            best_week[t] = None
        results = [r for r in sub["result"].tolist() if r in ("W", "L")]
        if results:
            last = results[-1]
            n = 0
            for r in reversed(results):
                if r == last:
                    n += 1
                else:
                    break
            streaks[t] = {"type": last, "length": n}
        else:
            streaks[t] = {"type": "-", "length": 0}

    # all-play records: each week, compare every team's line against every
    # other team's line that week (win the "game" by taking more categories)
    all_play = {t: {"w": 0, "l": 0, "t": 0} for t in teams}
    for w in weeks:
        wk = df[df["Week"] == w]
        lines = {r["Team"]: r for _, r in wk.iterrows()}
        ts = list(lines)
        for a in ts:
            for b in ts:
                if a == b:
                    continue
                ra, rb = lines[a], lines[b]
                cw = sum((ra[c] < rb[c]) if LOWER[c] else (ra[c] > rb[c]) for c in CAT_KEYS)
                cl = sum((ra[c] > rb[c]) if LOWER[c] else (ra[c] < rb[c]) for c in CAT_KEYS)
                if cw > cl:
                    all_play[a]["w"] += 1
                elif cw < cl:
                    all_play[a]["l"] += 1
                else:
                    all_play[a]["t"] += 1

    # power rankings: blend record (40%), all-play (40%), last-3 form (20%);
    # luck = actual win% minus all-play win% (schedule fortune)
    power = []
    for t in teams:
        ap = all_play[t]
        gp = ap["w"] + ap["l"] + ap["t"]
        ap_pct = (ap["w"] + ap["t"] / 2) / gp if gp else 0
        st = next(s for s in standings if s["team"] == t)
        res = [r for r in df[df["Team"] == t].sort_values("Week")["result"].tolist()
               if r in ("W", "L", "T")]
        last3 = res[-3:]
        l3 = ((sum(1 for r in last3 if r == "W") + 0.5 * sum(1 for r in last3 if r == "T"))
              / max(len(last3), 1))
        power.append({
            "team": t, "name": names.get(t, t),
            "score": rnd(100 * (0.4 * st["pct"] + 0.4 * ap_pct + 0.2 * l3), 1),
            "apW": ap["w"], "apL": ap["l"], "apT": ap["t"], "apPct": rnd(ap_pct, 3),
            "last3": last3, "luck": rnd(st["pct"] - ap_pct, 3), "recRank": st["rank"],
        })
    power.sort(key=lambda x: -x["score"])
    for i, p in enumerate(power):
        p["rank"] = i + 1
        p["delta"] = p["recRank"] - p["rank"]

    # head-to-head grid: season record of every team against every other
    h2h = {a: {} for a in teams}
    for _, r in df.iterrows():
        o = r["OppTeam"]
        cell = h2h[r["Team"]].setdefault(o, {"w": 0, "l": 0, "t": 0})
        if r["result"] == "W":
            cell["w"] += 1
        elif r["result"] == "L":
            cell["l"] += 1
        elif r["result"] == "T":
            cell["t"] += 1

    # season superlatives
    blow, close = None, None
    for (w_, mid), grp in df.groupby(["Week", "M.ID"]):
        if len(grp) != 2:
            continue
        g = grp.sort_values("catwins", ascending=False)
        top, bot = g.iloc[0], g.iloc[1]
        margin = float(top["catwins"] - bot["catwins"])
        rec = {"week": int(w_), "team": top["Team"], "opp": bot["Team"],
               "cw": rnd(top["catwins"], 1), "ocw": rnd(bot["catwins"], 1),
               "margin": rnd(margin, 1)}
        if margin > 0:
            if blow is None or margin > blow["margin"]:
                blow = rec
            if close is None or margin < close["margin"]:
                close = rec
    bi = df["catwins"].idxmax()
    br = df.loc[bi]
    best_any = {"team": br["Team"], "week": int(br["Week"]), "cw": rnd(br["catwins"], 1),
                "opp": br["OppTeam"]}
    long_streak = None
    for t in teams:
        res = [r for r in df[df["Team"] == t].sort_values("Week")["result"].tolist()
               if r in ("W", "L")]
        run, best_run = 0, 0
        for r in res:
            run = run + 1 if r == "W" else 0
            best_run = max(best_run, run)
        if best_run and (long_streak is None or best_run > long_streak["length"]):
            long_streak = {"team": t, "length": best_run}
    superlatives = {"blowout": blow, "closest": close, "bestWeek": best_any,
                    "longestStreak": long_streak}

    # per-team category win rates (% of weeks the category was won; ties excluded)
    cat_win_rate = {}
    for t in teams:
        sub = df[df["Team"] == t]
        cat_win_rate[t] = {}
        for c in CAT_KEYS:
            m = sub["R." + c].mean()   # NaN-safe: ties are NaN and skipped
            cat_win_rate[t][c] = rnd(m * 100, 0) if pd.notna(m) else None

    team_detail = {}
    for t in teams:
        sub = df[df["Team"] == t].sort_values("Week")
        st = next(s for s in standings if s["team"] == t)

        # expected chance to win each category posting an average week (league model)
        exp_cat = {}
        for c in CAT_KEYS:
            if c in models:
                val = (season_val[t][c] if c in ("OBP", "ERA", "WHIP")
                       else season_val[t][c] / max(len(sub), 1))
                exp_cat[c] = rnd(float(models[c].predict_proba([[val]])[0, 1]), 3)
            else:
                exp_cat[c] = None

        # recent form: last 3 weeks vs season per-week average
        last3 = sub.tail(3)
        form = {}
        for c in CAT_KEYS:
            season_pw = (season_val[t][c] if c in ("OBP", "ERA", "WHIP")
                         else season_val[t][c] / max(len(sub), 1))
            recent = float(last3[c].mean()) if len(last3) else np.nan
            delta = recent - season_pw
            good = None
            if abs(delta) > 1e-9:
                good = bool(delta < 0) if LOWER[c] else bool(delta > 0)
            form[c] = {"recent": rnd(recent, DECIMALS[c]),
                       "delta": rnd(delta, max(DECIMALS[c], 1)), "good": good}

        # clutch: record in matchups decided by <=1 category (incl. ties)
        close = sub[sub["margin"].abs() <= 1]
        clutch = {"w": int((close["result"] == "W").sum()),
                  "l": int((close["result"] == "L").sum()),
                  "t": int((close["result"] == "T").sum())}

        log = []
        for _, r in sub.iterrows():
            log.append({"week": int(r["Week"]), "opp": names.get(r["OppTeam"], r["OppTeam"]),
                        "oppAbbr": r["OppTeam"], "cw": rnd(r["catwins"], 1),
                        "result": r["result"] if pd.notna(r["result"]) else "-",
                        "stats": {c: rnd(r[c], DECIMALS[c]) for c in CAT_KEYS}})
        team_detail[t] = {
            "abbr": t, "name": names.get(t, t), "logo": logos.get(t),
            "record": {"w": st["w"], "l": st["l"], "t": st["t"], "pct": st["pct"],
                       "rank": st["rank"], "catWins": st["catWins"]},
            "weeks": [int(w) for w in sub["Week"].tolist()],
            "series": {c: [rnd(v, DECIMALS[c]) for v in sub[c].tolist()] for c in CAT_KEYS},
            "comboBat": [rnd(r["R"] + r["HR"] + r["RBI"] + r["SB"] + r["OBP"] * 100, 1)
                         for _, r in sub.iterrows()],
            "comboPit": [rnd((r["K"] / 2) + (7 - r["ERA"]) * 1.5 + (2 - r["WHIP"]) * 2 +
                             2 * r["SVHD"] + 2 * r["QS"], 1) for _, r in sub.iterrows()],
            "radar": [cat_rank[t][c] for c in CAT_KEYS],
            "seasonAvg": {c: rnd(season_val[t][c] if c in ("OBP", "ERA", "WHIP")
                                 else season_val[t][c] / max(len(sub), 1), DECIMALS[c])
                          for c in CAT_KEYS},
            "oppAvg": luck[t],
            "sos": next(s for s in sos if s["team"] == t),
            "batpitch": next(b for b in batpitch if b["team"] == t),
            "avgTeamWins": next(a for a in avg_team if a["team"] == t)["wins"],
            "bestWeek": best_week[t],
            "streak": streaks[t],
            "catWinRate": cat_win_rate[t],
            "expCat": exp_cat,
            "form": form,
            "clutch": clutch,
            "vsOpp": sorted(
                [{"opp": o, "name": names.get(o, o), **rec} for o, rec in h2h[t].items()],
                key=lambda x: (-(x["w"] - x["l"]), x["name"])),
            "allPlay": next(p for p in power if p["team"] == t),
            "log": log,
        }

    return {
        "meta": {"weeks": weeks, "nTeams": len(teams), "lastWeek": max(weeks),
                 "regWeeks": reg_weeks or weeks,
                 "generated": _dt.date.today().isoformat(),
                 "leagueName": league_name or "Atlantic City Hambinos"},
        "cats": [{"key": k, "label": LABEL[k], "lower": LOWER[k], "dec": DECIMALS[k]}
                 for k in CAT_KEYS],
        "bat": BAT, "pit": PIT, "teamOrder": teams, "standings": standings,
        "league": league, "avgTeam": avg_team, "sos": sos, "batpitch": batpitch,
        "probCurves": prob_curves, "weeklyHighs": weekly_highs, "teams": team_detail,
        "power": power, "h2h": h2h, "superlatives": superlatives,
        "race": race, "lastWeek": last_week,
        "logos": {t: logos.get(t) for t in teams},
    }


def _np(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")


def main():
    src = (sys.argv[1] if len(sys.argv) > 1 else "espn").lower()
    if src == "excel":
        path = sys.argv[2] if len(sys.argv) > 2 else "FantasyWeeklyStats.xlsx"
        base, names, league_name, logos = base_from_excel(path)
    elif src == "espn":
        lid = os.environ.get("ESPN_LEAGUE_ID", DEFAULT_LEAGUE_ID)
        year = os.environ.get("ESPN_YEAR", str(_dt.date.today().year))
        base, names, league_name, logos = base_from_espn(lid, year, os.environ.get("ESPN_S2"),
                                     os.environ.get("ESPN_SWID"))
    else:
        raise SystemExit("Usage: python generate_site.py [espn|excel] [xlsx-path]")

    data = build(add_opponents(base), names, league_name, logos)
    html = Path("template.html").read_text().replace("/*__DATA__*/",
                                                      json.dumps(data, allow_nan=False, default=_np))
    Path(OUT).write_text(html)
    print(f"Wrote {OUT} ({len(html)//1024} KB) — {data['meta']['nTeams']} teams, "
          f"{len(data['meta']['weeks'])} weeks, source={src}")


if __name__ == "__main__":
    main()
