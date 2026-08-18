import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
plt.rcParams["figure.dpi"] = 72
from scipy import stats
from xgboost import XGBRegressor

st.set_page_config(page_title="La Liga Intelligence", layout="wide")

# Transfer window dates shift with the season — approximate per La Liga calendar
def get_transfer_windows(season):
    return [
        (f"{season}-07-01",   f"{season}-09-02",   "#acd4ff", "Summer\nWindow"),
        (f"{season+1}-01-01", f"{season+1}-02-03", "#ffd9a0", "Winter\nWindow"),
    ]

def _shade_transfer_windows(ax, tl_df, season):
    """Add lightly shaded transfer window bands to a GW-axis plot."""
    if tl_df.empty:
        return
    for start_str, end_str, color, label in get_transfer_windows(season):
        start = pd.Timestamp(start_str)
        end   = pd.Timestamp(end_str)
        mask  = (tl_df["date"] >= start) & (tl_df["date"] <= end)
        if not mask.any():
            continue
        gw_lo = tl_df.loc[mask, "gw"].min() - 0.5
        gw_hi = tl_df.loc[mask, "gw"].max() + 0.5
        ax.axvspan(gw_lo, gw_hi, alpha=0.07, color=color, zorder=0)
        ax.text((gw_lo + gw_hi) / 2, 1.01, label,
                transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=6.5, color="grey", alpha=0.75)


NAME_FIX = {
    "FC Barcelona": "Barcelona", "Real Madrid": "Real Madrid",
    "CA Osasuna": "Osasuna", "RCD Espanyol Barcelona": "Espanyol",
    "Real Valladolid CF": "Real Valladolid", "RCD Mallorca": "Mallorca",
    "UD Las Palmas": "Las Palmas", "Girona FC": "Girona",
    "Getafe CF": "Getafe", "Celta de Vigo": "Celta Vigo",
    "Athletic Bilbao": "Athletic Club", "Villarreal CF": "Villarreal",
    "Valencia CF": "Valencia", "Sevilla FC": "Sevilla",
    "Levante UD": "Levante", "Elche CF": "Elche",
    "Rayo Vallecano": "Rayo Vallecano", "Real Oviedo": "Real Oviedo",
    "CD Leganes": "Leganes", "SD Eibar": "Eibar",
    "SD Huesca": "SD Huesca", "Granada CF": "Granada",
    # ASCII-stripped versions of accented names (encode ascii ignore drops accents)
    "Atltico de Madrid": "Atletico Madrid",
    "Real Betis Balompi": "Real Betis",
    "CD Legans": "Leganes",
    "Deportivo Alavs": "Alaves",
    "Cdiz CF": "Cadiz",
    "UD Almera": "Almeria",
}

def clean_tm_name(s):
    s = s.encode("ascii", "ignore").decode().strip()
    return NAME_FIX.get(s, s)

# ── Poisson xPts helper ──────────────────────────────────────────────────
_GOALS = np.arange(9)   # 0..8 goals

def _compute_xpts(xgf_arr, xga_arr):
    """Vectorised Poisson xPts: for each match row return expected pts for the team."""
    xgf_arr = np.maximum(np.asarray(xgf_arr, dtype=float), 1e-9)
    xga_arr = np.maximum(np.asarray(xga_arr, dtype=float), 1e-9)
    out = np.empty(len(xgf_arr))
    for k in range(len(xgf_arr)):
        pf  = stats.poisson.pmf(_GOALS, xgf_arr[k])
        pa  = stats.poisson.pmf(_GOALS, xga_arr[k])
        mat = np.outer(pf, pa)          # mat[i,j] = P(score i, concede j)
        p_win  = np.tril(mat, -1).sum() # i > j  → team wins
        p_draw = np.trace(mat)          # i == j → draw
        out[k] = 3.0 * p_win + p_draw
    return out

# ── data loaders ─────────────────────────────────────────────────────────
@st.cache_data
def load_match_data():
    df = pd.read_csv("laliga_xg.csv", parse_dates=["date"])
    home = df[["season","date","home_team","away_team","home_xg","away_xg",
               "home_goals","away_goals","result"]].copy()
    home.columns = ["season","date","team","opponent","xg_for","xg_against",
                    "goals_for","goals_against","match_result"]
    home["home"] = 1
    home["pts"]  = home.match_result.map({"H":3,"D":1,"A":0})
    away = df[["season","date","away_team","home_team","away_xg","home_xg",
               "away_goals","home_goals","result"]].copy()
    away.columns = ["season","date","team","opponent","xg_for","xg_against",
                    "goals_for","goals_against","match_result"]
    away["home"] = 0
    away["pts"]  = away.match_result.map({"A":3,"D":1,"H":0})
    tm = pd.concat([home, away], ignore_index=True).sort_values(["team","date"]).reset_index(drop=True)
    tm["xg_diff"]    = tm.xg_for - tm.xg_against
    tm["goals_diff"] = tm.goals_for - tm.goals_against
    g = tm.groupby(["season","team"])
    for w in [5, 10]:
        base = g[["xg_for","xg_against","xg_diff","pts","goals_diff"]].transform(
            lambda x: x.shift(1).rolling(w, min_periods=w).mean())
        tm[f"xgf_roll{w}"] = base["xg_for"];   tm[f"xga_roll{w}"] = base["xg_against"]
        tm[f"xgd_roll{w}"] = base["xg_diff"];  tm[f"pts_roll{w}"] = base["pts"]
        tm[f"gd_roll{w}"]  = base["goals_diff"]
        std = g[["xg_diff","pts"]].transform(
            lambda x: x.shift(1).rolling(w, min_periods=w).std())
        tm[f"xgd_std{w}"] = std["xg_diff"]; tm[f"pts_std{w}"] = std["pts"]
    tm["luck5"]  = tm["pts_roll5"]  - (tm["xgd_roll5"]  * 1.5 + 5)
    tm["luck10"] = tm["pts_roll10"] - (tm["xgd_roll10"] * 1.5 + 5)
    tm["pts_next5"] = g["pts"].transform(
        lambda x: x.shift(-1).rolling(5, min_periods=5).sum())
    # ── Poisson-based xPts per match ──────────────────────────────────────
    tm["xpts"] = _compute_xpts(tm["xg_for"].values, tm["xg_against"].values)
    return tm


@st.cache_data
def load_squad_values():
    sv = pd.read_csv("squad_values.csv")
    sv["team"] = sv["tm_name"].apply(clean_tm_name)
    return sv


@st.cache_data
def load_managers():
    """Load manager appointment/sacking data from managers.csv."""
    try:
        mgr = pd.read_csv("managers.csv")
        mgr["date_from"] = pd.to_datetime(mgr["date_from"], errors="coerce")
        mgr["date_to"]   = pd.to_datetime(mgr["date_to"],   errors="coerce")
        return mgr
    except FileNotFoundError:
        return pd.DataFrame(columns=["team","manager","date_from","date_to","status"])


@st.cache_data
def build_overperformance(_tm, _sv):
    """
    Per-team-season squad value + xG performance.
      value_rank = squad budget expectation (Transfermarkt rank, 1 = most expensive)
                   kept only as informational context (e.g. "ranked 3rd by spend")

    The actual "manager skill" metric is NOT computed here — see
    fit_value_xpts_model() / manager_skill_xpts, which compares actual xPts
    against a continuous squad-value trend line instead of a rank difference.
    Rank differences have a hard ceiling at value_rank=1 (can never show
    positive skill) and floor at value_rank=20 (can never show negative skill),
    which the continuous version avoids.

    Actual points are excluded from the skill metric — too noisy to compare
    performance against.
    """
    season_stats = (
        _tm.groupby(["season","team"])
           .agg(pts=("pts","sum"), xpts=("xpts","sum"))
           .reset_index()
    )
    merged = season_stats.merge(
        _sv[["season","team","squad_value_m"]], on=["season","team"], how="inner"
    ).dropna(subset=["squad_value_m"])

    merged["value_rank"] = merged.groupby("season")["squad_value_m"].rank(ascending=False)
    return merged


@st.cache_data
def fit_value_xpts_model(_overperf):
    """
    Continuous trend line: xPts ~ m * ln(squad_value_m) + b, fit across every
    team-season. Used to predict "expected xPts" for any squad value, so a
    manager's over/underperformance can be measured in actual xPts — with no
    artificial ceiling/floor the way rank-based comparisons have (a team
    already ranked #1 by value can never show positive rank-skill; this can).
    """
    x = np.log(_overperf["squad_value_m"].values)
    y = _overperf["xpts"].values
    m, b = np.polyfit(x, y, 1)
    return m, b


def _predicted_xpts(m, b, squad_value_m):
    return m * np.log(squad_value_m) + b


@st.cache_data
def build_manager_stints(_tm, _mgr, _overperf, _value_model, min_games=5):
    """
    Per-manager-stint skill, isolated from whoever managed the team before/after
    them in the same season. A "stint" = one manager's continuous run of games
    for one team within one season.
    """
    if _mgr.empty:
        return pd.DataFrame()

    m_coef, b_coef = _value_model
    mgr2 = _mgr.copy()
    mgr2["date_to_filled"] = mgr2["date_to"].fillna(pd.Timestamp("2100-01-01"))

    season_meta = (
        _tm.groupby(["season", "team"])
           .agg(games=("xpts", "count"))
           .reset_index()
    )
    games_lookup_all = season_meta.set_index(["season", "team"])["games"]
    value_lookup = _overperf.set_index(["season", "team"])["squad_value_m"]

    rows = []
    for season in sorted(_tm["season"].unique()):
        smeta = season_meta[season_meta.season == season]

        for team in smeta["team"]:
            if (season, team) not in value_lookup.index:
                continue
            squad_value_m = value_lookup[(season, team)]
            predicted_xpts = _predicted_xpts(m_coef, b_coef, squad_value_m)

            tdf = _tm[(_tm.team == team) & (_tm.season == season)].sort_values("date").copy()
            tmgrs = mgr2[mgr2.team == team]
            if tmgrs.empty:
                continue

            def _lookup(d):
                cand = tmgrs[(tmgrs.date_from <= d) & (tmgrs.date_to_filled >= d)]
                return cand.sort_values("date_from").iloc[-1]["manager"] if len(cand) else None

            tdf["manager"] = tdf["date"].apply(_lookup)
            tdf = tdf.dropna(subset=["manager"])
            if tdf.empty:
                continue
            tdf["block"] = (tdf["manager"] != tdf["manager"].shift()).cumsum()

            for _, block in tdf.groupby("block"):
                n = len(block)
                if n < min_games:
                    continue
                mgr_name = block["manager"].iloc[0]
                avg_xpts_pg = block["xpts"].mean()
                avg_pts_pg  = block["pts"].mean()
                games_in_season = games_lookup_all[(season, team)]
                projected_xpts = avg_xpts_pg * games_in_season
                skill = projected_xpts - predicted_xpts

                rows.append({
                    "manager": mgr_name, "team": team, "season": season,
                    "games": n, "avg_xpts_pg": avg_xpts_pg, "avg_pts_pg": avg_pts_pg,
                    "projected_xpts": projected_xpts, "predicted_xpts": predicted_xpts,
                    "squad_value_m": squad_value_m, "stint_skill": skill,
                })

    return pd.DataFrame(rows)


@st.cache_data
def load_wages():
    try:
        return pd.read_csv("wages_capology.csv")
    except FileNotFoundError:
        return pd.DataFrame(columns=["season","team","wage_m"])


@st.cache_data
def build_predictive_correlations(_overperf, _wages):
    from scipy import stats as _stats
    rows = []
    seasons = sorted(_overperf["season"].unique())
    for i, s in enumerate(seasons[:-1]):
        curr = _overperf[_overperf.season == s]
        nxt  = _overperf[_overperf.season == seasons[i+1]]
        merged = curr.merge(nxt[["team","pts"]], on="team", suffixes=("","_next"))
        if len(merged) < 8: continue
        for col, label in [("xpts","xPts this season"),("pts","Actual pts this season"),
                           ("squad_value_m","Squad value"),("predicted_xpts","Budget prediction")]:
            if col in merged.columns:
                r, _ = _stats.pearsonr(merged[col], merged["pts_next"])
                rows.append({"metric": label, "season": s, "r": r})
        wm = _wages[_wages.season == s] if not _wages.empty else pd.DataFrame()
        wm2 = curr.merge(wm[["team","wage_m"]], on="team") if not wm.empty else pd.DataFrame()
        if len(wm2) >= 8:
            wm2n = wm2.merge(nxt[["team","pts"]], on="team", suffixes=("","_next"))
            if len(wm2n) >= 8:
                r2, _ = _stats.pearsonr(wm2n["wage_m"], wm2n["pts_next"])
                rows.append({"metric": "Wages","season": s, "r": r2})
    return pd.DataFrame(rows)


@st.cache_data
def build_justice_table(_overperf):
    rows = []
    for s, g in _overperf.groupby("season"):
        g2 = g.copy().sort_values("xpts", ascending=False).reset_index(drop=True)
        g2["xpts_rank"] = g2.index + 1
        g3 = g2.sort_values("pts", ascending=False).reset_index(drop=True)
        g3["pts_rank"] = g3.index + 1
        merged = g2[["team","season","xpts","xpts_rank"]].merge(
            g3[["team","pts_rank","pts"]], on="team")
        merged["rank_change"] = merged["pts_rank"] - merged["xpts_rank"]
        merged["luck"]        = merged["pts"] - merged["xpts"]
        rows.append(merged)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


@st.cache_data
def build_pre_season_accuracy(_overperf):
    rows = []
    for s, g in _overperf.groupby("season"):
        g2 = g.copy()
        g2["sv_rank"]  = g2["squad_value_m"].rank(ascending=False)
        g2["pts_rank"] = g2["pts"].rank(ascending=True)
        bottom3_pred = set(g2.nsmallest(3,"squad_value_m")["team"])
        bottom3_act  = set(g2.nsmallest(3,"pts")["team"])
        top6_pred    = set(g2.nlargest(6,"squad_value_m")["team"])
        top6_act     = set(g2.nlargest(6,"pts")["team"])
        rows.append({"season": s,
                     "bottom3_hits": len(bottom3_pred & bottom3_act),
                     "top6_hits":    len(top6_pred & top6_act)})
    return pd.DataFrame(rows)


@st.cache_data
def load_squad_data():
    """Join squad_stats (age, minutes) with player_values (market value)."""
    try:
        ss = pd.read_csv("squad_stats.csv")
        ss["player_id"] = ss["player_id"].astype(str)
    except FileNotFoundError:
        return pd.DataFrame()

    try:
        pv = pd.read_csv("player_values.csv")
        pv["player_id"] = pv["player_id"].astype(str)
        ss = ss.merge(
            pv[["player_id", "season", "team", "value_m"]],
            on=["player_id", "season", "team"],
            how="left",
        )
    except FileNotFoundError:
        ss["value_m"] = None

    def _band(age):
        if pd.isna(age):
            return "Unknown"
        if age <= 23:
            return "Development (≤23)"
        if age <= 30:
            return "Peak (24–30)"
        return "Veteran (31+)"

    ss["age_band"] = ss["age"].apply(_band)
    return ss


@st.cache_data
def load_transfer_fees():
    """Real transfer fees scraped from Transfermarkt (collect_transfer_fees.py)."""
    try:
        tf = pd.read_csv("transfer_fees.csv")
        tf["team"] = tf["team"].replace({
            "CD Leganés":            "Leganes",
            "RCD Espanyol Barcelona": "Espanyol",
            "Real Valladolid CF":    "Real Valladolid",
        })
        return tf
    except FileNotFoundError:
        return pd.DataFrame(columns=["season", "team", "direction", "player_id", "player_name",
                                      "age", "other_club", "fee_type", "fee_eur", "fee_m"])


@st.cache_resource
def train_model(_tm):
    FEATURES = [
        "xgf_roll5","xga_roll5","xgd_roll5","xgd_std5",
        "xgf_roll10","xga_roll10","xgd_roll10","xgd_std10",
        "home",
    ]
    clean = _tm[FEATURES + ["pts_next5","season"]].dropna()
    train = clean
    xgb = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    xgb.fit(train[FEATURES], train["pts_next5"])
    return xgb, train["pts_next5"].mean(), FEATURES


def get_verdict(pred, xgd5, xgd10, benchmark):
    if pred < benchmark * 0.75 and xgd10 < -0.35: return "ALARM"
    elif pred < benchmark * 0.90:                  return "WATCH"
    elif xgd5 > xgd10 + 0.3 and pred >= benchmark * 0.85: return "RECOVERING"
    return "STABLE"


def build_timeline(tm, xgb, benchmark, FEATURES, team, season):
    tdf = tm[(tm.team == team) & (tm.season == season)].reset_index(drop=True)
    rows = []
    cum_pts = 0
    cum_xpts = 0
    for i, row in tdf.iterrows():
        cum_pts  += row["pts"]
        cum_xpts += row["xpts"]
        feat = row[FEATURES].values.reshape(1, -1)
        if pd.isnull(feat).any(): continue
        pred    = float(xgb.predict(feat)[0])
        verdict = get_verdict(pred, row["xgd_roll5"], row["xgd_roll10"], benchmark)
        rows.append({
            "gw":       i - tdf.index[0] + 1,
            "date":     row["date"],
            "pred":     pred,
            "xgd5":     row["xgd_roll5"],
            "xgd10":    row["xgd_roll10"],
            "xgf5":     row["xgf_roll5"],
            "xga5":     row["xga_roll5"],
            "pts5":     row["pts_roll5"],
            "pts":      row["pts"],
            "xg_for":   row["xg_for"],
            "xg_against": row["xg_against"],
            "xpts":     row["xpts"],
            "cum_pts":  cum_pts,
            "cum_xpts": cum_xpts,
            "home":     int(row["home"]),
            "verdict":  verdict,
        })
    return pd.DataFrame(rows)


@st.cache_data
def get_standings(season=2025):
    """
    Official La Liga tiebreak order when teams are level on points:
      1. Head-to-head results among ONLY the tied teams (mini-table: points,
         then goal difference, then goals scored — within those matches)
      2. (handled automatically since #1 resolves nearly all ties)
    Plain "most points, then overall goal difference" is NOT how La Liga
    breaks ties — head-to-head comes first, and it can flip the order
    overall-GD alone would give.
    """
    s = tm[tm.season == season]
    base = s.groupby("team").agg(
        pts=("pts", "sum"), gf=("goals_for", "sum"), ga=("goals_against", "sum"),
    ).reset_index()
    base["gd"] = base["gf"] - base["ga"]

    def _h2h_order(teams_tied):
        if len(teams_tied) == 1:
            return teams_tied
        sub = s[s.team.isin(teams_tied) & s.opponent.isin(teams_tied)]
        rows = []
        for t in teams_tied:
            tt = sub[sub.team == t]
            gf_t, ga_t = tt["goals_for"].sum(), tt["goals_against"].sum()
            rows.append((t, tt["pts"].sum(), gf_t - ga_t, gf_t))
        rows.sort(key=lambda r: (r[1], r[2], r[3]), reverse=True)
        return [r[0] for r in rows]

    order = []
    for pts_val in sorted(base["pts"].unique(), reverse=True):
        order.extend(_h2h_order(base.loc[base["pts"] == pts_val, "team"].tolist()))

    pts = base.set_index("team").loc[order].reset_index()
    pts.index = range(1, len(pts) + 1)
    pts["zone"] = pts.index.map(
        lambda i: "Title" if i == 1 else
                  ("Europe" if i <= 6 else ("Relegated" if i >= len(pts) - 2 else "Mid-table")))
    return pts


# ── initialise ───────────────────────────────────────────────────────────
tm          = load_match_data()
sv          = load_squad_values()
overperf    = build_overperformance(tm, sv)
xgb_model, AVG, FEATURES = train_model(tm)

managers     = load_managers()
value_xpts_model = fit_value_xpts_model(overperf)
overperf["predicted_xpts"]     = _predicted_xpts(*value_xpts_model, overperf["squad_value_m"])
overperf["manager_skill_xpts"] = overperf["xpts"] - overperf["predicted_xpts"]
mgr_stints   = build_manager_stints(tm, managers, overperf, value_xpts_model)
wages        = load_wages()
pred_corrs   = build_predictive_correlations(overperf, wages)
justice_table = build_justice_table(overperf)
preseas_acc   = build_pre_season_accuracy(overperf)
squad_data     = load_squad_data()
transfer_fees  = load_transfer_fees()
seasons_available = sorted(tm["season"].unique(), reverse=True)

# ── sidebar ──────────────────────────────────────────────────────────────
st.sidebar.title("La Liga Intelligence")
st.sidebar.caption(
    "Eight seasons of La Liga data, xG, squad value, and manager skill,"
    " distilled into one place. Track how a team's underlying quality compares to its budget,"
    " spot when results are masking a deeper problem, and see which managers"
    " are genuinely beating their resources."
)
st.sidebar.divider()
selected_season = st.sidebar.selectbox(
    "Select season", seasons_available,
    format_func=lambda s: f"{s}/{str(s+1)[-2:]}",
)
st.sidebar.markdown(f"**{selected_season}/{str(selected_season+1)[-2:]} Season**")
teams_for_season = sorted(tm[tm.season == selected_season]["team"].unique())
selected_team = st.sidebar.selectbox(
    "Select a team", teams_for_season, key=f"team_select_{selected_season}"
)
st.sidebar.divider()

with st.sidebar.expander("📖 New here? Start with the basics", expanded=False):
    st.markdown("""
**Expected Goals (xG)** measures how *good* a scoring chance was, not whether it
actually went in. A tap-in from 2 yards has high xG — it's expected to score almost
every time. A shot from 40 yards has very low xG. Adding up a team's xG across a
match shows how good their chances really were — a fairer picture of performance
than the final score, which can be skewed by one lucky deflection, a red card, or a
moment of individual brilliance.

**xGD** (xG Difference) = a team's xG created minus the xG they conceded. One number
that sums up how dominant a team's underlying play was in a game or over a stretch
of games. Positive = creating better chances than they're conceding.

**Expected Points (xPts)** turns match-by-match xG into the points a team *deserved*.
If a team created much better chances than their opponent, their xPts for that game
will be close to 3 (a deserved win) — even if the actual result was a 0-0 draw
because of bad finishing or a great goalkeeping performance.

- If **actual points > xPts** → the team got **lucky** (results flattered their play).
- If **actual points < xPts** → the team was **unlucky** (they played well but didn't
  get the results to show it).

**Squad value** is simply how much a team's whole squad is worth on the transfer
market (source: Transfermarkt) — a stand-in for "how big is this team's budget."

**Manager Skill** (used throughout this app) compares a team's actual xPts to what a
trend line says a squad of that value *should* produce, based on every team-season in
La Liga since 2018/19. A manager who gets more out of a modest squad than that line
predicts shows a positive number; one who underperforms a big budget shows a negative
one — regardless of whether their squad is the cheapest or the most expensive in the
league.
""")

standings    = get_standings(selected_season)

# ── page header ──────────────────────────────────────────────────────────
team_row  = standings[standings.team == selected_team]
final_pos = team_row.index[0] if len(team_row) else "?"
final_pts = int(team_row.pts.values[0]) if len(team_row) else "?"
zone      = team_row.zone.values[0] if len(team_row) else ""
relegated = (zone == "Relegated")

suffix = " — RELEGATED" if relegated else ""
st.title(f"{selected_team}{suffix}")
st.caption(f"Final position: **{final_pos}/20** · {final_pts} points · {zone} · "
           f"{selected_season}/{str(selected_season+1)[-2:]} season")

_sv_pre = overperf[(overperf.team == selected_team) & (overperf.season == selected_season)]
_pred_xpts_pre = float(_sv_pre["predicted_xpts"].values[0]) if len(_sv_pre) else None
budget_per5 = (_pred_xpts_pre / 38 * 5) if _pred_xpts_pre else AVG
tl = build_timeline(tm, xgb_model, budget_per5, FEATURES, selected_team, selected_season)

# squad value / overperformance data for selected team
sv_team    = overperf[overperf.team == selected_team].sort_values("season")
sv_season  = sv_team[sv_team.season == selected_season]

def _safe(df, col):
    return float(df[col].values[0]) if len(df) and col in df.columns else None

mgr_skill_val = _safe(sv_season, "manager_skill_xpts")
val_rank      = int(sv_season["value_rank"].values[0]) if len(sv_season) else None
val_eur       = float(sv[(sv.team == selected_team) & (sv.season == selected_season)]["squad_value_m"].values[0]) \
                if len(sv[(sv.team == selected_team) & (sv.season == selected_season)]) else None
season_xpts   = float(tl["cum_xpts"].iloc[-1]) if len(tl) else None

# ── KPI row ──────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("Squad value",
          f"€{val_eur:.0f}m" if val_eur else "—",
          help=f"What this squad is worth on the transfer market. Ranked {val_rank}/20 most expensive in the league this season." if val_rank else "")
c2.metric("xPts vs Actual",
          f"{season_xpts:.1f} xPts / {final_pts} pts" if season_xpts else "—",
          delta=f"{season_xpts - final_pts:+.1f}" if season_xpts else None,
          delta_color="inverse",
          help="xPts = points this team's chance quality deserved. The number on the right is xPts minus actual points: "
               "positive = unlucky (deserved more than they got), negative = lucky (got more than they deserved).")
if mgr_skill_val is not None:
    c3.metric("Manager skill",
              f"{mgr_skill_val:+.1f} xPts",
              delta=f"{mgr_skill_val:+.1f} vs trend",
              delta_color="normal",
              help="How many xPts this team produced above (+) or below (−) what a squad of this market value "
                   "typically produces league-wide. The closest thing this app has to grading the manager specifically, "
                   "separate from luck or budget.")
else:
    c3.metric("Manager skill", "—")

st.divider()

# ── tabs ─────────────────────────────────────────────────────────────────
# NOTE: this is a deliberately trimmed public build — Article 1, plus Groups
# 1-2 of Article 2 (Groups 3-4 unlock here as their articles publish); see
# dashboard.py in the private dev repo for the full version.
tab1, tab2 = st.tabs(["Article 1 — Framework", "Article 2 — Clubs"])


# ════════════════════════════════════════════════════════════════════════
# TAB 1 — season timeline
# ════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Article 1 — The Framework: What the Numbers Actually Measure")
    st.caption(
        "The methodology behind every chart in this dashboard: xG, xPts, the squad-value budget model, "
        "manager skill, and why xPts predicts next season better than anything else."
    )
    st.divider()
    st.subheader("What predicts next season's points?")
    st.caption(
        "For each pair of consecutive seasons in the dataset, we measure how strongly each metric "
        "this season correlates with actual points the following season (Pearson r). "
        "A value near +1 means it is a reliable leading indicator; near 0 means it tells you nothing "
        "about what comes next. This answers the question: is squad value actually the best predictor, "
        "or does xPts this season tell you more?"
    )

    if not pred_corrs.empty:
        corr_avg = (
            pred_corrs[pred_corrs["metric"] != "Budget prediction"]
            .groupby("metric")["r"]
            .agg(avg_r="mean", min_r="min", max_r="max", seasons="count")
            .reset_index()
            .sort_values("avg_r", ascending=False)
        )

        fig_pc, ax_pc = plt.subplots(figsize=(8, max(3, len(corr_avg) * 0.55)))
        colors_pc = ["#2ecc71" if r >= 0.5 else "#e67e22" if r >= 0.3 else "#e74c3c"
                     for r in corr_avg["avg_r"]]
        bars_pc = ax_pc.barh(corr_avg["metric"], corr_avg["avg_r"],
                             color=colors_pc, edgecolor="white", alpha=0.85)
        ax_pc.errorbar(corr_avg["avg_r"], range(len(corr_avg)),
                       xerr=[corr_avg["avg_r"] - corr_avg["min_r"],
                             corr_avg["max_r"] - corr_avg["avg_r"]],
                       fmt="none", color="black", capsize=4, lw=1.2, alpha=0.6)
        for bar, val in zip(bars_pc, corr_avg["avg_r"]):
            ax_pc.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                       f"r = {val:.2f}", va="center", fontsize=9)
        ax_pc.axvline(0, color="black", lw=0.8)
        ax_pc.axvline(0.5, color="grey", lw=0.8, ls=":", alpha=0.6, label="r = 0.5 (strong)")
        ax_pc.set_xlabel("Average Pearson r (correlation with next season's points)", fontsize=9)
        ax_pc.set_title("Predictive validity — averaged across all season pairs", fontsize=11)
        ax_pc.legend(fontsize=8)
        ax_pc.grid(axis="x", alpha=0.2)
        ax_pc.set_xlim(-0.1, 1.0)
        plt.tight_layout()
        st.pyplot(fig_pc)
        plt.close()

        st.caption(
            "Error bars show the range across different season pairs — a wide bar means the metric "
            "is more reliable in some seasons than others. Green = strong predictor (r ≥ 0.5), "
            "orange = moderate (r ≥ 0.3), red = weak."
        )

    else:
        st.info(
            "No predictive correlation data available. This requires at least two consecutive seasons "
            "of data for the same teams."
        )

    # ── Squad value vs wages — which explains xPts better? ─────────────────
    st.divider()
    st.subheader("Squad value vs wages — which explains xPts better?")
    st.caption(
        "Two ways to measure a club's financial firepower: squad market value (Transfermarkt) and "
        "total wage bill (Capology). This compares how well each one correlates with xPts in the "
        "same season — i.e. which is the better proxy for on-pitch quality."
    )

    m_x, b_x = value_xpts_model
    wm_curr = overperf.merge(wages, on=["season", "team"], how="inner") if not wages.empty else pd.DataFrame()

    r_sv_xpts, _ = stats.pearsonr(overperf["squad_value_m"], overperf["xpts"])

    if not wm_curr.empty and len(wm_curr) >= 5:
        r_wg_xpts, _ = stats.pearsonr(wm_curr["wage_m"], wm_curr["xpts"])
        r_sv_wm, _   = stats.pearsonr(wm_curr["squad_value_m"], wm_curr["xpts"])

        fig_sw, (ax_sv, ax_wg) = plt.subplots(1, 2, figsize=(13, 5))

        # Squad value vs xPts
        ax_sv.scatter(wm_curr["squad_value_m"], wm_curr["xpts"],
                      alpha=0.55, color="#3498db", edgecolors="white", lw=0.5, s=50)
        _xs = np.linspace(wm_curr["squad_value_m"].min(), wm_curr["squad_value_m"].max(), 100)
        ax_sv.plot(_xs, m_x * np.log(_xs) + b_x, color="black", lw=1.5, ls="--")
        ax_sv.set_xlabel("Squad value (€m)", fontsize=9)
        ax_sv.set_ylabel("xPts", fontsize=9)
        ax_sv.set_title(f"Squad value vs xPts  (r = {r_sv_wm:.2f})", fontsize=10)
        ax_sv.grid(alpha=0.2)

        # Wages vs xPts
        ax_wg.scatter(wm_curr["wage_m"], wm_curr["xpts"],
                      alpha=0.55, color="#e67e22", edgecolors="white", lw=0.5, s=50)
        _mw, _bw = np.polyfit(np.log(wm_curr["wage_m"]), wm_curr["xpts"], 1)
        _xw = np.linspace(wm_curr["wage_m"].min(), wm_curr["wage_m"].max(), 100)
        ax_wg.plot(_xw, _mw * np.log(_xw) + _bw, color="black", lw=1.5, ls="--")
        ax_wg.set_xlabel("Total wage bill (€m)", fontsize=9)
        ax_wg.set_ylabel("xPts", fontsize=9)
        ax_wg.set_title(f"Wages vs xPts  (r = {r_wg_xpts:.2f})", fontsize=10)
        ax_wg.grid(alpha=0.2)

        plt.tight_layout()
        st.pyplot(fig_sw)
        plt.close()

        winner = "Squad value" if r_sv_wm >= r_wg_xpts else "Wages"
        diff   = abs(r_sv_wm - r_wg_xpts)
        st.caption(
            f"**{winner}** correlates more strongly with xPts (r = {max(r_sv_wm, r_wg_xpts):.2f} "
            f"vs r = {min(r_sv_wm, r_wg_xpts):.2f}, difference of {diff:.2f}). "
            f"Both use a log scale fit since the relationship between spending and performance "
            f"flattens at the top end — doubling a €200m squad doesn't double the xPts."
        )
    else:
        # No wages data — just show squad value vs xPts
        fig_sv2, ax_sv2 = plt.subplots(figsize=(7, 5))
        ax_sv2.scatter(overperf["squad_value_m"], overperf["xpts"],
                       alpha=0.5, color="#3498db", edgecolors="white", lw=0.5, s=50)
        _xs2 = np.linspace(overperf["squad_value_m"].min(), overperf["squad_value_m"].max(), 100)
        ax_sv2.plot(_xs2, m_x * np.log(_xs2) + b_x, color="black", lw=1.5, ls="--")
        ax_sv2.set_xlabel("Squad value (€m)", fontsize=9)
        ax_sv2.set_ylabel("xPts", fontsize=9)
        ax_sv2.set_title(f"Squad value vs xPts  (r = {r_sv_xpts:.2f})", fontsize=10)
        ax_sv2.grid(alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig_sv2)
        plt.close()
        st.info("No wages data found (wages_capology.csv missing) — showing squad value only.")


    st.divider()
    season_label = f"{selected_season}/{str(selected_season+1)[-2:]}"

    st.subheader("Squad value vs final position — all seasons")
    st.caption(
        "Does spending more money actually buy more points? This chart plots every team's squad "
        "value against their results across all 8 seasons of data. The orange dashed curve is the "
        "**trend line** for xPts: it shows what a squad of a given value typically produces, "
        "league-wide. Dots above the curve over-performed their budget that season; dots below it "
        "under-performed. The gap between a team's *actual points* and their *xPts* (the two dot "
        "colours/shapes) is separately where **luck** comes in — see the sidebar for that distinction."
    )

    r_val, _ = stats.pearsonr(overperf["squad_value_m"], overperf["pts"])
    r_xpts, _ = stats.pearsonr(overperf["squad_value_m"], overperf["xpts"])
    r_xpts_log, _ = stats.pearsonr(np.log(overperf["squad_value_m"]), overperf["xpts"])
    with st.expander("Technical detail: correlation strength"):
        st.markdown(
            f"**Pearson r = {r_val:.3f}** (actual pts, raw value) | **r = {r_xpts:.3f}** (xPts, raw value) | "
            f"**r = {r_xpts_log:.3f}** (xPts, log value — the fit used for the trend line and Manager Skill) "
            f"between squad market value and season points across {len(overperf)} team-seasons. "
            f"(r ranges from -1 to 1; closer to 1 means budget predicts results more reliably.)"
        )

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # scatter: value vs actual pts (grey) + xPts (orange)
    ax = axes[0]
    sc = ax.scatter(overperf["squad_value_m"], overperf["pts"],
                    c=overperf["season"], cmap="plasma", alpha=0.4, s=28, label="Actual pts")
    ax.scatter(overperf["squad_value_m"], overperf["xpts"],
               c=overperf["season"], cmap="plasma", alpha=0.4, s=28, marker="^", label="xPts")
    # highlight selected team
    team_pts = overperf[overperf.team == selected_team]
    ax.scatter(team_pts["squad_value_m"], team_pts["pts"],
               color="red", s=100, zorder=6, edgecolors="white", linewidths=1.2,
               label=f"{selected_team} (actual)")
    ax.scatter(team_pts["squad_value_m"], team_pts["xpts"],
               color="darkorange", s=100, zorder=6, edgecolors="white", linewidths=1.2,
               marker="^", label=f"{selected_team} (xPts)")
    m, b = np.polyfit(overperf["squad_value_m"], overperf["pts"], 1)
    x_line = np.linspace(1, overperf["squad_value_m"].max(), 200)
    ax.plot(x_line, m*x_line+b, color="crimson", linewidth=1.5, linestyle="--", alpha=0.8, label="Pts trend (linear)")
    m_x, b_x = value_xpts_model
    ax.plot(x_line, m_x*np.log(x_line)+b_x, color="darkorange", linewidth=1.5, linestyle="--", alpha=0.8,
            label="xPts trend (log) — used for Manager Skill")
    ax.set_xlabel("Squad market value (€m)", fontsize=10)
    ax.set_ylabel("Points", fontsize=10)
    ax.set_title(f"Squad value vs pts  (r={r_val:.3f} actual, r={r_xpts:.3f} xPts raw, r={r_xpts_log:.3f} xPts log)", fontsize=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"€{x:.0f}m"))
    ax.legend(fontsize=8)
    plt.colorbar(sc, ax=ax, label="Season")

    # history bar chart: manager skill per season
    ax = axes[1]
    if len(sv_team) > 0:
        x = np.arange(len(sv_team))
        colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in sv_team["manager_skill_xpts"]]
        ax.bar(x, sv_team["manager_skill_xpts"], 0.6, color=colors, edgecolor="white", alpha=0.88)
        ax.axhline(0,  color="black", linewidth=1.2)
        ax.axhline(3,  color="grey",  linewidth=0.8, linestyle="--", alpha=0.5)
        ax.axhline(-3, color="grey",  linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s}/{str(s+1)[-2:]}" for s in sv_team["season"]], fontsize=8)
        ax.set_ylabel("xPts above/below value-predicted trend", fontsize=9)
        ax.set_title(f"{selected_team} — Manager skill per season (actual xPts vs. value trend)", fontsize=10)
        for i, v in enumerate(sv_team["manager_skill_xpts"]):
            ax.text(i, v + (0.3 if v >= 0 else -0.6), f"{v:+.1f}", ha="center", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No squad value data available", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)

    plt.suptitle(f"Squad Value Analysis — {selected_team}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.success(
        "**Manager skill** (see sidebar for the full idea) compares this team's actual xPts each "
        "season to what the orange trend line above predicts for a squad of that value. "
        "🟢 Green bar = beat the budget's expectation that season. 🔴 Red bar = fell short of it."
    )

    # ── Predictive correlations ─────────────────────────────────────────────
    st.divider()
    st.subheader(f"{season_label} — Manager skill vs squad budget")
    st.caption(
        "Every manager who took charge this season, ranked from worst to best on **Manager skill** — "
        "how many xPts their stint produced above (🟢) or below (🔴) what's typical for a squad of "
        "that market value. Per-stint isolation means a mid-season change splits credit correctly: "
        "each manager is only judged on their own games, not the team's whole season."
    )
    _mgr_season = mgr_stints[mgr_stints.season == selected_season].copy() if not mgr_stints.empty else pd.DataFrame()
    if _mgr_season.empty:
        st.warning(f"No manager stint data available for {season_label}.")
    else:
        _mgr_season = _mgr_season.sort_values("stint_skill")
        _mgr_season["label"] = _mgr_season["manager"] + " (" + _mgr_season["team"] + ")"
        colors_bar_mgr = ["#2ecc71" if v >= 0 else "#e74c3c" for v in _mgr_season["stint_skill"]]
        x = np.arange(len(_mgr_season))
        fig, ax = plt.subplots(figsize=(14, 7))
        bars = ax.bar(x, _mgr_season["stint_skill"].round(1), 0.6, color=colors_bar_mgr, edgecolor="white", alpha=0.88)
        ax.bar_label(bars, fmt="%+.1f", fontsize=7.5, padding=2)
        ax.set_xticks(x)
        ax.set_xticklabels(_mgr_season["label"], rotation=45, ha="right", fontsize=8)
        ax.axhline(0,  color="black",  linewidth=1.2)
        ax.axhline(3,  color="grey",   linewidth=0.8, linestyle="--", alpha=0.6)
        ax.axhline(-3, color="grey",   linewidth=0.8, linestyle="--", alpha=0.6)
        for idx, t in enumerate(_mgr_season["team"]):
            if t == selected_team:
                ax.axvspan(idx - 0.5, idx + 0.5, color="yellow", alpha=0.15, zorder=0)
        ax.set_ylabel("xPts above/below value-predicted trend", fontsize=10)
        ax.set_title(f"La Liga {season_label} — Manager Skill by manager (xPts vs. value-predicted trend)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


    st.divider()
    # ── cumulative xPts vs actual pts ───────────────────────────────────────
    st.subheader("Cumulative points vs Expected Points (xPts)")
    st.caption(
        "Running total across the season. There are three separate ideas stacked on this chart:\n\n"
        "**1. What actually happened** (blue solid line) — the real points the team collected match by match.\n\n"
        "**2. What the football deserved** (orange dashed line, xPts) — points based on the quality of "
        "chances created and conceded each game, calculated with a Poisson model. "
        "Green shading between blue and orange = the team ran **lucky** (actual results better than "
        "the football justified). Red shading = **unlucky** (the football deserved more points than they got).\n\n"
        "**3. What the squad value predicted** (purple dash-dot line) — a completely separate model that "
        "uses only squad market value (Transfermarkt data) to estimate where a team of that budget "
        "should end up by the end of the season, then scales that down gameweek by gameweek as a "
        "diagonal pace line. Purple shading above the orange line = the team is generating **better "
        "chance quality than their budget suggests** (manager overperformance). Red shading below it = "
        "underperforming their resources, regardless of luck."
    )

    gw       = tl["gw"].values
    cum_pts  = tl["cum_pts"].values
    cum_xpts = tl["cum_xpts"].values
    luck_diff = cum_pts - cum_xpts        # positive = lucky
    pred_xpts_val = _safe(sv_season, "predicted_xpts")

    fig2, ax = plt.subplots(figsize=(16, 4))
    ax.plot(gw, cum_pts,  color="steelblue", linewidth=2.5, label="Actual pts (cumulative)", zorder=4)
    ax.plot(gw, cum_xpts, color="#e67e22",   linewidth=2,   linestyle="--",
            label="xPts (cumulative — Poisson model)", zorder=4)
    if pred_xpts_val and len(gw) > 0:
        budget_pace = [pred_xpts_val * (i + 1) / len(gw) for i in range(len(gw))]
        ax.plot(gw, budget_pace, color="#8e44ad", linewidth=1.4, linestyle="-.",
                label=f"Budget prediction ({pred_xpts_val:.1f} xPts season total)", zorder=3)
        ax.fill_between(gw, cum_xpts, budget_pace,
                        where=[x >= b for x, b in zip(cum_xpts, budget_pace)],
                        alpha=0.08, color="#8e44ad",
                        label="xPts above budget (manager skill)")
        ax.fill_between(gw, cum_xpts, budget_pace,
                        where=[x < b for x, b in zip(cum_xpts, budget_pace)],
                        alpha=0.08, color="#e74c3c",
                        label="xPts below budget (underperforming resources)")
    ax.fill_between(gw, cum_pts, cum_xpts,
                    where=(luck_diff >= 0), alpha=0.15, color="#2ecc71",
                    label="Lucky (actual > xPts)")
    ax.fill_between(gw, cum_pts, cum_xpts,
                    where=(luck_diff <  0), alpha=0.15, color="#e74c3c",
                    label="Unlucky (actual < xPts)")
    ax.set_xlabel("Gameweek", fontsize=10)
    ax.set_ylabel("Cumulative points", fontsize=10)
    ax.set_title(
        f"{selected_team}  |  Season total: {final_pts} actual pts  /  "
        f"{cum_xpts[-1]:.1f} xPts  →  "
        f"{'Lucky +' if luck_diff[-1]>=0 else 'Unlucky '}{abs(luck_diff[-1]):.1f} pts",
        fontsize=11
    )
    _shade_transfer_windows(ax, tl, selected_season)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(gw[0] - 0.5, gw[-1] + 0.5)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()


# ════════════════════════════════════════════════════════════════════════
# TAB 2 — Article 2, Group 1 only (Club Rankings & Deserved Performance)
# ════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Article 2 — The Clubs: Who's Beating the Budget and Who Isn't")
    st.caption(
        "Club-level overperformance and underperformance, and pre-season prediction accuracy."
    )
    st.divider()
    st.subheader("Group 1 — Club Rankings & Deserved Performance")
    st.caption(
        "How every club stacks up against their budget, and what the table would look like if it only rewarded quality of play."
    )
    st.markdown("#### Club Rankings — xPts above/below budget expectation")
    st.caption(
        "Each club's average xPts residual across all La Liga seasons in the dataset: "
        "how many xPts per season they produced above or below what their squad value predicted. "
        "Positive = consistently overperforms budget. Negative = consistently underperforms."
    )

    cl_all_seasons = st.checkbox(
        "Show all seasons combined (instead of just the season selected in the sidebar)",
        value=True, key="cl_all_seasons"
    )
    if cl_all_seasons:
        cl_df_raw = overperf.copy()
        cl_scope_label = "all seasons (2018/19-2025/26)"
        cl_min_season_slider_max = 8
    else:
        cl_df_raw = overperf[overperf["season"] == selected_season].copy()
        cl_scope_label = f"{selected_season}/{str(selected_season+1)[-2:]} only"
        cl_min_season_slider_max = 1
    if cl_all_seasons:
        cl_min_seasons = st.slider("Minimum seasons in La Liga", 1, cl_min_season_slider_max, 3, key="cl_min_seasons")
    else:
        cl_min_seasons = 1
    cl_df = cl_df_raw.copy()
    cl_df["xpts_residual"] = cl_df["xpts"] - cl_df["predicted_xpts"]
    cl_df["pts_residual"]  = cl_df["pts"]  - cl_df["predicted_xpts"]

    cl_agg = (
        cl_df.groupby("team")
        .apply(lambda g: pd.Series({
            "seasons":       len(g),
            "avg_residual":  g["xpts_residual"].mean(),
            "avg_pts_res":   g["pts_residual"].mean(),
            "seasons_above": (g["xpts_residual"] > 0).sum(),
            "best_season":   g["xpts_residual"].max(),
            "worst_season":  g["xpts_residual"].min(),
        }))
        .reset_index()
    )
    cl_agg = cl_agg[cl_agg["seasons"] >= cl_min_seasons].sort_values("avg_residual", ascending=False).reset_index(drop=True)
    cl_agg.index += 1

    if cl_agg.empty:
        st.info("No clubs meet that threshold.")
    else:
        fig_cl, ax_cl = plt.subplots(figsize=(9, max(4, len(cl_agg) * 0.42)))
        colors_cl = ["#2ecc71" if v >= 0 else "#e74c3c" for v in cl_agg["avg_residual"]]
        bars = ax_cl.barh(
            cl_agg["team"][::-1],
            cl_agg["avg_residual"][::-1],
            color=colors_cl[::-1],
            edgecolor="white", linewidth=0.5
        )
        ax_cl.axvline(0, color="grey", linewidth=0.9, linestyle="--", alpha=0.6)
        for bar, val in zip(bars, cl_agg["avg_residual"][::-1]):
            ax_cl.text(
                val + (0.15 if val >= 0 else -0.15),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.1f}",
                va="center", ha="left" if val >= 0 else "right", fontsize=8
            )
        ax_cl.set_xlabel("Avg xPts vs value-predicted trend (+ = overperforming)", fontsize=9)
        ax_cl.set_title(f"Club xPts residual - all seasons (2018/19-2025/26)", fontsize=11)
        ax_cl.grid(axis="x", alpha=0.18)
        plt.tight_layout()
        st.pyplot(fig_cl)
        plt.close()

    st.divider()

    # ── Justice League Table ─────────────────────────────────────────
    st.markdown("#### The Justice League Table")
    st.caption(
        "Replace every team's actual points with their full-season xPts. "
        "A team above the line got lucky (finished higher on pts than quality suggests); "
        "below the line was unlucky."
    )
    jt_season = st.selectbox(
        "Season", sorted(justice_table["season"].unique(), reverse=True),
        format_func=lambda s: f"{s}/{str(s+1)[-2:]}", key="jt_season"
    )
    jt = justice_table[justice_table.season == jt_season].copy()
    col_j1, col_j2 = st.columns(2)
    with col_j1:
        st.markdown("**Actual position vs deserved position (xPts)**")
        jt_sorted = jt.sort_values("xpts_rank").reset_index(drop=True)
        y_pos = np.arange(len(jt_sorted))
        fig_dm, ax_dm = plt.subplots(figsize=(6, max(4, len(jt_sorted) * 0.38)))
        for i, row in jt_sorted.iterrows():
            line_color = "#e74c3c" if row["luck"] > 0 else "#2ecc71"
            ax_dm.plot([row["pts_rank"], row["xpts_rank"]], [i, i],
                       color=line_color, lw=1.8, alpha=0.55, zorder=2)
        ax_dm.scatter(jt_sorted["pts_rank"], y_pos, color="#3498db", s=70, zorder=4,
                      edgecolors="white", linewidths=0.6, label="Actual position")
        ax_dm.scatter(jt_sorted["xpts_rank"], y_pos, color="#2c3e50", s=70, zorder=4,
                      marker="D", edgecolors="white", linewidths=0.6, label="Deserved position (xPts)")
        for i, row in jt_sorted.iterrows():
            ax_dm.annotate(f"{row['pts']:.0f} pts", (row["pts_rank"], i),
                           fontsize=6, color="#2471a3", xytext=(0, 6), textcoords="offset points",
                           ha="center", va="bottom")
            ax_dm.annotate(f"{row['xpts']:.1f} xPts", (row["xpts_rank"], i),
                           fontsize=6, color="#1c2833", xytext=(0, -6), textcoords="offset points",
                           ha="center", va="top")
        ax_dm.set_yticks(y_pos)
        ax_dm.set_yticklabels(jt_sorted["team"], fontsize=8)
        ax_dm.invert_yaxis()
        ax_dm.set_xlim(0.5, 20.5)
        ax_dm.margins(y=0.03)
        ax_dm.set_xlabel("League position (1 = best)", fontsize=9)
        ax_dm.set_title(f"Actual vs deserved — {jt_season}/{str(jt_season+1)[-2:]}", fontsize=10)
        ax_dm.legend(fontsize=7.5, loc="upper right")
        ax_dm.grid(axis="x", alpha=0.15)
        plt.tight_layout()
        st.pyplot(fig_dm)
        plt.close()
    with col_j2:
        st.markdown("**Luck distribution**")
        jt_luck = jt.sort_values("luck", ascending=False).copy()
        fig_jl, ax_jl = plt.subplots(figsize=(6, max(4, len(jt_sorted) * 0.38)))
        colors_jl = ["#e74c3c" if v > 0 else "#2ecc71" for v in jt_luck["luck"]]
        ax_jl.barh(jt_luck["team"][::-1], jt_luck["luck"][::-1], color=colors_jl[::-1], edgecolor="white")
        ax_jl.axvline(0, color="grey", linewidth=0.8, linestyle="--")
        ax_jl.set_xlabel("Luck = Actual Pts − xPts", fontsize=9)
        ax_jl.set_title(f"Luck distribution {jt_season}/{str(jt_season+1)[-2:]}", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig_jl)
        plt.close()

    # ── Pre-season prediction accuracy ───────────────────────────────
    st.divider()
    st.markdown("#### Pre-season prediction accuracy")
    st.caption(
        "If you ranked all clubs by squad value before the season: how many predicted-bottom-3 "
        "clubs were actually relegated? How many predicted-top-6 actually made Europe?"
    )
    if not preseas_acc.empty:
        fig_ps, axes_ps = plt.subplots(1, 2, figsize=(12, 4))
        ps_labels = [f"{s}/{str(s+1)[-2:]}" for s in preseas_acc["season"]]
        for ax, col, title, max_val in [
            (axes_ps[0], "bottom3_hits", "Bottom 3 correct (max 3)", 3),
            (axes_ps[1], "top6_hits",    "Top 6 correct (max 6)",    6),
        ]:
            colors_ps = ["#2ecc71" if v >= max_val*0.67 else "#f39c12" if v >= 1 else "#e74c3c"
                         for v in preseas_acc[col]]
            bars_ps = ax.bar(ps_labels, preseas_acc[col], color=colors_ps, edgecolor="white", alpha=0.88)
            ax.bar_label(bars_ps, fmt="%d", padding=2, fontsize=9)
            ax.set_ylim(0, max_val+0.5)
            ax.set_ylabel("Teams correctly predicted", fontsize=9)
            ax.set_title(title, fontsize=10)
            ax.set_xticklabels(ps_labels, rotation=45, ha="right", fontsize=8)
            ax.axhline(max_val*0.67, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
        plt.suptitle("How well does squad value predict relegation and Europe?", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig_ps)
        plt.close()

    st.markdown("**Why the top is easier to predict than the bottom**")
    st.caption(
        "Average squad value by budget rank, across all 8 seasons, with the season-to-season range "
        "shaded. The gap between rank 1 and rank 6 is enormous — those clubs are financially miles "
        "apart, which is why the top-6 prediction lands so often. Down at ranks 15-20, the values "
        "bunch together into a near-flat line: several clubs are separated by pocket change in "
        "transfer-budget terms, so who actually goes down is much closer to a coin flip."
    )
    _vr = overperf.dropna(subset=["value_rank"]).copy()
    _vr["value_rank"] = _vr["value_rank"].astype(int)
    _vr_agg = (
        _vr.groupby("value_rank")["squad_value_m"]
        .agg(mean_value="mean", min_value="min", max_value="max")
        .reset_index()
        .sort_values("value_rank")
    )
    if not _vr_agg.empty:
        _vr_map = _vr_agg.set_index("value_rank")["mean_value"]
        gap_top    = _vr_map.get(1, np.nan) - _vr_map.get(6, np.nan)
        gap_bottom = _vr_map.get(15, np.nan) - _vr_map.get(20, np.nan)

        c_vr1, c_vr2 = st.columns(2)
        if not np.isnan(gap_top):
            c_vr1.metric("Value gap: rank 1 → rank 6", f"€{gap_top:.0f}m")
        if not np.isnan(gap_bottom):
            c_vr2.metric("Value gap: rank 15 → rank 20", f"€{gap_bottom:.0f}m")

        fig_vr, ax_vr = plt.subplots(figsize=(10, 5))
        ax_vr.axvspan(0.5, 6.5, color="#2ecc71", alpha=0.06, zorder=0)
        ax_vr.axvspan(17.5, 20.5, color="#e74c3c", alpha=0.06, zorder=0)
        ax_vr.fill_between(_vr_agg["value_rank"], _vr_agg["min_value"], _vr_agg["max_value"],
                            alpha=0.15, color="#3498db", label="Range across seasons")
        ax_vr.plot(_vr_agg["value_rank"], _vr_agg["mean_value"], marker="o",
                   color="#2980b9", lw=2.2, label="Average squad value")
        _ymax = _vr_agg["max_value"].max()
        ax_vr.text(3.5, _ymax * 0.97, "Top 6", fontsize=8, color="#27ae60",
                   ha="center", va="top", style="italic", alpha=0.8)
        ax_vr.text(19, _ymax * 0.97, "Bottom 3", fontsize=8, color="#c0392b",
                   ha="center", va="top", style="italic", alpha=0.8)
        ax_vr.set_xlabel("Squad value rank (1 = most expensive)", fontsize=9)
        ax_vr.set_ylabel("Squad value (€m)", fontsize=9)
        ax_vr.set_title("Squad value drops fast at the top, flattens at the bottom", fontsize=10)
        ax_vr.set_xticks(range(1, 21))
        ax_vr.legend(fontsize=8)
        ax_vr.grid(alpha=0.15)
        plt.tight_layout()
        st.pyplot(fig_vr)
        plt.close()

    if squad_data.empty:
        st.warning(
            "No squad stats yet. Run:  \n"
            "```\ncd D:\\ManagerSacking\npython collect_squad_stats.py\n```\n"
            "Takes ~10 minutes (160 Transfermarkt pages). "
            "The file saves checkpoints so you can resume if interrupted."
        )
    else:
        st.divider()
        st.markdown("#### Youth Pipeline")
        st.caption(
            "Players who were aged ≤23 at a club at some point in our dataset "
            "AND still at that same club once they turned 24+."
        )

        if not squad_data.empty:
            _pc = (
                squad_data.dropna(subset=["age"])
                .groupby(["player_id", "team"])
                .agg(
                    min_age      = ("age", "min"),
                    max_age      = ("age", "max"),
                    seasons_held = ("season", "nunique"),
                    player_name  = ("player_name", "first"),
                )
                .reset_index()
            )

            _grads = _pc[
                (_pc["min_age"] <= 23) &
                (_pc["max_age"] >= 24) &
                (_pc["seasons_held"] >= 2)
            ].copy()

            _team_seasons = squad_data.groupby("team")["season"].nunique().reset_index(name="n_seasons")

            _pipe = (
                _grads.groupby("team")
                .agg(graduates=("player_id", "count"))
                .reset_index()
            )
            _pipe = _pipe.merge(_team_seasons, on="team", how="left")
            _pipe = _pipe[_pipe["n_seasons"] >= 3].copy()  # need enough seasons for "retained to 24+" to even be measurable
            _pipe["graduates_per_season"] = _pipe["graduates"] / _pipe["n_seasons"]
            _pipe = _pipe.sort_values("graduates_per_season", ascending=False)

            fig, ax = plt.subplots(figsize=(12, 5))
            _bar_cols = [
                "#2ecc71" if t == selected_team else "#3498db"
                for t in _pipe["team"]
            ]
            bars = ax.bar(_pipe["team"], _pipe["graduates_per_season"],
                          color=_bar_cols, edgecolor="white", alpha=0.87)
            for bar, row in zip(bars, _pipe.itertuples()):
                ax.text(bar.get_x() + bar.get_width() / 2, row.graduates_per_season + 0.02,
                        f"{row.graduates_per_season:.2f}\n({row.graduates}/{row.n_seasons}s)",
                        ha="center", va="bottom", fontsize=6.5)
            ax.set_ylabel("Players developed from ≤23 → 24+, per season in the dataset", fontsize=9)
            ax.set_title("Youth pipeline (2018–2025)", fontsize=10)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
            ax.grid(axis="y", alpha=0.18)
            if selected_team in _pipe["team"].values:
                ax.annotate(
                    f"← {selected_team}",
                    xy=(list(_pipe["team"]).index(selected_team),
                        _pipe[_pipe.team == selected_team]["graduates_per_season"].values[0]),
                    fontsize=8, color="#1e8449", xytext=(5, 5),
                    textcoords="offset points",
                )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.divider()
        st.markdown("#### Development Output")
        st.caption(
            "For each player aged ≤23 with recorded minutes at a club, "
            "we measure how much their market value changed the following season. "
            "**Value growth per 90 min** reveals which clubs actually improve players — "
            "not just play them."
        )

        # Next-season value lookup built from squad_data itself
        _pv_all  = squad_data.dropna(subset=["value_m"])
        _pv_next = (
            _pv_all.groupby(["player_id", "season"])["value_m"]
            .max()
            .reset_index()
            .rename(columns={"value_m": "next_val"})
        )
        _pv_next["prev_season"] = _pv_next["season"] - 1

        _u23 = squad_data[
            squad_data["age"].notna() &
            (squad_data["age"] <= 23) &
            squad_data["minutes"].notna() &
            (squad_data["minutes"] > 0) &
            squad_data["value_m"].notna()
        ].copy()

        _u23 = _u23.merge(
            _pv_next[["player_id", "prev_season", "next_val"]].rename(
                columns={"prev_season": "season"}
            ),
            on=["player_id", "season"],
            how="inner",
        )

        if not _u23.empty:
            _u23["growth"]    = _u23["next_val"] - _u23["value_m"]
            _u23["growth_p90"] = _u23["growth"] / (_u23["minutes"] / 90)
            _u23 = _u23[_u23["growth_p90"].abs() <= 8]  # drop extreme outliers

            _dev_club = (
                _u23.groupby("team")
                .agg(
                    avg_g90      = ("growth_p90", "mean"),
                    n_players    = ("player_id",  "nunique"),
                    total_mins   = ("minutes",    "sum"),
                    total_growth = ("growth",     "sum"),
                )
                .reset_index()
                .sort_values("avg_g90", ascending=False)
            )

            c_d1, c_d2 = st.columns(2)

            with c_d1:
                st.markdown("**Development efficiency — €m value growth per 90 min (≤23 players)**")
                st.caption("Averaged across all seasons. Which clubs convert youth minutes into market value?")
                _cols_de = ["#e74c3c" if t == selected_team else "#2ecc71"
                            for t in _dev_club["team"]]
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.barh(_dev_club["team"][::-1], _dev_club["avg_g90"][::-1],
                        color=_cols_de[::-1], edgecolor="white", alpha=0.87)
                ax.axvline(0, color="black", lw=0.8, alpha=0.4)
                ax.set_xlabel("Avg €m growth per 90 min (≤23 players)", fontsize=9)
                ax.set_title("Who extracts the most development value from youth minutes?", fontsize=9)
                ax.tick_params(axis="y", labelsize=7)
                ax.grid(axis="x", alpha=0.15)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with c_d2:
                st.markdown("**Youth minutes vs total value growth (scatter)**")
                st.caption("x = total youth (≤23) minutes across all seasons  ·  y = total value growth of those players")
                _other_d = _dev_club[_dev_club["team"] != selected_team]
                _sel_d   = _dev_club[_dev_club["team"] == selected_team]
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.scatter(_other_d["total_mins"], _other_d["total_growth"],
                           c="#3498db", alpha=0.7, s=60, edgecolors="white", zorder=3)
                if not _sel_d.empty:
                    ax.scatter(_sel_d["total_mins"], _sel_d["total_growth"],
                               c="#e74c3c", alpha=0.95, s=110, edgecolors="white",
                               zorder=5, label=selected_team)
                for _, dr in _dev_club.iterrows():
                    ax.annotate(
                        dr["team"][:3].upper(),
                        (dr["total_mins"], dr["total_growth"]),
                        fontsize=6.5, xytext=(3, 3), textcoords="offset points",
                        color="#2c3e50", alpha=0.85, zorder=4,
                    )
                ax.axhline(0, color="grey", lw=0.8, ls="--", alpha=0.4)
                ax.set_xlabel("Total youth (≤23) minutes — all seasons", fontsize=9)
                ax.set_ylabel("Total value growth of those players (€m)", fontsize=9)
                ax.set_title("Youth investment vs development return", fontsize=9)
                if not _sel_d.empty:
                    ax.legend(fontsize=8)
                ax.grid(alpha=0.1)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            st.divider()
            st.markdown('#### Youth "Grade-Out" Rate')
            st.caption(
                "Of all players who arrived at a club aged ≤23 and stayed 2+ seasons, what % eventually "
                "broke into meaningful minutes (1,000+ in a single season) while still there?"
            )
            _go = squad_data.dropna(subset=["age"]).copy()
            _go_first = _go.sort_values("season").groupby(["player_id", "team"]).first().reset_index()
            _go_first = _go_first[_go_first["age"] <= 23][["player_id", "team"]]
            _go_seasons = _go.groupby(["player_id", "team"])["season"].nunique().reset_index(name="n_seasons")
            _go_eligible = _go_first.merge(_go_seasons, on=["player_id", "team"])
            _go_eligible = _go_eligible[_go_eligible["n_seasons"] >= 2]

            _go_max_mins = _go.groupby(["player_id", "team"])["minutes"].max().reset_index(name="max_minutes")
            _go_eligible = _go_eligible.merge(_go_max_mins, on=["player_id", "team"])
            _go_eligible["graded_out"] = _go_eligible["max_minutes"] >= 1000

            _go_rate = (
                _go_eligible.groupby("team")
                .agg(n_eligible=("graded_out", "count"), n_graded=("graded_out", "sum"))
                .reset_index()
            )
            _go_rate = _go_rate[_go_rate["n_eligible"] >= 5]
            _go_rate["rate"] = _go_rate["n_graded"] / _go_rate["n_eligible"] * 100
            _go_rate = _go_rate.sort_values("rate", ascending=False)

            if not _go_rate.empty:
                _cols_go = ["#e74c3c" if t == selected_team else "#2ecc71" for t in _go_rate["team"]]
                fig, ax = plt.subplots(figsize=(9, max(4, len(_go_rate) * 0.35)))
                ax.barh(_go_rate["team"][::-1], _go_rate["rate"][::-1],
                        color=_cols_go[::-1], edgecolor="white", alpha=0.88)
                ax.set_xlabel("% of eligible youth players who reached 1,000+ minutes in a season", fontsize=9)
                ax.set_title("Youth grade-out rate (clubs with 5+ eligible players)", fontsize=10)
                ax.grid(axis="x", alpha=0.15)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.info("Not enough qualifying players to compute this across clubs.")

            # ── Does developing players actually pay off economically? ──
            st.divider()
            st.markdown("**Does playing youth minutes actually pay off — in money or results?**")
            st.caption(
                "Per club, career-long: **share of total squad minutes given to ≤23 players**, averaged "
                "across its whole history in the dataset, plotted against two separate payoffs — **total "
                "career transfer sale revenue** (the financial case, from actual Transfermarkt fees) and "
                "**career-average manager skill** (the results case), every club with at least 4 seasons "
                "of data."
            )
            _all_mins_career = squad_data[squad_data["minutes"].notna()].groupby(
                "team"
            )["minutes"].sum().reset_index(name="total_minutes")
            _youth_mins_career = squad_data[
                squad_data["age"].notna() & (squad_data["age"] <= 23) & squad_data["minutes"].notna()
            ].groupby("team")["minutes"].sum().reset_index(name="youth_minutes")
            _n_seasons_dev = squad_data[squad_data["minutes"].notna()].groupby("team")["season"].nunique().reset_index(name="n_seasons")
            _dev_by_club = _all_mins_career.merge(_youth_mins_career, on="team", how="left")
            _dev_by_club["youth_minutes"] = _dev_by_club["youth_minutes"].fillna(0)
            _dev_by_club["youth_share"] = _dev_by_club["youth_minutes"] / _dev_by_club["total_minutes"] * 100
            _dev_by_club = _dev_by_club.merge(_n_seasons_dev, on="team")

            _tf_known_eco = transfer_fees.dropna(subset=["fee_m"])
            _sale_rev_career = _tf_known_eco[_tf_known_eco["direction"] == "departure"].groupby(
                "team"
            )["fee_m"].sum().reset_index(name="sale_revenue")

            _skill_career = overperf.groupby("team")["manager_skill_xpts"].mean().reset_index(name="career_skill")
            _n_seasons_skill = overperf.groupby("team").size().reset_index(name="n_overperf_seasons")
            _skill_career = _skill_career.merge(_n_seasons_skill, on="team")

            _econ = _dev_by_club.merge(_sale_rev_career, on="team", how="inner")
            _econ = _econ.merge(_skill_career, on="team", how="inner")
            _econ = _econ[(_econ["n_seasons"] >= 4) & (_econ["n_overperf_seasons"] >= 4)]

            if len(_econ) >= 10:
                c_e1, c_e2 = st.columns(2)
                with c_e1:
                    _r_eg, _p_eg = stats.pearsonr(_econ["youth_share"], _econ["sale_revenue"])
                    fig, ax = plt.subplots(figsize=(6, 5))
                    ax.scatter(_econ["youth_share"], _econ["sale_revenue"], color="#8e44ad",
                               alpha=0.6, s=50, edgecolors="white", zorder=3)
                    _sl, _in = np.polyfit(_econ["youth_share"], _econ["sale_revenue"], 1)
                    _xr = np.linspace(_econ["youth_share"].min(), _econ["youth_share"].max(), 50)
                    ax.plot(_xr, _sl*_xr+_in, color="grey", lw=1.5, ls="--", alpha=0.7)
                    ax.axhline(0, color="black", lw=0.6, alpha=0.3)
                    ax.set_xlabel("Career share of squad minutes given to ≤23 players (%)", fontsize=9)
                    ax.set_ylabel("Total career sale revenue (€m)", fontsize=9)
                    ax.set_title(f"Youth minutes vs. money made — career-long  (r={_r_eg:.2f}, p={_p_eg:.3f})", fontsize=9)
                    ax.grid(alpha=0.15)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                with c_e2:
                    _r_es, _p_es = stats.pearsonr(_econ["youth_share"], _econ["career_skill"])
                    fig, ax = plt.subplots(figsize=(6, 5))
                    ax.scatter(_econ["youth_share"], _econ["career_skill"], color="#2980b9",
                               alpha=0.6, s=50, edgecolors="white", zorder=3)
                    _sl2, _in2 = np.polyfit(_econ["youth_share"], _econ["career_skill"], 1)
                    _xr2 = np.linspace(_econ["youth_share"].min(), _econ["youth_share"].max(), 50)
                    ax.plot(_xr2, _sl2*_xr2+_in2, color="grey", lw=1.5, ls="--", alpha=0.7)
                    ax.axhline(0, color="black", lw=0.6, alpha=0.3)
                    ax.set_xlabel("Career share of squad minutes given to ≤23 players (%)", fontsize=9)
                    ax.set_ylabel("Career-average manager skill (xPts vs. budget)", fontsize=9)
                    ax.set_title(f"Youth minutes vs. results — career-long  (r={_r_es:.2f}, p={_p_es:.3f})", fontsize=9)
                    ax.grid(alpha=0.15)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                _sig_eg = "a real relationship" if _p_eg < 0.05 else "not statistically significant"
                _sig_es = "a real relationship" if _p_es < 0.05 else "not statistically significant"
                st.caption(
                    f"**Money: r={_r_eg:.2f}, p={_p_eg:.3f}** against career sale revenue ({_sig_eg}, "
                    f"n={len(_econ)} clubs). **Results: r={_r_es:.2f}, p={_p_es:.3f}** against career manager "
                    f"skill ({_sig_es})."
                )
            else:
                st.info("Not enough overlapping data to test this relationship yet.")

            # ── Is growing your own actually worth it vs buying? ──
            st.divider()
            st.markdown("**How long does growing your own actually take?**")
            st.caption(
                "Of every player who broke through to a 1,000+ minute season, how many years after their "
                "first appearance did it happen?"
            )

            _tf_known = transfer_fees.dropna(subset=["fee_m"])
            _arr = _tf_known[_tf_known["direction"] == "arrival"].groupby(["team", "season"])["fee_m"].sum().reset_index(name="spend_in")
            _dep = _tf_known[_tf_known["direction"] == "departure"].groupby(["team", "season"])["fee_m"].sum().reset_index(name="spend_out")
            _ns = _arr.merge(_dep, on=["team", "season"], how="outer").fillna(0)
            _ns["net_spend_real"] = _ns["spend_in"] - _ns["spend_out"]
            _ns = _ns.merge(overperf[["team", "season", "manager_skill_xpts", "squad_value_m"]], on=["team", "season"], how="inner")

            if len(_ns) >= 15:
                _r_ns, _p_ns = stats.pearsonr(_ns["net_spend_real"], _ns["manager_skill_xpts"])
                _ns["bucket"] = pd.qcut(_ns["net_spend_real"], 3, labels=["Net sellers / grow-your-own", "Middle", "Heavy net buyers"])
                _bucket_agg = _ns.groupby("bucket")["manager_skill_xpts"].agg(mean_skill="mean", n="count").reset_index()

                # ── How long does "growing your own" actually take? ──
                _go2 = squad_data.dropna(subset=["age"]).copy()
                _first2 = _go2.sort_values("season").groupby(["player_id", "team"]).first().reset_index()
                _first2 = _first2[_first2["age"] <= 23][["player_id", "team", "season"]].rename(columns={"season": "first_season"})
                _seasons2 = _go2.groupby(["player_id", "team"])["season"].nunique().reset_index(name="n_seasons")
                _elig2 = _first2.merge(_seasons2, on=["player_id", "team"])
                _elig2 = _elig2[_elig2["n_seasons"] >= 2]
                _crossed2 = _go2[_go2["minutes"] >= 1000].groupby(["player_id", "team"])["season"].min().reset_index(name="graded_season")
                _elig2 = _elig2.merge(_crossed2, on=["player_id", "team"], how="left")
                _elig2["years_to_grade"] = _elig2["graded_season"] - _elig2["first_season"]
                _graded2 = _elig2.dropna(subset=["years_to_grade"])

                if not _graded2.empty:
                    _yrs_counts = _graded2["years_to_grade"].value_counts().sort_index()
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.bar(_yrs_counts.index.astype(int).astype(str), _yrs_counts.values,
                           color="#2980b9", edgecolor="white", alpha=0.88)
                    ax.set_xlabel("Years from first appearance to 1,000+ min season", fontsize=9)
                    ax.set_ylabel("Number of players", fontsize=9)
                    ax.set_title(f"How long does 'growing your own' actually take?  (n={len(_graded2)})", fontsize=9)
                    ax.grid(axis="y", alpha=0.15)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                # ── Actual trading profit: developed/free vs. bought-and-resold ──
                st.markdown("**The direct financial test: actual profit from reselling players**")
                st.caption(
                    "For every player who arrived and later left the same club (real fees on both ends), "
                    "profit = sale fee minus purchase fee (arrivals with no fee — free, loan, or academy — "
                    "have a €0 cost basis)."
                )
                _arr_tf = transfer_fees[transfer_fees["direction"] == "arrival"][
                    ["player_id", "team", "season", "age", "fee_type", "fee_m"]
                ].rename(columns={"season": "arrival_season", "age": "arrival_age",
                                   "fee_type": "arrival_fee_type", "fee_m": "arrival_fee"})
                _arr_tf["player_id"] = _arr_tf["player_id"].astype(str)
                # Only permanent/free departures are a real exit — a loan departure is temporary,
                # not the club losing the player (see caveat above).
                _dep_tf = transfer_fees[
                    (transfer_fees["direction"] == "departure") &
                    (transfer_fees["fee_type"].isin(["permanent", "free"]))
                ][["player_id", "team", "season", "fee_m"]].rename(columns={"season": "departure_season", "fee_m": "departure_fee"})
                _dep_tf["player_id"] = _dep_tf["player_id"].astype(str)

                _trade = _arr_tf.merge(_dep_tf, on=["player_id", "team"], how="inner")
                _trade = _trade[_trade["departure_season"] > _trade["arrival_season"]]
                _trade = _trade.sort_values("departure_season").groupby(
                    ["player_id", "team", "arrival_season"]
                ).first().reset_index()
                _trade = _trade.dropna(subset=["departure_fee"])
                _trade["realized"] = True

                # A loan-tagged arrival only counts as a genuine "returning loanee" if this
                # player was already at this same club in an earlier season — otherwise it's
                # a real loan-in from another club, not the club's own player coming home.
                _first_season_at_team = squad_data.groupby(["player_id", "team"])["season"].min().to_dict()

                def _is_returning_loan(row):
                    _first = _first_season_at_team.get((row["player_id"], row["team"]))
                    return _first is not None and _first < row["arrival_season"]

                def _classify_trade(row):
                    if row["arrival_fee_type"] in ("free", "unknown") and row["arrival_age"] <= 21:
                        return "Developed/free\n(young, no fee in)"
                    elif row["arrival_fee_type"] == "permanent" and row["arrival_age"] <= 23:
                        return "Bought young (≤23),\nresold"
                    elif row["arrival_fee_type"] == "permanent":
                        return "Bought established (24+),\nresold"
                    elif row["arrival_fee_type"] == "loan" and _is_returning_loan(row):
                        return "Returning loanee,\nlater resold"
                    else:
                        return "Other"

                def _is_genuine_loan_in(row):
                    # A loan-tagged arrival that ISN'T a returning loanee is a real loan-in
                    # from another club — a fundamentally different, temporary arrangement,
                    # not an "acquisition" in the sense this chart is measuring. Excluded
                    # entirely below, not folded into "Other."
                    return row["arrival_fee_type"] == "loan" and not _is_returning_loan(row)

                if not _trade.empty:
                    # ── Long-term extension: players never (yet) permanently sold ──
                    # A player bought young who's still at the club isn't a failure just because
                    # there's no resale to measure. If his current market value is well above what
                    # was paid, that's real value the club is holding — just unrealized, not a "loss."
                    _sold_keys = set(zip(_trade["player_id"], _trade["team"], _trade["arrival_season"]))
                    _arr_tf["_key"] = list(zip(_arr_tf["player_id"], _arr_tf["team"], _arr_tf["arrival_season"]))
                    _unsold = _arr_tf[~_arr_tf["_key"].isin(_sold_keys)].drop(columns=["_key"]).copy()

                    _latest_season = squad_data["season"].max()
                    _current_val = squad_data.dropna(subset=["value_m"])
                    _current_val = _current_val[_current_val["season"] == _latest_season][["player_id", "team", "value_m"]].copy()
                    _current_val["player_id"] = _current_val["player_id"].astype(str)

                    _still_held = _unsold.merge(_current_val, on=["player_id", "team"], how="inner")
                    _still_held = _still_held.rename(columns={"value_m": "departure_fee"})
                    _still_held["departure_season"] = _latest_season + 1
                    _still_held["realized"] = False

                    _trade_lt = pd.concat([_trade, _still_held], ignore_index=True)
                    _trade_lt["category"] = _trade_lt.apply(_classify_trade, axis=1)
                    _trade_lt["cost_basis"] = _trade_lt["arrival_fee"].where(_trade_lt["arrival_fee_type"] == "permanent", 0).fillna(0)
                    _trade_lt["profit"] = _trade_lt["departure_fee"] - _trade_lt["cost_basis"]
                    _trade_lt["years_held"] = (_trade_lt["departure_season"] - _trade_lt["arrival_season"]).clip(lower=1)
                    _trade_lt["profit_per_year"] = _trade_lt["profit"] / _trade_lt["years_held"]

                    # keep the original realized-only view too (used by the buy/sell-by-age
                    # section further down, which is about a real completed transaction's timing)
                    _trade["category"] = _trade.apply(_classify_trade, axis=1)
                    _trade["cost_basis"] = _trade["arrival_fee"].where(_trade["arrival_fee_type"] == "permanent", 0).fillna(0)
                    _trade["profit"] = _trade["departure_fee"] - _trade["cost_basis"]
                    _trade["years_held"] = (_trade["departure_season"] - _trade["arrival_season"]).clip(lower=1)
                    _trade["profit_per_year"] = _trade["profit"] / _trade["years_held"]

                    _trade_lt_for_chart = _trade_lt[~_trade_lt.apply(_is_genuine_loan_in, axis=1)]
                    _trade_summary = _trade_lt_for_chart.groupby("category").agg(
                        n=("profit", "count"), avg_profit=("profit", "mean"),
                        avg_profit_yr=("profit_per_year", "mean"), avg_years=("years_held", "mean"),
                        pct_realized=("realized", "mean"),
                    ).reset_index().sort_values("avg_profit_yr", ascending=False)

                    _cols_tp = ["#2ecc71" if v >= 0 else "#e74c3c" for v in _trade_summary["avg_profit_yr"]]
                    fig, ax = plt.subplots(figsize=(11, 5))
                    bars = ax.barh(_trade_summary["category"], _trade_summary["avg_profit_yr"],
                                   color=_cols_tp, edgecolor="white", alpha=0.88)
                    ax.axvline(0, color="black", lw=0.8, alpha=0.5)
                    _rng_tp = max(abs(_trade_summary["avg_profit_yr"].max()), abs(_trade_summary["avg_profit_yr"].min()), 0.1)
                    ax.set_xlim(_trade_summary["avg_profit_yr"].min() - _rng_tp * 1.6, _trade_summary["avg_profit_yr"].max() + _rng_tp * 1.6)
                    for bar, row in zip(bars, _trade_summary.itertuples()):
                        ax.text(row.avg_profit_yr + (_rng_tp*0.08 if row.avg_profit_yr >= 0 else -_rng_tp*0.08), bar.get_y()+bar.get_height()/2,
                                f"€{row.avg_profit_yr:+.1f}m/yr (n={row.n}, {row.avg_years:.1f}yrs held, {row.pct_realized*100:.0f}% sold)", va="center",
                                ha="left" if row.avg_profit_yr >= 0 else "right", fontsize=8)
                    ax.set_xlabel("Avg value created per player, per year held (€m)", fontsize=9)
                    ax.set_title("Long-term value by acquisition type — per year held", fontsize=9)
                    ax.tick_params(axis="y", labelsize=8)
                    ax.grid(axis="x", alpha=0.15)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                    _dev_row = _trade_summary[_trade_summary["category"].str.startswith("Developed")]
                    _est_row = _trade_summary[_trade_summary["category"].str.startswith("Bought established")]
                    if not _dev_row.empty and not _est_row.empty:
                        st.caption(
                            f"**Growing your own wins, even per year held.** A developed/free acquisition creates "
                            f"an average of **€{_dev_row['avg_profit_yr'].iloc[0]:+.1f}m a year** it's held "
                            f"(n={int(_dev_row['n'].iloc[0])}, {_dev_row['pct_realized'].iloc[0]*100:.0f}% actually sold so far). "
                            f"Buying an established player and reselling him later loses an average of "
                            f"**€{_est_row['avg_profit_yr'].iloc[0]:+.1f}m a year** (n={int(_est_row['n'].iloc[0])}, "
                            f"{_est_row['pct_realized'].iloc[0]*100:.0f}% sold). "
                            "Zoom in to a single player, though, and the financial case stops being ambiguous: "
                            "developing your own is reliably profitable, and buying an established star and "
                            "reselling him later reliably loses money."
                        )
