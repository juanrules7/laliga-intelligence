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
# NOTE: this is a deliberately trimmed public build — only Article 1's tab is
# included. Articles 2-6 unlock here as each one is published; see dashboard.py
# in the private dev repo for the full 6-tab version.
tab1, = st.tabs(["Article 1 — Framework"])


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
