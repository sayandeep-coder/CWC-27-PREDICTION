from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "cleaned"
OUT = ROOT / "outputs"
CHARTS = ROOT / "charts"
MODELS = ROOT / "models"
REPORT = ROOT / "report"
HOSTS = {"South Africa", "Zimbabwe", "Namibia"}
ALIASES = {
    "U.S.A.": "United States of America", "USA": "United States of America",
    "United States": "United States of America", "UAE": "United Arab Emirates",
    "West Indies Cricket Board President's XI": "West Indies",
}
WINDOWS = (5, 10, 20, 50)


def clean_text(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().replace(ALIASES)


def load_and_clean() -> dict[str, pd.DataFrame]:
    paths = {
        "world_cup": RAW / "world_cup_matches.csv",
        "odi": RAW / "odi_history.csv",
        "recent": RAW / "recent_100_confirmed.csv",
        "host": RAW / "host_performance.csv",
    }
    data = {k: pd.read_csv(v) for k, v in paths.items()}
    team_cols = {
        "world_cup": ["team1", "team2", "winner"],
        "odi": ["team1", "team2", "winner", "toss_winner", "innings1_team", "innings2_team"],
        "recent": ["confirmed_team", "opponent", "winner"],
        "host": ["team", "opponent", "winner"],
    }
    audits = {}
    for name, df in data.items():
        original = len(df)
        df.columns = df.columns.str.strip().str.lower()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for col in team_cols[name]:
            if col in df:
                df[col] = clean_text(df[col])
        for col in ["venue", "city", "country", "host_country"]:
            if col in df:
                df[col] = clean_text(df[col])
        dup_subset = [c for c in ["match_id", "date", "team1", "team2", "team", "opponent"] if c in df]
        exact_dupes = int(df.duplicated().sum())
        key_dupes = int(df.duplicated(dup_subset).sum()) if dup_subset else exact_dupes
        df = df.drop_duplicates().copy()
        audits[name] = {
            "rows_raw": original, "rows_clean": len(df), "exact_duplicates_removed": exact_dupes,
            "duplicate_keys_flagged": key_dupes, "date_min": str(df.date.min().date()),
            "date_max": str(df.date.max().date()), "invalid_dates": int(df.date.isna().sum()),
            "missing_fraction": {c: round(float(v), 4) for c, v in df.isna().mean().items() if v > 0},
        }
        data[name] = df.sort_values("date").reset_index(drop=True)
        data[name].to_csv(CLEAN / f"{name}_clean.csv", index=False, date_format="%Y-%m-%d")
    (CLEAN / "data_quality_report.json").write_text(json.dumps(audits, indent=2), encoding="utf-8")
    return data


@dataclass
class TeamState:
    results: list[float] = field(default_factory=list)
    dates: list[pd.Timestamp] = field(default_factory=list)
    runs_for: list[float] = field(default_factory=list)
    runs_against: list[float] = field(default_factory=list)
    wickets_lost: list[float] = field(default_factory=list)
    wickets_taken: list[float] = field(default_factory=list)
    bat_first: list[float] = field(default_factory=list)
    chase: list[float] = field(default_factory=list)
    opponent_strength: list[float] = field(default_factory=list)
    host_results: list[float] = field(default_factory=list)

    def rate(self, n=None):
        x = self.results[-n:] if n else self.results
        return float(np.mean(x)) if x else 0.5

    def ewm(self, alpha=.18):
        value = .5
        for x in self.results:
            value = alpha * x + (1 - alpha) * value
        return value

    @staticmethod
    def avg(values, n=20, default=0.0):
        return float(np.mean(values[-n:])) if values else default


def wc_rate(wc: pd.DataFrame, team: str, date: pd.Timestamp) -> float:
    x = wc[(wc.date < date) & ((wc.team1 == team) | (wc.team2 == team))]
    decisive = x[x.result_type.isin(["win", "awarded"])]
    return float((decisive.winner == team).mean()) if len(decisive) else .5


BASE_METRICS = ["overall", "form5", "form10", "form20", "form50", "ewm", "wc", "h2h",
                "host", "venue", "bat_first", "chase", "runs_for", "runs_against",
                "wickets_lost", "wickets_taken", "opp_strength", "rest_days"]


def snapshot(state: TeamState, wc: pd.DataFrame, team: str, opponent: str, date: pd.Timestamp,
             h2h: dict, venue_results: dict, venue: str) -> dict[str, float]:
    key = tuple(sorted((team, opponent)))
    h = h2h.get((key, team), [])
    rest = min((date - state.dates[-1]).days, 60) if state.dates else 30
    return {
        "overall": state.rate(), "form5": state.rate(5), "form10": state.rate(10),
        "form20": state.rate(20), "form50": state.rate(50), "ewm": state.ewm(),
        "wc": wc_rate(wc, team, date), "h2h": float(np.mean(h)) if h else .5,
        "host": TeamState.avg(state.host_results, 30, .5),
        "venue": float(np.mean(venue_results.get((team, venue), []))) if venue_results.get((team, venue)) else .5,
        "bat_first": TeamState.avg(state.bat_first, 30, .5), "chase": TeamState.avg(state.chase, 30, .5),
        "runs_for": TeamState.avg(state.runs_for), "runs_against": TeamState.avg(state.runs_against),
        "wickets_lost": TeamState.avg(state.wickets_lost), "wickets_taken": TeamState.avg(state.wickets_taken),
        "opp_strength": TeamState.avg(state.opponent_strength, 20, .5), "rest_days": float(rest),
    }


def build_features(odi: pd.DataFrame, wc: pd.DataFrame, host: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    states = defaultdict(TeamState)
    h2h, venue_results = defaultdict(list), defaultdict(list)
    host_by_match = host.drop_duplicates("match_id").set_index("match_id")["host_country"].to_dict()
    rows = []
    for _, m in odi.sort_values(["date", "match_id"]).iterrows():
        if pd.isna(m.date) or pd.isna(m.team1) or pd.isna(m.team2):
            continue
        # Stable orientation avoids target/order artefacts while keeping reproducibility.
        flip = int(hashlib.sha1(str(m.match_id).encode()).hexdigest(), 16) % 2
        a, b = (m.team2, m.team1) if flip else (m.team1, m.team2)
        venue = str(m.venue) if pd.notna(m.venue) else "Unknown"
        host_country = host_by_match.get(m.match_id, "")
        sa = snapshot(states[a], wc, a, b, m.date, h2h, venue_results, venue)
        sb = snapshot(states[b], wc, b, a, m.date, h2h, venue_results, venue)
        row = {"match_id": m.match_id, "date": m.date, "team_a": a, "team_b": b,
               "venue": venue, "event_name": m.event_name, "result_type": m.result_type,
               "winner": m.winner}
        for f in BASE_METRICS:
            row[f"diff_{f}"] = sa[f] - sb[f]
        row["host_advantage_diff"] = float(a == host_country) - float(b == host_country)
        row["toss_a"] = float(m.toss_winner == a) if pd.notna(m.toss_winner) else 0.0
        row["toss_bat_a"] = float(m.toss_winner == a and m.toss_decision == "bat") - float(m.toss_winner == b and m.toss_decision == "bat")
        row["target"] = float(m.winner == a) if m.result_type == "win" and pd.notna(m.winner) else np.nan
        rows.append(row)

        decisive = m.result_type == "win" and pd.notna(m.winner)
        if decisive:
            for t, opp in [(m.team1, m.team2), (m.team2, m.team1)]:
                win = float(m.winner == t)
                st = states[t]
                opp_rate = states[opp].rate()
                st.results.append(win); st.dates.append(m.date); st.opponent_strength.append(opp_rate)
                if host_country in HOSTS:
                    st.host_results.append(win)
                venue_results[(t, venue)].append(win)
                key = tuple(sorted((t, opp))); h2h[(key, t)].append(win)
                is_i1 = m.innings1_team == t
                rf = m.innings1_runs if is_i1 else m.innings2_runs
                ra = m.innings2_runs if is_i1 else m.innings1_runs
                wl = m.innings1_wickets if is_i1 else m.innings2_wickets
                wt = m.innings2_wickets if is_i1 else m.innings1_wickets
                if pd.notna(rf): st.runs_for.append(float(rf))
                if pd.notna(ra): st.runs_against.append(float(ra))
                if pd.notna(wl): st.wickets_lost.append(float(wl))
                if pd.notna(wt): st.wickets_taken.append(float(wt))
                (st.bat_first if is_i1 else st.chase).append(win)
    features = pd.DataFrame(rows)
    features.to_csv(CLEAN / "modeling_features.csv", index=False, date_format="%Y-%m-%d")
    return features, {"states": states, "h2h": h2h, "venue_results": venue_results}


def metrics(y, p, threshold=.5):
    pred = (p >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, p), "log_loss": log_loss(y, p),
        "brier": brier_score_loss(y, p),
    }


def train_models(features: pd.DataFrame):
    df = features.dropna(subset=["target"]).sort_values("date").reset_index(drop=True)
    pre_cols = [c for c in df if c.startswith("diff_")] + ["host_advantage_diff"]
    post_cols = pre_cols + ["toss_a", "toss_bat_a"]
    n = len(df); i1, i2 = int(n * .70), int(n * .85)
    splits = {"train": df.iloc[:i1], "validation": df.iloc[i1:i2], "test": df.iloc[i2:]}
    model_specs = {
        "logistic_regression": LogisticRegression(C=.5, max_iter=3000, random_state=27),
        "random_forest": RandomForestClassifier(n_estimators=350, min_samples_leaf=12, max_features=.7, class_weight="balanced", n_jobs=-1, random_state=27),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=250, learning_rate=.045, max_leaf_nodes=15, l2_regularization=2.0, random_state=27),
    }
    results, fitted = [], {}
    for mode, cols in [("pre_match", pre_cols), ("post_toss", post_cols)]:
        for name, estimator in model_specs.items():
            pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                             ("scale", StandardScaler() if name == "logistic_regression" else "passthrough"),
                             ("model", clone(estimator))])
            pipe.fit(splits["train"][cols], splits["train"].target)
            fitted[(mode, name)] = (pipe, cols)
            for split in ["validation", "test"]:
                p = pipe.predict_proba(splits[split][cols])[:, 1]
                results.append({"mode": mode, "model": name, "split": split,
                                **metrics(splits[split].target, p)})
    comp = pd.DataFrame(results)
    comp.to_csv(OUT / "model_comparison.csv", index=False)
    val = comp[(comp["mode"] == "pre_match") & (comp.split == "validation")].copy()
    val["selection_score"] = val.log_loss + val.brier
    best_name = val.sort_values("selection_score").iloc[0].model
    best, best_cols = fitted[("pre_match", best_name)]
    post, post_cols = fitted[("post_toss", best_name)]
    joblib.dump({"pipeline": best, "features": best_cols, "mode": "pre_match", "cutoffs": {
        "train_end": str(splits["train"].date.max().date()), "validation_end": str(splits["validation"].date.max().date()),
        "test_end": str(splits["test"].date.max().date())}}, MODELS / "best_pre_match.joblib")
    joblib.dump({"pipeline": post, "features": post_cols, "mode": "post_toss"}, MODELS / "best_post_toss.joblib")
    test = splits["test"].copy(); test["win_probability_team_a"] = best.predict_proba(test[best_cols])[:, 1]
    test[["match_id", "date", "team_a", "team_b", "winner", "win_probability_team_a"]].to_csv(OUT / "match_predictions.csv", index=False)
    # Walk-forward World Cup checks: each edition is predicted using only earlier ODIs.
    backtests = []
    edition_names = {2015: "ICC Cricket World Cup", 2019: "World Cup", 2023: "ICC Cricket World Cup"}
    for year, event_name in edition_names.items():
        edition = df[(df.date.dt.year == year) & df.event_name.eq(event_name)]
        if len(edition) < 2:
            continue
        prior = df[df.date < edition.date.min()]
        bt = Pipeline([("imputer", SimpleImputer(strategy="median")),
                       ("scale", StandardScaler() if best_name == "logistic_regression" else "passthrough"),
                       ("model", clone(model_specs[best_name]))])
        bt.fit(prior[best_cols], prior.target)
        bp = bt.predict_proba(edition[best_cols])[:, 1]
        backtests.append({"edition": year, "matches": len(edition), **metrics(edition.target, bp)})
    pd.DataFrame(backtests).to_csv(OUT / "world_cup_backtests.csv", index=False)
    return best_name, best, best_cols, post, post_cols, splits, comp


def current_team_table(recent: pd.DataFrame, wc: pd.DataFrame, host: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for team, g in recent.groupby("confirmed_team"):
        g = g.sort_values("date")
        decisive = g[g.team_result.isin(["win", "loss"])]
        vals = (decisive.team_result == "win").astype(float).tolist()
        rows.append({"team": team, "matches": len(decisive), "overall": np.mean(vals) if vals else .5,
                     "form5": np.mean(vals[-5:]) if vals else .5, "form10": np.mean(vals[-10:]) if vals else .5,
                     "form20": np.mean(vals[-20:]) if vals else .5, "form50": np.mean(vals[-50:]) if vals else .5,
                     "ewm": pd.Series(vals).ewm(alpha=.18, adjust=False).mean().iloc[-1] if vals else .5,
                     "wc": wc_rate(wc, team, pd.Timestamp("2027-01-01"))})
    table = pd.DataFrame(rows)
    host_rates = host[host.team_result.isin(["win", "loss"])].groupby("team").team_result.apply(lambda x: (x == "win").mean())
    table["host"] = table.team.map(host_rates).fillna(.5)
    for i in range(1, 5):
        table.loc[len(table)] = {"team": f"Qualifier {i}", "matches": 0, "overall": .5, "form5": .5,
                                 "form10": .5, "form20": .5, "form50": .5, "ewm": .5, "wc": .5, "host": .5}
    return table


def pair_features(a: pd.Series, b: pd.Series, cols: list[str]) -> pd.DataFrame:
    d = {}
    for c in cols:
        if c == "host_advantage_diff": d[c] = float(a.name in HOSTS) - float(b.name in HOSTS)
        elif c.startswith("diff_"):
            metric = c[5:]
            av = a.get(metric, .5); bv = b.get(metric, .5)
            # Neutral defaults for statistics unavailable in the team-centric recent file.
            d[c] = float(av if pd.notna(av) else .5) - float(bv if pd.notna(bv) else .5)
        else: d[c] = 0.0
    return pd.DataFrame([d])


def simulate(table: pd.DataFrame, model, cols: list[str], runs=10000, seed=2027):
    rng = np.random.default_rng(seed); teams = table.team.tolist(); index = table.set_index("team")
    probs = {}
    for i, a in enumerate(teams):
        for b in teams[i+1:]:
            p = float(model.predict_proba(pair_features(index.loc[a], index.loc[b], cols))[0, 1])
            probs[(a, b)] = np.clip(p, .05, .95); probs[(b, a)] = 1 - probs[(a, b)]
    semi = defaultdict(int); final = defaultdict(int); champ = defaultdict(int)
    for _ in range(runs):
        points = {t: 0 for t in teams}; wins = {t: 0 for t in teams}
        for i, a in enumerate(teams):
            for b in teams[i+1:]:
                w = a if rng.random() < probs[(a, b)] else b; points[w] += 2; wins[w] += 1
        ranked = sorted(teams, key=lambda t: (points[t], wins[t], rng.random()), reverse=True)
        top = ranked[:4]
        for t in top: semi[t] += 1
        finalists = []
        for a, b in [(top[0], top[3]), (top[1], top[2])]:
            w = a if rng.random() < probs[(a, b)] else b; finalists.append(w); final[w] += 1
        a, b = finalists; w = a if rng.random() < probs[(a, b)] else b; champ[w] += 1
    out = pd.DataFrame({"team": teams, "semi_final_probability": [semi[t]/runs for t in teams],
                        "final_probability": [final[t]/runs for t in teams],
                        "win_probability": [champ[t]/runs for t in teams]}).sort_values("win_probability", ascending=False)
    out.to_csv(OUT / "tournament_probabilities.csv", index=False)
    return out, probs


def save_charts(comp, splits, model, cols, sim, team_table, wc, host):
    plt.style.use("seaborn-v0_8-whitegrid")
    def bar(df, x, y, title, path, color="#155E75"):
        fig, ax = plt.subplots(figsize=(10, 6)); q = df.sort_values(y)
        ax.barh(q[x], q[y], color=color); ax.set_title(title, loc="left", weight="bold"); ax.set_xlabel(y.replace("_", " ").title())
        fig.tight_layout(); fig.savefig(CHARTS / path, dpi=160); plt.close(fig)
    bar(sim, "team", "win_probability", "Provisional 2027 World Cup winning probability", "winning_probability.png")
    q = sim.sort_values("semi_final_probability")
    fig, ax = plt.subplots(figsize=(10, 7)); ax.barh(q.team, q.semi_final_probability, label="Semi-final"); ax.barh(q.team, q.final_probability, label="Final")
    ax.legend(); ax.set_title("Qualification probabilities", loc="left", weight="bold"); fig.tight_layout(); fig.savefig(CHARTS / "qualification_probabilities.png", dpi=160); plt.close(fig)
    bar(team_table[~team_table.team.str.startswith("Qualifier")], "team", "form10", "Recent ODI win rate (last 10 decisive matches)", "recent_win_rate.png")
    titles = wc[wc.stage.str.casefold().eq("final") & (wc.result_type.isin(["win", "awarded"]))].groupby("winner").size().reset_index(name="titles")
    bar(titles, "winner", "titles", "Historical World Cup final wins", "historical_world_cup_wins.png")
    hp = host[host.team_result.isin(["win", "loss"])].groupby("team").team_result.apply(lambda x: (x == "win").mean()).nlargest(15).reset_index(name="host_win_rate")
    bar(hp, "team", "host_win_rate", "ODI win rate in 2027 host countries", "host_country_performance.png")
    test = splits["test"]; p = model.predict_proba(test[cols])[:, 1]
    fig, ax = plt.subplots(figsize=(6, 6)); ConfusionMatrixDisplay(confusion_matrix(test.target, p >= .5)).plot(ax=ax, colorbar=False); ax.set_title("Test confusion matrix"); fig.tight_layout(); fig.savefig(CHARTS / "confusion_matrix.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 6)); CalibrationDisplay.from_predictions(test.target, p, n_bins=8, strategy="quantile", ax=ax); ax.set_title("Test probability calibration"); fig.tight_layout(); fig.savefig(CHARTS / "calibration_curve.png", dpi=160); plt.close(fig)
    q = comp[comp.split == "test"].copy(); q["label"] = q["mode"] + "\n" + q["model"]
    bar(q, "label", "log_loss", "Model comparison on chronological test set", "model_comparison.png", "#7C3AED")
    imp = None
    est = model.named_steps["model"]
    if hasattr(est, "coef_"): imp = pd.DataFrame({"feature": cols, "importance": np.abs(est.coef_[0])})
    elif hasattr(est, "feature_importances_"): imp = pd.DataFrame({"feature": cols, "importance": est.feature_importances_})
    if imp is None:
        from sklearn.inspection import permutation_importance
        r = permutation_importance(model, test[cols], test.target, n_repeats=5, random_state=27, scoring="neg_log_loss")
        imp = pd.DataFrame({"feature": cols, "importance": r.importances_mean})
    imp.sort_values("importance", ascending=False).to_csv(OUT / "feature_importance.csv", index=False)
    bar(imp.nlargest(15, "importance"), "feature", "importance", "Most important predictive factors", "feature_importance.png", "#D97706")


def render_report(best_name, comp, sim, audits):
    test = comp[(comp.model == best_name) & (comp["mode"] == "pre_match") & (comp.split == "test")].iloc[0]
    rows = "".join(f"<tr><td>{html.escape(r.team)}</td><td>{r.semi_final_probability:.1%}</td><td>{r.final_probability:.1%}</td><td>{r.win_probability:.1%}</td></tr>" for _, r in sim.iterrows())
    metric_rows = "".join(f"<tr><th>{k.replace('_',' ').title()}</th><td>{test[k]:.3f}</td></tr>" for k in ["accuracy","precision","recall","f1","roc_auc","log_loss","brier"])
    report = f"""<!doctype html><html><head><meta charset='utf-8'><title>CWC 2027 prediction</title><style>
    body{{font:16px system-ui;max-width:1100px;margin:40px auto;color:#17202a}} h1,h2{{color:#083344}} .note{{background:#fff7ed;padding:14px;border-left:4px solid #d97706}} table{{border-collapse:collapse;width:100%;margin:16px 0}} td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:left}} img{{max-width:100%;margin:14px 0}} .grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}</style></head><body>
    <h1>ICC Men's ODI World Cup 2027 prediction</h1><p>Data cutoff: 2 November 2026 for the match-level ODI history. Selected pre-match model: <b>{best_name.replace('_',' ').title()}</b>.</p>
    <div class='note'><b>Provisional simulation.</b> No verified 2027 fixture schedule was supplied. The simulation uses a 14-team round robin, top-four semi-finals (1 v 4, 2 v 3), then a final. Four unknown teams are represented by neutral Qualifier placeholders. It does not model net run rate or fixture-specific venues.</div>
    <h2>Chronological test metrics</h2><table>{metric_rows}</table><h2>Tournament probabilities</h2><table><tr><th>Team</th><th>Semi-final</th><th>Final</th><th>Champion</th></tr>{rows}</table>
    <div class='grid'><img src='../charts/winning_probability.png'><img src='../charts/qualification_probabilities.png'><img src='../charts/model_comparison.png'><img src='../charts/calibration_curve.png'><img src='../charts/feature_importance.png'><img src='../charts/confusion_matrix.png'></div>
    <h2>Data and leakage controls</h2><p>All rolling, head-to-head, venue, opposition, rest and World Cup features use records strictly earlier than the match. Splits are chronological. Ties and no-results remain in cleaned files but are excluded from binary training. The recent team-centric file is used for current rankings and simulation snapshots rather than being appended to match-level training, avoiding double counting.</p>
    <h2>Limitations</h2><p>Cricsheet match-level coverage begins in 2002 and is not a complete history of every ODI. Recent-team data does not contain full venue, toss, wickets and opponent-strength histories, so missing current snapshot factors are neutral in the provisional simulation. Qualifier identities and the official schedule were not supplied. These are model estimates, not certainties.</p>
    <h2>Reproducibility</h2><p>Run <code>python -m src.pipeline --runs 10000</code>. Replace or append source CSVs using the documented schemas, then rerun when qualifiers or fixtures become available.</p></body></html>"""
    (REPORT / "cwc_2027_report.html").write_text(report, encoding="utf-8")


def main(runs=10000):
    for d in [CLEAN, OUT, CHARTS, MODELS, REPORT]: d.mkdir(parents=True, exist_ok=True)
    data = load_and_clean()
    features, _ = build_features(data["odi"], data["world_cup"], data["host"])
    best_name, model, cols, post, post_cols, splits, comp = train_models(features)
    teams = current_team_table(data["recent"], data["world_cup"], data["host"])
    teams.sort_values(["form20", "overall"], ascending=False).to_csv(OUT / "team_rankings.csv", index=False)
    sim, _ = simulate(teams, model, cols, runs=runs)
    save_charts(comp, splits, model, cols, sim, teams, data["world_cup"], data["host"])
    audits = json.loads((CLEAN / "data_quality_report.json").read_text())
    render_report(best_name, comp, sim, audits)
    summary = {"best_model": best_name, "simulation_runs": runs,
               "test_metrics": comp[(comp.model == best_name) & (comp["mode"] == "pre_match") & (comp.split == "test")].iloc[0].to_dict(),
               "leaders": sim.head(5).to_dict("records")}
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--runs", type=int, default=10000)
    main(parser.parse_args().runs)

