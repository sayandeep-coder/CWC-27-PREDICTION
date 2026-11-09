"""Leakage-controlled advanced ODI model and 2027 tournament simulation."""
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
from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.calibration import calibration_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .pipeline import (ALIASES, CHARTS, CLEAN, HOSTS, MODELS, OUT, RAW, REPORT,
                       clean_text, load_and_clean)
from .venue_mapping import build_venue_mapping

RNG_SEED = 2701
FULL_MEMBERS = {"Afghanistan", "Australia", "Bangladesh", "England", "India", "Ireland",
                "New Zealand", "Pakistan", "South Africa", "Sri Lanka", "West Indies", "Zimbabwe"}


def safe_mean(values, default=.5):
    x = [v for v in values if np.isfinite(v)]
    return float(np.mean(x)) if x else float(default)


def weighted_mean(values, n, decay=.86, default=.5):
    x = np.asarray(values[-n:], dtype=float)
    if not len(x): return float(default)
    w = decay ** np.arange(len(x) - 1, -1, -1)
    return float(np.average(x, weights=w))


@dataclass
class State:
    results: list[float] = field(default_factory=list)
    adjusted_results: list[float] = field(default_factory=list)
    opponent_elos: list[float] = field(default_factory=list)
    dates: list[pd.Timestamp] = field(default_factory=list)
    rpwi_z: list[float] = field(default_factory=list)
    rpcw_z: list[float] = field(default_factory=list)
    run_rate_z: list[float] = field(default_factory=list)
    economy_z: list[float] = field(default_factory=list)
    nrr_z: list[float] = field(default_factory=list)
    bat_first: list[float] = field(default_factory=list)
    chase: list[float] = field(default_factory=list)
    home: list[float] = field(default_factory=list)
    away: list[float] = field(default_factory=list)
    neutral: list[float] = field(default_factory=list)


class RunningDistribution:
    def __init__(self): self.values = defaultdict(list)
    def z(self, key, value):
        vals = self.values[key]
        if not np.isfinite(value) or len(vals) < 12: return 0.0
        mu, sd = np.mean(vals), np.std(vals)
        return float((value - mu) / max(sd, 1e-6))
    def update(self, key, value):
        if np.isfinite(value): self.values[key].append(float(value))


def expected(r_a, r_b): return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def classify_surface(venue_history, venue):
    own = venue_history.get(venue, [])
    all_vals = [v for values in venue_history.values() for v in values]
    if len(own) < 3 or len(all_vals) < 30: return "unknown"
    q1, q2 = np.quantile(all_vals, [.33, .67]); v = np.mean(own)
    return "low_scoring" if v <= q1 else ("high_scoring" if v >= q2 else "balanced")


def wc_snapshot(wc, team, date):
    x = wc[(wc.date < date) & ((wc.team1 == team) | (wc.team2 == team)) & wc.result_type.isin(["win", "awarded"])]
    if x.empty: return {"wc_rate": .5, "wc_experience": 0.0}
    age = np.maximum(0, date.year - x.edition_year.astype(int)); w = np.exp(-age / 10.0)
    wins = (x.winner == team).astype(float).to_numpy()
    return {"wc_rate": float(np.average(wins, weights=w)), "wc_experience": float(w.sum())}


FEATURE_NAMES = [
    "elo", "surface_elo", "host_elo", "overall", "weighted_form5", "weighted_form10",
    "weighted_form20", "weighted_form50", "opponent_adjusted_form", "recent_opponent_elo",
    "rpwi_z", "rpcw_z", "run_rate_z", "economy_z", "nrr_z", "bat_first", "chase",
    "home", "away", "neutral", "venue_familiarity", "rest_days", "h2h_bayes",
    "wc_rate", "wc_experience", "matches_played",
]


def team_snapshot(team, opp, date, venue, host_country, host_team, surface, states, elos, surface_elos,
                  host_elos, venue_counts, h2h, wc):
    st = states[team]; pair = tuple(sorted((team, opp))); hh = h2h[(pair, team)]
    rest = min((date - st.dates[-1]).days, 120) if st.dates else 30
    role = "home" if team == host_team else ("away" if opp == host_team else "neutral")
    wc_s = wc_snapshot(wc, team, date)
    return {
        "elo": elos[team], "surface_elo": surface_elos[surface][team],
        "host_elo": host_elos[host_country][team] if host_country else 1500.0,
        "overall": (sum(st.results) + 3) / (len(st.results) + 6),
        "weighted_form5": weighted_mean(st.results, 5), "weighted_form10": weighted_mean(st.results, 10),
        "weighted_form20": weighted_mean(st.results, 20), "weighted_form50": weighted_mean(st.results, 50),
        "opponent_adjusted_form": weighted_mean(st.adjusted_results, 20),
        "recent_opponent_elo": safe_mean(st.opponent_elos[-20:], 1500),
        "rpwi_z": safe_mean(st.rpwi_z[-20:], 0), "rpcw_z": safe_mean(st.rpcw_z[-20:], 0),
        "run_rate_z": safe_mean(st.run_rate_z[-20:], 0), "economy_z": safe_mean(st.economy_z[-20:], 0),
        "nrr_z": safe_mean(st.nrr_z[-20:], 0), "bat_first": (sum(st.bat_first)+2)/(len(st.bat_first)+4),
        "chase": (sum(st.chase)+2)/(len(st.chase)+4), "home": (sum(st.home)+2)/(len(st.home)+4),
        "away": (sum(st.away)+2)/(len(st.away)+4), "neutral": (sum(st.neutral)+2)/(len(st.neutral)+4),
        "venue_familiarity": math.log1p(venue_counts[(team, venue)]), "rest_days": float(rest),
        "h2h_bayes": (sum(hh)+2)/(len(hh)+4), "wc_rate": wc_s["wc_rate"],
        "wc_experience": wc_s["wc_experience"], "matches_played": math.log1p(len(st.results)),
        "role": role,
    }


def innings_values(m, team):
    first = m.innings1_team == team
    rf = m.innings1_runs if first else m.innings2_runs
    ra = m.innings2_runs if first else m.innings1_runs
    wl = m.innings1_wickets if first else m.innings2_wickets
    wt = m.innings2_wickets if first else m.innings1_wickets
    balls_f = m.innings1_legal_balls if first else m.innings2_legal_balls
    balls_a = m.innings2_legal_balls if first else m.innings1_legal_balls
    def num(x): return float(x) if pd.notna(x) else np.nan
    rf, ra, wl, wt, balls_f, balls_a = map(num, [rf, ra, wl, wt, balls_f, balls_a])
    rpwi = rf / max(wl, 1) if np.isfinite(rf) and np.isfinite(wl) else np.nan
    rpcw = ra / max(wt, 1) if np.isfinite(ra) and np.isfinite(wt) else np.nan
    rr = rf / (balls_f / 6) if np.isfinite(rf) and balls_f > 0 else np.nan
    econ = ra / (balls_a / 6) if np.isfinite(ra) and balls_a > 0 else np.nan
    return first, rpwi, rpcw, rr, econ, rr - econ if np.isfinite(rr) and np.isfinite(econ) else np.nan


def build_advanced_features(odi, wc, host, location_map=None):
    states = defaultdict(State); elos = defaultdict(lambda: 1500.0)
    surface_elos = defaultdict(lambda: defaultdict(lambda: 1500.0))
    host_elos = defaultdict(lambda: defaultdict(lambda: 1500.0))
    venue_counts, h2h, venue_history = defaultdict(int), defaultdict(list), defaultdict(list)
    dist = RunningDistribution()
    if location_map is None:
        location_map, _, _, _ = build_venue_mapping(odi, wc, host)
    host_one = location_map.drop_duplicates("match_id").set_index("match_id")
    rows, orientation_errors = [], 0
    source = odi.sort_values(["date", "match_id"]).drop_duplicates("match_id")
    for _, m in source.iterrows():
        if pd.isna(m.date) or pd.isna(m.team1) or pd.isna(m.team2): continue
        venue = str(m.venue) if pd.notna(m.venue) else "Unknown"
        host_country = str(host_one.loc[m.match_id, "country"]) if m.match_id in host_one.index and pd.notna(host_one.loc[m.match_id, "country"]) else ""
        host_team = str(host_one.loc[m.match_id, "host_team"]) if m.match_id in host_one.index and pd.notna(host_one.loc[m.match_id, "host_team"]) else ""
        surface = classify_surface(venue_history, venue)
        flip = int(hashlib.sha256(f"orientation:{m.match_id}".encode()).hexdigest(), 16) & 1
        a, b = (m.team2, m.team1) if flip else (m.team1, m.team2)
        sa = team_snapshot(a, b, m.date, venue, host_country, host_team, surface, states, elos, surface_elos,
                           host_elos, venue_counts, h2h, wc)
        sb = team_snapshot(b, a, m.date, venue, host_country, host_team, surface, states, elos, surface_elos,
                           host_elos, venue_counts, h2h, wc)
        row = {"match_id": m.match_id, "date": m.date, "team_a": a, "team_b": b, "winner": m.winner,
               "result_type": m.result_type, "event_name": m.event_name, "venue": venue,
               "host_country": host_country, "host_team": host_team, "surface": surface,
               "is_full_member_match": int(a in FULL_MEMBERS and b in FULL_MEMBERS),
               "is_world_cup": int(str(m.event_name) in {"ICC World Cup", "ICC Cricket World Cup", "World Cup"}),
               "is_2027_host_country": int(host_country in HOSTS)}
        for f in FEATURE_NAMES: row[f"diff_{f}"] = sa[f] - sb[f]
        row["home_advantage_diff"] = float(sa["role"] == "home") - float(sb["role"] == "home")
        row["neutral_match"] = float(sa["role"] == "neutral" and bool(host_country))
        row["interaction_elo_surface"] = row["diff_elo"] * row["diff_surface_elo"] / 10000
        row["interaction_elo_venue"] = row["diff_elo"] * row["diff_venue_familiarity"] / 100
        row["interaction_strength_host"] = row["diff_elo"] * row["home_advantage_diff"] / 100
        row["interaction_form_opposition"] = row["diff_weighted_form20"] * row["diff_recent_opponent_elo"] / 100
        row["target"] = float(m.winner == a) if m.result_type == "win" and pd.notna(m.winner) else np.nan
        if pd.notna(row["target"]) and ((row["target"] == 1) != (m.winner == a)): orientation_errors += 1
        rows.append(row)

        decisive = m.result_type == "win" and pd.notna(m.winner)
        if decisive:
            old1, old2 = elos[m.team1], elos[m.team2]; exp1 = expected(old1, old2)
            y1 = float(m.winner == m.team1); k = 24
            elos[m.team1] += k*(y1-exp1); elos[m.team2] += k*((1-y1)-(1-exp1))
            s1, s2 = surface_elos[surface][m.team1], surface_elos[surface][m.team2]; es1 = expected(s1, s2)
            surface_elos[surface][m.team1] += 18*(y1-es1); surface_elos[surface][m.team2] += 18*((1-y1)-(1-es1))
            if host_country:
                h1, h2 = host_elos[host_country][m.team1], host_elos[host_country][m.team2]; eh1 = expected(h1, h2)
                host_elos[host_country][m.team1] += 18*(y1-eh1); host_elos[host_country][m.team2] += 18*((1-y1)-(1-eh1))
            for team, opp, opp_old, exp_win in [(m.team1,m.team2,old2,exp1),(m.team2,m.team1,old1,1-exp1)]:
                win = float(m.winner == team); st = states[team]
                first, rpwi, rpcw, rr, econ, nrr = innings_values(m, team)
                vals = {"rpwi": rpwi, "rpcw": rpcw, "rr": rr, "econ": econ, "nrr": nrr}
                z = {k0: dist.z(k0, v) for k0,v in vals.items()}
                st.results.append(win); st.adjusted_results.append(np.clip(.5 + win - exp_win, 0, 1))
                st.opponent_elos.append(opp_old); st.dates.append(m.date); st.rpwi_z.append(z["rpwi"])
                st.rpcw_z.append(z["rpcw"]); st.run_rate_z.append(z["rr"]); st.economy_z.append(z["econ"]); st.nrr_z.append(z["nrr"])
                (st.bat_first if first else st.chase).append(win)
                role = "home" if team == host_team else ("away" if opp == host_team else "neutral")
                if host_country: getattr(st, role).append(win)
                venue_counts[(team, venue)] += 1; pair = tuple(sorted((team, opp))); h2h[(pair, team)].append(win)
            # Update era distributions only after both pre-match snapshots and team normalizations.
            first_runs = float(m.innings1_runs) if pd.notna(m.innings1_runs) else np.nan
            if np.isfinite(first_runs): venue_history[venue].append(first_runs)
            for team in [m.team1, m.team2]:
                _, rpwi, rpcw, rr, econ, nrr = innings_values(m, team)
                for k0,v in {"rpwi":rpwi,"rpcw":rpcw,"rr":rr,"econ":econ,"nrr":nrr}.items(): dist.update(k0,v)
    df = pd.DataFrame(rows).sort_values(["date", "match_id"]).reset_index(drop=True)
    df.to_csv(CLEAN / "advanced_modeling_features.csv", index=False, date_format="%Y-%m-%d")
    audit = {"source_odi_rows": len(odi), "unique_match_rows": len(df), "duplicate_match_ids": int(df.match_id.duplicated().sum()),
             "orientation_errors": orientation_errors, "decisive_rows": int(df.target.notna().sum()),
             "feature_update_order": "snapshot before current-match update", "orientation": "stable SHA-256 reversal",
             "neutral_rule": "neutral only when supplied host_country is known and neither team is host",
             "unknown_host_matches": int(df.host_country.eq("").sum()), "world_cup_match_rows": int(df.is_world_cup.sum()),
             "recent_team_centric_rows_used_in_training": 0, "tournament_summary_rows_used_in_training": 0}
    (CLEAN / "advanced_pipeline_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    final = {t: team_snapshot(t, "Unknown", pd.Timestamp("2027-01-01"), "Unknown", "", "", "unknown",
                              states, elos, surface_elos, host_elos, venue_counts, h2h, wc) for t in states}
    return df, final, audit


def metric_set(y, p):
    y=np.asarray(y,int); p=np.clip(np.asarray(p,float),1e-6,1-1e-6); pred=p>=.5
    return {"accuracy":accuracy_score(y,pred),"precision":precision_score(y,pred,zero_division=0),
            "recall":recall_score(y,pred,zero_division=0),"f1":f1_score(y,pred,zero_division=0),
            "roc_auc":roc_auc_score(y,p) if len(np.unique(y))>1 else np.nan,
            "log_loss":log_loss(y,p,labels=[0,1]),"brier":brier_score_loss(y,p)}


def symmetric_frame(X):
    return pd.concat([X, -X], ignore_index=True)


def fit_model(estimator, X, y, weights=None):
    pipe=Pipeline([("imputer",SimpleImputer(strategy="median",add_indicator=True)),
                   ("scale",StandardScaler() if isinstance(estimator,LogisticRegression) else "passthrough"),
                   ("model",clone(estimator))])
    Xa=symmetric_frame(X); ya=np.r_[np.asarray(y),1-np.asarray(y)]
    fit_args={}
    if weights is not None: fit_args["model__sample_weight"]=np.r_[weights,weights]
    pipe.fit(Xa,ya,**fit_args); return pipe


def sym_predict(model, X):
    p=model.predict_proba(X)[:,1]; pr=model.predict_proba(-X)[:,1]
    return np.clip(.5*(p+1-pr),1e-6,1-1e-6)


def expanding_folds(n):
    bounds=[(.42,.50),(.50,.60),(.60,.70)]
    return [(np.arange(int(n*a)),np.arange(int(n*a),int(n*b))) for a,b in bounds]


def candidates():
    return {
      "regularized_logistic":[LogisticRegression(C=c,max_iter=3000,random_state=RNG_SEED) for c in [.02,.08,.25,1.0]],
      "hist_gradient_boosting":[HistGradientBoostingClassifier(max_iter=220,learning_rate=lr,max_leaf_nodes=leaf,l2_regularization=3,random_state=RNG_SEED) for lr,leaf in [(.025,7),(.04,15),(.025,15)]],
      "random_forest":[RandomForestClassifier(n_estimators=450,min_samples_leaf=leaf,max_features=mf,n_jobs=-1,random_state=RNG_SEED,class_weight="balanced") for leaf,mf in [(10,.5),(18,.7)]],
      "xgboost":[XGBClassifier(n_estimators=300,max_depth=d,learning_rate=lr,subsample=.8,colsample_bytree=.8,reg_lambda=5,min_child_weight=8,n_jobs=4,random_state=RNG_SEED,eval_metric="logloss") for d,lr in [(2,.025),(3,.02)]],
      "catboost":[CatBoostClassifier(iterations=350,depth=d,learning_rate=lr,l2_leaf_reg=6,verbose=False,random_seed=RNG_SEED,thread_count=4,allow_writing_files=False) for d,lr in [(4,.025),(5,.02)]],
    }


def recency_weights(dates, half_life_years):
    if half_life_years is None: return None
    age=(dates.max()-dates).dt.days.to_numpy(); return np.maximum(.15,np.exp(-math.log(2)*age/(365.25*half_life_years)))


def tune_models(train, features):
    rows=[]; best={}
    folds=expanding_folds(len(train))
    for family, specs in candidates().items():
        for idx, est in enumerate(specs):
            for half in [None,4.0]:
                scores=[]
                for ti,vi in folds:
                    tr,va=train.iloc[ti],train.iloc[vi]
                    model=fit_model(est,tr[features],tr.target,recency_weights(tr.date,half))
                    p=sym_predict(model,va[features]); scores.append(metric_set(va.target,p))
                row={"model":family,"candidate":idx,"recency_half_life_years":half or 0,
                     **{f"cv_{k}":np.mean([s[k] for s in scores]) for k in scores[0]}}
                row["selection_score"]=row["cv_log_loss"]+row["cv_brier"]; rows.append(row)
        fam=[r for r in rows if r["model"]==family]; best[family]=min(fam,key=lambda r:r["selection_score"])
    return pd.DataFrame(rows).sort_values("selection_score"),best


class Calibrator:
    def __init__(self, kind="none"): self.kind=kind; self.model=None
    def fit(self,p,y):
        p=np.clip(np.asarray(p),1e-6,1-1e-6)
        if self.kind=="isotonic": self.model=IsotonicRegression(out_of_bounds="clip").fit(p,y)
        elif self.kind=="sigmoid": self.model=LogisticRegression(C=1e6).fit(np.log(p/(1-p)).reshape(-1,1),y)
        return self
    def predict(self,p):
        p=np.clip(np.asarray(p),1e-6,1-1e-6)
        if self.kind=="isotonic": return np.clip(self.model.predict(p),1e-6,1-1e-6)
        if self.kind=="sigmoid": return self.model.predict_proba(np.log(p/(1-p)).reshape(-1,1))[:,1]
        return p


def baseline_predictions(train,test):
    outputs={}
    # Historical stronger team, deliberately simple and probabilistic.
    outputs["historically_stronger"] = np.where(test.diff_overall>=0,.60,.40)
    for name,col in [("elo_only","diff_elo"),("overall_win_rate","diff_overall"),("home_host_advantage","home_advantage_diff")]:
        m=fit_model(LogisticRegression(C=.25,max_iter=2000),train[[col]],train.target)
        outputs[name]=sym_predict(m,test[[col]])
    blend_cols=["diff_elo","diff_weighted_form20"]
    blend=fit_model(LogisticRegression(C=.25,max_iter=2000),train[blend_cols],train.target)
    outputs["blended_elo_recent_form"]=sym_predict(blend,test[blend_cols])
    legacy_cols=[c for c in train if c in {"diff_overall","diff_weighted_form5","diff_weighted_form10","diff_weighted_form20","diff_weighted_form50","diff_wc_rate","diff_h2h_bayes","diff_bat_first","diff_chase","diff_rest_days"}]
    m=fit_model(LogisticRegression(C=.5,max_iter=2000),train[legacy_cols],train.target)
    outputs["existing_logistic_rebuilt"]=sym_predict(m,test[legacy_cols])
    return outputs


def bootstrap_metrics(y,p,baseline_probs,n=1500):
    rng=np.random.default_rng(RNG_SEED); y=np.asarray(y); p=np.asarray(p); names=list(metric_set(y,p))
    vals={k:[] for k in names}; diffs={b:{"log_loss":[],"brier":[]} for b in baseline_probs}
    for _ in range(n):
        ix=rng.integers(0,len(y),len(y)); yy=y[ix]; pp=p[ix]
        if len(np.unique(yy))<2: continue
        mm=metric_set(yy,pp)
        for k in names: vals[k].append(mm[k])
        for b,bp in baseline_probs.items():
            bm=metric_set(yy,np.asarray(bp)[ix])
            diffs[b]["log_loss"].append(mm["log_loss"]-bm["log_loss"])
            diffs[b]["brier"].append(mm["brier"]-bm["brier"])
    ci=pd.DataFrame([{"metric":k,"estimate":metric_set(y,p)[k],"ci_low":np.percentile(v,2.5),"ci_high":np.percentile(v,97.5)} for k,v in vals.items()])
    sig=[]
    for b,d in diffs.items():
        for k,v in d.items():
            lo,hi=np.percentile(v,[2.5,97.5]); sig.append({"baseline":b,"metric":k,"improved_minus_baseline":np.mean(v),"ci_low":lo,"ci_high":hi,"meaningfully_better":bool(hi<0)})
    return ci,pd.DataFrame(sig)


def evaluate_scopes(test,p):
    scopes={"all_final_test":np.ones(len(test),bool),"full_member_vs_full_member":test.is_full_member_match.eq(1).to_numpy(),
            "matches_2015_onward":test.date.ge("2015-01-01").to_numpy(),"world_cup_matches":test.is_world_cup.eq(1).to_numpy(),
            "south_africa_zimbabwe_namibia":test.is_2027_host_country.eq(1).to_numpy()}
    rows=[]
    for name,mask in scopes.items():
        if mask.sum()<10 or test.loc[mask,"target"].nunique()<2:
            rows.append({"scope":name,"matches":int(mask.sum()),"note":"insufficient final-test matches"}); continue
        rows.append({"scope":name,"matches":int(mask.sum()),**metric_set(test.loc[mask,"target"],p[mask])})
    return pd.DataFrame(rows)


def world_cup_walk_forward(df, features, estimator, half_life, calibration_kind="none"):
    pieces=[]; names={2015:"ICC Cricket World Cup",2019:"World Cup",2023:"ICC Cricket World Cup"}
    for year,event in names.items():
        ev=df[(df.date.dt.year==year)&df.event_name.eq(event)&df.target.notna()]
        if ev.empty: continue
        prior=df[(df.date<ev.date.min())&df.target.notna()].copy()
        cut=int(len(prior)*.85); tr,ca=prior.iloc[:cut],prior.iloc[cut:]
        m=fit_model(estimator,tr[features],tr.target,recency_weights(tr.date,half_life))
        cal=Calibrator(calibration_kind).fit(sym_predict(m,ca[features]),ca.target)
        pred=cal.predict(sym_predict(m,ev[features]))
        pieces.append(pd.DataFrame({"target":ev.target.to_numpy(),"probability":pred,"edition":year}))
    if not pieces: return pd.DataFrame()
    return pd.concat(pieces,ignore_index=True)


def current_pair_features(a,b,feature_cols,host_country=""):
    row={}
    for c in feature_cols:
        if c.startswith("diff_"):
            k=c[5:]; row[c]=float(a.get(k,0))-float(b.get(k,0))
        elif c=="home_advantage_diff": row[c]=float(a.get("team")==host_country)-float(b.get("team")==host_country)
        elif c=="neutral_match": row[c]=0.0
        elif c=="interaction_elo_surface": row[c]=(a.get("elo",1500)-b.get("elo",1500))*(a.get("surface_elo",1500)-b.get("surface_elo",1500))/10000
        elif c=="interaction_elo_venue": row[c]=0.0
        elif c=="interaction_strength_host": row[c]=(a.get("elo",1500)-b.get("elo",1500))*row.get("home_advantage_diff",0)/100
        elif c=="interaction_form_opposition": row[c]=(a.get("weighted_form20",.5)-b.get("weighted_form20",.5))*(a.get("recent_opponent_elo",1500)-b.get("recent_opponent_elo",1500))/100
        else: row[c]=0.0
    return pd.DataFrame([row])


def simulate_tournament(team_snapshots,model,calibrator,features,runs=20000):
    confirmed=sorted(pd.read_csv(RAW/"recent_100_confirmed.csv").confirmed_team.unique())
    neutral={k:(1500.0 if "elo" in k else (.5 if any(x in k for x in ["form","overall","rate","bat_first","chase","home","away","neutral","h2h"]) else 0.0)) for k in FEATURE_NAMES}
    table={t:{**neutral,**team_snapshots.get(t,{}),"team":t} for t in confirmed}
    for i in range(1,5): table[f"Qualifier {i}"]={**neutral,"team":f"Qualifier {i}"}
    teams=list(table); probs={}
    for i,a in enumerate(teams):
        for b in teams[i+1:]:
            host=a if a in HOSTS else (b if b in HOSTS else "")
            x=current_pair_features(table[a],table[b],features,host)
            pr=float(calibrator.predict(sym_predict(model,x))[0]); probs[a,b]=np.clip(pr,.04,.96); probs[b,a]=1-probs[a,b]
    rng=np.random.default_rng(2027); batches=40; per=max(250,runs//batches); batch_rows=[]
    totals={t:[0,0,0] for t in teams}
    for batch in range(batches):
        counts={t:[0,0,0] for t in teams}
        for _ in range(per):
            pts={t:0 for t in teams}; wins={t:0 for t in teams}
            for i,a in enumerate(teams):
                for b in teams[i+1:]:
                    w=a if rng.random()<probs[a,b] else b; pts[w]+=2; wins[w]+=1
            top=sorted(teams,key=lambda t:(pts[t],wins[t],rng.random()),reverse=True)[:4]
            for t in top: counts[t][0]+=1; totals[t][0]+=1
            finalists=[]
            for a,b in [(top[0],top[3]),(top[1],top[2])]:
                w=a if rng.random()<probs[a,b] else b; finalists.append(w); counts[w][1]+=1; totals[w][1]+=1
            a,b=finalists; w=a if rng.random()<probs[a,b] else b; counts[w][2]+=1; totals[w][2]+=1
        for t in teams: batch_rows.append({"batch":batch,"team":t,"win_probability":counts[t][2]/per})
    n=batches*per; batch_df=pd.DataFrame(batch_rows)
    out=[]
    for t in teams:
        vals=batch_df[batch_df.team==t].win_probability
        out.append({"team":t,"semi_final_probability":totals[t][0]/n,"final_probability":totals[t][1]/n,"win_probability":totals[t][2]/n,
                    "win_probability_ci_low":vals.quantile(.025),"win_probability_ci_high":vals.quantile(.975)})
    return pd.DataFrame(out).sort_values("win_probability",ascending=False),probs


def render_outputs(comparison,ci,significance,scopes,calibration,simulation,audit,selected_name,importance):
    comparison.to_csv(OUT/"advanced_model_comparison.csv",index=False); ci.to_csv(OUT/"advanced_metric_confidence_intervals.csv",index=False)
    significance.to_csv(OUT/"advanced_baseline_significance.csv",index=False); scopes.to_csv(OUT/"advanced_subgroup_evaluation.csv",index=False)
    calibration.to_csv(OUT/"advanced_calibration_comparison.csv",index=False); simulation.to_csv(OUT/"advanced_tournament_probabilities.csv",index=False)
    importance.to_csv(OUT/"advanced_feature_importance.csv",index=False)
    plt.style.use("seaborn-v0_8-whitegrid")
    q=comparison.sort_values("log_loss",ascending=True)
    fig,ax=plt.subplots(figsize=(10,7)); ax.barh(q.model,q.log_loss,color="#155e75"); ax.invert_yaxis(); ax.set_title("Chronological test log loss",loc="left",weight="bold"); fig.tight_layout(); fig.savefig(CHARTS/"advanced_model_comparison.png",dpi=170); plt.close(fig)
    q=simulation.sort_values("win_probability")
    low=np.minimum(q.win_probability,q.win_probability_ci_low); high=np.maximum(q.win_probability,q.win_probability_ci_high)
    fig,ax=plt.subplots(figsize=(10,7)); err=np.vstack([q.win_probability-low,high-q.win_probability]); ax.barh(q.team,q.win_probability,xerr=err,color="#0f766e",alpha=.9); ax.set_title("Provisional 2027 title probability with simulation intervals",loc="left",weight="bold"); fig.tight_layout(); fig.savefig(CHARTS/"advanced_winning_probability.png",dpi=170); plt.close(fig)
    q=importance.nlargest(18,"importance").sort_values("importance")
    fig,ax=plt.subplots(figsize=(10,7)); ax.barh(q.feature,q.importance,color="#b45309"); ax.set_title("Advanced-model permutation importance",loc="left",weight="bold"); fig.tight_layout(); fig.savefig(CHARTS/"advanced_feature_importance.png",dpi=170); plt.close(fig)
    q=significance[significance.metric.eq("log_loss")].sort_values("improved_minus_baseline")
    fig,ax=plt.subplots(figsize=(10,6)); x=q.improved_minus_baseline; colors=np.where(q.meaningfully_better,"#0f766e","#94a3b8")
    for i,(_,r) in enumerate(q.iterrows()): ax.errorbar(r.improved_minus_baseline,i,xerr=[[r.improved_minus_baseline-r.ci_low],[r.ci_high-r.improved_minus_baseline]],fmt="o",color=colors[i],ecolor=colors[i],capsize=4)
    ax.axvline(0,color="#111827",lw=1); ax.set_yticks(np.arange(len(q)),q.baseline); ax.set_xlabel("CatBoost minus baseline log loss (lower is better)"); ax.set_title("Paired bootstrap differences with 95% intervals",loc="left",weight="bold"); fig.tight_layout(); fig.savefig(CHARTS/"advanced_metric_differences.png",dpi=170); plt.close(fig)
    rows="".join(f"<tr><td>{html.escape(r.team)}</td><td>{r.semi_final_probability:.1%}</td><td>{r.final_probability:.1%}</td><td>{r.win_probability:.1%}</td><td>{r.win_probability_ci_low:.1%}–{r.win_probability_ci_high:.1%}</td></tr>" for _,r in simulation.iterrows())
    comp_html=comparison.round(4).to_html(index=False); ci_html=ci.round(4).to_html(index=False); sig_html=significance.round(4).to_html(index=False); scope_html=scopes.round(4).to_html(index=False)
    report=f"""<!doctype html><html><head><meta charset='utf-8'><title>Advanced CWC 2027 model</title><style>body{{font:15px system-ui;max-width:1200px;margin:36px auto;color:#17202a}}h1,h2{{color:#083344}}table{{border-collapse:collapse;width:100%;margin:12px 0}}th,td{{padding:7px;border-bottom:1px solid #ddd;text-align:left}}.note{{padding:12px;background:#fff7ed;border-left:4px solid #d97706}}img{{max-width:100%}}</style></head><body><h1>Advanced ICC Men's ODI World Cup 2027 model</h1><p>Selected model: <b>{html.escape(selected_name)}</b>. All final-test rows remain chronological and untouched during tuning and calibration.</p><div class='note'>No official fixture list was supplied. Tournament results remain a provisional 14-team round-robin simulation with four neutral qualifier placeholders. Intervals show Monte Carlo batch uncertainty, not full structural model uncertainty.</div><h2>Final-test model and baselines</h2>{comp_html}<h2>Bootstrap 95% confidence intervals</h2>{ci_html}<h2>Paired improvement tests</h2>{sig_html}<h2>Separate evaluations</h2>{scope_html}<h2>Provisional tournament</h2><table><tr><th>Team</th><th>Semi-final</th><th>Final</th><th>Champion</th><th>Champion interval</th></tr>{rows}</table><img src='../charts/advanced_model_comparison.png'><img src='../charts/advanced_metric_differences.png'><img src='../charts/advanced_calibration_reliability.png'><img src='../charts/advanced_winning_probability.png'><img src='../charts/advanced_feature_importance.png'><h2>Audit conclusions</h2><pre>{html.escape(json.dumps(audit,indent=2))}</pre><h2>Limitations and next data</h2><p>The match-level source begins in 2002 and does not cover every ODI. Host-country labels are available only for South Africa, Zimbabwe and Namibia, so home/away/neutral features outside those countries remain unknown rather than inferred. Surface is a leakage-safe scoring-environment proxy, not measured pitch composition. The largest likely improvements would come from complete historical ODI coverage, verified venue countries and pitch types, player/squad availability, injuries, rankings as-of-match, weather, and the official 2027 schedule.</p></body></html>"""
    (REPORT/"advanced_cwc_2027_report.html").write_text(report,encoding="utf-8")


def main(runs=20000, location_mode="direct", reuse_tuning=False):
    data=load_and_clean(); expanded_map,_,_,location_summary=build_venue_mapping(data["odi"],data["world_cup"],data["host"])
    if location_mode == "expanded":
        model_map = expanded_map
    else:
        direct = data["host"].drop_duplicates("match_id")[["match_id","host_country"]].rename(columns={"host_country":"country"})
        direct["host_team"] = direct.country.replace({"Wales":"England"})
        model_map = data["odi"][["match_id"]].merge(direct,on="match_id",how="left")
    df,team_snapshots,audit=build_advanced_features(data["odi"],data["world_cup"],data["host"],model_map)
    audit["location_coverage"] = location_summary; audit["model_location_mode"] = location_mode
    decisive=df.dropna(subset=["target"]).reset_index(drop=True); n=len(decisive); i1=int(.70*n); i2=int(.85*n)
    train,cal,test=decisive.iloc[:i1].copy(),decisive.iloc[i1:i2].copy(),decisive.iloc[i2:].copy()
    excluded={"match_id","date","team_a","team_b","winner","result_type","event_name","venue","host_country","host_team","surface","target","is_full_member_match","is_world_cup","is_2027_host_country"}
    features=[c for c in decisive.columns if c not in excluded]
    if reuse_tuning and (OUT/"advanced_cv_tuning.csv").exists():
        tuning=pd.read_csv(OUT/"advanced_cv_tuning.csv")
        best_by_family={k:g.sort_values("selection_score").iloc[0].to_dict() for k,g in tuning.groupby("model")}
    else:
        tuning,best_by_family=tune_models(train,features); tuning.to_csv(OUT/"advanced_cv_tuning.csv",index=False)
    fitted={}; val_rows=[]
    spec_map=candidates()
    for family,choice in best_by_family.items():
        est=spec_map[family][int(choice["candidate"])]; half=choice["recency_half_life_years"] or None
        m=fit_model(est,train[features],train.target,recency_weights(train.date,half)); fitted[family]=m
        vp=sym_predict(m,cal[features]); vm=metric_set(cal.target,vp); val_rows.append({"model":family,"split":"calibration_validation","calibration":"none",**vm})
    selected_family=min(val_rows,key=lambda r:r["log_loss"]+r["brier"])["model"]
    base=fitted[selected_family]; val_base=sym_predict(base,cal[features])
    # Calibration choice is made on the later part of validation; calibrators are fit only
    # on the earlier part, then the chosen method is refit on all validation observations.
    vc=int(len(cal)*.60); cal_fit,cal_select=cal.iloc[:vc],cal.iloc[vc:]
    fit_probs=sym_predict(base,cal_fit[features]); select_probs=sym_predict(base,cal_select[features])
    selection=[]
    for kind in ["none","sigmoid","isotonic"]:
        c=Calibrator(kind).fit(fit_probs,cal_fit.target); pp=c.predict(select_probs)
        selection.append({"method":kind,**metric_set(cal_select.target,pp)})
    selection_df=pd.DataFrame(selection); chosen_kind=selection_df.assign(score=lambda x:x.log_loss+x.brier).sort_values("score").iloc[0].method
    calibration_rows=[]; calibrators={}
    for kind in ["none","sigmoid","isotonic"]:
        c=Calibrator(kind).fit(val_base,cal.target); calibrators[kind]=c; pp=c.predict(val_base); calibration_rows.append({"method":kind,**metric_set(cal.target,pp)})
    calibration_df=pd.DataFrame(calibration_rows).merge(selection_df[["method","log_loss","brier"]].rename(columns={"log_loss":"selection_log_loss","brier":"selection_brier"}),on="method")
    calibrator=calibrators[chosen_kind]; test_base=sym_predict(base,test[features]); test_p=calibrator.predict(test_base)
    base_preds=baseline_predictions(train,test)
    base_preds["regularized_logistic"] = sym_predict(fitted["regularized_logistic"], test[features])
    comparison=[]
    for name,p in base_preds.items(): comparison.append({"model":name,"type":"baseline","calibration":"none",**metric_set(test.target,p)})
    for family,m in fitted.items():
        p=sym_predict(m,test[features]); comparison.append({"model":family,"type":"candidate","calibration":"none",**metric_set(test.target,p)})
    comparison.append({"model":f"improved_{selected_family}","type":"selected","calibration":chosen_kind,**metric_set(test.target,test_p)})
    comparison.append({"model":"original_reported_model","type":"original","calibration":"none","accuracy":.5775,"precision":np.nan,"recall":np.nan,"f1":np.nan,"roc_auc":.6067,"log_loss":.6930,"brier":.2483})
    comparison=pd.DataFrame(comparison)
    legacy=pd.read_csv(OUT/"match_predictions.csv")[["match_id","team_a","win_probability_team_a"]]
    aligned=test[["match_id","team_a"]].merge(legacy,on="match_id",how="left",suffixes=("","_legacy"))
    original_p=np.where(aligned.team_a.eq(aligned.team_a_legacy),aligned.win_probability_team_a,1-aligned.win_probability_team_a)
    original_p=np.where(pd.isna(original_p),.5,original_p)
    ci,significance=bootstrap_metrics(test.target,test_p,{**base_preds,"original_reported_model":np.clip(original_p,1e-6,1-1e-6)})
    scopes=evaluate_scopes(test,test_p)
    selected_est=spec_map[selected_family][int(best_by_family[selected_family]["candidate"])]
    wc_oof=world_cup_walk_forward(decisive,features,selected_est,best_by_family[selected_family]["recency_half_life_years"] or None,chosen_kind)
    if not wc_oof.empty:
        scopes=scopes[scopes.scope!="world_cup_matches"]
        scopes=pd.concat([scopes,pd.DataFrame([{"scope":"world_cup_matches_walk_forward_2015_2019_2023","matches":len(wc_oof),**metric_set(wc_oof.target,wc_oof.probability)}])],ignore_index=True)
    # Reliability curves are diagnostics only; they do not change selection.
    fig,ax=plt.subplots(figsize=(7,7)); test_cal_rows=[]
    for kind,c in calibrators.items():
        cp=c.predict(test_base); frac,mean=calibration_curve(test.target,cp,n_bins=8,strategy="quantile"); ax.plot(mean,frac,marker="o",label=kind)
        test_cal_rows.append({"method":kind,**metric_set(test.target,cp)})
    ax.plot([0,1],[0,1],"--",color="black",label="ideal"); ax.set(xlabel="Mean predicted probability",ylabel="Observed win rate",title="Chronological test reliability"); ax.legend(); fig.tight_layout(); fig.savefig(CHARTS/"advanced_calibration_reliability.png",dpi=170); plt.close(fig)
    pd.DataFrame(test_cal_rows).to_csv(OUT/"advanced_calibration_test_diagnostics.csv",index=False)
    # Ordering audit is measured on the final test without changing any labels or split.
    raw=base.predict_proba(test[features])[:,1]; reverse=base.predict_proba(-test[features])[:,1]
    served=sym_predict(base,test[features]); served_reverse=sym_predict(base,-test[features])
    audit.update({"train_end":str(train.date.max().date()),"calibration_end":str(cal.date.max().date()),"test_start":str(test.date.min().date()),"test_end":str(test.date.max().date()),
                  "selected_family":selected_family,"selected_calibration":chosen_kind,"raw_model_mean_order_error":float(np.mean(np.abs(raw+reverse-1))),
                  "served_prediction_mean_order_error":float(np.mean(np.abs(served+served_reverse-1))),
                  "train_mean_first_innings_runs":float(data["odi"][data["odi"].date<=train.date.max()].innings1_runs.mean()),
                  "test_mean_first_innings_runs":float(data["odi"][data["odi"].date>=test.date.min()].innings1_runs.mean())})
    (CLEAN/"advanced_pipeline_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    # Production model uses all observed rows after honest final evaluation; validation-fitted calibration is retained.
    choice=best_by_family[selected_family]; production=fit_model(spec_map[selected_family][int(choice["candidate"])],decisive[features],decisive.target,recency_weights(decisive.date,choice["recency_half_life_years"] or None))
    simulation,_=simulate_tournament(team_snapshots,production,calibrator,features,runs)
    from sklearn.inspection import permutation_importance
    # Wrapper-free permutation importance uses the selected base model's direct score; symmetry is checked separately.
    perm=permutation_importance(base,test[features],test.target,n_repeats=8,random_state=RNG_SEED,scoring="neg_log_loss")
    importance=pd.DataFrame({"feature":features,"importance":perm.importances_mean}).sort_values("importance",ascending=False)
    render_outputs(comparison,ci,significance,scopes,calibration_df,simulation,audit,f"{selected_family} + {chosen_kind}",importance)
    serialized_calibrator = None if chosen_kind == "none" else calibrator
    joblib.dump({"model":production,"calibrator":serialized_calibrator,"features":features,"selected_family":selected_family,"calibration":chosen_kind,"audit":audit,
                 "candidate":int(choice["candidate"]),"recency_half_life_years":choice["recency_half_life_years"] or None,"team_snapshots":team_snapshots},MODELS/"advanced_pre_match.joblib")
    joblib.dump({"model":base,"calibrator":serialized_calibrator,"features":features,"selected_family":selected_family,"calibration":chosen_kind,"audit":audit},MODELS/"advanced_evaluation_model.joblib")
    test_out=test[["match_id","date","team_a","team_b","winner"]].copy(); test_out["win_probability_team_a"]=test_p; test_out.to_csv(OUT/"advanced_match_predictions.csv",index=False)
    summary={"selected_model":selected_family,"calibration":chosen_kind,"test_metrics":metric_set(test.target,test_p),"audit":audit,"leaders":simulation.head(5).to_dict("records")}
    (OUT/"advanced_run_summary.json").write_text(json.dumps(summary,indent=2,default=str),encoding="utf-8"); print(json.dumps(summary,indent=2,default=str))


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--runs",type=int,default=20000); p.add_argument("--location-mode",choices=["direct","expanded"],default="direct"); p.add_argument("--reuse-tuning",action="store_true"); a=p.parse_args(); main(a.runs,a.location_mode,a.reuse_tuning)
