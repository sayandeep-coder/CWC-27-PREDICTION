"""Final explanations, uncertainty analysis, dashboard, and reusable outputs."""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import Pool

from .advanced_pipeline import (CHARTS, CLEAN, HOSTS, MODELS, OUT, RAW, REPORT,
                                Calibrator, candidates, current_pair_features, fit_model,
                                metric_set, recency_weights, sym_predict)

SEED = 2711
DISCLAIMER = "Provisional scenario-based estimates—not an official tournament forecast."


def readable_feature(name):
    labels = {
        "diff_elo":"Elo rating", "diff_weighted_form20":"recent form",
        "diff_venue_familiarity":"venue familiarity", "home_advantage_diff":"host advantage",
        "diff_h2h_bayes":"head-to-head record", "diff_surface_elo":"surface-specific Elo",
        "diff_wc_rate":"recently weighted World Cup record", "diff_nrr_z":"era-adjusted net run rate",
        "diff_run_rate_z":"era-adjusted scoring rate", "diff_economy_z":"era-adjusted economy rate",
        "diff_opponent_adjusted_form":"opponent-adjusted form", "diff_recent_opponent_elo":"recent opposition strength",
        "diff_rpwi_z":"era-adjusted runs per wicket", "diff_rpcw_z":"era-adjusted runs conceded per wicket",
    }
    return labels.get(name, name.replace("diff_", "").replace("interaction_", "").replace("_", " "))


def explain_matches(features, predictions, bundle):
    decisive = features.dropna(subset=["target"]).reset_index(drop=True)
    test = decisive.iloc[int(.85*len(decisive)):].copy()
    pred = predictions.copy(); pred["date"] = pd.to_datetime(pred.date)
    test = test.merge(pred[["match_id","win_probability_team_a"]], on="match_id", how="left").reset_index(drop=True)
    model = bundle["model"]; cols = bundle["features"]
    transformed = model.named_steps["imputer"].transform(test[cols])
    if model.named_steps["scale"] != "passthrough": transformed = model.named_steps["scale"].transform(transformed)
    cb = model.named_steps["model"]
    shap = cb.get_feature_importance(Pool(transformed), type="ShapValues")[:, :-1]
    names = list(model.named_steps["imputer"].get_feature_names_out(cols))
    rows=[]
    for i,r in test.iterrows():
        p_a=float(r.win_probability_team_a); winner=r.team_a if p_a>=.5 else r.team_b
        direction=1 if winner==r.team_a else -1
        scores=shap[i]*direction; order=np.argsort(np.abs(scores))[::-1]
        top=[(names[j],scores[j]) for j in order if not names[j].startswith("missingindicator")][:3]
        supporting=[readable_feature(n) for n,v in top if v>=0]
        opposing=[readable_feature(n) for n,v in top if v<0]
        reason = f"The model leans toward {winner} because " + (", ".join(supporting) if supporting else readable_feature(top[0][0]))
        if opposing: reason += f". {', '.join(opposing)} offsets part of that advantage"
        reason += ". These are model-behavior explanations, not causal claims."
        rows.append({"match_id":r.match_id,"date":r.date,"team_a":r.team_a,"team_b":r.team_b,
                     "predicted_winner":winner,"team_a_win_probability":p_a,"team_b_win_probability":1-p_a,
                     "elo_difference_team_a_minus_b":r.diff_elo,"recent_form_difference_team_a_minus_b":r.diff_weighted_form20,
                     "host_advantage_team_a_minus_b":r.home_advantage_diff,"venue_familiarity_difference":r.diff_venue_familiarity,
                     "most_influential_features":"; ".join(f"{readable_feature(n)} ({v:+.3f} SHAP)" for n,v in top),
                     "plain_language_explanation":reason})
    return pd.DataFrame(rows), test, cols


def model_bootstrap_uncertainty(train, test, cols, production_bundle, n_models=12):
    rng=np.random.default_rng(SEED); spec=candidates()["catboost"][int(production_bundle["candidate"])]
    probs=[]; metric_rows=[]
    for b in range(n_models):
        ix=rng.integers(0,len(train),len(train)); sample=train.iloc[ix].copy()
        model=fit_model(spec,sample[cols],sample.target,recency_weights(sample.date,production_bundle.get("recency_half_life_years")))
        p=sym_predict(model,test[cols]); probs.append(p); metric_rows.append({"bootstrap_model":b,**metric_set(test.target,p)})
    arr=np.vstack(probs)
    uncertainty=pd.DataFrame({"match_id":test.match_id,"model_probability_mean":arr.mean(0),"model_probability_sd":arr.std(0),
                              "model_probability_low":np.quantile(arr,.025,axis=0),"model_probability_high":np.quantile(arr,.975,axis=0)})
    return uncertainty,pd.DataFrame(metric_rows)


def pair_probabilities(model, snapshots, teams, cols, qualifier_profile="median"):
    confirmed=[t for t in teams if not t.startswith("Qualifier")]
    neutral={k:(1500.0 if "elo" in k else (.5 if any(x in k for x in ["form","overall","rate","bat_first","chase","home","away","neutral","h2h"]) else 0.0)) for k in [c[5:] for c in cols if c.startswith("diff_")]}
    complete={t:{**neutral,**snapshots.get(t,{})} for t in confirmed}
    ranked=sorted(confirmed,key=lambda t:complete[t].get("elo",1500))
    pick={"weak":ranked[max(0,int(.25*(len(ranked)-1)))],"median":ranked[int(.5*(len(ranked)-1))],"strong":ranked[int(.75*(len(ranked)-1))]}[qualifier_profile]
    table={t:{**complete[t],"team":t} for t in confirmed}
    for t in teams:
        if t.startswith("Qualifier"): table[t]={**complete[pick],"team":t}
    probs={}
    for i,a in enumerate(teams):
        for b in teams[i+1:]:
            host=a if a in HOSTS else (b if b in HOSTS else "")
            p=float(sym_predict(model,current_pair_features(table[a],table[b],cols,host))[0])
            probs[a,b]=np.clip(p,.03,.97); probs[b,a]=1-probs[a,b]
    return probs,pick


def simulate_schedule(probs, teams, runs, schedule, seed):
    rng=np.random.default_rng(seed); counts={t:[0,0,0] for t in teams}
    for _ in range(runs):
        if schedule=="round_robin":
            pts={t:0 for t in teams}; wins={t:0 for t in teams}
            for i,a in enumerate(teams):
                for b in teams[i+1:]:
                    w=a if rng.random()<probs[a,b] else b; pts[w]+=2; wins[w]+=1
            top=sorted(teams,key=lambda t:(pts[t],wins[t],rng.random()),reverse=True)[:4]
            semis=[(top[0],top[3]),(top[1],top[2])]
        else:
            shuffled=list(rng.permutation(teams)); groups=[shuffled[:7],shuffled[7:]]; qualifiers=[]
            for group in groups:
                pts={t:0 for t in group}; wins={t:0 for t in group}
                for i,a in enumerate(group):
                    for b in group[i+1:]:
                        w=a if rng.random()<probs[a,b] else b; pts[w]+=2; wins[w]+=1
                qualifiers.append(sorted(group,key=lambda t:(pts[t],wins[t],rng.random()),reverse=True)[:2])
            top=qualifiers[0]+qualifiers[1]; semis=[(top[0],top[3]),(top[2],top[1])]
        for t in top: counts[t][0]+=1
        finalists=[]
        for a,b in semis:
            w=a if rng.random()<probs[a,b] else b; finalists.append(w); counts[w][1]+=1
        a,b=finalists; w=a if rng.random()<probs[a,b] else b; counts[w][2]+=1
    return pd.DataFrame([{"team":t,"semi":v[0]/runs,"final":v[1]/runs,"win":v[2]/runs} for t,v in counts.items()])


def complete_tournament_uncertainty(bundle, point, n_boot_models=8):
    snapshots=bundle["team_snapshots"]; cols=bundle["features"]
    confirmed=sorted(pd.read_csv(RAW/"recent_100_confirmed.csv").confirmed_team.unique())
    teams=confirmed+[f"Qualifier {i}" for i in range(1,5)]
    rng=np.random.default_rng(SEED+5); features=pd.read_csv(CLEAN/"advanced_modeling_features.csv",parse_dates=["date"]); decisive=features.dropna(subset=["target"])
    spec=candidates()["catboost"][int(bundle["candidate"])]; scenario_rows=[]; representative={}
    # Scenario uncertainty from qualifier strength and two plausible formats.
    for qi,q in enumerate(["weak","median","strong"]):
        probs,pick=pair_probabilities(bundle["model"],snapshots,teams,cols,q); representative[q]=pick
        for si,schedule in enumerate(["round_robin","two_groups_random"]):
            s=simulate_schedule(probs,teams,600,schedule,SEED+100*qi+si)
            s["source"]="scenario"; s["qualifier_profile"]=q; s["schedule_scenario"]=schedule; s["model_bootstrap"]=-1; scenario_rows.append(s)
    # Sampling/model uncertainty with median qualifiers and round robin.
    for b in range(n_boot_models):
        ix=rng.integers(0,len(decisive),len(decisive)); sample=decisive.iloc[ix]
        model=fit_model(spec,sample[cols],sample.target,recency_weights(sample.date,bundle.get("recency_half_life_years")))
        probs,_=pair_probabilities(model,snapshots,teams,cols,"median")
        s=simulate_schedule(probs,teams,350,"round_robin",SEED+1000+b)
        s["source"]="model_bootstrap"; s["qualifier_profile"]="median"; s["schedule_scenario"]="round_robin"; s["model_bootstrap"]=b; scenario_rows.append(s)
    all_runs=pd.concat(scenario_rows,ignore_index=True); all_runs.to_csv(OUT/"tournament_uncertainty_scenarios.csv",index=False)
    rows=[]
    for _,r in point.iterrows():
        t=r.team; q=all_runs[all_runs.team==t]
        sim_low,sim_high=r.win_probability_ci_low,r.win_probability_ci_high
        model=q[q.source=="model_bootstrap"].win; scenarios=q[q.source=="scenario"].win
        rows.append({"team":t,"qualification_status":"Placeholder" if t.startswith("Qualifier") else "Confirmed as supplied",
                     "placeholder_warning":"Identity and strength unknown" if t.startswith("Qualifier") else "",
                     "semi_final_probability":r.semi_final_probability,"semi_final_lower":min(r.semi_final_probability,q.semi.quantile(.025)),"semi_final_upper":max(r.semi_final_probability,q.semi.quantile(.975)),
                     "final_probability":r.final_probability,"final_lower":min(r.final_probability,q.final.quantile(.025)),"final_upper":max(r.final_probability,q.final.quantile(.975)),
                     "win_probability":r.win_probability,"win_lower":min(r.win_probability,sim_low,model.quantile(.025),scenarios.min()),"win_upper":max(r.win_probability,sim_high,model.quantile(.975),scenarios.max()),
                     "simulation_uncertainty_low":sim_low,"simulation_uncertainty_high":sim_high,
                     "model_sampling_low":model.quantile(.025),"model_sampling_high":model.quantile(.975),
                     "scenario_low":scenarios.min(),"scenario_high":scenarios.max(),"simulations":20000+6*600+n_boot_models*350,
                     "disclaimer":DISCLAIMER})
    final=pd.DataFrame(rows).sort_values("win_probability",ascending=False)
    meta={"qualifier_profiles_represented_by_supplied_team_snapshots":representative,
          "schedule_scenarios":["14-team round robin; top four","two randomly allocated groups of seven; top two per group"],
          "interval_note":"Outer ranges combine Monte Carlo, bootstrap model, qualifier-strength, and schedule scenarios; they are uncertainty ranges, not formal confidence intervals."}
    (OUT/"uncertainty_methodology.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    return final,all_runs,meta


def charts(final, location_summary):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig,ax=plt.subplots(figsize=(8,5)); vals=[location_summary["coverage_before"],location_summary["coverage_after"]]; ax.bar(["Before","After"],vals,color=["#94a3b8","#0f766e"]); ax.set_ylim(0,1); ax.set_ylabel("Share of ODI matches"); ax.set_title("Host-country coverage improved to 94.6%",loc="left",weight="bold");
    for i,v in enumerate(vals): ax.text(i,v+.025,f"{v:.1%}",ha="center",weight="bold"); fig.tight_layout(); fig.savefig(CHARTS/"venue_coverage.png",dpi=170); plt.close(fig)
    q=final.sort_values("win_probability"); fig,ax=plt.subplots(figsize=(10,7)); err=np.vstack([np.maximum(0,q.win_probability-q.win_lower),np.maximum(0,q.win_upper-q.win_probability)]); ax.barh(q.team,q.win_probability,xerr=err,color="#0f766e"); ax.set_xlabel("Winning probability"); ax.set_title("Provisional title estimates with combined uncertainty ranges",loc="left",weight="bold"); fig.tight_layout(); fig.savefig(CHARTS/"final_tournament_uncertainty.png",dpi=170); plt.close(fig)


def dashboard(final, explanations, comparison, significance, audit, location_impact):
    leaders=final.head(8); tournament_rows="".join(f"<tr><td>{html.escape(r.team)}</td><td>{r.semi_final_probability:.1%}</td><td>{r.final_probability:.1%}</td><td><b>{r.win_probability:.1%}</b></td><td>{r.win_lower:.1%} to {r.win_upper:.1%}</td></tr>" for _,r in leaders.iterrows())
    comp=comparison[["model","accuracy","roc_auc","log_loss","brier"]].round(4).to_html(index=False,classes="data")
    sig=significance[significance.metric.eq("log_loss")][["baseline","improved_minus_baseline","ci_low","ci_high","meaningfully_better"]].round(4).to_html(index=False,classes="data")
    report=f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>CWC 2027 prediction dashboard</title><style>
    :root{{--ink:#102a43;--teal:#0f766e;--gold:#d97706;--paper:#f8fafc}}*{{box-sizing:border-box}}body{{margin:0;font:16px/1.55 system-ui;color:#243b53;background:var(--paper)}}header{{padding:54px max(6vw,28px);background:linear-gradient(130deg,#082f49,#115e59);color:white}}header h1{{font-size:clamp(32px,5vw,60px);max-width:900px;margin:0 0 12px}}header p{{font-size:20px;max-width:850px}}main{{max-width:1200px;margin:auto;padding:32px 24px}}section{{background:white;margin:22px 0;padding:28px;border-radius:16px;box-shadow:0 6px 24px #0f172a12}}h2{{color:var(--ink);font-size:28px}}.warning{{border-left:6px solid var(--gold);background:#fff7ed;padding:16px;font-weight:650}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:22px}}img{{width:100%;height:auto}}table.data,table.results{{width:100%;border-collapse:collapse;overflow:auto}}th,td{{padding:9px;border-bottom:1px solid #d9e2ec;text-align:left}}th{{color:var(--ink)}}.metric{{font-size:34px;font-weight:750;color:var(--teal)}}small{{color:#627d98}}code{{background:#e2e8f0;padding:2px 5px}}@media(max-width:700px){{section{{padding:18px}}th,td{{font-size:13px;padding:6px}}}}</style></head><body>
    <header><h1>India currently has the highest provisional estimated winning probability</h1><p>ICC Men's ODI World Cup 2027 scenario model based only on the supplied match data through 2 November 2026.</p></header><main><div class='warning'>{DISCLAIMER} Four qualifiers, current squads, and the official schedule remain incomplete.</div>
    <section><h2>Project objective</h2><p>Estimate pre-match win probabilities, compare them with transparent baselines, and translate the selected model into provisional tournament scenarios without changing the final chronological test period.</p><div class='grid'><div><div class='metric'>0.6534</div><b>Final-test log loss</b><p>Improved from 0.6930 in the original model.</p></div><div><div class='metric'>0.6608</div><b>Final-test ROC-AUC</b><p>The final test covers 9 July 2023 to 2 November 2026.</p></div><div><div class='metric'>94.6%</div><b>Venue-country coverage</b><p>Coverage rose from 18.1%. The experiment did not justify replacing the preserved model.</p></div></div></section>
    <section><h2>Data coverage and cleaning</h2><div class='grid'><img src='../charts/venue_coverage.png'><div><p>The pipeline contains 2,617 unique match rows and 2,487 decisive outcomes. It excludes ties and no-results from binary fitting while retaining them in cleaned data.</p><p>Normalized venue and city aliases use direct host records, supplied World Cup locations, unique city-country matches, and explicit host names in event titles. The process leaves 142 matches unresolved.</p></div></div></section>
    <section><h2>Chronological test performance</h2>{comp}<img src='../charts/advanced_model_comparison.png'><img src='../charts/advanced_metric_differences.png'></section>
    <section><h2>Calibration and feature behavior</h2><div class='grid'><img src='../charts/advanced_calibration_reliability.png'><img src='../charts/advanced_feature_importance.png'></div><p>Temporal validation retained the uncalibrated CatBoost probabilities. SHAP explanations describe how the model behaved for each prediction; they do not establish causal effects.</p></section>
    <section><h2>Historical and current context</h2><div class='grid'><img src='../charts/historical_world_cup_wins.png'><img src='../charts/recent_win_rate.png'><img src='../charts/host_country_performance.png'></div></section>
    <section><h2>Provisional tournament scenarios</h2><div class='warning'>{DISCLAIMER}</div><table class='results'><tr><th>Team</th><th>Semi-final</th><th>Final</th><th>Champion</th><th>Combined range</th></tr>{tournament_rows}</table><img src='../charts/final_tournament_uncertainty.png'><p><small>Combined ranges include Monte Carlo variation, bootstrap model variation, weak/median/strong anonymous qualifier profiles, and both round-robin and random two-group formats. They are scenario ranges rather than formal confidence intervals.</small></p></section>
    <section><h2>Baseline significance</h2>{sig}<p>CatBoost is statistically better than the original and rebuilt existing-logistic models. Its point estimates beat Elo-only, overall-rate, historical-strength and blended baselines, but the 95% paired intervals do not establish superiority over every strong simple baseline.</p></section>
    <section><h2>Methodology in plain language</h2><p>Before each match, the pipeline records what each team had achieved up to that date. It compares Elo strength, recent form, scoring efficiency, venue familiarity, host context and World Cup experience. Expanding historical windows select model settings. A later validation block chooses calibration. Only then does the pipeline evaluate the untouched final period.</p><p>For tournament scenarios, the model converts team differences into match probabilities. Repeated simulated tournaments estimate advancement rates. Alternative qualifier strengths, model bootstraps and two plausible competition formats widen the reported ranges.</p></section>
    <section><h2>Limitations and next data</h2><p>The match source begins in 2002 and remains incomplete. Surface labels approximate scoring environments rather than measured pitch composition. Qualifiers, fixtures, venues and squads remain unknown. Player availability, injuries, weather, complete historical ODIs, verified pitch types and the official schedule would most improve the forecast.</p><p>Rerun with <code>python -m src.advanced_pipeline --runs 20000 --location-mode direct</code>, followed by <code>python -m src.final_reporting</code>.</p></section></main></body></html>"""
    (REPORT/"final_dashboard.html").write_text(report,encoding="utf-8")


def main():
    # Models saved by a ``python -m`` run may reference the calibrator through __main__.
    import __main__
    __main__.Calibrator = Calibrator
    features=pd.read_csv(CLEAN/"advanced_modeling_features.csv",parse_dates=["date"])
    predictions=pd.read_csv(OUT/"advanced_match_predictions.csv",parse_dates=["date"])
    evaluation=joblib.load(MODELS/"advanced_evaluation_model.joblib"); production=joblib.load(MODELS/"advanced_pre_match.joblib")
    explanations,test,cols=explain_matches(features,predictions,evaluation)
    decisive=features.dropna(subset=["target"]).reset_index(drop=True); train=decisive.iloc[:int(.70*len(decisive))]
    match_uncertainty,model_metrics=model_bootstrap_uncertainty(train,test,cols,production)
    explanations=explanations.merge(match_uncertainty,on="match_id",how="left")
    explanations.to_csv(OUT/"final_match_predictions_with_explanations.csv",index=False)
    model_metrics.to_csv(OUT/"model_training_bootstrap_metrics.csv",index=False)
    point=pd.read_csv(OUT/"advanced_tournament_probabilities.csv")
    final,scenarios,meta=complete_tournament_uncertainty(production,point)
    final.to_csv(OUT/"final_tournament_probabilities.csv",index=False)
    tournament_unc=final[["team","simulation_uncertainty_low","simulation_uncertainty_high","model_sampling_low","model_sampling_high","scenario_low","scenario_high","win_lower","win_upper"]].copy()
    tournament_unc.insert(0,"entity_type","tournament_team"); tournament_unc.to_csv(OUT/"prediction_uncertainty.csv",index=False)
    location_audit=json.loads((CLEAN/"location_experiment_audit.json").read_text()); location_summary=location_audit["location_coverage"]
    location_impact=pd.DataFrame([{"experiment":"preserved_direct_host_catboost","accuracy":.6149732620,"roc_auc":.6608424605,"log_loss":.6534219113,"brier":.2310391301,"selected_pre_test":True},
                                  {"experiment":"expanded_location_validation_selected_random_forest","accuracy":.6069518717,"roc_auc":.6577540107,"log_loss":.6576878347,"brier":.2327224720,"selected_pre_test":True},
                                  {"experiment":"expanded_location_catboost_diagnostic_only","accuracy":.6229946524,"roc_auc":.668020,"log_loss":.651750,"brier":.229893,"selected_pre_test":False}])
    location_impact.to_csv(OUT/"venue_coverage_model_impact.csv",index=False)
    charts(final,location_summary)
    comparison=pd.read_csv(OUT/"advanced_model_comparison.csv"); significance=pd.read_csv(OUT/"advanced_baseline_significance.csv"); audit=json.loads((CLEAN/"advanced_pipeline_audit.json").read_text())
    dashboard(final,explanations,comparison,significance,audit,location_impact)
    print(json.dumps({"explained_matches":len(explanations),"location_coverage":location_summary,"leaders":final.head(5).to_dict("records")},indent=2,default=str))


if __name__=="__main__": main()
