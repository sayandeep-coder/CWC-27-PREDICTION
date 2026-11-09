# ICC Men's ODI World Cup 2027 prediction

A reproducible, leakage-controlled machine-learning project built only from the four supplied CSV files. The advanced pipeline adds dynamic Elo systems, era-normalized scoring features, expanding-window tuning, validation-only calibration, paired bootstrap uncertainty, and a 20,000-iteration provisional tournament simulation.

## Advanced pipeline

```powershell
.\.venv\Scripts\python.exe -m src.advanced_pipeline --runs 20000
.\.venv\Scripts\python.exe -m src.final_reporting
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The advanced run preserves the final chronological test interval (9 July 2023 through 2 November 2026). Model and hyperparameter selection use expanding windows within the earlier training period. Calibration is fit and selected only inside the validation period. The served probability is explicitly symmetrized so reversing team A and team B produces the complementary probability.

The selected CatBoost model achieved 0.6534 log loss, 0.2310 Brier score and 0.6608 ROC-AUC on the untouched final test, compared with 0.6930, 0.2483 and 0.6067 for the original reported model. Bootstrap tests find the improvement significant against the original and rebuilt existing-logistic model, but not against every stronger simple baseline. See the advanced report for the complete qualification.

## Headline methodology

- Match-level training source: 2,617 Cricsheet-available men's ODIs dated 27 June 2002 to 2 November 2026.
- Historical World Cup source: 502 matches from 1975 through the 2023 final.
- Current-team snapshots: the supplied last-100 file for ten confirmed teams.
- Host analysis: supplied South Africa, Zimbabwe and Namibia team/venue records.
- Binary training excludes ties and no-results but retains them in cleaned exports.
- Every rolling feature is calculated before updating state with the current match.
- Splits are chronological: first 70% train, next 15% validation, final 15% test.
- Model selection uses validation log loss plus Brier score. The final test period is untouched during selection.

## Features

Overall and exponentially weighted win rate; form over 5, 10, 20 and 50 matches; prior World Cup rate; head-to-head and venue rate; host-country rate; batting-first/chasing rate; recent runs and wickets for/against; prior opposition strength; rest days; and host advantage. Post-toss models additionally use toss winner and toss decision. Features are expressed as team-A minus team-B differences with a stable, hash-based orientation.

## Setup and run

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.pipeline --runs 10000
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The project-local environment used for this delivery is already present. Running the pipeline recreates all cleaned data, models, predictions, charts, and the HTML report.

## Outputs

- `models/advanced_pre_match.joblib` and `advanced_evaluation_model.joblib`: final production and chronological-evaluation model artifacts.
- `outputs/advanced_model_comparison.csv`, `advanced_metric_confidence_intervals.csv`, and `advanced_baseline_significance.csv`.
- `outputs/advanced_subgroup_evaluation.csv` and `advanced_calibration_test_diagnostics.csv`.
- `outputs/final_tournament_probabilities.csv`: central advancement estimates plus combined sampling, model, simulation, qualifier and schedule ranges.
- `outputs/final_match_predictions_with_explanations.csv`: both-team probabilities, SHAP feature contributions and plain-language behavior explanations.
- `outputs/prediction_uncertainty.csv` and `tournament_uncertainty_scenarios.csv`: final uncertainty components and scenario runs.
- `data/cleaned/venue_country_mapping.csv` and `unresolved_venues.csv`: reusable, provenance-labelled venue normalization outputs.
- `report/final_dashboard.html`: responsive presentation-ready dashboard.
- `presentation/CWC_2027_Prediction_Project_Final_v2.pptx`: validated 12-slide project presentation.

## Venue coverage experiment

The normalized venue mapping increased location coverage from 473 of 2,617 matches (18.1%) to 2,475 (94.6%). It uses direct host match IDs, supplied World Cup venue-country records, unique city-country matches, explicit host names in event titles and unanimous mappings learned from those resolved records. It leaves 142 matches unresolved rather than guessing.

Expanded location features did not justify replacing the preserved model under the predeclared validation selection. Validation selected random forest, which scored 0.6577 log loss and 0.2327 Brier on the unchanged final test, compared with 0.6534 and 0.2310 for the preserved CatBoost. An expanded-location CatBoost diagnostic scored slightly better, but selecting it after reviewing the final test would violate the methodology.

## Uncertainty interpretation

The final tournament file distinguishes the original Monte Carlo interval, bootstrap model and sampling variation, and scenario variation from anonymous weak, median and strong qualifier profiles plus two provisional competition formats. The outer bounds are scenario ranges rather than formal confidence intervals. No official schedule, qualifier identity, squad or injury information is invented.
- `data/cleaned/data_quality_report.json`: coverage, missingness and duplicate audit.

## Updating qualifiers or using an official schedule

Replace the `Qualifier 1`–`Qualifier 4` rows created by `current_team_table` with identified teams after adding their ODI histories to the supplied sources. Rerun the pipeline. The current simulator intentionally does **not** invent fixtures: it assumes a 14-team round robin and standard top-four knockouts. Once a verified schedule is supplied, replace the round-robin loop in `simulate` with those fixtures and pass fixture venues into `pair_features`.

## Cleaning decisions and limitations

- Column names are normalized; team, venue, city and country text is trimmed; known USA/UAE aliases are standardized; dates are parsed strictly with invalid values recorded.
- Exact duplicates are removed. Match IDs remain the authoritative key for match-level files; the host file intentionally has two team-perspective rows per match.
- Team-centric recent data is not appended to match-level history, because the same match may appear once per confirmed team. This prevents double counting and cross-split leakage.
- Cricsheet-available coverage starts in 2002 and is not guaranteed to contain every ODI.
- The provisional simulation uses neutral values where qualifier identities or current snapshot statistics are unavailable. It does not model net run rate, weather, injuries, squads, or an unsupplied official schedule.
- Model probabilities are estimates from supplied historical data, not forecasts based on invented results or fixtures.

## Project layout

`data/` raw and cleaned data; `src/` pipeline; `models/` serialized pipelines; `outputs/` CSV/JSON results; `charts/` PNG figures; `report/` HTML report; `tests/` validation suite.
