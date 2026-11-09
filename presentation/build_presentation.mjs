import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "C:/Users/User/CWC-27-PREDICTION";
const buildDir = path.join(workspaceDir, ".presentation-build");
const outputDir = path.join(workspaceDir, "presentation");
const skillDir = "C:/Users/User/.codex/plugins/cache/openai-primary-runtime/presentations/26.1105.12137/skills/presentations";
const pythonExe = "C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1105-1297-0_26.1105.1297.0_x64__3k8sg7r9htsxt/dependencies/python/python.exe";
const finalPath = path.join(outputDir, "CWC_2027_Prediction_Project_Final_v2.pptx");
const font = "Arial";
const DISCLAIMER = "Provisional scenario-based estimates—not an official tournament forecast.";
const C = { navy: "#082F49", teal: "#0F766E", gold: "#D97706", ink: "#102A43", slate: "#52667A", light: "#F3F7FA", white: "#FFFFFF", line: "#D9E2EC", red: "#B91C1C" };

async function bytes(rel) {
  const b = await fs.readFile(path.join(workspaceDir, rel));
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
}
function box(slide, left, top, width, height, fill="none") {
  return slide.shapes.add({geometry:"rect", position:{left,top,width,height}, fill, line:{fill:"none",width:0}});
}
function text(slide, value, left, top, width, height, size=24, color=C.ink, bold=false, align="left") {
  const s=slide.shapes.add({geometry:"textbox",position:{left,top,width,height},fill:"none",line:{fill:"none",width:0}});
  s.text=value; s.text.style={fontFamily:font,fontSize:size,color,bold,alignment:align,verticalAlignment:"middle",autoFit:"shrinkText"}; return s;
}
function title(slide, value, number) {
  text(slide,value,64,38,1080,56,34,C.ink,true);
  text(slide,String(number).padStart(2,"0"),1160,42,54,40,15,C.teal,true,"right");
  box(slide,64,100,1150,3,C.teal);
}
function footer(slide, note="Source: supplied project data and final model outputs") {
  text(slide,note,64,680,1050,22,11,C.slate,false);
}
async function image(slide, rel, left, top, width, height, alt) {
  slide.images.add({blob:await bytes(rel),contentType:"image/png",alt,fit:"contain",position:{left,top,width,height}});
}
function note(slide, value, left, top, width, height, color=C.gold) {
  box(slide,left,top,6,height,color); text(slide,value,left+18,top,width-18,height,18,C.ink,true);
}
function metric(slide, value, label, left, top, color=C.teal) {
  text(slide,value,left,top,250,64,38,color,true); text(slide,label,left,top+58,260,54,17,C.slate,false);
}

await fs.mkdir(buildDir,{recursive:true}); await fs.mkdir(outputDir,{recursive:true});
const p=Presentation.create({slideSize:{width:1280,height:720}});

// 1
{
  const s=p.slides.add(); s.background.fill=C.light; box(s,0,0,24,720,C.teal);
  text(s,"ICC Men's ODI World Cup 2027",72,92,590,70,46,C.navy,true);
  text(s,"Prediction project",72,164,500,54,34,C.teal,true);
  text(s,"A leakage-controlled model comparison and provisional tournament scenario analysis",72,242,520,104,22,C.slate,false);
  note(s,"Provisional scenario-based estimates—not an official tournament forecast.",72,404,520,62,C.gold);
  text(s,"Data through 2 November 2026",72,510,420,34,17,C.slate,true);
  await image(s,"charts/advanced_winning_probability.png",650,70,570,570,"Provisional winning probabilities");
  footer(s);
}
// 2
{
  const s=p.slides.add(); s.background.fill=C.white; title(s,"Data sources and coverage",2);
  metric(s,"2,617","match-level men's ODIs",76,134); metric(s,"502","World Cup matches since 1975",382,134,C.navy); metric(s,"1,000","recent team records",688,134,C.gold); metric(s,"946","host-country team rows",994,134,C.teal);
  await image(s,"charts/venue_coverage.png",78,300,520,330,"Venue country coverage before and after mapping");
  text(s,"The match source begins in 2002 and does not cover every ODI. The model retains ties and no-results in cleaned data but excludes them from binary fitting.",650,328,500,150,22,C.ink,false);
  note(s,"The final test contains 374 decisive matches from 9 July 2023 to 2 November 2026.",650,520,500,64,C.teal);
  footer(s);
}
// 3
{
  const s=p.slides.add(); s.background.fill=C.light; title(s,"Data cleaning and venue normalization",3);
  text(s,"Cleaning decisions",72,130,340,38,24,C.navy,true);
  text(s,"• Team and venue aliases standardized\n• Dates parsed and invalid values audited\n• One training row retained per match ID\n• Team-centric recent rows excluded from training\n• Unresolved venue countries left blank",72,182,470,260,21,C.ink,false);
  await image(s,"charts/venue_coverage.png",594,126,590,350,"Venue coverage improvement");
  note(s,"Coverage improved from 473 to 2,475 matches. 142 matches remain unresolved.",594,500,590,66,C.gold);
  text(s,"The expanded mapping experiment did not justify replacing the preserved CatBoost model.",72,510,450,74,21,C.ink,true);
  footer(s,"Source: supplied venue, city, country and event-name fields; unresolved locations were not guessed");
}
// 4
{
  const s=p.slides.add(); s.background.fill=C.white; title(s,"Feature engineering",4);
  text(s,"Team strength",72,136,250,34,23,C.teal,true); text(s,"Dynamic Elo, surface Elo, host-country Elo and opponent-adjusted form",72,178,300,126,19,C.ink,false);
  text(s,"Recent performance",382,136,250,34,23,C.teal,true); text(s,"Weighted form over 5, 10, 20 and 50 matches, recent opponent strength and rest days",382,178,300,126,19,C.ink,false);
  text(s,"Scoring efficiency",692,136,250,34,23,C.teal,true); text(s,"Era-adjusted runs per wicket, run rate, economy and net run-rate proxy",692,178,300,126,19,C.ink,false);
  text(s,"Context",1002,136,190,34,23,C.teal,true); text(s,"Home, away, neutral, venue familiarity, head-to-head and World Cup experience",1002,178,200,150,19,C.ink,false);
  await image(s,"charts/advanced_feature_importance.png",185,344,910,300,"Permutation feature importance"); footer(s);
}
// 5
{
  const s=p.slides.add(); s.background.fill=C.light; title(s,"Leakage prevention and chronological testing",5);
  box(s,100,210,290,120,C.navy); box(s,495,210,290,120,C.teal); box(s,890,210,290,120,C.gold);
  text(s,"TRAIN",100,214,290,36,17,C.white,true,"center"); text(s,"Through 11 Aug 2019",100,250,290,52,24,C.white,true,"center");
  text(s,"VALIDATION",495,214,290,36,17,C.white,true,"center"); text(s,"Through 7 Jul 2023",495,250,290,52,24,C.white,true,"center");
  text(s,"FINAL TEST",890,214,290,36,17,C.white,true,"center"); text(s,"9 Jul 2023 to 2 Nov 2026",890,250,290,52,22,C.white,true,"center");
  text(s,"Expanding windows tune models inside the training period. Validation selects calibration. The pipeline opens the final test only after those choices are fixed.",154,394,970,90,24,C.ink,true,"center");
  note(s,"Rolling features capture team state before the current match. Served probabilities are invariant to team order.",250,530,780,64,C.teal); footer(s);
}
// 6
{
  const s=p.slides.add(); s.background.fill=C.white; title(s,"Model comparison",6);
  await image(s,"charts/advanced_model_comparison.png",70,120,720,520,"Chronological test log loss by model");
  text(s,"CatBoost recorded the lowest final-test log loss",830,160,360,80,28,C.navy,true);
  metric(s,"0.6534","CatBoost log loss",830,276); metric(s,"0.6697","Elo-only log loss",830,394,C.slate);
  text(s,"Lower log loss indicates better probabilistic predictions.",830,514,350,72,19,C.ink,false); footer(s);
}
// 7
{
  const s=p.slides.add(); s.background.fill=C.light; title(s,"Final model performance",7);
  metric(s,"61.50%","Accuracy",84,142); metric(s,"0.6608","ROC-AUC",382,142,C.navy); metric(s,"0.6534","Log loss",680,142,C.teal); metric(s,"0.2310","Brier score",978,142,C.gold);
  await image(s,"charts/advanced_calibration_reliability.png",82,300,540,330,"Reliability curve");
  await image(s,"charts/advanced_metric_differences.png",648,300,550,330,"Paired bootstrap metric differences");
  footer(s);
}
// 8
{
  const s=p.slides.add(); s.background.fill=C.white; title(s,"Prediction explanations",8);
  await image(s,"charts/advanced_feature_importance.png",70,128,650,490,"Advanced model feature importance");
  text(s,"Every test-match output includes",770,142,400,40,25,C.navy,true);
  text(s,"• Winner and both team probabilities\n• Elo and recent-form differences\n• Host and venue context\n• Three largest SHAP contributions\n• Plain-language model explanation",770,200,400,250,21,C.ink,false);
  note(s,"SHAP values describe model behavior. They do not prove that a feature caused a result.",770,500,400,76,C.gold); footer(s);
}
// 9
{
  const s=p.slides.add(); s.background.fill=C.light; title(s,"Provisional tournament probabilities",9);
  await image(s,"charts/final_tournament_uncertainty.png",70,120,760,520,"Provisional tournament title probabilities with uncertainty");
  text(s,"India currently has the highest provisional estimated winning probability",860,154,340,130,27,C.navy,true);
  metric(s,"32.1%","Central winning estimate",860,324); text(s,"Combined scenario range",860,448,260,30,17,C.slate,true); text(s,"18.8% to 36.3%",860,482,300,52,29,C.teal,true);
  note(s,DISCLAIMER,860,566,330,62,C.gold); footer(s);
}
// 10
{
  const s=p.slides.add(); s.background.fill=C.white; title(s,"Prediction uncertainty and limitations",10);
  text(s,"Uncertainty sources",76,136,330,40,25,C.navy,true);
  text(s,"Sampling variation from retraining on bootstrap samples\n\nModel variation in match probabilities\n\nMonte Carlo tournament variation\n\nQualifier-strength and schedule scenarios",76,196,500,310,21,C.ink,false);
  text(s,"Material limitations",660,136,330,40,25,C.navy,true);
  text(s,"Four qualifiers remain unknown\n\nOfficial fixtures and venues are unavailable\n\nCurrent squads and injuries are unavailable\n\nSurface labels approximate scoring conditions\n\nHistorical ODI coverage is incomplete",660,196,500,330,21,C.ink,false);
  note(s,"Reported outer ranges are scenario ranges, not formal confidence intervals.",280,562,720,64,C.gold); footer(s);
}
// 11
{
  const s=p.slides.add(); s.background.fill=C.light; title(s,"Future improvements and rerun process",11);
  text(s,"Highest-value additional data",78,142,420,40,25,C.navy,true);
  text(s,"Complete historical ODI coverage\nVerified venue countries and pitch types\nPlayer and squad availability\nInjuries and weather\nOfficial 2027 fixtures",78,200,420,270,22,C.ink,false);
  text(s,"Rerun inputs",688,142,300,40,25,C.navy,true);
  text(s,"Final 14 teams\nOfficial groups or fixtures\nConfirmed venues\nCurrent squads",688,200,340,220,22,C.ink,false);
  note(s,"The pipeline exposes venue mappings, model artifacts, prediction outputs and simulation scenarios as reusable files.",250,522,780,78,C.teal); footer(s);
}
// 12
{
  const s=p.slides.add(); s.background.fill=C.navy;
  text(s,"Final conclusion",72,74,500,56,34,C.white,true);
  text(s,"The preserved CatBoost model improves probability quality over the original project while retaining the same chronological final test.",72,174,690,140,31,C.white,true);
  text(s,"Its advantage is statistically clear against the original and rebuilt logistic models. The evidence does not establish superiority over every strong simple baseline.",72,352,690,130,23,"#D9E2EC",false);
  box(s,72,536,6,82,C.gold); text(s,"India currently has the highest provisional estimated winning probability. Qualifiers, squads and the official schedule remain incomplete.",92,536,1020,82,18,C.white,true);
  text(s,"CWC 2027 prediction project",930,92,280,34,17,"#A7C7C4",true,"right"); text(s,"Source: supplied project data and final model outputs",72,680,680,22,11,"#A7C7C4",false);
}

for (const [i,s] of p.slides.items.entries()) {
  const png=await p.export({slide:s,format:"png",scale:1});
  await fs.writeFile(path.join(buildDir,`slide-${String(i+1).padStart(2,"0")}.png`),new Uint8Array(await png.arrayBuffer()));
}
const montage=await p.export({format:"webp",montage:true}); await fs.writeFile(path.join(buildDir,"montage.webp"),new Uint8Array(await montage.arrayBuffer()));

const { finalizePresentation } = await import(pathToFileURL(path.join(skillDir,"container_tools/artifact_tool_utils.mjs")).href);
const candidatePath=path.join(buildDir,"candidate.pptx"); await (await PresentationFile.exportPptx(p)).save(candidatePath);
const requirements={explicitTotalSlideCount:12,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],materializeLiteralChartWorkbooks:false};
await finalizePresentation({...requirements,workspaceDir,candidatePath,finalPath,pythonExecutable:pythonExe,
  integrityValidatorPath:path.join(skillDir,"container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath:path.join(skillDir,"container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs:["--expected-slide-size-emu","12192000,6858000","--validate-heading-fit"],
  fontPolicy:{basis:"design",families:[font]},verifyArtifactToolImport:true,
  receiptPath:path.join(buildDir,"presentation_final_v2.validation.json")});
console.log(finalPath);
