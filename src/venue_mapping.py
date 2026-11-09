"""Create reusable, provenance-labelled venue and host-country mappings."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "cleaned"

COUNTRY_ALIASES = {
    "UAE": "United Arab Emirates", "USA": "United States of America",
    "Scotland": "Scotland", "Netherlands": "Netherlands", "Hong Kong": "Hong Kong",
    "West Indies": "West Indies", "England": "England", "Ireland": "Ireland",
}

# Renames observed as separate venue strings in the supplied ODI file.
VENUE_ALIASES = {
    "arun jaitley stadium": "Feroz Shah Kotla",
    "feroz shah kotla delhi": "Feroz Shah Kotla",
    "narendra modi stadium ahmedabad": "Sardar Patel Stadium",
    "sardar patel stadium motera ahmedabad": "Sardar Patel Stadium",
    "national bank cricket arena karachi": "National Stadium Karachi",
    "national stadium karachi": "National Stadium Karachi",
    "sky stadium wellington": "Wellington Stadium",
    "westpac stadium wellington": "Wellington Stadium",
    "st george s park gqeberha": "St George's Park",
    "st george s park port elizabeth": "St George's Park",
}

CITY_ALIASES = {"port elizabeth": "gqeberha", "chittagong": "chattogram"}


def norm(value):
    if pd.isna(value): return ""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode().casefold()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def canonical_city(value):
    x = norm(value); return CITY_ALIASES.get(x, x)


def canonical_venue(value):
    x = norm(value)
    if x in VENUE_ALIASES: return VENUE_ALIASES[x]
    return str(value).strip() if pd.notna(value) else "Unknown"


def event_country(event, candidates):
    e = norm(event)
    if not e: return ""
    found = []
    for country in sorted(candidates, key=len, reverse=True):
        c = norm(country)
        if f"tour of {c}" in e or f" in {c} " in f" {e} " or e.endswith(f" in {c}"):
            found.append(country)
    # Longer names take precedence, but ambiguity remains unresolved.
    unique = list(dict.fromkeys(found))
    return unique[0] if len(unique) == 1 else ""


def _unique_lookup(frame, key, country):
    x = frame[(frame[key] != "") & (frame[country] != "")]
    counts = x.groupby(key)[country].nunique()
    valid = counts[counts == 1].index
    return x[x[key].isin(valid)].drop_duplicates(key).set_index(key)[country].to_dict()


def build_venue_mapping(odi, wc, host):
    known = []
    for df, country_col, source in [(wc, "country", "world_cup_file"), (host, "host_country", "host_file")]:
        q = df[["venue", "city", country_col]].copy()
        q["venue_key"] = q.venue.map(norm); q["city_key"] = q.city.map(canonical_city)
        q["country"] = q[country_col].astype("string").fillna("").str.strip().replace(COUNTRY_ALIASES)
        q["source"] = source; known.append(q[["venue_key", "city_key", "country", "source"]])
    known = pd.concat(known, ignore_index=True)
    venue_lookup = _unique_lookup(known, "venue_key", "country")
    city_lookup = _unique_lookup(known, "city_key", "country")
    direct = host.drop_duplicates("match_id").set_index("match_id")["host_country"].astype(str).to_dict()
    candidates = sorted(set(odi.team1.dropna()) | set(odi.team2.dropna()) | set(wc.country.dropna()) | set(host.host_country.dropna()))
    candidates = [COUNTRY_ALIASES.get(c, c) for c in candidates]

    rows = []
    for _, m in odi.iterrows():
        vk, ck = norm(m.venue), canonical_city(m.city)
        country, method = "", "unresolved"
        if m.match_id in direct:
            country, method = direct[m.match_id], "direct_host_match_id"
        elif vk in venue_lookup:
            country, method = venue_lookup[vk], "supplied_venue_country"
        elif ck and ck in city_lookup:
            country, method = city_lookup[ck], "supplied_unique_city_country"
        else:
            ec = event_country(m.event_name, candidates)
            if ec: country, method = ec, "event_name_host"
        rows.append({"match_id": m.match_id, "venue": m.venue, "city": m.city,
                     "venue_normalized": vk, "city_normalized": ck,
                     "canonical_venue": canonical_venue(m.venue), "country": country,
                     "mapping_method": method, "event_name": m.event_name})
    match_map = pd.DataFrame(rows)

    # Learn reusable venue/city mappings only from already resolved matches with unanimous countries.
    resolved = match_map[match_map.country.ne("")]
    learned_venue = _unique_lookup(resolved, "venue_normalized", "country")
    learned_city = _unique_lookup(resolved, "city_normalized", "country")
    for idx, r in match_map[match_map.country.eq("")].iterrows():
        if r.venue_normalized in learned_venue:
            match_map.loc[idx, ["country", "mapping_method"]] = [learned_venue[r.venue_normalized], "consistent_resolved_venue"]
        elif r.city_normalized and r.city_normalized in learned_city:
            match_map.loc[idx, ["country", "mapping_method"]] = [learned_city[r.city_normalized], "consistent_resolved_city"]
    match_map["host_team"] = match_map.country.replace({"Wales": "England"})

    reusable = (match_map.groupby(["venue", "city", "venue_normalized", "city_normalized", "canonical_venue"], dropna=False)
                .agg(country=("country", lambda x: x[x.ne("")].mode().iat[0] if x.ne("").any() and x[x.ne("")].nunique()==1 else ""),
                     host_team=("host_team", lambda x: x[x.ne("")].mode().iat[0] if x.ne("").any() and x[x.ne("")].nunique()==1 else ""),
                     mapping_method=("mapping_method", lambda x: ";".join(sorted(set(x)))), matches=("match_id", "size"))
                .reset_index().sort_values(["country", "canonical_venue"], ascending=[False, True]))
    reusable.to_csv(CLEAN / "venue_country_mapping.csv", index=False)
    unresolved = (match_map[match_map.country.eq("")].groupby(["venue", "city", "venue_normalized", "city_normalized"], dropna=False)
                  .agg(matches=("match_id", "size"), sample_event=("event_name", "first")).reset_index().sort_values("matches", ascending=False))
    unresolved.to_csv(CLEAN / "unresolved_venues.csv", index=False)
    match_map.to_csv(CLEAN / "match_host_country_mapping.csv", index=False)
    before = int(odi.match_id.isin(direct).sum()); after = int(match_map.country.ne("").sum())
    summary = {"matches": len(odi), "covered_before": before, "covered_after": after,
               "coverage_before": before/len(odi), "coverage_after": after/len(odi),
               "unresolved_matches": len(odi)-after, "unique_unresolved_venue_city_pairs": len(unresolved)}
    return match_map, reusable, unresolved, summary

