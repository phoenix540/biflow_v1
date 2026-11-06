"""BIFlow - BI Analyst Agent.

Transforme les KPI en analyses : tendances, baisses, anomalies, explications et recommandations.

Deux couches :
  1. DÉTECTION (Python, déterministe) : chaque constat devient un "signal" numéroté (S1, S2...)
     avec ses chiffres exacts (facts). Rien n'est inventé à cette étape.
  2. EXPLICATION (LLM, facultatif) : le LLM rédige les insights à partir des signaux, en citant
     leurs identifiants. Python vérifie ensuite que les signaux cités existent et que les nombres
     écrits se retrouvent dans les faits. Sans LLM (--no-llm, quota, réseau), des insights sont
     générés par règles : le pipeline ne bloque jamais.

Entrées : data/kpi/kpi_results.json, data/curated/fact_orders.csv, data/curated/dim_date.csv
Sorties : data/analyst/analyst_insights.json, data/analyst/analyst_report.md
"""
import hashlib
import json
import os
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
# Certains modèles Gemini ignorent "temperature" et le signalent à chaque appel : avertissement sans effet
warnings.filterwarnings("ignore", message=".*fixed sampling defaults.*")

CURATED = ROOT / "data" / "curated"
KPI_FILE = ROOT / "data" / "kpi" / "kpi_results.json"
OUT = ROOT / "data" / "analyst"
INSIGHTS_FILE = OUT / "analyst_insights.json"

# Seuils de détection (choix de conception, à justifier dans le rapport)
THRESHOLDS = {
    "month_deviation_pct": 25,     # mois anormal si écart >= 25 % vs médiane des 3 mois fiables précédents
    "daily_robust_z": 3.5,         # jour anormal si score robuste (médiane/MAD sur 28 jours) >= 3,5
    "daily_window_days": 28,
    "max_daily_episodes": 5,
    "episode_gap_days": 3,         # jours atypiques à <= 3 jours d'écart = un même épisode
    "event_window_days": 3,        # un événement à +/- 3 jours peut expliquer un épisode (ex. Cyber Monday)
    "state_min_orders": 500,       # États trop petits ignorés (bruit)
    "state_late_ratio": 1.5,       # État en difficulté si retard >= 1,5 x la moyenne nationale
    "latest_month_drop_pct": -10,  # alerte si le dernier mois fiable baisse de plus de 10 % sur un mois
}
SEVERITY_RANK = {"alert": 0, "warning": 1, "opportunity": 2, "info": 3}


# ============================================================================
# 1. DÉTECTION
# ============================================================================
class Signals:
    def __init__(self):
        self.items = []

    def add(self, kind, severity, title, summary, facts, hint=""):
        sid = f"S{len(self.items) + 1}"
        self.items.append({"id": sid, "kind": kind, "severity": severity, "title": title,
                           "summary": summary, "facts": facts, "recommendation_hint": hint})
        return sid


def r1(x):
    return None if x is None or pd.isna(x) else round(float(x), 1)


def load_inputs():
    if not KPI_FILE.exists():
        raise FileNotFoundError(f"KPI results missing: {KPI_FILE} (run the KPI agent first)")
    kpis = json.loads(KPI_FILE.read_text(encoding="utf-8"))
    if "headline" not in kpis:
        raise ValueError("kpi_results.json has an old format: rerun the KPI agent.")

    orders = pd.read_csv(CURATED / "fact_orders.csv",
                         usecols=["order_id", "purchase_date", "is_revenue_order", "revenue_brl"])
    orders = orders[orders["is_revenue_order"].astype(str).str.lower() == "true"]
    orders["purchase_date"] = pd.to_datetime(orders["purchase_date"], errors="coerce")
    daily = orders.groupby("purchase_date").agg(revenue=("revenue_brl", "sum"), orders=("order_id", "size"))

    cal = pd.read_csv(CURATED / "dim_date.csv", usecols=["date", "is_reliable_month", "event"])
    cal["date"] = pd.to_datetime(cal["date"])
    cal["is_reliable_month"] = cal["is_reliable_month"].astype(str).str.lower() == "true"
    cal["event"] = cal["event"].fillna("")
    cal = cal.set_index("date").join(daily, how="left").fillna({"revenue": 0.0, "orders": 0})
    cal["year_month"] = cal.index.to_period("M").astype(str)
    return kpis, cal


def detect_trend(k, sig):
    rows = [m for m in k["breakdowns"]["monthly"] if m["yoy_growth_pct"] is not None]
    y = k["breakdowns"]["yoy_summary"]
    if len(rows) >= 3:
        first, last = rows[0], rows[-1]
        slope = np.polyfit(range(len(rows)), [m["yoy_growth_pct"] for m in rows], 1)[0]
        decel = last["yoy_growth_pct"] < first["yoy_growth_pct"]
        sig.add("trend", "warning" if decel else "opportunity",
                "Croissance annuelle en ralentissement" if decel else "Croissance annuelle en accélération",
                f"Le CA progresse de {r1(y['growth_pct'])} % sur un an ({y['period']} vs {y['compared_to']}), "
                f"mais la croissance passe de {r1(first['yoy_growth_pct'])} % ({first['month']}) à "
                f"{r1(last['yoy_growth_pct'])} % ({last['month']}), soit {r1(slope)} points par mois.",
                {"yoy_growth_pct": r1(y["growth_pct"]), "period": y["period"], "compared_to": y["compared_to"],
                 "first_month": first["month"], "first_yoy_pct": r1(first["yoy_growth_pct"]),
                 "last_month": last["month"], "last_yoy_pct": r1(last["yoy_growth_pct"]),
                 "slope_points_per_month": r1(slope)},
                "Anticiper la fin de la phase d'hypercroissance : fidélisation et nouveaux segments.")


def detect_monthly_anomalies(k, cal, sig):
    rel = [m for m in k["breakdowns"]["monthly"] if m["is_reliable_month"]]
    for i in range(3, len(rel)):
        m = rel[i]
        base = float(np.median([x["revenue_brl"] for x in rel[i - 3:i]]))
        dev = (m["revenue_brl"] / base - 1) * 100 if base else 0
        if abs(dev) < THRESHOLDS["month_deviation_pct"]:
            continue
        month = cal[cal["year_month"] == m["month"]]
        ev = month[month["event"] != ""]
        ev_share = ev["revenue"].sum() / month["revenue"].sum() * 100 if month["revenue"].sum() else 0
        events = sorted(set(ev["event"]))
        spike = dev > 0
        sig.add("monthly_anomaly", "info" if spike else "alert",
                f"{'Pic' if spike else 'Chute'} de CA en {m['month']}",
                f"{m['month']} : CA de {r1(m['revenue_brl'])} BRL, {r1(dev)} % vs la médiane des 3 mois fiables "
                f"précédents ({r1(base)} BRL). Événements du mois : {', '.join(events) or 'aucun'} "
                f"({r1(ev_share)} % du CA du mois réalisé ces jours-là).",
                {"month": m["month"], "revenue_brl": r1(m["revenue_brl"]), "baseline_brl": r1(base),
                 "deviation_pct": r1(dev), "events_in_month": events, "event_days_revenue_share_pct": r1(ev_share)},
                "Reproduire le mécanisme du pic." if spike else "Analyser la cause de la chute (catégories, régions).")


def detect_daily_anomalies(cal, sig):
    s = cal.loc[cal["is_reliable_month"], "revenue"]
    w = THRESHOLDS["daily_window_days"]
    hist = s.shift(1)
    med = hist.rolling(w, min_periods=w // 2).median()
    mad = hist.rolling(w, min_periods=w // 2).apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
    z = (0.6745 * (s - med) / mad.replace(0, np.nan)).dropna()
    flagged = sorted(z[z.abs() >= THRESHOLDS["daily_robust_z"]].index)

    # Regroupe les jours atypiques proches en épisodes (ex. toute la semaine du Black Friday)
    episodes = []
    for d in flagged:
        if episodes and (d - episodes[-1][-1]).days <= THRESHOLDS["episode_gap_days"]:
            episodes[-1].append(d)
        else:
            episodes.append([d])
    episodes.sort(key=lambda ep: max(abs(z[d]) for d in ep), reverse=True)

    win = pd.Timedelta(days=THRESHOLDS["event_window_days"])
    for ep in episodes[: THRESHOLDS["max_daily_episodes"]]:
        peak = max(ep, key=lambda d: abs(z[d]))
        near = cal.loc[ep[0] - win: ep[-1] + win]
        near = sorted({f"{e} ({i:%Y-%m-%d})" for i, e in near["event"].items() if e})
        dev = (s[peak] / med[peak] - 1) * 100 if med[peak] else None
        up = s[peak] > med[peak]
        days = [{"date": f"{d:%Y-%m-%d}", "revenue_brl": r1(s[d]),
                 "deviation_pct": r1((s[d] / med[d] - 1) * 100 if med[d] else None), "robust_z": r1(z[d])}
                for d in ep]
        span = f"{ep[0]:%Y-%m-%d}" + (f" -> {ep[-1]:%Y-%m-%d}" if len(ep) > 1 else "")
        sig.add("daily_anomaly", "info" if near or up else "warning",
                f"{'Pic' if up else 'Creux'} atypique : {span}",
                f"{len(ep)} jour(s) atypique(s) ({span}). Pic le {peak:%Y-%m-%d} : {r1(s[peak])} BRL, "
                f"{r1(dev)} % vs la médiane des {w} jours précédents ({r1(med[peak])} BRL). "
                + (f"Événement(s) à {THRESHOLDS['event_window_days']} jours près : {', '.join(near)}."
                   if near else "Aucun événement connu à proximité."),
                {"start": f"{ep[0]:%Y-%m-%d}", "end": f"{ep[-1]:%Y-%m-%d}", "days": days,
                 "peak_date": f"{peak:%Y-%m-%d}", "peak_revenue_brl": r1(s[peak]),
                 "peak_baseline_brl": r1(med[peak]), "peak_deviation_pct": r1(dev),
                 "nearby_events": near, "explained_by_event": bool(near)},
                "" if near else "Vérifier une cause externe (promotion, incident, problème de collecte).")


def detect_latest_month(k, cal, sig):
    rel = [m for m in k["breakdowns"]["monthly"] if m["is_reliable_month"]]
    # mois du calendrier après le dernier mois fiable (y compris ceux sans aucune commande livrée)
    unreliable_after = sorted(set(cal.loc[~cal["is_reliable_month"], "year_month"])
                              - {m["month"] for m in rel})
    unreliable_after = [m for m in unreliable_after if rel and m > rel[-1]["month"]]
    if len(rel) < 2:
        return
    last, prev = rel[-1], rel[-2]
    mom = (last["revenue_brl"] / prev["revenue_brl"] - 1) * 100
    alert = mom <= THRESHOLDS["latest_month_drop_pct"] or (last["yoy_growth_pct"] or 0) < 0
    sig.add("latest_month", "alert" if alert else "info", f"Dernier mois fiable : {last['month']}",
            f"{last['month']} : {r1(last['revenue_brl'])} BRL, {r1(mom)} % vs {prev['month']} et "
            f"{r1(last['yoy_growth_pct'])} % sur un an. Mois suivants exclus (volume trop faible, données "
            f"incomplètes) : {', '.join(unreliable_after) or 'aucun'}.",
            {"month": last["month"], "revenue_brl": r1(last["revenue_brl"]), "mom_pct": r1(mom),
             "yoy_pct": r1(last["yoy_growth_pct"]), "excluded_months_after": unreliable_after},
            "Ne pas interpréter les derniers mois incomplets comme une baisse.")


def detect_categories(k, sig):
    c = k["breakdowns"]["categories"]
    for x in c["declining"]:
        sig.add("category_decline", "alert", f"Catégorie en baisse : {x['category']}",
                f"{x['category']} : {r1(x['yoy_growth_pct'])} % sur un an, alors que le CA global progresse ; "
                f"note moyenne {x['avg_review_score']} / 5.",
                {"category": x["category"], "yoy_growth_pct": r1(x["yoy_growth_pct"]),
                 "revenue_brl": r1(x["revenue_brl"]), "avg_review_score": x["avg_review_score"]},
                "Revoir l'offre, les prix et les vendeurs de la catégorie.")
    if c["fastest_growing"]:
        sig.add("category_growth", "opportunity", "Catégories en forte croissance",
                "Plus fortes croissances sur un an : " + ", ".join(
                    f"{x['category']} ({r1(x['yoy_growth_pct'])} %)" for x in c["fastest_growing"]) + ".",
                {"categories": [{"category": x["category"], "yoy_growth_pct": r1(x["yoy_growth_pct"])}
                                for x in c["fastest_growing"]]},
                "Renforcer le catalogue et la visibilité de ces catégories.")
    if c["lowest_rated"]:
        sig.add("category_rating", "warning", "Catégories les moins bien notées",
                "Notes les plus basses (au moins 1 000 articles) : " + ", ".join(
                    f"{x['category']} ({x['avg_review_score']} / 5)" for x in c["lowest_rated"]) + ".",
                {"categories": [{"category": x["category"], "avg_review_score": x["avg_review_score"]}
                                for x in c["lowest_rated"]],
                 "overall_review_score": k["headline"]["average_review_score"]},
                "Examiner qualité produit et délais des vendeurs de ces catégories.")


def detect_regions(k, sig):
    nat = k["headline"]["late_delivery_rate_pct"]
    bad = [s for s in k["breakdowns"]["by_customer_state"]
           if s["orders"] >= THRESHOLDS["state_min_orders"]
           and s["late_delivery_rate_pct"] >= THRESHOLDS["state_late_ratio"] * nat]
    if bad:
        bad.sort(key=lambda s: s["late_delivery_rate_pct"], reverse=True)
        sig.add("region_delivery", "alert", "États mal desservis",
                f"Retard national : {r1(nat)} %. États (>= {THRESHOLDS['state_min_orders']} commandes) au moins "
                f"{THRESHOLDS['state_late_ratio']} fois au-dessus : " + ", ".join(
                    f"{s['state']} ({r1(s['late_delivery_rate_pct'])} % de retard, {s['median_delivery_days']} j, "
                    f"note {s['avg_review_score']})" for s in bad) + ".",
                {"national_late_rate_pct": r1(nat), "states": [
                    {k_: s[k_] for k_ in ("state", "orders", "late_delivery_rate_pct", "median_delivery_days",
                                          "freight_to_revenue_pct", "avg_review_score")} for s in bad]},
                "Revoir transporteurs et délais annoncés pour ces États ; envisager des vendeurs ou stocks locaux.")


def detect_delivery_satisfaction(k, sig):
    ins = k["insights"]
    late, ok = ins.get("review_score_late_orders"), ins.get("review_score_on_time_orders")
    if late is not None and ok is not None:
        sig.add("delivery_satisfaction", "alert", "Le retard de livraison détruit la satisfaction",
                f"Note moyenne : {late} / 5 pour une commande en retard contre {ok} / 5 à l'heure "
                f"(écart {r1(ok - late)} point) ; {r1(k['headline']['late_delivery_rate_pct'])} % des commandes "
                "sont en retard. Association observée, pas une preuve de causalité.",
                {"review_late": late, "review_on_time": ok, "gap_points": r1(ok - late),
                 "late_delivery_rate_pct": r1(k["headline"]["late_delivery_rate_pct"])},
                "Faire de la ponctualité un objectif prioritaire (dates estimées réalistes, suivi des transporteurs).")


def detect_sellers(k, sig):
    s = k["breakdowns"]["sellers"]
    c = s["concentration"]
    sig.add("seller_concentration", "warning", "CA très concentré",
            f"Les 10 % meilleurs vendeurs ({c['active_sellers']} vendeurs actifs) réalisent "
            f"{r1(c['top_10pct_sellers_revenue_share_pct'])} % du CA ; les vendeurs de {c['top_seller_state']} "
            f"{r1(c['top_seller_state_revenue_share_pct'])} %.",
            {k_: c[k_] for k_ in c}, "Réduire la dépendance : recruter et accompagner des vendeurs hors de cet État.")
    nat = k["headline"]["late_delivery_rate_pct"]
    risky = [x for x in s["top_sellers"] if (x["late_delivery_rate_pct"] or 0) > nat]
    if risky:
        sig.add("seller_quality", "warning", "Gros vendeurs plus souvent en retard",
                f"Parmi les 10 premiers vendeurs, au-dessus du retard national ({r1(nat)} %) : " + ", ".join(
                    f"{x['seller_name']} ({r1(x['late_delivery_rate_pct'])} %, note {x['avg_review_score']})"
                    for x in risky) + ".",
                {"national_late_rate_pct": r1(nat), "sellers": [
                    {k_: x[k_] for k_ in ("seller_name", "state", "revenue_brl", "late_delivery_rate_pct",
                                          "avg_review_score")} for x in risky]},
                "Plan d'amélioration logistique avec ces vendeurs clés.")


def detect_events(k, sig):
    ev = k["breakdowns"]["events"]
    commercial = [e for e in ev if "commercial" in e["type"]]
    holidays = [e for e in ev if e["type"] == "jour_ferie" and e["vs_normal_day_pct"] is not None]
    for e in commercial:
        sig.add("event_commercial", "opportunity", f"Effet {e['event']}",
                f"{e['event']} ({e['date']}) : {r1(e['revenue_brl'])} BRL, {r1(e['vs_normal_day_pct'])} % vs un "
                "jour normal du même mois.",
                {k_: e[k_] for k_ in ("date", "event", "revenue_brl", "orders", "vs_normal_day_pct")},
                "Préparer stocks, logistique et service client pour les prochains événements commerciaux.")
    if holidays:
        vals = [e["vs_normal_day_pct"] for e in holidays]
        sig.add("event_holidays", "info", "Les jours fériés font baisser les ventes",
                f"Sur {len(vals)} jours fériés, écart médian {r1(np.median(vals))} % vs un jour normal du même mois ; "
                f"{sum(v < 0 for v in vals)} sur {len(vals)} sont en baisse (min {r1(min(vals))} %, max {r1(max(vals))} %).",
                {"holidays_analyzed": len(vals), "median_vs_normal_pct": r1(np.median(vals)),
                 "below_normal_count": sum(v < 0 for v in vals), "min_pct": r1(min(vals)), "max_pct": r1(max(vals))},
                "Ne pas lire une baisse un jour férié comme une alerte ; planifier les promotions hors fériés.")


def detect_weekday_and_customers(k, sig):
    wd = {w["weekday"]: w["revenue_share_pct"] for w in k["breakdowns"]["weekday_revenue_share"]}
    weekend = (wd.get("Saturday") or 0) + (wd.get("Sunday") or 0)
    early = sum(wd.get(d) or 0 for d in ("Monday", "Tuesday", "Wednesday"))
    sig.add("weekday", "info", "Ventes concentrées en début de semaine",
            f"Lundi à mercredi : {r1(early)} % du CA ; samedi et dimanche : {r1(weekend)} %.",
            {"monday_to_wednesday_pct": r1(early), "weekend_pct": r1(weekend), "by_weekday": wd},
            "Placer les campagnes en début de semaine ; tester des offres week-end.")
    h = k["headline"]
    sig.add("retention", "alert" if h["repeat_customer_rate_pct"] < 10 else "info", "Fidélisation très faible",
            f"Seulement {r1(h['repeat_customer_rate_pct'])} % des {h['unique_customers']} clients ont passé plus "
            f"d'une commande livrée ; commandes perdues (annulées ou indisponibles) : {r1(h['lost_order_rate_pct'])} %.",
            {"repeat_customer_rate_pct": r1(h["repeat_customer_rate_pct"]), "unique_customers": h["unique_customers"],
             "lost_order_rate_pct": r1(h["lost_order_rate_pct"])},
            "Programme de réachat (e-mails post-livraison, offres de fidélité).")


def caveats(k):
    out = [k["margin_simulated"]["warning"]]
    unreliable = [m["month"] for m in k["breakdowns"]["monthly"] if not m["is_reliable_month"]]
    if unreliable:
        out.append("Mois exclus des tendances (volume trop faible) : " + ", ".join(unreliable) + ".")
    out += [x for x in k.get("limitations", []) if "association" in x.lower()]
    return out


def detect_all(k, cal):
    sig = Signals()
    detect_trend(k, sig)
    detect_latest_month(k, cal, sig)
    detect_monthly_anomalies(k, cal, sig)
    detect_daily_anomalies(cal, sig)
    detect_categories(k, sig)
    detect_regions(k, sig)
    detect_delivery_satisfaction(k, sig)
    detect_sellers(k, sig)
    detect_events(k, sig)
    detect_weekday_and_customers(k, sig)
    return sig.items


# ============================================================================
# 2. EXPLICATION (LLM) + garde-fous
# ============================================================================
SYSTEM_PROMPT = """Tu es le BI Analyst Agent d'un pipeline BI multi-agents (projet BIFlow, marketplace Olist).
On te donne des SIGNAUX détectés par Python (id, gravité, résumé, faits chiffrés) et des précautions.
Ta mission : transformer ces signaux en insights utiles à un responsable métier.

RÈGLES :
- Chaque insight cite les signaux qui le justifient dans "signal_ids" (au moins un id existant).
- N'utilise QUE des nombres présents dans les faits ; recopie-les tels quels (même arrondi). N'invente rien.
- Explique les variations par les faits fournis (événements, jours fériés, saisonnalité, délais, régions).
  Une corrélation n'est pas une causalité : dis "associé à", pas "causé par".
- Respecte les précautions (marge simulée = aucune conclusion ; mois incomplets = pas une baisse).
- Regroupe les signaux liés (ex. retards par État + note des commandes en retard).
- Chaque signal de gravité "alert" DOIT être cité par au moins un insight (baisses, retards, fidélisation...).
- Maximum 8 insights, triés par priorité décroissante. Recommandation concrète et actionnable pour chacun.

SORTIE : UNIQUEMENT un JSON valide :
{"executive_summary": "4 à 6 phrases en français",
 "insights": [{"id": "A1", "title": "...", "type": "trend|alert|anomaly|opportunity|risk",
               "priority": "high|medium|low", "finding": "constat chiffré",
               "explanation": "pourquoi (selon les faits)", "recommendation": "action",
               "signal_ids": ["S1"]}],
 "watchlist": ["indicateur à surveiller et pourquoi"]}"""


def _norm(text: str) -> str:
    text = re.sub(r"(?<=\d)[   ](?=\d{3}\b)", "", text)
    return re.sub(r"(?<=\d),(?=\d)", ".", text)


def validate(result: dict, signals: list) -> dict:
    """Vérifie les signaux cités et les nombres écrits (un nombre absent des faits est signalé)."""
    ids = {s["id"] for s in signals}
    facts_text = _norm(json.dumps(signals, ensure_ascii=False))
    for ins in result.get("insights", []):
        cited = [i for i in ins.get("signal_ids", []) if i in ids]
        ins["unknown_signal_ids"] = [i for i in ins.get("signal_ids", []) if i not in ids]
        text = _norm(" ".join(str(ins.get(f, "")) for f in ("finding", "explanation", "recommendation")))
        nums = {n for n in re.findall(r"\d+(?:\.\d+)?", text) if len(n.replace(".", "")) >= 2}
        ins["ungrounded_numbers"] = sorted(n for n in nums if n not in facts_text)
        ins["verified"] = bool(cited) and not ins["unknown_signal_ids"] and not ins["ungrounded_numbers"]
    return result


def llm_insights(signals: list, cav: list):
    model = os.environ.get("LLM_MODEL")
    if not model or "--no-llm" in sys.argv:
        return None
    try:
        from langchain.chat_models import init_chat_model
        from langchain_core.messages import HumanMessage, SystemMessage
        from profiler_agent import invoke_with_retry, parse_json, text_of
        llm = init_chat_model(model, temperature=float(os.environ.get("LLM_TEMPERATURE", "0")))
        payload = {"signals": signals, "precautions": cav}
        for attempt in range(2):
            ai = invoke_with_retry(llm, [SystemMessage(SYSTEM_PROMPT),
                                         HumanMessage(json.dumps(payload, ensure_ascii=False, default=str))])
            try:
                return parse_json(text_of(ai))
            except ValueError:
                if attempt:
                    raise
    except Exception as e:   # quota, réseau, JSON invalide : on passe aux règles
        print(f"  (LLM indisponible pour l'analyse, insights par règles : {str(e)[:150]})")
    return None


def rule_insights(signals: list) -> dict:
    """Repli sans LLM : un insight par signal important, texte déterministe."""
    kind_type = {"trend": "trend", "latest_month": "trend", "monthly_anomaly": "anomaly",
                 "daily_anomaly": "anomaly", "category_decline": "alert", "category_growth": "opportunity",
                 "event_commercial": "opportunity", "region_delivery": "alert", "delivery_satisfaction": "alert",
                 "seller_concentration": "risk", "seller_quality": "risk", "retention": "alert"}
    prio = {"alert": "high", "warning": "medium", "opportunity": "medium", "info": "low"}
    ranked = sorted(signals, key=lambda s: SEVERITY_RANK[s["severity"]])
    ranked = [s for s in ranked if s["severity"] != "info" or s["kind"] in ("event_holidays", "weekday")][:10]
    insights = [{"id": f"A{i}", "title": s["title"], "type": kind_type.get(s["kind"], "trend"),
                 "priority": prio[s["severity"]], "finding": s["summary"], "explanation": "",
                 "recommendation": s["recommendation_hint"], "signal_ids": [s["id"]]}
                for i, s in enumerate(ranked, 1)]
    alerts = [s["title"] for s in signals if s["severity"] == "alert"]
    return {"executive_summary": f"{len(signals)} signaux détectés, dont {len(alerts)} alertes : "
                                 + "; ".join(alerts) + ". (Synthèse générée par règles, sans LLM.)",
            "insights": insights,
            "watchlist": ["Croissance sur un an", "Taux de retard par État", "Catégories en baisse"]}


# ============================================================================
# 3. AGENT
# ============================================================================
def kpi_fingerprint(k: dict) -> str:
    """Empreinte du CONTENU des KPI (sans la date de calcul) : le KPI agent réécrit le fichier à
    chaque exécution, mais si les chiffres n'ont pas changé, l'analyse précédente reste valable."""
    # Seuls les CHIFFRES comptent : reformuler une définition ne rend pas l'analyse obsolète
    body = {key: k.get(key) for key in ("headline", "insights", "breakdowns")}
    body["margin"] = {key: v for key, v in (k.get("margin_simulated") or {}).items() if key != "warning"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def previous_analysis(fingerprint: str):
    """Analyse précédente produite par le LLM sur exactement les mêmes KPI, sinon None."""
    if not INSIGHTS_FILE.exists():
        return None
    prev = json.loads(INSIGHTS_FILE.read_text(encoding="utf-8"))
    ok = prev.get("kpi_fingerprint") == fingerprint and prev.get("generated_by") == "llm"
    return prev if ok else None


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    k, cal = load_inputs()
    fingerprint = kpi_fingerprint(k)

    prev = previous_analysis(fingerprint)
    if "--reuse" in sys.argv and prev:
        print("  Analyse existante réutilisée (KPI identiques, aucun appel LLM).")
        return summary_of(prev, reused=True)

    signals = detect_all(k, cal)
    cav = caveats(k)
    print(f"  {len(signals)} signaux détectés")

    result = llm_insights(signals, cav)
    generated_by = "llm"
    if (not result or not result.get("insights")) and prev and "--no-llm" not in sys.argv:
        # LLM en échec (quota, surcharge) mais une analyse LLM existe pour les mêmes KPI : on la garde
        print("  LLM indisponible : l'analyse LLM précédente (mêmes KPI) est conservée.")
        return summary_of(prev, reused=True)
    if not result or not result.get("insights"):
        result, generated_by = rule_insights(signals), "rules"
    result = validate(result, signals)

    # Garde-fou : aucune alerte ne doit disparaître de l'analyse (ex. catégories en baisse)
    cited = {sid for i in result["insights"] for sid in i.get("signal_ids", [])}
    missing = [s for s in signals if s["severity"] == "alert" and s["id"] not in cited]
    if missing:
        extra = validate(rule_insights(missing), signals)["insights"]
        for n, ins in enumerate(extra, len(result["insights"]) + 1):
            ins.update(id=f"A{n}", added_by="rules (alerte non couverte par le LLM)")
        result["insights"] += extra
        print(f"  {len(extra)} alerte(s) non couverte(s) par le LLM ajoutée(s) par règles")

    data = {"agent": "analyst_agent", "generated_at": datetime.now().isoformat(timespec="seconds"),
            "generated_by": generated_by, "model": os.environ.get("LLM_MODEL") if generated_by == "llm" else None,
            "source": str(KPI_FILE.relative_to(ROOT)), "kpi_fingerprint": fingerprint,
            "thresholds": THRESHOLDS,
            **result, "caveats": cav, "signals": signals}
    INSIGHTS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    write_report(data)
    print(f"  {len(data['insights'])} insights ({generated_by}) -> data/analyst/analyst_report.md")
    return summary_of(data, reused=False)


def summary_of(data, reused):
    ins = data.get("insights", [])
    return {"ok": True, "reused": reused, "generated_by": data.get("generated_by"),
            "insights": len(ins), "high_priority": sum(i.get("priority") == "high" for i in ins),
            "unverified_insights": sum(not i.get("verified") for i in ins),
            "alerts": sum(s["severity"] == "alert" for s in data.get("signals", [])),
            "outputs": ["data/analyst/analyst_insights.json", "data/analyst/analyst_report.md"]}


def write_report(d):
    L = ["# Rapport du BI Analyst Agent", "",
         f"- Généré le {d['generated_at']} ({'LLM ' + str(d['model']) if d['generated_by'] == 'llm' else 'règles, sans LLM'})",
         "", "## Synthèse", "", d.get("executive_summary", ""), "", "## Insights", ""]
    for i in d.get("insights", []):
        flags = []
        if i.get("unknown_signal_ids"):
            flags.append("signaux inconnus : " + ", ".join(i["unknown_signal_ids"]))
        if i.get("ungrounded_numbers"):
            flags.append("nombres non retrouvés dans les faits : " + ", ".join(i["ungrounded_numbers"]))
        warn = f" [À CONTRÔLER : {' ; '.join(flags)}]" if flags else ""
        if i.get("added_by"):
            warn += f" _(ajouté par {i['added_by']})_"
        L += [f"### {i.get('id')}. {i.get('title')} ({i.get('type')}, priorité {i.get('priority')}){warn}", "",
              f"- **Constat :** {i.get('finding')}"]
        if i.get("explanation"):
            L.append(f"- **Explication :** {i.get('explanation')}")
        L += [f"- **Recommandation :** {i.get('recommendation')}",
              f"- **Signaux :** {', '.join(i.get('signal_ids', []))}", ""]
    L += ["## À surveiller", ""] + [f"- {w}" for w in d.get("watchlist", [])]
    L += ["", "## Précautions", ""] + [f"- {c}" for c in d.get("caveats", [])]
    L += ["", "## Signaux détectés (traçabilité)", "", "| Id | Gravité | Type | Constat |", "|---|---|---|---|"]
    L += [f"| {s['id']} | {s['severity']} | {s['kind']} | {s['summary']} |" for s in d.get("signals", [])]
    (OUT / "analyst_report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    print("Le BI Analyst Agent analyse les KPI...")
    print(run())
