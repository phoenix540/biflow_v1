"""Passerelle entre Olist Pilot et les agents du pipeline BIFlow.

L'assistant ne refait pas le travail des agents : il les appelle.
  - Agent KPI      : kpi_agent.filter_frames + kpi_agent.compute (mêmes formules que kpi_results.json)
  - BI Analyst     : analyst_agent.detect_* (signaux déterministes) sur les KPI du périmètre demandé,
                     + les insights rédigés et vérifiés publiés dans analyst_insights.json
  - XAI            : xai_engine.explain_kpi_change (décomposition prix / volume / mix de la variation du CA)
                     xai_engine.shap_attributions (facteurs du risque de mauvaise note, modèle ML + SHAP)
                     modèle review_model.joblib pour le risque d'une commande précise ou d'un scénario
"""
from __future__ import annotations

import json
import sys
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
AGENTS = ROOT / "agents"
if str(AGENTS) not in sys.path:
    sys.path.insert(0, str(AGENTS))

import analyst_agent as AA  # noqa: E402
import kpi_agent as KA      # noqa: E402
import xai_engine as XE     # noqa: E402

warnings.filterwarnings("ignore", message=".*fixed sampling defaults.*")
SEVERITY_FR = {"alert": "alerte", "warning": "vigilance", "opportunity": "opportunité", "info": "info"}


def _stamp() -> tuple:
    return tuple(p.stat().st_mtime for p in (KA.CURATED / "fact_orders.csv", KA.CURATED / "fact_sales.csv",
                                             KA.CURATED / "dim_date.csv") if p.exists())


@lru_cache(maxsize=1)
def _tables(stamp):
    return KA.load()


def tables():
    """(orders, sales, dates) tels que chargés et contrôlés par l'agent KPI."""
    return _tables(_stamp())


# --------------------------------------------------------------------------
# Agent KPI
# --------------------------------------------------------------------------
@lru_cache(maxsize=64)
def _kpis(stamp, period, states, cats):
    o, s, d = _tables(stamp)
    o, s = KA.filter_frames(o, s, states, cats)
    return KA.compute(o, s, d, period)


def kpis(period=None, states=(), cats=()) -> dict:
    """KPI de l'agent KPI sur un périmètre (period = ("AAAA-MM", "AAAA-MM") ou None = toute la période)."""
    return _kpis(_stamp(), tuple(period) if period else None, tuple(states), tuple(cats))


# --------------------------------------------------------------------------
# BI Analyst
# --------------------------------------------------------------------------
def _calendar(period, states, cats) -> pd.DataFrame:
    """Calendrier journalier (CA, commandes, événements) comme analyst_agent.load_inputs, sur le périmètre."""
    o, s, d = tables()
    o, _ = KA.filter_frames(o, s, states, cats)
    rev = o[o["is_revenue_order"]]
    daily = rev.groupby("purchase_date").agg(revenue=("revenue_brl", "sum"), orders=("order_id", "size"))
    cal = d[["date", "is_reliable_month", "event"]].copy().set_index("date")
    cal = cal.join(daily, how="left").fillna({"revenue": 0.0, "orders": 0})
    cal["year_month"] = cal.index.to_period("M").astype(str)
    if period:
        cal = cal[cal["year_month"].between(period[0], period[1])]
    return cal


DETECTORS = [  # même ordre que analyst_agent.detect_all
    ("trend", lambda k, c, s: AA.detect_trend(k, s)),
    ("latest_month", lambda k, c, s: AA.detect_latest_month(k, c, s)),
    ("monthly", lambda k, c, s: AA.detect_monthly_anomalies(k, c, s)),
    ("daily", lambda k, c, s: AA.detect_daily_anomalies(c, s)),
    ("categories", lambda k, c, s: AA.detect_categories(k, s)),
    ("regions", lambda k, c, s: AA.detect_regions(k, s)),
    ("delivery", lambda k, c, s: AA.detect_delivery_satisfaction(k, s)),
    ("sellers", lambda k, c, s: AA.detect_sellers(k, s)),
    ("events", lambda k, c, s: AA.detect_events(k, s)),
    ("weekday", lambda k, c, s: AA.detect_weekday_and_customers(k, s)),
]


@lru_cache(maxsize=32)
def _signals(stamp, period, states, cats):
    k = _kpis(stamp, period, states, cats)
    cal = _calendar(period, states, cats)
    sig = AA.Signals()
    skipped = []
    for name, fn in DETECTORS:   # un détecteur sans données suffisantes (petit périmètre) ne bloque pas les autres
        try:
            fn(k, cal, sig)
        except Exception as e:  # noqa: BLE001
            skipped.append(f"{name} ({type(e).__name__})")
    items = sorted(sig.items, key=lambda x: AA.SEVERITY_RANK.get(x["severity"], 9))
    try:
        cav = AA.caveats(k)
    except Exception:  # noqa: BLE001
        cav = []
    return items, cav, skipped


def analyst_signals(period=None, states=(), cats=()) -> dict:
    items, cav, skipped = _signals(_stamp(), tuple(period) if period else None, tuple(states), tuple(cats))
    return {"signaux": [{"id": s["id"], "type": s["kind"], "gravite": SEVERITY_FR.get(s["severity"], s["severity"]),
                         "titre": s["title"], "constat": s["summary"], "piste": s.get("recommendation_hint", "")}
                        for s in items],
            "precautions": cav, "detecteurs_non_applicables": skipped,
            "seuils": AA.THRESHOLDS}


def published_insights() -> dict:
    """Insights rédigés par le BI Analyst (LLM ou règles), nombres vérifiés par Python, période complète."""
    try:
        a = json.loads(AA.INSIGHTS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"disponible": False}
    return {"disponible": True, "genere_le": a.get("generated_at"), "redige_par": a.get("generated_by"),
            "synthese": a.get("executive_summary"),
            "insights": [{k: i.get(k) for k in ("title", "priority", "type", "finding", "explanation", "recommendation", "verified")}
                         for i in a.get("insights", [])],
            "a_surveiller": a.get("watchlist", []), "precautions": a.get("caveats", [])}


# --------------------------------------------------------------------------
# XAI : pourquoi le CA a changé (prix / volume / mix)
# --------------------------------------------------------------------------
XAI_DIMS = {"categorie": "category", "etat_client": "customer_state", "etat_vendeur": "seller_state", "vendeur": "seller_name"}


@lru_cache(maxsize=1)
def _xai_sales(stamp):
    s = pd.read_csv(KA.CURATED / "fact_sales.csv", usecols=[
        "order_id", "purchase_date", "is_revenue_order", "price", "revenue_brl", "category", "customer_state",
        "seller_state", "seller_name"])
    s["is_revenue_order"] = KA.read_bool(s["is_revenue_order"])
    s["category"] = s["category"].fillna("non_defini")
    return s


def explain_revenue_change(current=None, previous=None, dimension="categorie", states=(), cats=(), n=5) -> dict:
    s = _xai_sales(_stamp())
    if states:
        s = s[s["customer_state"].isin(states)]
    if cats:
        s = s[s["category"].isin(cats)]
    dim = XAI_DIMS.get(dimension, "category")
    res = XE.explain_kpi_change(s, dim=dim, current=current, previous=previous)
    if not res or not res.get("available"):
        return {"disponible": False, "raison": (res or {}).get("reason")}
    b = res["bridge"]
    keep = ["delta", "volume_effect", "price_effect", "rev_prev", "rev_curr", "units_prev", "units_curr",
            "avg_price_prev", "avg_price_curr"]
    mv = XE.top_movers(b, n)
    return {"disponible": True, "phrase": res["sentence"], "dimension": dimension,
            "mois_precedent": b["period_prev"], "mois_courant": b["period_curr"],
            "ca_precedent": b["revenue_prev"], "ca_courant": b["revenue_curr"],
            "variation_brl": b["total_delta"], "variation_pct": b["pct_change"],
            "effets": {"volume": b["effects"]["volume"], "prix": b["effects"]["price"],
                       "interaction": b["effects"]["interaction"]},
            "reconcilie": b["reconciles"],
            "plus_fortes_hausses": [{dim: r[dim], **{k: r[k] for k in keep}} for r in mv["up"]],
            "plus_fortes_baisses": [{dim: r[dim], **{k: r[k] for k in keep}} for r in mv["down"]]}


# --------------------------------------------------------------------------
# XAI : risque de mauvaise note (modèle ML + SHAP)
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _shap_global(model_mtime):
    return XE.shap_attributions()


def bad_review_drivers() -> dict:
    if not XE.MODEL_PATH.exists():
        try:
            XE.train_review_model()
        except Exception as e:  # noqa: BLE001
            return {"available": False, "reason": f"modèle indisponible : {e}"}
    return _shap_global(XE.MODEL_PATH.stat().st_mtime)


@lru_cache(maxsize=1)
def _model(model_mtime):
    import joblib
    import shap
    model = joblib.load(XE.MODEL_PATH)
    return model, shap.TreeExplainer(model)


@lru_cache(maxsize=1)
def _review_frame(stamp):
    o = pd.read_csv(KA.CURATED / "fact_orders.csv")
    o["is_revenue_order"] = KA.read_bool(o["is_revenue_order"])
    s = pd.read_csv(KA.CURATED / "fact_sales.csv", usecols=["order_id", "seller_id"])
    X, y, ids = XE.review_dataset(o, s)
    X.index = ids.values
    return X, y.set_axis(ids.values)


SCENARIO_ARGS = {"retard_jours": "delay_days", "delai_jours": "delivery_days", "nb_articles": "items_count",
                 "nb_vendeurs": "n_sellers", "valeur_brl": "items_value", "fret_brl": "freight_total",
                 "nb_echeances": "max_installments", "paiement_boleto": "pay_boleto", "client_sp": "customer_sp"}


def order_risk(order_id: str | None = None, **scenario) -> dict:
    """Probabilité d'avis 1-2 étoiles + contributions SHAP, pour une commande réelle ou un scénario
    (valeurs médianes des commandes, modifiées par les paramètres fournis)."""
    if not XE.MODEL_PATH.exists():
        return {"disponible": False, "raison": "modèle XAI absent : lancer l'agent Auditor / xai_engine.train_review_model"}
    model, explainer = _model(XE.MODEL_PATH.stat().st_mtime)
    feats = list(model.feature_names_in_)
    X, y = _review_frame(_stamp())
    if order_id:
        hits = [i for i in X.index if str(i).startswith(order_id.strip())]
        if not hits:
            return {"disponible": False, "raison": f"commande « {order_id} » introuvable parmi les commandes livrées avec avis"}
        row = X.loc[[hits[0]], feats]
        source = {"commande": hits[0], "avis_reel_mauvais": bool(y.loc[hits[0]])}
    else:
        row = X[feats].median().to_frame().T
        changed = {}
        for k, v in scenario.items():
            if v is None or k not in SCENARIO_ARGS:
                continue
            row[SCENARIO_ARGS[k]] = float(v)
            changed[k] = v
        if "retard_jours" in changed and "delai_jours" not in changed:
            # un retard allonge le délai : délai = délai médian promis + retard
            row["delivery_days"] = float(X["delivery_days"].median() - X["delay_days"].median() + changed["retard_jours"])
        source = {"scenario": changed or "commande type (valeurs médianes)"}
    proba = float(model.predict_proba(row)[0, 1])
    sv = np.asarray(explainer.shap_values(row)).reshape(-1, len(feats))[-1]   # 1 ligne x n variables (classe « mauvaise note »)
    contrib = sorted(zip(feats, sv, row.iloc[0].values), key=lambda t: -abs(t[1]))
    base = float(y.mean())
    return {"disponible": True, **source, "probabilite_mauvaise_note_pct": round(proba * 100, 1),
            "taux_moyen_pct": round(base * 100, 1), "risque_relatif": round(proba / base, 2) if base else None,
            "facteurs": [{"facteur": XE.FEATURES_FR.get(f, f), "valeur": round(float(v), 2),
                          "effet": "augmente le risque" if c > 0 else "réduit le risque", "poids_shap": round(float(c), 3)}
                         for f, c, v in contrib[:5]],
            "modele": json.loads(XE.METRICS_PATH.read_text(encoding="utf-8")) if XE.METRICS_PATH.exists() else {}}
