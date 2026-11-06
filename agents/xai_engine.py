"""
Moteur d'explication XAI — projet BIFlow (utilisé par l'agent Auditor XAI).

Deux niveaux d'explication :

1. Pourquoi un KPI a CHANGÉ (explication métier, sans modèle ML)
   - Découpage en périodes (mois) à partir de fact_sales.purchase_date.
   - Décomposition Prix-Volume-Mix (bridge) de la variation de chiffre d'affaires,
     par dimension (par défaut : category). La somme des effets reconcilie
     EXACTEMENT la variation totale -> explication auditable.
     ΔCA par segment = effet_volume + effet_prix + effet_interaction
       effet_volume      = (u1 - u0) * p0
       effet_prix        = (p1 - p0) * u0
       effet_interaction = (p1 - p0) * (u1 - u0)
     où u = unités, p = prix moyen, indice 0 = période précédente, 1 = période courante.
   - Top movers : segments qui contribuent le plus (en + et en -) à la variation.

2. Pourquoi un MODÈLE a prédit (attribution de features, SHAP) — OPTIONNEL
   - Actif seulement si `shap` est installé ET qu'un modèle + échantillon sont fournis.
   - Sinon ignoré proprement (le projet n'a pas encore de modèle ML).

Sortie : dict structuré (repris par auditor_agent dans data/audit/xai_explanations.json).
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CURATED = ROOT / "data" / "curated"


# ---------------------------------------------------------------------------
# Périodes
# ---------------------------------------------------------------------------
def with_period(sales: pd.DataFrame, granularity: str = "M") -> pd.DataFrame:
    """Ajoute une colonne 'period' (ex. '2018-03') à partir de purchase_date."""
    s = sales.copy()
    s["_dt"] = pd.to_datetime(s["purchase_date"], errors="coerce")
    s = s.dropna(subset=["_dt"])
    if "is_revenue_order" in s.columns:
        s = s[s["is_revenue_order"] == True]  # noqa: E712
    freq = {"M": "%Y-%m", "Q": None, "Y": "%Y"}.get(granularity, "%Y-%m")
    if granularity == "Q":
        s["period"] = s["_dt"].dt.to_period("Q").astype(str)
    else:
        s["period"] = s["_dt"].dt.strftime(freq)
    return s


def pick_periods(s: pd.DataFrame, current: str | None, previous: str | None):
    """Choisit (précédente, courante). Par défaut : les deux derniers mois présents."""
    if current and previous:
        return previous, current
    # Mois à trop faible volume (début/fin de données incomplets) exclus : même règle que l'ETL
    # (< 30 % du volume médian), sinon une fin de période incomplète passerait pour une chute du CA.
    by_period = s.groupby("period").size()
    periods = sorted(by_period[by_period >= 0.3 * by_period.median()].index)
    if len(periods) < 2:
        return None, None
    return periods[-2], periods[-1]


# ---------------------------------------------------------------------------
# Décomposition Prix-Volume-Mix (bridge)
# ---------------------------------------------------------------------------
def pvm_bridge(s: pd.DataFrame, prev: str, curr: str, dim: str = "category") -> dict:
    """Décompose la variation de CA entre deux périodes, par segment `dim`."""
    def agg(period):
        d = s[s["period"] == period]
        g = d.groupby(dim).agg(units=("price", "size"), revenue=("revenue_brl", "sum")).reset_index()
        g["avg_price"] = g["revenue"] / g["units"].where(g["units"] > 0, 1)
        return g.set_index(dim)

    a0, a1 = agg(prev), agg(curr)
    segments = sorted(set(a0.index) | set(a1.index))
    rows, tot = [], {"volume": 0.0, "price": 0.0, "interaction": 0.0, "delta": 0.0}
    for seg in segments:
        u0 = float(a0.loc[seg, "units"]) if seg in a0.index else 0.0
        u1 = float(a1.loc[seg, "units"]) if seg in a1.index else 0.0
        p0 = float(a0.loc[seg, "avg_price"]) if seg in a0.index else 0.0
        p1 = float(a1.loc[seg, "avg_price"]) if seg in a1.index else 0.0
        vol = (u1 - u0) * p0
        pri = (p1 - p0) * u0
        inter = (p1 - p0) * (u1 - u0)
        delta = (u1 * p1) - (u0 * p0)
        rows.append({dim: seg, "rev_prev": round(u0 * p0, 2), "rev_curr": round(u1 * p1, 2),
                     "delta": round(delta, 2), "volume_effect": round(vol, 2),
                     "price_effect": round(pri, 2), "interaction_effect": round(inter, 2),
                     "units_prev": u0, "units_curr": u1,
                     "avg_price_prev": round(p0, 2), "avg_price_curr": round(p1, 2)})
        for k, v in (("volume", vol), ("price", pri), ("interaction", inter), ("delta", delta)):
            tot[k] += v

    rev0 = float(s[s["period"] == prev]["revenue_brl"].sum())
    rev1 = float(s[s["period"] == curr]["revenue_brl"].sum())
    return {
        "dimension": dim, "period_prev": prev, "period_curr": curr,
        "revenue_prev": round(rev0, 2), "revenue_curr": round(rev1, 2),
        "total_delta": round(rev1 - rev0, 2),
        "pct_change": round((rev1 - rev0) / rev0 * 100, 2) if rev0 else None,
        "effects": {k: round(v, 2) for k, v in tot.items() if k != "delta"},
        # Contrôle de réconciliation : somme des effets == variation totale (aux arrondis près).
        "reconciles": abs((tot["volume"] + tot["price"] + tot["interaction"]) - (rev1 - rev0)) < 1.0,
        "by_segment": rows,
    }


def top_movers(bridge: dict, n: int = 3) -> dict:
    rows = sorted(bridge["by_segment"], key=lambda r: r["delta"])
    return {"down": rows[:n], "up": list(reversed(rows[-n:]))}


# ---------------------------------------------------------------------------
# Phrase d'explication (gabarit ; LLM optionnel branché par l'auditeur)
# ---------------------------------------------------------------------------
def explain_sentence(bridge: dict) -> str:
    d = bridge["total_delta"]
    pct = bridge["pct_change"]
    direction = "augmenté" if d >= 0 else "diminué"
    movers = top_movers(bridge, n=1)
    main = (movers["up"] if d >= 0 else movers["down"])[0]
    eff = bridge["effects"]
    lead = max(eff, key=lambda k: abs(eff[k]))
    lead_fr = {"volume": "les volumes vendus", "price": "les prix moyens",
               "interaction": "un effet combiné prix/volume"}[lead]
    return (f"Le CA a {direction} de {abs(d):,.0f} BRL "
            f"({pct:+.1f} %) entre {bridge['period_prev']} et {bridge['period_curr']}, "
            f"principalement porté par {lead_fr} ; "
            f"segment le plus déterminant : {main[bridge['dimension']]} "
            f"({main['delta']:+,.0f} BRL).")


# ---------------------------------------------------------------------------
# Explication d'un KPI changeant
# ---------------------------------------------------------------------------
def explain_kpi_change(sales: pd.DataFrame, dim: str = "category",
                       current: str | None = None, previous: str | None = None,
                       granularity: str = "M") -> dict | None:
    s = with_period(sales, granularity)
    prev, curr = pick_periods(s, current, previous)
    if not prev or not curr:
        return {"available": False, "reason": "Moins de deux périodes exploitables dans fact_sales."}
    bridge = pvm_bridge(s, prev, curr, dim)
    return {"available": True, "kpi": "revenue_total_brl", "bridge": bridge,
            "top_movers": top_movers(bridge), "sentence": explain_sentence(bridge)}


# ---------------------------------------------------------------------------
# Modèle ML « mauvaise note » + SHAP
# Cible : l'avis client sera-t-il mauvais (1 ou 2 étoiles) ? Sur les commandes livrées.
# Le modèle est ré-entraîné seulement si fact_orders est plus récent que lui.
# ---------------------------------------------------------------------------
XAI_DIR = ROOT / "data" / "xai"
MODEL_PATH = XAI_DIR / "review_model.joblib"
SAMPLE_PATH = XAI_DIR / "review_sample.csv"      # commandes de test expliquées (+ id, vérité, proba)
METRICS_PATH = XAI_DIR / "review_model.json"
SAMPLE_SIZE = 500

FEATURES_FR = {
    "delivery_days": "Délai de livraison (jours)",
    "delay_days": "Retard vs date promise (jours)",
    "items_count": "Nombre d'articles",
    "n_sellers": "Nombre de vendeurs",
    "items_value": "Valeur des articles (BRL)",
    "freight_total": "Frais de port (BRL)",
    "max_installments": "Nombre d'échéances",
    "pay_boleto": "Paiement par boleto",
    "customer_sp": "Client à São Paulo",
}


def review_dataset(orders: pd.DataFrame, sales: pd.DataFrame):
    """(X, y, order_ids) : commandes livrées avec avis ; NaN remplacés par la médiane."""
    o = orders[(orders["is_revenue_order"] == True) & orders["review_score"].notna()]  # noqa: E712
    o = o.join(sales.groupby("order_id")["seller_id"].nunique().rename("n_sellers"), on="order_id")
    X = pd.DataFrame({
        "delivery_days": o["delivery_days"], "delay_days": o["delay_days"],
        "items_count": o["items_count"], "n_sellers": o["n_sellers"],
        "items_value": o["items_value"], "freight_total": o["freight_total"],
        "max_installments": o["max_installments"],
        "pay_boleto": (o["main_payment_type"] == "boleto").astype(int),
        "customer_sp": (o["customer_state"] == "SP").astype(int),
    })
    return X.fillna(X.median()), (o["review_score"] <= 2).astype(int), o["order_id"]


def train_review_model(force: bool = False) -> dict:
    """Entraîne le classifieur, sauvegarde modèle + échantillon de test + métriques."""
    import json
    import joblib
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    src = CURATED / "fact_orders.csv"
    if not force and MODEL_PATH.exists() and METRICS_PATH.exists() \
            and MODEL_PATH.stat().st_mtime >= src.stat().st_mtime:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    X, y, ids = review_dataset(pd.read_csv(src),
                               pd.read_csv(CURATED / "fact_sales.csv", usecols=["order_id", "seller_id"]))
    Xtr, Xte, ytr, yte, _, id_te = train_test_split(X, y, ids, test_size=0.25,
                                                    random_state=42, stratify=y)
    model = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42).fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]

    XAI_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    sample = Xte.assign(order_id=id_te.values, bad_review=yte.values, proba_bad=proba.round(4))
    sample.sample(min(SAMPLE_SIZE, len(sample)), random_state=42).to_csv(SAMPLE_PATH, index=False)

    metrics = {"target": "avis 1-2 étoiles", "algorithm": "GradientBoostingClassifier",
               "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
               "base_rate": round(float(y.mean()), 4),
               # AUC plutôt que l'exactitude : classes déséquilibrées (~13 % de mauvaises notes)
               "auc_test": round(float(roc_auc_score(yte, proba)), 4)}
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def shap_attributions(model_path: str | None = None, sample_path: str | None = None,
                      max_features: int = 10) -> dict:
    """Attributions SHAP (TreeExplainer) du modèle sur l'échantillon de test.
    Par défaut : modèle « mauvaise note » du projet ; XAI_MODEL_PATH / XAI_SAMPLE_PATH le remplacent."""
    model_path = model_path or os.environ.get("XAI_MODEL_PATH") or MODEL_PATH
    sample_path = sample_path or os.environ.get("XAI_SAMPLE_PATH") or SAMPLE_PATH
    if not Path(model_path).exists() or not Path(sample_path).exists():
        return {"available": False, "reason": "Modèle ML absent (lancer train_review_model)."}
    try:
        import joblib
        import numpy as np
        import shap
    except Exception:
        return {"available": False, "reason": "Paquet `shap` non installé."}
    try:
        model = joblib.load(model_path)
        features = list(model.feature_names_in_)
        X = pd.read_csv(sample_path)[features]
        explainer = shap.TreeExplainer(model)
        sv = np.asarray(explainer.shap_values(X))
        mean_abs = np.abs(sv).mean(axis=0)
        order = mean_abs.argsort()[::-1][:max_features]
        feats = []
        for i in order:
            # sens : une valeur plus haute de la variable augmente-t-elle le risque ?
            corr = np.corrcoef(X.iloc[:, i], sv[:, i])[0, 1] if X.iloc[:, i].std() > 0 else 0.0
            feats.append({"feature": features[i], "label": FEATURES_FR.get(features[i], features[i]),
                          "importance": round(float(mean_abs[i]), 5),
                          "direction": "hausse du risque" if corr > 0 else "baisse du risque"})
        base = float(np.ravel(explainer.expected_value)[0])
        sentence = ""
        if len(feats) >= 3:
            sentence = (f"Le facteur qui influence le plus le risque de mauvaise note est "
                        f"« {feats[0]['label']} » (une valeur plus élevée entraîne une "
                        f"{feats[0]['direction']}), suivi de « {feats[1]['label']} » "
                        f"et « {feats[2]['label']} ».")
        metrics = {}
        if Path(model_path) == MODEL_PATH and METRICS_PATH.exists():
            import json
            metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        return {"available": True, "n_samples": int(len(X)), "model": metrics,
                "base_value_logodds": round(base, 4),
                "base_proba": round(float(1 / (1 + np.exp(-base))), 4),
                "top_features": feats, "sentence": sentence}
    except Exception as e:
        return {"available": False, "reason": f"Échec SHAP : {str(e)[:200]}"}


# ---------------------------------------------------------------------------
# Agrégateur appelé par l'auditeur
# ---------------------------------------------------------------------------
def run_xai(sales: pd.DataFrame, dims=("category",),
            current: str | None = None, previous: str | None = None) -> dict:
    changes = {}
    for dim in dims:
        try:
            changes[dim] = explain_kpi_change(sales, dim=dim, current=current, previous=previous)
        except Exception as e:
            changes[dim] = {"available": False, "reason": str(e)[:200]}
    try:
        train_review_model()
    except Exception as e:
        print(f"  (XAI : entraînement du modèle impossible : {str(e)[:120]})")
    return {"kpi_change_explanations": changes, "model_explanations": shap_attributions()}


if __name__ == "__main__":
    import json
    df = pd.read_csv(CURATED / "fact_sales.csv")
    print(json.dumps(run_xai(df), indent=2, ensure_ascii=False, default=str))
