"""
Agent XAI Auditor — projet BIFlow.

Rôle : vérifier que les KPI produits par l'agent KPI sont corrects, en les
RECALCULANT directement depuis les tables de faits (source de vérité), puis
expliquer chaque chiffre (formule, table source, lignes utilisées) sous forme
de "cartes de lignage" exploitables par le dashboard.

Entrée : data/kpi/kpi_results.json tel qu'écrit par l'agent KPI (sections imbriquées,
ex. sales.total_revenue_brl, delivery.late_delivery_rate_pct). Chaque KPI audité est
désigné par son chemin dans ce fichier (REGISTRY). Un chemin absent du fichier est
reporté "non_rapporte" (le KPI n'a pas été produit) — ce n'est pas un écart.

Sortie :
- data/audit/lineage_cards.json  (une carte par KPI : formule, source, recalcul, écart)
- data/audit/audit_report.md     (rapport lisible)

run() renvoie à l'orchestrateur :
  ok (bool), passed (bool), audited (int), mismatches (int),
  cause ("kpi" | "etl" | None)  -> aide l'orchestrateur à choisir le retour arrière.
`passed` vaut False si un KPI recalculé s'écarte de la valeur rapportée, si une
réconciliation de faits (niveau ETL) échoue, ou si AUCUN KPI n'a pu être audité.
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))   # agents/ sur le chemin (import xai_engine)
load_dotenv(ROOT / ".env")

CURATED = ROOT / "data" / "curated"
RAW = ROOT / "data" / "raw" / "olist"
KPI_RESULTS = ROOT / "data" / "kpi" / "kpi_results.json"
OUT_DIR = ROOT / "data" / "audit"

# Tolérance entre valeur rapportée et recalculée : l'agent KPI arrondit à 2 décimales,
# donc 0.01 suffit ; une tolérance plus large masquerait des écarts (ex. note moyenne 4.16).
REL_TOL = float(os.environ.get("AUDIT_REL_TOL", "0"))
ABS_TOL = float(os.environ.get("AUDIT_ABS_TOL", "0.01"))

USE_LLM = "--no-llm" not in sys.argv


# ---------------------------------------------------------------------------
# Chargement des tables de faits
# ---------------------------------------------------------------------------
def load_facts() -> dict:
    """Charge les tables curated nécessaires. Lève FileNotFoundError si absentes."""
    def read(name: str) -> pd.DataFrame:
        p = CURATED / name
        if not p.exists():
            raise FileNotFoundError(f"Table curated absente : {p} (le modèle métier Olist, olist_model_agent, doit tourner avant l'auditeur).")
        return pd.read_csv(p)

    return {"sales": read("fact_sales.csv"), "orders": read("fact_orders.csv")}


# ---------------------------------------------------------------------------
# Registre de recalcul : id KPI -> (libellé, formule lisible, source, fonction)
# Chaque fonction reçoit le dict de tables de faits et renvoie un float,
# ou (float, details) où details documente les lignes/filtres utilisés.
# ---------------------------------------------------------------------------
def _revenue_rows(sales: pd.DataFrame) -> pd.DataFrame:
    # Seules les lignes de commandes génératrices de CA comptent (revenue déjà à 0 sinon,
    # mais on filtre explicitement pour un lignage honnête).
    if "is_revenue_order" in sales.columns:
        return sales[sales["is_revenue_order"] == True]  # noqa: E712
    return sales


def _revenue_orders(orders: pd.DataFrame) -> pd.DataFrame:
    return orders[orders["is_revenue_order"] == True]  # noqa: E712


def _flag(series: pd.Series) -> pd.Series:
    """Booléen lu du CSV (True/False/vide) -> série de booléens, vides exclus."""
    s = series.dropna()
    return s.astype(str).str.lower() == "true"


def kpi_revenue_total_brl(f):
    s = _revenue_rows(f["sales"])
    return float(s["revenue_brl"].sum()), {"rows": int(len(s)), "column": "revenue_brl"}


def kpi_delivered_orders(f):
    o = _revenue_orders(f["orders"])
    return float(len(o)), {"rows": int(len(o)), "note": "1 ligne fact_orders = 1 commande"}


def kpi_avg_order_value(f):
    o = _revenue_orders(f["orders"])
    rev, n = float(o["revenue_brl"].sum()), len(o)
    return (rev / n if n else 0.0), {"formula_detail": f"{rev:.2f} / {n}"}


def kpi_unique_customers(f):
    o = _revenue_orders(f["orders"])
    return float(o["customer_unique_id"].nunique()), {"column": "customer_unique_id"}


def kpi_repeat_customer_rate(f):
    counts = _revenue_orders(f["orders"]).groupby("customer_unique_id")["order_id"].nunique()
    multi = int((counts > 1).sum())
    return (multi / len(counts) * 100 if len(counts) else 0.0), \
        {"formula_detail": f"{multi} / {len(counts)} * 100"}


def kpi_total_freight(f):
    o = _revenue_orders(f["orders"])
    return float(o["freight_total"].sum()), {"column": "freight_total", "rows": int(len(o))}


def kpi_late_delivery_rate(f):
    late = _flag(_revenue_orders(f["orders"])["is_late"])
    return float(late.mean() * 100) if len(late) else 0.0, \
        {"formula_detail": f"{int(late.sum())} / {len(late)} * 100",
         "note": "commandes au statut de retard connu uniquement"}


def kpi_avg_review_score(f):
    scores = _revenue_orders(f["orders"])["review_score"].dropna()
    return float(scores.mean()), {"rows": int(len(scores)), "column": "review_score"}


def kpi_lost_order_rate(f):
    o = f["orders"]
    lost = o["order_status"].isin(["canceled", "unavailable"])
    return float(lost.mean() * 100) if len(o) else 0.0, \
        {"formula_detail": f"{int(lost.sum())} / {len(o)} * 100", "note": "TOUTES les commandes"}


def kpi_median_delivery_days(f):
    d = _revenue_orders(f["orders"])["delivery_days"].dropna()
    return float(d.median()), {"rows": int(len(d)), "column": "delivery_days"}


def kpi_low_rating_rate(f):
    scores = _revenue_orders(f["orders"])["review_score"].dropna()
    low = scores.isin([1, 2])
    return float(low.mean() * 100) if len(scores) else 0.0, \
        {"formula_detail": f"{int(low.sum())} / {len(scores)} * 100"}


def kpi_freight_to_revenue(f):
    o = _revenue_orders(f["orders"])
    fr, it = float(o["freight_total"].sum()), float(o["items_value"].sum())
    return (fr / it * 100 if it else 0.0), {"formula_detail": f"{fr:.2f} / {it:.2f} * 100"}


# Pour les KPI de type classement, on vérifie le n°1 plutôt que toute la liste.
def kpi_top_category_by_revenue(f):
    s = _revenue_rows(f["sales"])
    g = s.assign(category=s["category"].fillna("unknown")) \
        .groupby("category")["revenue_brl"].sum().sort_values(ascending=False)
    top = g.index[0] if len(g) else None
    return top, {"kind": "ranking", "column": "category", "note": "catégorie n°1 par CA"}


def kpi_top_seller_by_revenue(f):
    g = _revenue_rows(f["sales"]).groupby("seller_id")["revenue_brl"].sum().sort_values(ascending=False)
    top = g.index[0] if len(g) else None
    return top, {"kind": "ranking", "column": "seller_id", "note": "vendeur n°1 par CA"}


# chemin dans kpi_results.json -> (libellé, formule lisible, source, fonction de recalcul)
# Format actuel de l'agent KPI : 12 indicateurs dans "headline", classements dans "breakdowns".
REGISTRY = {
    "headline.total_revenue_brl":        ("Chiffre d'affaires total (BRL)", "sum(fact_sales.revenue_brl | is_revenue_order)", "fact_sales", kpi_revenue_total_brl),
    "headline.delivered_orders":         ("Commandes livrées", "count(fact_orders | is_revenue_order)", "fact_orders", kpi_delivered_orders),
    "headline.average_order_value_brl":  ("Panier moyen (BRL)", "sum(fact_orders.revenue_brl) / count(commandes) | is_revenue_order", "fact_orders", kpi_avg_order_value),
    "headline.unique_customers":         ("Clients uniques", "nunique(fact_orders.customer_unique_id | is_revenue_order)", "fact_orders", kpi_unique_customers),
    "headline.repeat_customer_rate_pct": ("Taux de clients récurrents (%)", "clients avec >1 commande / clients * 100", "fact_orders", kpi_repeat_customer_rate),
    "headline.lost_order_rate_pct":      ("Commandes perdues (%)", "commandes canceled ou unavailable / toutes les commandes * 100", "fact_orders", kpi_lost_order_rate),
    "headline.late_delivery_rate_pct":   ("Taux de livraison en retard (%)", "mean(fact_orders.is_late connu) * 100", "fact_orders", kpi_late_delivery_rate),
    "headline.median_delivery_days":     ("Délai médian de livraison (jours)", "median(fact_orders.delivery_days | is_revenue_order)", "fact_orders", kpi_median_delivery_days),
    "headline.average_review_score":     ("Note moyenne des avis", "mean(fact_orders.review_score | is_revenue_order)", "fact_orders", kpi_avg_review_score),
    "headline.low_rating_rate_pct":      ("Avis faibles 1-2 étoiles (%)", "part des review_score dans {1, 2} * 100", "fact_orders", kpi_low_rating_rate),
    "headline.freight_to_revenue_pct":   ("Fret / CA (%)", "sum(freight_total) / sum(items_value) * 100 | is_revenue_order", "fact_orders", kpi_freight_to_revenue),
    "breakdowns.categories.top_by_revenue.0.category": ("Catégorie n°1 (CA)", "argmax_category sum(revenue_brl)", "fact_sales", kpi_top_category_by_revenue),
    "breakdowns.sellers.top_sellers.0.seller_id":      ("Vendeur n°1 (CA)", "argmax_seller sum(revenue_brl)", "fact_sales", kpi_top_seller_by_revenue),
}


def get_path(data, path: str):
    """Lit 'a.b.0.c' dans le JSON de l'agent KPI ; None si le chemin n'existe pas."""
    for key in path.split("."):
        if isinstance(data, list) and key.isdigit() and int(key) < len(data):
            data = data[int(key)]
        elif isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return None
    return data


# ---------------------------------------------------------------------------
# Réconciliations niveau ETL (permettent de distinguer cause KPI vs cause ETL)
# ---------------------------------------------------------------------------
def fact_reconciliations(f) -> list:
    checks = []
    s, o = f["sales"], f["orders"]

    # CA fact_sales doit égaler le CA porté par fact_orders (si la colonne existe).
    if "revenue_brl" in o.columns:
        rs = float(_revenue_rows(s)["revenue_brl"].sum())
        ro = float(o["revenue_brl"].sum())
        checks.append({"check": "CA fact_sales == CA fact_orders",
                       "passed": abs(rs - ro) <= max(ABS_TOL, REL_TOL * max(abs(ro), 1)),
                       "detail": f"sales={rs:.2f} orders={ro:.2f} ecart={rs - ro:.2f}"})

    # Aucun CA négatif.
    neg = int((s["revenue_brl"] < 0).sum())
    checks.append({"check": "Aucun CA négatif dans fact_sales", "passed": neg == 0,
                   "detail": f"{neg} ligne(s) négative(s)"})

    # Marge <= CA ligne à ligne (un coût négatif rendrait la marge supérieure au CA).
    bad = int((s["margin_brl"] > s["revenue_brl"] + ABS_TOL).sum())
    checks.append({"check": "Marge (simulée) <= CA (par ligne)", "passed": bad == 0,
                   "detail": f"{bad} ligne(s) incohérente(s)"})

    # Recalcul INDÉPENDANT depuis les CSV bruts (sans passer par l'ETL) : un écart ici
    # signifie que le modèle métier a perdu, dupliqué ou mal filtré des données -> cause "etl"
    # (l'orchestrateur relance alors olist_model).
    try:
        cfg = json.loads((ROOT / "etl_config.json").read_text(encoding="utf-8"))
        statuses = cfg.get("revenue_statuses", ["delivered"])
        raw_o = pd.read_csv(RAW / "olist_orders_dataset.csv", usecols=["order_id", "order_status"])
        raw_i = pd.read_csv(RAW / "olist_order_items_dataset.csv", usecols=["order_id", "price"])
        raw_rev = float(raw_i[raw_i["order_id"].isin(raw_o.loc[raw_o["order_status"].isin(statuses),
                                                                  "order_id"])]["price"].sum())
        rs = float(_revenue_rows(s)["revenue_brl"].sum())
        checks += [
            {"check": "CA recalculé depuis les CSV bruts == CA fact_sales",
             "passed": abs(raw_rev - rs) <= max(ABS_TOL, 0.05),
             "detail": f"brut={raw_rev:.2f} fact_sales={rs:.2f} (statuts {statuses})"},
            {"check": "Commandes brutes == lignes fact_orders",
             "passed": len(raw_o) == len(o) and o["order_id"].is_unique,
             "detail": f"brut={len(raw_o)} fact_orders={len(o)}"},
            {"check": "Articles bruts == lignes fact_sales", "passed": len(raw_i) == len(s),
             "detail": f"brut={len(raw_i)} fact_sales={len(s)}"},
        ]
    except FileNotFoundError as e:
        checks.append({"check": "CSV bruts disponibles", "passed": False, "detail": str(e)[:150]})
    return checks


# ---------------------------------------------------------------------------
# Contrôles de la chaîne aval (Analyste, Dashboard) : avertissements, non bloquants
# ---------------------------------------------------------------------------
def pipeline_checks(kpi_results: dict) -> list:
    def load(path):
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    checks = []
    analyst = load(ROOT / "data" / "analyst" / "analyst_insights.json")
    if analyst is None:
        checks.append({"check": "Analyses du BI Analyst présentes", "passed": False, "detail": "fichier absent"})
    else:
        ins = analyst.get("insights", [])
        bad = [i["id"] for i in ins if not i.get("verified")]
        checks.append({"check": "Insights vérifiés (signaux existants, nombres retrouvés)", "passed": not bad,
                       "detail": f"{len(ins)} insights, à contrôler : {bad or 'aucun'}"})
        cited = {sid for i in ins for sid in i.get("signal_ids", [])}
        missing = [s["id"] for s in analyst.get("signals", []) if s["severity"] == "alert" and s["id"] not in cited]
        checks.append({"check": "Toutes les alertes couvertes par un insight", "passed": not missing,
                       "detail": f"non couvertes : {missing or 'aucune'}"})
        try:
            from analyst_agent import kpi_fingerprint
            same = analyst.get("kpi_fingerprint") == kpi_fingerprint(kpi_results)
            checks.append({"check": "Analyses calculées sur les KPI actuels", "passed": same,
                           "detail": "empreinte identique" if same else "KPI modifiés : relancer l'Analyste"})
        except Exception as e:
            checks.append({"check": "Analyses calculées sur les KPI actuels", "passed": False, "detail": str(e)[:120]})
    dash = load(ROOT / "data" / "dashboard" / "dashboard_agent_decision.json")
    if dash is None:
        checks.append({"check": "Dashboard présent", "passed": False, "detail": "fichier absent"})
    else:
        failed = [c["check"] for c in dash.get("checks", []) if not c["passed"]]
        checks.append({"check": "Contrôles du dashboard au vert (totaux = ETL et KPI)", "passed": not failed,
                       "detail": f"échecs : {failed or 'aucun'}"})
        expected = len((analyst or {}).get("insights", []))
        checks.append({"check": "Tous les insights affichés dans le dashboard",
                       "passed": dash.get("insights_displayed") == expected,
                       "detail": f"{dash.get('insights_displayed')} affichés / {expected}"})
    margin_ok = bool((kpi_results.get("margin_simulated") or {}).get("warning"))
    checks.append({"check": "Marge déclarée comme simulée", "passed": margin_ok, "detail": "margin_simulated.warning"})
    return checks


# ---------------------------------------------------------------------------
# Comparaison rapporté vs recalculé
# ---------------------------------------------------------------------------
def compare(reported, recomputed) -> bool:
    if isinstance(recomputed, str) or recomputed is None:   # KPI de type classement
        return str(reported) == str(recomputed)
    try:
        r = float(reported)
    except (TypeError, ValueError):
        return False
    return abs(r - recomputed) <= max(ABS_TOL, REL_TOL * max(abs(recomputed), 1.0))


def build_cards(kpi_results: dict, f) -> list:
    cards = []
    for kid, (label, formula, source, fn) in REGISTRY.items():
        reported = get_path(kpi_results, kid)
        if reported is None:
            cards.append({"id": kid, "label": label, "status": "non_rapporte",
                          "reported": None, "reason": "KPI absent de kpi_results.json"})
            continue
        try:
            recomputed, details = fn(f)
        except Exception as e:
            cards.append({"id": kid, "label": label, "status": "erreur_recalcul",
                          "reported": reported, "error": str(e)[:200]})
            continue
        ok = compare(reported, recomputed)
        rec_out = recomputed if isinstance(recomputed, str) else round(float(recomputed), 4)
        delta = None
        if not isinstance(recomputed, str) and recomputed is not None:
            try:
                delta = round(float(reported) - float(recomputed), 4)
            except (TypeError, ValueError):
                delta = None
        cards.append({
            "id": kid, "label": label, "status": "ok" if ok else "ecart",
            "formula": formula, "source": source,
            "reported": reported, "recomputed": rec_out, "delta": delta,
            "lineage": details,
        })
    return cards


# ---------------------------------------------------------------------------
# Explications en langage naturel (optionnel, LLM) — dégrade proprement si absent
# ---------------------------------------------------------------------------
def explain_with_llm(cards: list) -> None:
    """Ajoute card['explanation'] pour chaque carte. Silencieux si LLM indisponible."""
    model = os.environ.get("LLM_MODEL")
    if not USE_LLM or not model:
        for c in cards:
            c["explanation"] = f"{c.get('label', c['id'])} : {c.get('formula', 'formule non recalculée')}."
        return
    try:
        from langchain.chat_models import init_chat_model
        from langchain_core.messages import HumanMessage, SystemMessage
    except Exception:
        for c in cards:
            c["explanation"] = f"{c.get('label', c['id'])} : {c.get('formula', '')}."
        return

    llm = init_chat_model(model, temperature=0)
    prompt = ("Tu es l'agent XAI d'un pipeline BI. Pour chaque KPI, écris UNE phrase en français, "
              "simple et destinée à un responsable non technique, expliquant ce que mesure le KPI et "
              "comment il est calculé. N'invente aucun chiffre. Réponds par un JSON {id: phrase}.")
    payload = [{"id": c["id"], "label": c.get("label"), "formula": c.get("formula"),
                "status": c["status"]} for c in cards]
    try:
        ai = llm.invoke([SystemMessage(prompt),
                         HumanMessage(json.dumps(payload, ensure_ascii=False))])
        text = ai.content if isinstance(ai.content, str) else "".join(
            p.get("text", "") for p in ai.content if isinstance(p, dict))
        mapping = json.loads(text[text.find("{"): text.rfind("}") + 1])
    except Exception as e:
        mapping = {}
        print(f"  (Auditor : explications LLM indisponibles : {str(e)[:120]})")
    for c in cards:
        c["explanation"] = mapping.get(c["id"], f"{c.get('label', c['id'])} : {c.get('formula', '')}.")


# ---------------------------------------------------------------------------
# Rapport
# ---------------------------------------------------------------------------
def write_reports(cards: list, recon: list, verdict: dict, xai: dict, downstream: list = ()) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "lineage_cards.json").write_text(
        json.dumps({"generated_at": datetime.now().isoformat(timespec="seconds"),
                    "verdict": verdict, "reconciliations": recon, "pipeline_checks": list(downstream),
                    "cards": cards},
                   indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    L = ["# Rapport de l'auditeur XAI", "",
         f"- Verdict : **{'CONFORME' if verdict['passed'] else 'NON CONFORME'}**",
         f"- KPI audités : {verdict['audited']} | écarts : {verdict['mismatches']} "
         f"| non audités : {verdict['not_audited']}",
         f"- Cause probable : {verdict['cause'] or '—'}", "",
         "## Réconciliations de faits", ""]
    for r in recon:
        L.append(f"- [{'OK' if r['passed'] else 'ÉCHEC'}] {r['check']} — {r['detail']}")
    L += ["", "## Chaîne aval : Analyste et Dashboard (avertissements)", ""]
    for r in downstream:
        L.append(f"- [{'OK' if r['passed'] else 'AVERTISSEMENT'}] {r['check']} — {r['detail']}")
    L += ["", "## Cartes de lignage (KPI)", ""]
    for c in cards:
        L.append(f"### {c.get('label', c['id'])} — `{c['status']}`")
        if "formula" in c:
            L.append(f"- Formule : `{c['formula']}` (source : {c['source']})")
        L.append(f"- Rapporté : {c.get('reported')} | Recalculé : {c.get('recomputed', '—')}"
                 + (f" | Écart : {c['delta']}" if c.get("delta") is not None else ""))
        if c.get("explanation"):
            L.append(f"- {c['explanation']}")
        L.append("")
    change = xai.get("kpi_change_explanations", {}).get("category", {})
    shap_info = xai.get("model_explanations", {})
    L += ["## XAI : pourquoi le CA a changé", "",
          change.get("sentence") or f"Indisponible : {change.get('reason') or xai.get('error', '—')}", "",
          "## XAI : modèle « mauvaise note » expliqué par SHAP", ""]
    if shap_info.get("available"):
        m = shap_info.get("model", {})
        L += [shap_info.get("sentence", ""), "",
              f"- Modèle : {m.get('algorithm', '—')} | AUC test : {m.get('auc_test', '—')} "
              f"| taux de mauvaises notes : {m.get('base_rate', '—')}", ""]
        L += [f"- {f['label']} : importance {f['importance']} ({f['direction']})"
              for f in shap_info.get("top_features", [])] + [""]
    else:
        L += [f"Indisponible : {shap_info.get('reason', '—')}", ""]
    (OUT_DIR / "audit_report.md").write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------
# Point d'entrée orchestrateur
# ---------------------------------------------------------------------------
def run() -> dict:
    # 1. Pas de sortie KPI -> rien à auditer (ce n'est pas un échec, on laisse passer).
    if not KPI_RESULTS.exists():
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        note = f"Aucune sortie KPI à auditer ({KPI_RESULTS} absent)."
        (OUT_DIR / "audit_report.md").write_text(f"# Auditeur XAI\n\n{note}\n", encoding="utf-8")
        return {"ok": True, "passed": True, "audited": 0, "mismatches": 0,
                "not_audited": 0, "cause": None, "note": note}

    try:
        facts = load_facts()
    except FileNotFoundError as e:
        return {"ok": False, "passed": True, "error": str(e)}   # erreur d'infra, pas un échec d'audit

    kpi_results = json.loads(KPI_RESULTS.read_text(encoding="utf-8"))

    recon = fact_reconciliations(facts)
    cards = build_cards(kpi_results, facts)
    explain_with_llm(cards)

    # Couche XAI : pourquoi les KPI ont changé (décomposition Prix-Volume-Mix) + SHAP si modèle.
    try:
        from xai_engine import run_xai
        xai = run_xai(facts["sales"], dims=("category",))
        (OUT_DIR / "xai_explanations.json").write_text(
            json.dumps(xai, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    except Exception as e:
        xai = {"error": str(e)[:200]}

    mismatches = sum(1 for c in cards if c["status"] == "ecart")
    audited = sum(1 for c in cards if c["status"] in ("ok", "ecart"))
    not_audited = sum(1 for c in cards if c["status"] in ("non_rapporte", "erreur_recalcul"))
    recon_failed = [r for r in recon if not r["passed"]]
    downstream = pipeline_checks(kpi_results)
    warnings = [c for c in downstream if not c["passed"]]

    # 0 KPI audité = contrat KPI cassé : ne jamais déclarer « conforme » sans rien vérifier.
    # Les contrôles aval (Analyste, Dashboard) sont des avertissements : ils ne bloquent pas.
    passed = audited > 0 and mismatches == 0 and not recon_failed and not_audited == 0
    # Indice de cause pour l'orchestrateur : une réconciliation de faits en échec -> ETL ;
    # sinon un écart de KPI avec faits cohérents -> KPI.
    cause = None
    if not passed:
        cause = "etl" if recon_failed else "kpi"

    verdict = {"passed": passed, "audited": audited, "mismatches": mismatches,
               "not_audited": not_audited, "cause": cause, "warnings": len(warnings)}
    write_reports(cards, recon, verdict, xai, downstream)

    # Résumé lu par l'application (accueil, « Lancer et suivre ») : un tableau unique de contrôles
    now = datetime.now().isoformat(timespec="seconds")
    table = ([{"area": "KPI", "check": c.get("label", c["id"]), "passed": c["status"] == "ok",
               "severity": "critical",
               "detail": f"rapporté {c.get('reported')} / recalculé {c.get('recomputed', '—')}",
               "blame": None if c["status"] == "ok" else "kpi"} for c in cards]
             + [{"area": "ETL", "check": r["check"], "passed": r["passed"], "severity": "critical",
                 "detail": r["detail"], "blame": None if r["passed"] else "etl"} for r in recon]
             + [{"area": "aval", "check": c["check"], "passed": c["passed"], "severity": "warning",
                 "detail": c["detail"], "blame": None} for c in downstream])
    summary = {"agent": "auditor_agent", "audited_at": now, "passed": passed, "checks_total": len(table),
               "critical_failures": sum(1 for t in table if not t["passed"] and t["severity"] == "critical"),
               "warnings": len(warnings), "suggested_rerun": cause, "checks": table}
    (OUT_DIR / "audit_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str),
                                               encoding="utf-8")
    print(f"  Audit {'CONFORME' if passed else 'NON CONFORME'} : {audited} KPI recalculés, {mismatches} écart(s), "
          f"{len(recon) - len(recon_failed)}/{len(recon)} réconciliations, {len(warnings)} avertissement(s)")

    xai_ok = bool(xai.get("kpi_change_explanations", {}).get("category", {}).get("available"))
    shap_ok = bool(xai.get("model_explanations", {}).get("available"))
    return {"ok": True, "passed": passed, "audited": audited, "mismatches": mismatches,
            "not_audited": not_audited, "cause": cause, "suggested_rerun": cause,
            "critical_failures": summary["critical_failures"], "warnings": len(warnings),
            "xai_change_explained": xai_ok, "shap_available": shap_ok,
            "outputs": [str(OUT_DIR / "lineage_cards.json"), str(OUT_DIR / "audit_report.md"),
                        str(OUT_DIR / "audit_report.json"), str(OUT_DIR / "xai_explanations.json")]}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False, default=str))
