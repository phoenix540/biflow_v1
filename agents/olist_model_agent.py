"""Olist Model Agent BIFlow (couche métier) : construit le modèle en étoile métier d'Olist.

Le socle générique (etl_agent.py, kpi_agent.py) fonctionne sur n'importe quel dossier de CSV.
Cette couche métier, propre au domaine Olist, ajoute les règles qu'aucun algorithme générique ne peut
deviner : statuts comptés dans le CA, coût et marge, retards (comparaison en jours), un avis par commande,
traductions validées par un humain, calendrier avec jours fériés et événements.

Elle produit le modèle utilisé par l'agent KPI métier, le BI Analyst, le Dashboard, l'Auditor,
le moteur XAI et l'application manager :
  data/curated/fact_sales.csv, fact_orders.csv, dim_*.csv, reviews_clean.csv
  data/etl/olist_model_decision.json, olist_correction_log.json, olist_model_report.md

Si les fichiers Olist sont absents (autre jeu de données), l'agent se signale "non applicable" et
l'orchestrateur termine après le socle générique.

Les transformations sont déterministes (reproductibles). L'agent :
  - lit la décision du Profiler et n'applique que les traductions validées par un humain,
  - journalise chaque correction (pour l'Auditor),
  - valide le résultat par des contrôles,
  - (optionnel) demande au LLM si le plan du Profiler est couvert. Le LLM ne touche jamais aux données.
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
# Certains modèles Gemini ignorent "temperature" et le signalent à chaque appel : avertissement sans effet
import warnings  # noqa: E402
warnings.filterwarnings("ignore", message=".*fixed sampling defaults.*")

RAW = ROOT / "data" / "raw" / "olist"
CURATED = ROOT / "data" / "curated"
OUT = ROOT / "data" / "etl"
DECISION_FILE = OUT / "olist_model_decision.json"
LOG_FILE = OUT / "olist_correction_log.json"
REPORT_FILE = OUT / "olist_model_report.md"
REQUIRED_FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "geo": "olist_geolocation_dataset.csv",
    "translation": "product_category_name_translation.csv",
    "reviews": "olist_order_reviews_dataset.csv",
}
PROFILER_DECISION = ROOT / "data" / "profile" / "profiler_agent_decision.json"
CONFIG_PATH = ROOT / "etl_config.json"
VALIDATED = ROOT / "data" / "validated_translations.json"

# --------------------------------------------------------------------------
# Configuration (modifiable sans toucher au code)
# --------------------------------------------------------------------------
DEFAULT_CONFIG = {
    # Définition du chiffre d'affaires : somme des prix des articles de ces statuts
    "revenue_statuses": ["delivered"],
    # Source du coût : "production_cost" = colonne production_cost_brl ajoutée à olist_products_dataset
    # (repli sur le % du prix si le coût manque) ; "estimated_pct" = coût = % du prix (ancienne hypothèse)
    "cost_source": "production_cost",
    "default_cost_pct": 0.70,
    "cost_pct_by_category": {},          # exceptions par catégorie, à justifier avant de remplir
    # Devise : à vérifier et dater
    "brl_to_tnd": 0.62,
    "brl_to_tnd_date": "A_COMPLETER",
    # Traductions validées par un humain (le Profiler les propose, tu les valides ici)
    "approved_translations": {
        "pc_gamer": "pc_gamer",
        "portateis_cozinha_e_preparadores_de_alimentos": "portable_kitchen_and_food_preparers",
    },
    # Coordonnées valides pour le Brésil (le reste est considéré comme aberrant)
    "brazil_bounds": {"lat_min": -34.0, "lat_max": 6.0, "lng_min": -75.0, "lng_max": -28.0},
    # Un mois avec moins de 30 % du volume médian est marqué "non fiable" (risque de fausses alertes)
    "low_volume_threshold": 0.3,
    # Événements pour expliquer les variations : à compléter avec des dates vérifiées
    "events": [
        {"name": "Black Friday 2017", "start": "2017-11-24", "end": "2017-11-24", "source": "à vérifier"},
    ],
}


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(json.dumps(DEFAULT_CONFIG, indent=2, ensure_ascii=False), encoding="utf-8")
    saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg = {**DEFAULT_CONFIG, **saved}   # une clé absente prend la valeur par défaut
    # Traductions : défauts < translation_overrides (config) < validation humaine (page Streamlit)
    approved = {**DEFAULT_CONFIG["approved_translations"],
                **saved.get("approved_translations", {}),
                **saved.get("translation_overrides", {})}
    if VALIDATED.exists():
        human = json.loads(VALIDATED.read_text(encoding="utf-8"))
        approved.update(human.get("translations", {}))
        for rejected in human.get("rejected", []):
            approved.pop(rejected, None)
    cfg["approved_translations"] = approved
    return cfg


# --------------------------------------------------------------------------
# Journal des corrections
# --------------------------------------------------------------------------
class CorrectionLog:
    def __init__(self):
        self.entries = []

    def add(self, step, action, rows_affected, reason):
        self.entries.append({
            "step": step, "action": action, "rows_affected": int(rows_affected),
            "reason": reason, "timestamp": datetime.now().isoformat(timespec="seconds"),
        })
        print(f"  [{step}] {action} -> {int(rows_affected)} lignes")


# --------------------------------------------------------------------------
# OUTILS (fonctions déterministes)
# --------------------------------------------------------------------------
def is_applicable() -> bool:
    """La couche métier ne s'applique qu'au jeu de données Olist (tous les fichiers attendus présents)."""
    return all((RAW / f).exists() for f in REQUIRED_FILES.values())


def load_tables() -> dict:
    tables = {}
    for name, fname in REQUIRED_FILES.items():
        path = RAW / fname
        if not path.exists():
            raise FileNotFoundError(f"Fichier manquant : {path}")
        tables[name] = pd.read_csv(path)
    # Jours fériés : facultatifs (le calendrier fonctionne sans)
    hol = RAW / "br_holidays.csv"
    tables["holidays"] = pd.read_csv(hol) if hol.exists() else pd.DataFrame(
        columns=["date", "localName", "name"])
    return tables


def tool_clean_orders(orders, cfg, log):
    """Dates en vraies dates, statut de CA, délais et retards de livraison."""
    for c in ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
              "order_delivered_customer_date", "order_estimated_delivery_date"]:
        before = orders[c].isna().sum()
        orders[c] = pd.to_datetime(orders[c], errors="coerce")
        log.add("dates", f"Conversion texte -> date : {c}", len(orders),
                f"{orders[c].isna().sum() - before} valeurs illisibles mises à vide")

    orders["purchase_date"] = orders["order_purchase_timestamp"].dt.normalize()
    orders["is_revenue_order"] = orders["order_status"].isin(cfg["revenue_statuses"])
    log.add("statuts", "Commandes hors statuts du CA (conservées, marquées is_revenue_order=False)",
            (~orders["is_revenue_order"]).sum(), f"CA = statuts {cfg['revenue_statuses']} uniquement")

    p, d, e = (orders["order_purchase_timestamp"], orders["order_delivered_customer_date"],
               orders["order_estimated_delivery_date"])
    orders["delivery_days"] = (d - p).dt.days
    # La date estimée n'a pas d'heure (minuit) : on compare des JOURS, sinon une livraison
    # le jour promis à 15 h serait comptée en retard.
    d_day = d.dt.normalize()
    orders["delay_days"] = (d_day - e.dt.normalize()).dt.days
    late = (d_day > e.dt.normalize()).astype("boolean")
    late[d.isna() | e.isna()] = pd.NA
    orders["is_late"] = late
    log.add("livraison", "Colonnes delivery_days, delay_days, is_late calculées", d.notna().sum(),
            "Seulement pour les commandes qui ont une date de livraison")

    no_date = ((orders["order_status"] == "delivered") & d.isna()).sum()
    log.add("qualite", "Commandes 'delivered' sans date de livraison (signalées, conservées)", no_date,
            "Comptées dans le CA, mais sans délai de livraison")
    return orders


def tool_categories(products, translation, cfg, log):
    """Traductions : applique seulement celles validées par un humain. Aucun filtre de domaine."""
    p = products.merge(translation, on="product_category_name", how="left")
    no_cat = p["product_category_name"].isna().sum()
    log.add("categories", "Produits sans catégorie -> 'non_defini'", no_cat, "Valeur manquante dans la source")

    approved = cfg.get("approved_translations", {})
    missing = p["product_category_name_english"].isna() & p["product_category_name"].notna()
    fill = missing & p["product_category_name"].isin(approved.keys())
    p.loc[fill, "product_category_name_english"] = p.loc[fill, "product_category_name"].map(approved)
    log.add("categories", "Traductions manquantes complétées (validées par un humain)", fill.sum(),
            "Source : etl_config.json + data/validated_translations.json (validation humaine)")

    still = missing & ~p["product_category_name"].isin(approved.keys())
    untranslated = {str(k): int(v) for k, v in p.loc[still, "product_category_name"].value_counts().items()}
    log.add("categories", "Catégories encore sans traduction (nom portugais conservé)", still.sum(),
            "À valider par un humain : " + ", ".join(untranslated) if untranslated else "aucune")

    p["category_pt"] = p["product_category_name"].fillna("non_defini")
    p["category"] = p["product_category_name_english"].fillna(p["product_category_name"]).fillna("non_defini")

    zero_w = p["product_weight_g"] == 0
    p.loc[zero_w, "product_weight_g"] = np.nan
    log.add("produits", "Poids = 0 remplacé par vide (valeur impossible)", zero_w.sum(),
            "Évite de fausser les moyennes de poids")

    # Colonnes ajoutées au jeu d'origine : nom du produit et coût de production unitaire
    if "product_name" not in p.columns:
        p["product_name"] = pd.NA
    p["product_name"] = p["product_name"].fillna("non_defini")
    if "production_cost_brl" not in p.columns:
        p["production_cost_brl"] = np.nan
    p["production_cost_brl"] = pd.to_numeric(p["production_cost_brl"], errors="coerce")
    bad_cost = p["production_cost_brl"].isna() | (p["production_cost_brl"] <= 0)
    p.loc[bad_cost, "production_cost_brl"] = np.nan
    log.add("produits", "Coût de production manquant ou <= 0 (mis à vide, repli sur le % du prix)",
            bad_cost.sum(), "Colonne production_cost_brl")

    dim = p.rename(columns={"product_weight_g": "weight_g", "product_length_cm": "length_cm",
                            "product_height_cm": "height_cm", "product_width_cm": "width_cm",
                            "product_photos_qty": "photos_qty",
                            # « lenght » : faute de frappe d'origine dans les colonnes Olist
                            "product_description_lenght": "description_length"})
    if "description_length" not in dim.columns:
        dim["description_length"] = np.nan
    cols = ["product_id", "product_name", "category", "category_pt", "production_cost_brl",
            "weight_g", "length_cm", "height_cm", "width_cm", "photos_qty", "description_length"]
    return dim[cols], untranslated


def tool_geo(geo, cfg, log):
    """Retire les coordonnées hors Brésil, puis une position par code postal."""
    b = cfg["brazil_bounds"]
    inside = (geo["geolocation_lat"].between(b["lat_min"], b["lat_max"])
              & geo["geolocation_lng"].between(b["lng_min"], b["lng_max"]))
    log.add("geo", "Lignes aux coordonnées hors du Brésil supprimées", (~inside).sum(),
            "Latitude ou longitude impossible pour le Brésil")
    geo = geo[inside]
    before = len(geo)
    out = (geo.groupby("geolocation_zip_code_prefix")
              .agg(lat=("geolocation_lat", "median"), lng=("geolocation_lng", "median"),
                   city=("geolocation_city", "first"), state=("geolocation_state", "first"))
              .reset_index().rename(columns={"geolocation_zip_code_prefix": "zip_code_prefix"}))
    log.add("geo", "Dédoublonnage : une position par code postal", before - len(out),
            f"{before} -> {len(out)} lignes")
    return out


def tool_reviews(reviews, log):
    """Un avis par commande (le plus récent). Renvoie les avis nettoyés et la version par commande."""
    for c in ["review_creation_date", "review_answer_timestamp"]:
        reviews[c] = pd.to_datetime(reviews[c], errors="coerce")
    log.add("avis", "Conversion texte -> date des colonnes d'avis", len(reviews), "2 colonnes de dates")
    dup = reviews.duplicated().sum()
    reviews = reviews.drop_duplicates()
    log.add("avis", "Lignes entièrement dupliquées supprimées", dup, "Lignes identiques")
    per_order = (reviews.sort_values("review_answer_timestamp")
                 .drop_duplicates("order_id", keep="last")[["order_id", "review_score"]])
    log.add("avis", "Commandes avec plusieurs avis : seul le plus récent est gardé pour fact_orders",
            len(reviews) - len(per_order), "Évite de compter une commande plusieurs fois")
    return reviews, per_order


def tool_build_orders(orders, items, payments, reviews_per_order, customers, log):
    """Table fact_orders : UNE ligne par commande (commandes, panier moyen, délais, satisfaction)."""
    it = (items.groupby("order_id")
               .agg(items_count=("order_item_id", "count"), items_value=("price", "sum"),
                    freight_total=("freight_value", "sum")).reset_index())
    main = (payments.sort_values("payment_value", ascending=False)
            .drop_duplicates("order_id")[["order_id", "payment_type"]]
            .rename(columns={"payment_type": "main_payment_type"}))
    pay = (payments.groupby("order_id")
                   .agg(payment_total=("payment_value", "sum"),
                        max_installments=("payment_installments", "max")).reset_index()
                   .merge(main, on="order_id"))

    fo = orders.merge(customers[["customer_id", "customer_unique_id", "customer_state"]],
                      on="customer_id", how="left")
    fo = (fo.merge(it, on="order_id", how="left")
            .merge(pay, on="order_id", how="left")
            .merge(reviews_per_order, on="order_id", how="left"))
    fo["items_count"] = fo["items_count"].fillna(0).astype(int)
    fo["items_value"] = fo["items_value"].fillna(0.0)
    fo["freight_total"] = fo["freight_total"].fillna(0.0)
    fo["revenue_brl"] = fo["items_value"].where(fo["is_revenue_order"], 0.0)
    fo["has_payment"] = fo["payment_total"].notna()
    comparable = fo["has_payment"] & (fo["items_count"] > 0)
    fo["payment_gap"] = (fo["items_value"] + fo["freight_total"] - fo["payment_total"]).where(comparable)
    fo["has_payment_gap"] = fo["payment_gap"].abs() > 1

    log.add("commandes", "Table fact_orders construite (1 ligne par commande)", len(fo),
            "Commandes, panier moyen, délais, retards, satisfaction")
    log.add("qualite", "Commandes sans aucun article (conservées, items_count = 0)",
            (fo["items_count"] == 0).sum(), "Surtout unavailable / canceled")
    log.add("qualite", "Commandes sans paiement (conservées)", (~fo["has_payment"]).sum(), "Signalées")
    log.add("qualite", "Commandes dont articles + livraison diffèrent du total payé (écart > 1)",
            fo["has_payment_gap"].sum(), "Marquées has_payment_gap ; le CA suit les prix des articles")

    cols = ["order_id", "customer_id", "customer_unique_id", "customer_state", "order_status",
            "is_revenue_order", "purchase_date", "order_delivered_customer_date",
            "order_estimated_delivery_date", "delivery_days", "delay_days", "is_late",
            "items_count", "items_value", "freight_total", "revenue_brl", "payment_total",
            "main_payment_type", "max_installments", "payment_gap", "has_payment_gap", "review_score"]
    return fo[cols]


def tool_build_sales(items, fact_orders, products_dim, sellers, cfg, log):
    """Table fact_sales : UNE ligne par article vendu (CA, coût, marge)."""
    o = fact_orders[["order_id", "customer_id", "customer_unique_id", "customer_state",
                     "order_status", "is_revenue_order", "purchase_date"]]
    seller_cols = [c for c in ("seller_id", "seller_name", "seller_state") if c in sellers.columns]
    f = (items.merge(o, on="order_id", how="left")
              .merge(products_dim[["product_id", "product_name", "category", "production_cost_brl"]],
                     on="product_id", how="left")
              .merge(sellers[seller_cols], on="seller_id", how="left"))
    if "seller_name" not in f.columns:
        f["seller_name"] = pd.NA
    f["seller_name"] = f["seller_name"].fillna(f["seller_id"])

    f["revenue_brl"] = f["price"].where(f["is_revenue_order"], 0.0)
    # Coût unitaire : 1 ligne d'article Olist = 1 unité vendue
    cost_pct = f["category"].map(cfg["cost_pct_by_category"]).fillna(cfg["default_cost_pct"])
    pct_cost = f["price"] * cost_pct
    if cfg.get("cost_source", "production_cost") == "production_cost":
        has_cost = f["production_cost_brl"].notna()
        unit_cost = f["production_cost_brl"].where(has_cost, pct_cost)
        f["cost_source"] = np.where(has_cost, "production_cost", "estimated_pct")
    else:
        unit_cost = pct_cost
        f["cost_source"] = "estimated_pct"
    f["cost_brl"] = unit_cost.where(f["is_revenue_order"], 0.0)
    f["margin_brl"] = f["revenue_brl"] - f["cost_brl"]
    f["is_sold_below_cost"] = f["is_revenue_order"] & (unit_cost > f["price"])
    f["revenue_tnd"] = f["revenue_brl"] * cfg["brl_to_tnd"]

    rev = f["is_revenue_order"]
    log.add("faits", "Table fact_sales construite (1 ligne par article vendu)", len(f),
            f"Source du coût : {cfg.get('cost_source')} ; "
            f"{int((f.loc[rev, 'cost_source'] == 'estimated_pct').sum())} lignes du CA en repli "
            f"{int(cfg['default_cost_pct'] * 100)} % du prix")
    log.add("qualite", "Articles du CA vendus sous le coût (coût > prix, marge négative, signalés)",
            f["is_sold_below_cost"].sum(),
            f"sur {int(rev.sum())} lignes du CA ; vérifier l'origine de production_cost_brl")
    cols = ["order_id", "order_item_id", "product_id", "product_name", "seller_id", "seller_name",
            "customer_id", "customer_unique_id", "customer_state", "seller_state", "category",
            "purchase_date", "order_status", "is_revenue_order", "price", "freight_value",
            "revenue_brl", "cost_brl", "cost_source", "margin_brl", "is_sold_below_cost", "revenue_tnd"]
    return f[cols]


def tool_build_dim_date(fact_orders, holidays, cfg, log):
    """Calendrier avec mois non fiables (volume trop faible), jours fériés et événements."""
    dates = fact_orders["purchase_date"].dropna()
    dim = pd.DataFrame({"date": pd.date_range(dates.min(), dates.max(), freq="D")})
    dim["year"], dim["month"], dim["day"] = dim.date.dt.year, dim.date.dt.month, dim.date.dt.day
    dim["quarter"], dim["weekday"] = dim.date.dt.quarter, dim.date.dt.day_name()
    dim["year_month"] = dim["date"].dt.to_period("M").astype(str)

    monthly = dates.dt.to_period("M").astype(str).value_counts()
    counts = pd.Series({m: int(monthly.get(m, 0)) for m in dim["year_month"].unique()})
    reliable = counts >= cfg["low_volume_threshold"] * counts.median()
    dim["orders_in_month"] = dim["year_month"].map(counts)
    dim["is_reliable_month"] = dim["year_month"].map(reliable)
    log.add("calendrier", "Mois à volume très faible marqués is_reliable_month=False", (~reliable).sum(),
            "Mois : " + ", ".join(counts.index[~reliable]) + " (risque de fausses alertes de baisse)")

    # Événements commerciaux complémentaires (etl_config.json, ex. Black Friday)
    commercial = pd.Series("", index=dim.index)
    for ev in cfg.get("events", []):
        mask = (dim["date"] >= pd.Timestamp(ev["start"])) & (dim["date"] <= pd.Timestamp(ev["end"]))
        commercial[mask] = ev["name"]

    # Jours fériés brésiliens (br_holidays.csv) : source principale des événements
    h = holidays.copy()
    h["date"] = pd.to_datetime(h["date"], errors="coerce")
    h = h.dropna(subset=["date"])
    label = h["name"].fillna(h.get("localName", "")).astype(str)
    names = label.groupby(h["date"]).agg(" / ".join)          # plusieurs fériés le même jour
    dim["holiday_name"] = dim["date"].map(names).fillna("")
    dim["is_holiday"] = dim["holiday_name"] != ""
    covered = set(h["date"].dt.year)
    uncovered = sorted(set(dim["year"]) - covered)
    log.add("calendrier", "Jours fériés (br_holidays.csv) appliqués au calendrier", dim["is_holiday"].sum(),
            f"Années sans données de jours fériés : {uncovered}" if uncovered else "Toutes les années couvertes")

    # Colonne event = TOUS les événements du jour : jours fériés + événements commerciaux
    dim["event"] = [" / ".join(x for x in (h_, c_) if x) for h_, c_ in zip(dim["holiday_name"], commercial)]
    dim["event_type"] = np.select(
        [dim["is_holiday"] & (commercial != ""), dim["is_holiday"], commercial != ""],
        ["jour_ferie+commercial", "jour_ferie", "commercial"], default="")
    log.add("calendrier", "Événements appliqués au calendrier (jours fériés + événements commerciaux)",
            (dim["event"] != "").sum(),
            f"{int(dim['is_holiday'].sum())} jours fériés (br_holidays.csv) + "
            f"{int((commercial != '').sum())} jours d'événements commerciaux (etl_config.json)")
    return dim


def tool_build_dim_events(holidays, cfg):
    """Événements de la configuration + jours fériés, dans un format unique."""
    ev = pd.DataFrame(cfg.get("events", []), columns=["name", "start", "end", "source"])
    ev["type"] = "commercial"
    h = holidays.dropna(subset=["date"])
    hol = pd.DataFrame({"name": h["name"].fillna(h["localName"]), "start": h["date"], "end": h["date"],
                        "source": "br_holidays.csv", "type": "jour_ferie"})
    return pd.concat([ev, hol], ignore_index=True).sort_values("start", ignore_index=True)


def tool_validate(items, orders, fact_sales, fact_orders, products_dim, customers, sellers, log):
    checks = []

    def check(name, ok, detail):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    check("Nb lignes items = nb lignes fact_sales", len(items) == len(fact_sales),
          f"{len(items)} vs {len(fact_sales)}")
    check("Nb lignes orders = nb lignes fact_orders", len(orders) == len(fact_orders),
          f"{len(orders)} vs {len(fact_orders)}")
    check("order_id unique dans fact_orders", fact_orders["order_id"].is_unique, "")
    for col, ref, label in [("product_id", products_dim, "produit"), ("seller_id", sellers, "vendeur"),
                            ("customer_id", customers, "client")]:
        bad = (~fact_sales[col].isin(ref[col])).sum()
        check(f"Aucun {label} orphelin", bad == 0, f"{bad} orphelins")
    check("Aucune date d'achat vide", fact_sales["purchase_date"].notna().all(),
          f"{fact_sales['purchase_date'].isna().sum()} vides")
    check("Aucune catégorie vide", fact_sales["category"].notna().all(),
          f"{fact_sales['category'].isna().sum()} vides")

    delivered = orders.loc[orders["is_revenue_order"], "order_id"]
    expected = items.loc[items["order_id"].isin(delivered), "price"].sum()
    got = fact_sales["revenue_brl"].sum()
    check("CA fact_sales = somme des prix des commandes comptées", abs(expected - got) < 0.01,
          f"attendu {expected:,.2f} / obtenu {got:,.2f}")
    got_orders = fact_orders["revenue_brl"].sum()
    check("CA fact_orders = CA fact_sales", abs(got_orders - got) < 0.01,
          f"{got_orders:,.2f} vs {got:,.2f}")
    check("Aucun CA négatif", (fact_sales["revenue_brl"] >= 0).all(), "")
    rev = fact_sales["is_revenue_order"]
    check("Coût renseigné pour toutes les lignes du CA", fact_sales.loc[rev, "cost_brl"].notna().all(),
          f"{fact_sales.loc[rev, 'cost_brl'].isna().sum()} vides")
    check("Marge = CA - coût", np.allclose(fact_sales["margin_brl"],
                                            fact_sales["revenue_brl"] - fact_sales["cost_brl"]), "")

    for c in checks:
        log.add("validation", f"{'OK ' if c['passed'] else 'ECHEC'} {c['check']}", 0, c["detail"])
    return checks


# --------------------------------------------------------------------------
# AGENT : plan du Profiler, lecture LLM (optionnelle), rapports
# --------------------------------------------------------------------------
def read_profiler_plan() -> dict:
    if PROFILER_DECISION.exists():
        return json.loads(PROFILER_DECISION.read_text(encoding="utf-8"))
    print("  (profiler_agent_decision.json introuvable : l'ETL tourne sans plan du Profiler)")
    return {}


def text_of(msg) -> str:
    c = msg.content
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c


def llm_plan_review(plan: dict, log_entries: list, checks: list):
    """Le LLM compare le plan du Profiler aux corrections réellement faites. Il ne modifie aucune donnée."""
    model = os.environ.get("LLM_MODEL")
    if not model or "--no-llm" in sys.argv or not plan:
        return None
    try:
        from langchain.chat_models import init_chat_model
        from langchain_core.messages import HumanMessage, SystemMessage
        payload = {
            "plan_du_profiler": plan.get("etl_plan", []),
            "problemes": [{k: i.get(k) for k in ("id", "table", "column", "severity", "action_for_etl")}
                          for i in plan.get("issues", [])],
            "corrections_faites": [{"step": e["step"], "action": e["action"], "lignes": e["rows_affected"]}
                                   for e in log_entries],
            "controles": checks,
        }
        system = ("Tu es l'ETL Agent d'un pipeline BI. Compare le plan du Profiler aux corrections réellement "
                  "faites. Pour chaque étape du plan, dis en une ligne si elle est couverte (et par quelle "
                  "correction) ou non. Termine par la liste de ce qui reste à traiter. N'invente aucun "
                  "chiffre : utilise seulement ceux fournis. Texte simple en français, sans JSON.")
        from profiler_agent import invoke_with_retry   # mêmes réessais que le Profiler (503, 429 par minute)
        llm = init_chat_model(model, temperature=float(os.environ.get("LLM_TEMPERATURE", "0")))
        ai = invoke_with_retry(llm, [SystemMessage(system),
                                     HumanMessage(json.dumps(payload, ensure_ascii=False, default=str))])
        return text_of(ai).strip()
    except Exception as e:
        print(f"  (LLM indisponible pour la revue du plan : {str(e)[:100]})")
        return None


def run_model() -> dict:
    if not is_applicable():
        missing = [f for f in REQUIRED_FILES.values() if not (RAW / f).exists()]
        print(f"Couche métier Olist non applicable : fichiers absents de {RAW} ({', '.join(missing[:3])}...)")
        return {"ok": True, "applicable": False, "status": "not_applicable",
                "reason": "jeu de données non Olist : seul le socle générique (ETL + KPI) s'applique"}
    for d in (CURATED, OUT):
        d.mkdir(parents=True, exist_ok=True)
    cfg, log = load_config(), CorrectionLog()
    plan = read_profiler_plan()

    print("1) Chargement des tables brutes")
    t = load_tables()
    rows_before = {k: len(v) for k, v in t.items()}
    items_raw = t["items"].copy()

    print("2) Nettoyage")
    orders = tool_clean_orders(t["orders"], cfg, log)
    products_dim, untranslated = tool_categories(t["products"], t["translation"], cfg, log)
    geo = tool_geo(t["geo"], cfg, log)
    reviews, reviews_per_order = tool_reviews(t["reviews"], log)

    pending = [x for x in plan.get("proposed_translations", [])
               if x.get("original") not in cfg.get("approved_translations", {})]
    if pending:
        log.add("categories", "Traductions proposées par le Profiler, NON appliquées (à valider)",
                len(pending), ", ".join(x.get("original", "?") for x in pending))

    print("3) Construction du modèle")
    fact_orders = tool_build_orders(orders, t["items"], t["payments"], reviews_per_order, t["customers"], log)
    fact_sales = tool_build_sales(t["items"], fact_orders, products_dim, t["sellers"], cfg, log)
    dim_date = tool_build_dim_date(fact_orders, t["holidays"], cfg, log)
    dim_events = tool_build_dim_events(t["holidays"], cfg)

    print("4) Validation")
    checks = tool_validate(items_raw, orders, fact_sales, fact_orders, products_dim,
                           t["customers"], t["sellers"], log)

    print("5) Écriture des fichiers")
    reviews.to_csv(CURATED / "reviews_clean.csv", index=False)
    fact_sales.to_csv(CURATED / "fact_sales.csv", index=False)
    fact_orders.to_csv(CURATED / "fact_orders.csv", index=False)
    products_dim.to_csv(CURATED / "dim_products.csv", index=False)
    t["customers"].to_csv(CURATED / "dim_customers.csv", index=False)
    t["sellers"].to_csv(CURATED / "dim_sellers.csv", index=False)
    geo.to_csv(CURATED / "dim_geolocation.csv", index=False)
    dim_date.to_csv(CURATED / "dim_date.csv", index=False)
    dim_events.to_csv(CURATED / "dim_events.csv", index=False)
    LOG_FILE.write_text(
        json.dumps(log.entries, indent=2, ensure_ascii=False), encoding="utf-8")

    all_ok = all(c["passed"] for c in checks)
    review = llm_plan_review(plan, log.entries, checks)
    rev = fact_sales["is_revenue_order"]
    below = int(fact_sales["is_sold_below_cost"].sum())
    cost_note = (
        "Coût = production_cost_brl (colonne SIMULÉE ajoutée à olist_products_dataset : ~70 % du prix "
        f"médian du produit), repli {int(cfg['default_cost_pct'] * 100)} % du prix si absent. "
        f"{below} des {int(rev.sum())} articles du CA sont vendus sous le coût (prix ponctuellement plus bas)"
        if cfg.get("cost_source", "production_cost") == "production_cost" else
        f"Marge ESTIMÉE : coût = {int(cfg['default_cost_pct'] * 100)} % du prix (cost_source = estimated_pct)")
    holiday_years = sorted(set(pd.to_datetime(t["holidays"]["date"], errors="coerce").dt.year.dropna().astype(int)))
    assumptions = [
        cost_note,
        "Stock inexistant dans Olist : rotation du stock à simuler ou à retirer (décision à prendre)",
        f"Taux BRL->TND = {cfg['brl_to_tnd']} (date : {cfg['brl_to_tnd_date']}) : à vérifier",
        f"Événements : jours fériés de br_holidays.csv (années {holiday_years}) + événements commerciaux "
        "de etl_config.json (ex. Black Friday)",
        f"CA = somme des prix des articles, statuts {cfg['revenue_statuses']} (hors livraison)",
    ]
    decision = {
        "agent": "olist_model_agent", "run_at": datetime.now().isoformat(timespec="seconds"),
        "status": "success" if all_ok else "failed_validation",
        "config_used": cfg, "rows_before": rows_before,
        "rows_fact_sales": len(fact_sales), "rows_fact_orders": len(fact_orders),
        "checks": checks,
        "untranslated_categories_left": untranslated,
        "translations_pending_validation": [x.get("original") for x in pending],
        "assumptions": assumptions,
        "plan_review_by_llm": review,
        "outputs": ["data/curated/fact_sales.csv", "data/curated/fact_orders.csv", "data/curated/dim_*.csv",
                    "data/curated/reviews_clean.csv", "data/etl/olist_correction_log.json",
                    "data/etl/olist_model_report.md"],
        "next_agent": "olist_kpi" if all_ok else "STOP - corriger le modèle métier",
    }
    DECISION_FILE.write_text(
        json.dumps(decision, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    L = ["# Modèle métier Olist", "", f"Statut : **{decision['status']}**", "",
         f"- fact_sales : {len(fact_sales)} lignes (1 par article vendu)",
         f"- fact_orders : {len(fact_orders)} lignes (1 par commande)", "", "## Contrôles", ""]
    L += [f"- {'OK' if c['passed'] else 'ECHEC'} : {c['check']} ({c['detail']})" for c in checks]
    L += ["", "## Corrections", ""]
    L += [f"- [{e['step']}] {e['action']} : {e['rows_affected']} lignes ({e['reason']})" for e in log.entries]
    L += ["", "## À valider par un humain", ""]
    L += [f"- Traduction en attente : {x}" for x in decision["translations_pending_validation"]] or []
    L += [f"- Catégorie encore sans traduction : {k} ({v} produits)" for k, v in untranslated.items()]
    L += ["", "## Hypothèses et limites", ""] + [f"- {a}" for a in assumptions]
    if review:
        L += ["", "## Plan du Profiler : couverture (lecture LLM, indicative)", "", review]
    REPORT_FILE.write_text("\n".join(L), encoding="utf-8")

    print(f"\nTerminé. Statut : {decision['status']}. Voir data/etl/olist_model_report.md")
    return {"ok": all_ok, "applicable": True, "status": decision["status"], "rows_fact_sales": len(fact_sales),
            "rows_fact_orders": len(fact_orders),
            "checks_failed": [c["check"] for c in checks if not c["passed"]]}


def run() -> dict:
    """Point d'entrée appelé par l'orchestrateur."""
    return run_model()


if __name__ == "__main__":
    run()