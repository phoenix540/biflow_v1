"""BIFlow KPI Agent.

Calcule, de façon déterministe, un jeu RÉDUIT d'indicateurs à partir des tables préparées :
  - "headline"   : 12 indicateurs clés (une valeur chacun)
  - "breakdowns" : les découpages qui produisent les analyses (mois, États, catégories,
                   vendeurs, produits, événements, jours de la semaine)
  - "coverage"   : nombre de lignes utilisées (technique, pour l'Auditor)

Point d'entrée : run(), appelé par l'orchestrateur.
Sorties : data/kpi/kpi_results.json et data/kpi/kpi_report.md
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
CURATED = ROOT / "data" / "curated"
OUT = ROOT / "data" / "kpi"

LOST_STATUSES = ["canceled", "unavailable"]   # commandes perdues (jamais livrées)
MIN_CATEGORY_BASE_BRL = 20_000                # croissance d'une catégorie calculée seulement au-dessus
TOP_N = 10
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def read_bool(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower().isin(["true", "1", "yes"])


def read_flag(series: pd.Series) -> pd.Series:
    """Booléen -> 1.0 / 0.0, valeur inconnue -> NaN (pour pouvoir faire des moyennes)."""
    return series.astype("string").str.strip().str.lower().map({"true": 1.0, "false": 0.0}).astype(float)


def rounded(value, digits=2):
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def share(part, total):
    return part / total * 100 if total else None


def growth(current, previous):
    return (current / previous - 1) * 100 if previous else None


def fmt_number(value, digits=2):
    value = rounded(value, digits)
    return f"{value:,.{digits}f}" if value is not None else "N/A"


def fmt_percent(value):
    value = rounded(value)
    return f"{value:+.2f}%" if value is not None and value < 0 else (
        f"{value:.2f}%" if value is not None else "N/A")


# -------------------------------------------------------------------
# Chargement
# -------------------------------------------------------------------

def load():
    paths = {n: CURATED / f"{n}.csv" for n in ("fact_orders", "fact_sales", "dim_date")}
    for p in paths.values():
        if not p.exists():
            raise FileNotFoundError(f"Required ETL output missing: {p}")

    orders = pd.read_csv(paths["fact_orders"], usecols=[
        "order_id", "customer_unique_id", "customer_state", "order_status", "is_revenue_order",
        "purchase_date", "delivery_days", "is_late", "items_value", "freight_total",
        "revenue_brl", "review_score"])
    sales = pd.read_csv(paths["fact_sales"], usecols=[
        "order_id", "product_id", "product_name", "seller_id", "seller_name", "seller_state",
        "customer_state", "category", "purchase_date", "is_revenue_order", "revenue_brl", "margin_brl"])
    dates = pd.read_csv(paths["dim_date"], usecols=[
        "date", "is_reliable_month", "event", "event_type"])

    for frame in (orders, sales):
        frame["is_revenue_order"] = read_bool(frame["is_revenue_order"])
        frame["purchase_date"] = pd.to_datetime(frame["purchase_date"], errors="coerce")
    orders["late"] = read_flag(orders["is_late"])
    for c in ("delivery_days", "items_value", "freight_total", "revenue_brl", "review_score"):
        orders[c] = pd.to_numeric(orders[c], errors="coerce")
    for c in ("revenue_brl", "margin_brl"):
        sales[c] = pd.to_numeric(sales[c], errors="coerce")
    sales["category"] = sales["category"].fillna("non_defini")
    sales["seller_name"] = sales["seller_name"].fillna(sales["seller_id"])

    dates["date"] = pd.to_datetime(dates["date"], errors="coerce")
    dates = dates.dropna(subset=["date"]).copy()
    dates["is_reliable_month"] = read_bool(dates["is_reliable_month"])
    dates["event"] = dates["event"].fillna("")
    dates["event_type"] = dates["event_type"].fillna("")
    dates["year_month"] = dates["date"].dt.to_period("M").astype(str)

    # Contrôles d'intégrité : on s'arrête plutôt que de publier des chiffres faux
    if orders["order_id"].duplicated().any():
        raise ValueError("fact_orders must contain exactly one row per order.")
    if sales["revenue_brl"].isna().any():
        raise ValueError("fact_sales contains missing revenue values.")
    rev_o = orders.loc[orders["is_revenue_order"], "revenue_brl"].sum()
    rev_s = sales.loc[sales["is_revenue_order"], "revenue_brl"].sum()
    if abs(rev_o - rev_s) >= 0.01:
        raise ValueError(f"Revenue mismatch between fact tables: {rev_o:.2f} vs {rev_s:.2f}")
    return orders, sales, dates


# -------------------------------------------------------------------
# Découpages (breakdowns)
# -------------------------------------------------------------------

def monthly_trend(rev_orders, dates, period=None):
    """CA et commandes par mois + croissance sur un an (même mois de l'année précédente).

    period = ("AAAA-MM", "AAAA-MM") : ne garde que les mois de la période, mais compare toujours au
    même mois N-1 (pris dans l'historique complet de rev_orders)."""
    rev_orders = rev_orders.assign(year_month=rev_orders["purchase_date"].dt.to_period("M").astype(str))
    m = (rev_orders.dropna(subset=["purchase_date"]).groupby("year_month")
         .agg(revenue_brl=("revenue_brl", "sum"), orders=("order_id", "size")))
    reliable = dates.drop_duplicates("year_month").set_index("year_month")["is_reliable_month"]

    rows, comparable = [], []
    for ym, r in m.iterrows():
        if period and not (period[0] <= ym <= period[1]):
            continue
        prev = str(pd.Period(ym, freq="M") - 12)
        ok = bool(reliable.get(ym, False))
        yoy = None
        if ok and bool(reliable.get(prev, False)) and prev in m.index:
            yoy = growth(r["revenue_brl"], m.loc[prev, "revenue_brl"])
            comparable.append((ym, prev))
        rows.append({"month": ym, "revenue_brl": rounded(r["revenue_brl"]), "orders": int(r["orders"]),
                     "is_reliable_month": ok, "yoy_growth_pct": rounded(yoy)})

    cur = m.loc[[c for c, _ in comparable], "revenue_brl"].sum()
    prev = m.loc[[p for _, p in comparable], "revenue_brl"].sum()
    summary = {
        "period": f"{comparable[0][0]} -> {comparable[-1][0]}" if comparable else None,
        "compared_to": f"{comparable[0][1]} -> {comparable[-1][1]}" if comparable else None,
        "revenue_brl": rounded(cur), "previous_revenue_brl": rounded(prev),
        "growth_pct": rounded(growth(cur, prev)),
    }
    return rows, summary, comparable


def state_breakdown(rev_orders):
    """Par État client : CA, commandes, retards, délai, poids du fret, note."""
    g = rev_orders.groupby("customer_state").agg(
        orders=("order_id", "size"), revenue_brl=("revenue_brl", "sum"),
        late_rate=("late", "mean"), median_delivery_days=("delivery_days", "median"),
        freight=("freight_total", "sum"), items=("items_value", "sum"),
        avg_review=("review_score", "mean")).sort_values("revenue_brl", ascending=False)
    return [{
        "state": str(s), "orders": int(r["orders"]), "revenue_brl": rounded(r["revenue_brl"]),
        "late_delivery_rate_pct": rounded(r["late_rate"] * 100),
        "median_delivery_days": rounded(r["median_delivery_days"], 1),
        "freight_to_revenue_pct": rounded(share(r["freight"], r["items"])),
        "avg_review_score": rounded(r["avg_review"]),
    } for s, r in g.iterrows()]


def category_breakdown(rev_sales, rev_orders, comparable, history_sales=None):
    """Par catégorie : CA, unités, note, croissance sur un an (mêmes mois).

    history_sales : ventes sans filtre de période (pour lire les mois N-1) ; par défaut rev_sales."""
    s = rev_sales.merge(rev_orders[["order_id", "review_score"]], on="order_id", how="left")
    s["year_month"] = s["purchase_date"].dt.to_period("M").astype(str)
    g = s.groupby("category").agg(revenue_brl=("revenue_brl", "sum"), units=("order_id", "size"),
                                  avg_review=("review_score", "mean"))
    hist = s if history_sales is None else history_sales.assign(
        year_month=history_sales["purchase_date"].dt.to_period("M").astype(str))
    cur = hist[hist["year_month"].isin([c for c, _ in comparable])].groupby("category")["revenue_brl"].sum()
    prev = hist[hist["year_month"].isin([p for _, p in comparable])].groupby("category")["revenue_brl"].sum()
    g["growth"] = [growth(cur.get(c, 0.0), prev.get(c, 0.0)) if prev.get(c, 0.0) >= MIN_CATEGORY_BASE_BRL
                   else None for c in g.index]

    def row(c, r):
        return {"category": str(c), "revenue_brl": rounded(r["revenue_brl"]), "units": int(r["units"]),
                "avg_review_score": rounded(r["avg_review"]), "yoy_growth_pct": rounded(r["growth"])}

    with_growth = g.dropna(subset=["growth"]).sort_values("growth")
    return {
        "top_by_revenue": [row(c, r) for c, r in g.sort_values("revenue_brl", ascending=False)
                           .head(TOP_N).iterrows()],
        "fastest_growing": [row(c, r) for c, r in with_growth.iloc[::-1].head(5).iterrows()],
        "declining": [row(c, r) for c, r in with_growth[with_growth["growth"] < 0].iterrows()],
        "lowest_rated": [row(c, r) for c, r in g[g["units"] >= 1000].sort_values("avg_review")
                         .head(5).iterrows()],
    }


def seller_breakdown(rev_sales, rev_orders):
    """Fiche des meilleurs vendeurs + concentration du CA."""
    per = rev_sales.groupby("seller_id").agg(
        seller_name=("seller_name", "first"), seller_state=("seller_state", "first"),
        revenue_brl=("revenue_brl", "sum"), orders=("order_id", "nunique")
    ).sort_values("revenue_brl", ascending=False)
    seller_orders = (rev_sales[["seller_id", "order_id"]].drop_duplicates()
                     .merge(rev_orders[["order_id", "late", "review_score"]], on="order_id", how="left")
                     .groupby("seller_id").agg(late_rate=("late", "mean"), avg_review=("review_score", "mean")))
    per = per.join(seller_orders)

    total = per["revenue_brl"].sum()
    top_decile = max(1, int(len(per) * 0.1))
    state_share = rev_sales.groupby("seller_state")["revenue_brl"].sum().sort_values(ascending=False)
    return {
        "top_sellers": [{
            "seller_id": str(sid), "seller_name": str(r["seller_name"]), "state": str(r["seller_state"]),
            "revenue_brl": rounded(r["revenue_brl"]), "orders": int(r["orders"]),
            "late_delivery_rate_pct": rounded(r["late_rate"] * 100),
            "avg_review_score": rounded(r["avg_review"]),
        } for sid, r in per.head(TOP_N).iterrows()],
        "concentration": {
            "active_sellers": int(len(per)),
            "top_10pct_sellers_revenue_share_pct": rounded(share(per["revenue_brl"].head(top_decile).sum(), total)),
            "top_seller_state": str(state_share.index[0]) if len(state_share) else None,
            "top_seller_state_revenue_share_pct": rounded(share(state_share.iloc[0], total))
            if len(state_share) else None,
        },
    }


def best_sellers(rev_sales):
    """Produits les plus vendus (1 ligne d'article = 1 unité), regroupés par product_id."""
    g = (rev_sales.groupby("product_id")
         .agg(product_name=("product_name", "first"), category=("category", "first"),
              units=("order_id", "size"), revenue_brl=("revenue_brl", "sum"))
         .sort_values(["units", "revenue_brl"], ascending=False).head(TOP_N))
    return [{"product_id": str(pid), "product_name": str(r["product_name"]), "category": str(r["category"]),
             "units": int(r["units"]), "revenue_brl": rounded(r["revenue_brl"])} for pid, r in g.iterrows()]


def event_breakdown(rev_orders, dates):
    """CA de chaque jour d'événement comparé au jour moyen SANS événement du même mois (mois fiables)."""
    daily = (rev_orders.dropna(subset=["purchase_date"]).groupby("purchase_date")
             .agg(revenue_brl=("revenue_brl", "sum"), orders=("order_id", "size")))
    cal = (dates[dates["is_reliable_month"]].set_index("date").join(daily, how="left")
           .fillna({"revenue_brl": 0.0, "orders": 0}))
    plain = cal[cal["event"] == ""]
    baseline = plain.groupby("year_month")["revenue_brl"].mean()
    return [{
        "date": d.strftime("%Y-%m-%d"), "event": str(r["event"]), "type": str(r["event_type"]),
        "revenue_brl": rounded(r["revenue_brl"]), "orders": int(r["orders"]),
        "vs_normal_day_pct": rounded(growth(r["revenue_brl"], baseline.get(r["year_month"]))),
    } for d, r in cal[cal["event"] != ""].iterrows()]


def weekday_breakdown(rev_orders):
    rev = rev_orders.dropna(subset=["purchase_date"])
    by_day = rev.groupby(rev["purchase_date"].dt.day_name())["revenue_brl"].sum()
    total = by_day.sum()
    return [{"weekday": d, "revenue_share_pct": rounded(share(by_day.get(d, 0.0), total))} for d in WEEKDAYS]


# -------------------------------------------------------------------
# Agent
# -------------------------------------------------------------------

def filter_frames(orders, sales, states=(), categories=()):
    """Restreint les tables à des États clients et/ou des catégories (filtres du dashboard).

    Filtre catégorie : on garde les articles de ces catégories et les commandes qui en contiennent ;
    le CA d'une commande devient la somme de SES articles filtrés (le reste du panier est exclu)."""
    if states:
        orders = orders[orders["customer_state"].isin(states)]
        sales = sales[sales["customer_state"].isin(states)]
    if categories:
        sales = sales[sales["category"].isin(categories)]
        rev_by_order = sales.groupby("order_id")["revenue_brl"].sum()
        orders = orders[orders["order_id"].isin(rev_by_order.index)].copy()
        orders["revenue_brl"] = orders["order_id"].map(rev_by_order).where(orders["is_revenue_order"], 0.0)
    return orders, sales


def compute(orders, sales, dates, period=None) -> dict:
    """Calcule tous les KPI (même structure que kpi_results.json), sans rien écrire.

    orders / sales peuvent être déjà filtrés (filter_frames) ; period = ("AAAA-MM", "AAAA-MM") limite la
    période analysée, l'historique restant utilisé pour les comparaisons sur un an."""
    hist_rev_orders = orders[orders["is_revenue_order"]].copy()
    hist_rev_sales = sales[sales["is_revenue_order"]].copy()
    if period:
        def in_period(df, col="purchase_date"):
            ym = df[col].dt.to_period("M").astype(str)
            return df[ym.between(period[0], period[1])]
        orders, sales = in_period(orders), in_period(sales)
        dates_p = dates[dates["year_month"].between(period[0], period[1])]
    else:
        dates_p = dates
    rev_orders = orders[orders["is_revenue_order"]].copy()
    rev_sales = sales[sales["is_revenue_order"]].copy()

    total_revenue = rev_orders["revenue_brl"].sum()
    n_orders = len(rev_orders)
    per_customer = rev_orders.dropna(subset=["customer_unique_id"]).groupby("customer_unique_id").size()
    scores = rev_orders["review_score"].dropna()
    known_late = rev_orders.dropna(subset=["late"])

    monthly, yoy, comparable = monthly_trend(hist_rev_orders, dates, period)

    headline = {
        "total_revenue_brl": rounded(total_revenue),
        "delivered_orders": int(n_orders),
        "average_order_value_brl": rounded(total_revenue / n_orders if n_orders else None),
        "unique_customers": int(len(per_customer)),
        "repeat_customer_rate_pct": rounded(share((per_customer > 1).sum(), len(per_customer))),
        "lost_order_rate_pct": rounded(share(orders["order_status"].isin(LOST_STATUSES).sum(), len(orders))),
        "yoy_revenue_growth_pct": yoy["growth_pct"],
        "late_delivery_rate_pct": rounded(known_late["late"].mean() * 100 if len(known_late) else None),
        "median_delivery_days": rounded(rev_orders["delivery_days"].median(), 1),
        "average_review_score": rounded(scores.mean()),
        "low_rating_rate_pct": rounded(scores.isin([1, 2]).mean() * 100 if len(scores) else None),
        "freight_to_revenue_pct": rounded(share(rev_orders["freight_total"].sum(),
                                                rev_orders["items_value"].sum())),
    }

    late_scores = known_late.groupby("late")["review_score"].mean()
    margin = rev_sales["margin_brl"].sum()

    kpis = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": {"orders": "data/curated/fact_orders.csv", "sales": "data/curated/fact_sales.csv",
                   "calendar": "data/curated/dim_date.csv"},
        "headline": headline,
        "insights": {
            "review_score_late_orders": rounded(late_scores.get(1.0)),
            "review_score_on_time_orders": rounded(late_scores.get(0.0)),
        },
        "breakdowns": {
            "monthly": monthly,
            "yoy_summary": yoy,
            "by_customer_state": state_breakdown(rev_orders),
            "categories": category_breakdown(rev_sales, rev_orders, comparable, history_sales=hist_rev_sales),
            "sellers": seller_breakdown(rev_sales, rev_orders),
            "best_sellers": best_sellers(rev_sales),
            "events": event_breakdown(rev_orders, dates_p),
            "weekday_revenue_share": weekday_breakdown(rev_orders),
        },
        "margin_simulated": {
            "gross_margin_brl": rounded(margin),
            "gross_margin_rate_pct": rounded(share(margin, total_revenue)),
            "warning": "Coût = production_cost_brl, SIMULÉ à partir du prix de vente (~70 %). "
                       "Aucune conclusion métier ne doit être tirée de cette marge.",
        },
        "coverage": {
            "orders_total": int(len(orders)),
            "orders_counted_in_revenue": int(n_orders),
            "orders_with_delivery_data": int(rev_orders["delivery_days"].notna().sum()),
            "orders_with_known_lateness": int(len(known_late)),
            "orders_with_review": int(len(scores)),
        },
        "definitions": {
            "revenue": "Somme des prix des articles des commandes livrées, hors fret.",
            "lost_order_rate": f"Part de TOUTES les commandes au statut {LOST_STATUSES}.",
            "yoy_growth": "CA comparé aux mêmes mois de l'année précédente ; seulement les mois fiables "
                          "(volume suffisant) des deux années.",
            "late_delivery": "Livrée après la date estimée, comparaison par jour calendaire.",
            "freight_to_revenue": "Fret payé par les clients / CA des articles. Ni un coût de transport, ni une marge.",
            "category_growth": f"Croissance sur un an (mois comparables) ; seulement les catégories ayant au moins "
                               f"R$ {MIN_CATEGORY_BASE_BRL:,} l'année précédente.",
            "events": "CA de chaque jour d'événement (jours fériés de br_holidays.csv + événements commerciaux de "
                      "etl_config.json) comparé au jour moyen sans événement du même mois.",
            "best_sellers": "Produits classés par unités vendues (une ligne d'article = une unité).",
        },
        "limitations": [
            "Marge simulée (coût dérivé du prix de vente) : ce n'est pas un résultat exploitable.",
            "Rotation du stock et taux de conversion non calculables (pas de données de stock ni de trafic web).",
            "Les comparaisons de notes (retard, catégorie, vendeur) montrent une association, pas une causalité.",
            "Les mois à trop faible volume sont exclus des calculs de croissance.",
        ],
    }
    if period:
        kpis["filters"] = {"period": list(period)}
    return kpis


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    orders, sales, dates = load()
    kpis = compute(orders, sales, dates)
    headline, yoy = kpis["headline"], kpis["breakdowns"]["yoy_summary"]
    total_revenue, n_orders = headline["total_revenue_brl"], headline["delivered_orders"]

    (OUT / "kpi_results.json").write_text(
        json.dumps(kpis, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    write_report(kpis)

    print("KPI Agent terminé.")
    print(f"CA : R$ {total_revenue:,.2f} | sur un an : {fmt_percent(yoy['growth_pct'])} "
          f"| retards : {fmt_percent(headline['late_delivery_rate_pct'])}")
    return {"ok": True, "status": "success", "total_revenue_brl": headline["total_revenue_brl"],
            "delivered_orders": n_orders, "yoy_revenue_growth_pct": yoy["growth_pct"],
            "outputs": ["data/kpi/kpi_results.json", "data/kpi/kpi_report.md"]}


# -------------------------------------------------------------------
# Rapport Markdown
# -------------------------------------------------------------------

def table(header, rows):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + \
           ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]


def write_report(k):
    h, b, ins = k["headline"], k["breakdowns"], k["insights"]
    y = b["yoy_summary"]
    L = ["# Rapport KPI BIFlow", "", f"- Calculé le : {k['generated_at']}", "", "## Indicateurs clés", "",
         f"- Chiffre d'affaires : R$ {fmt_number(h['total_revenue_brl'])} ({h['delivered_orders']:,} commandes livrées)",
         f"- Panier moyen : R$ {fmt_number(h['average_order_value_brl'])}",
         f"- Clients uniques : {h['unique_customers']:,} | taux de clients récurrents : {fmt_percent(h['repeat_customer_rate_pct'])}",
         f"- Commandes perdues (annulées / indisponibles) : {fmt_percent(h['lost_order_rate_pct'])}",
         f"- Croissance du CA sur un an ({y['period']} vs {y['compared_to']}) : {fmt_percent(h['yoy_revenue_growth_pct'])}",
         f"- Livraisons en retard : {fmt_percent(h['late_delivery_rate_pct'])} | délai médian : {h['median_delivery_days']} jours",
         f"- Note moyenne : {h['average_review_score']} / 5 | avis faibles (1-2) : {fmt_percent(h['low_rating_rate_pct'])}",
         f"- Fret / CA : {fmt_percent(h['freight_to_revenue_pct'])}",
         "", "## Insight clé : livraison et satisfaction", "",
         f"- Note moyenne des commandes en retard : {ins['review_score_late_orders']} / 5 "
         f"contre commandes à l'heure : {ins['review_score_on_time_orders']} / 5", "",
         "## CA mensuel", ""]
    L += table(["Mois", "CA (BRL)", "Commandes", "Fiable", "Sur un an"],
               [[m["month"], fmt_number(m["revenue_brl"]), f"{m['orders']:,}",
                 "Oui" if m["is_reliable_month"] else "Non", fmt_percent(m["yoy_growth_pct"])]
                for m in b["monthly"]])
    L += ["", "## Par État client", ""]
    L += table(["État", "Commandes", "CA", "Retards", "Délai médian", "Fret/CA", "Note"],
               [[s["state"], f"{s['orders']:,}", fmt_number(s["revenue_brl"]), fmt_percent(s["late_delivery_rate_pct"]),
                 s["median_delivery_days"], fmt_percent(s["freight_to_revenue_pct"]), s["avg_review_score"]]
                for s in b["by_customer_state"]])
    c = b["categories"]
    cat_rows = lambda rows: [[r["category"], fmt_number(r["revenue_brl"]), f"{r['units']:,}",
                              r["avg_review_score"], fmt_percent(r["yoy_growth_pct"])] for r in rows]
    hdr = ["Catégorie", "CA", "Unités", "Note", "Sur un an"]
    for title, key in (("Catégories principales (CA)", "top_by_revenue"), ("Plus forte croissance", "fastest_growing"),
                       ("Catégories en baisse", "declining"), ("Catégories les moins bien notées", "lowest_rated")):
        L += ["", f"## {title}", ""] + (table(hdr, cat_rows(c[key])) if c[key] else ["- aucune"])
    s = b["sellers"]
    conc = s["concentration"]
    L += ["", "## Vendeurs", "",
          f"- Vendeurs actifs : {conc['active_sellers']:,} | les 10 % meilleurs réalisent {fmt_percent(conc['top_10pct_sellers_revenue_share_pct'])} du CA",
          f"- Les vendeurs de {conc['top_seller_state']} réalisent {fmt_percent(conc['top_seller_state_revenue_share_pct'])} du CA", ""]
    L += table(["Vendeur", "État", "CA", "Commandes", "Retards", "Note"],
               [[x["seller_name"], x["state"], fmt_number(x["revenue_brl"]), f"{x['orders']:,}",
                 fmt_percent(x["late_delivery_rate_pct"]), x["avg_review_score"]] for x in s["top_sellers"]])
    L += ["", "## Best-sellers (unités)", ""]
    L += table(["Produit", "Catégorie", "Unités", "CA"],
               [[f"{p['product_name']} ({p['product_id'][:8]}…)", p["category"], f"{p['units']:,}",
                 fmt_number(p["revenue_brl"])] for p in b["best_sellers"]])
    L += ["", "## Événements (vs jour normal moyen du même mois)", ""]
    L += table(["Date", "Événement", "Type", "CA", "Commandes", "vs jour normal"],
               [[e["date"], e["event"], e["type"], fmt_number(e["revenue_brl"]), f"{e['orders']:,}",
                 fmt_percent(e["vs_normal_day_pct"])] for e in b["events"]])
    L += ["", "## Part du CA par jour de la semaine", ""]
    L += [f"- {w['weekday']}: {fmt_percent(w['revenue_share_pct'])}" for w in b["weekday_revenue_share"]]
    m = k["margin_simulated"]
    L += ["", "## Marge (simulée)", "",
          f"- Marge brute : R$ {fmt_number(m['gross_margin_brl'])} ({fmt_percent(m['gross_margin_rate_pct'])})",
          f"- {m['warning']}", "", "## Limites", ""] + [f"- {x}" for x in k["limitations"]] + [""]
    (OUT / "kpi_report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    run()
