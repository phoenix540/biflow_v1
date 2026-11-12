"""Analyses complémentaires de l'app manager : clients et fidélisation, paiements, produits, carte des ventes.

Même principe que metrics.py : calculs pandas déterministes sur data/curated/, sur le périmètre filtré
(période x États clients x catégories), avec les mêmes définitions que l'agent KPI (commandes du CA =
is_revenue_order ; CA = prix des articles, hors fret).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import metrics as M

PAYMENT_FR = {"credit_card": "Carte de crédit", "boleto": "Boleto", "voucher": "Bon d'achat",
              "debit_card": "Carte de débit", "not_defined": "Non défini"}


@dataclass
class Extra:
    """Dimensions utiles aux pages complémentaires (chargées une seule fois)."""
    products: pd.DataFrame      # product_id -> photos_qty, description_length, weight_g
    cust_zip: pd.Series         # customer_id -> code postal
    seller_zip: pd.Series       # seller_id -> code postal
    geo_zip: pd.DataFrame       # code postal -> lat, lng, city, state


def load_extra() -> Extra:
    cols = ["product_id", "photos_qty", "weight_g"]
    head = pd.read_csv(M.CUR / "dim_products.csv", nrows=0).columns
    p = pd.read_csv(M.CUR / "dim_products.csv",
                    usecols=cols + (["description_length"] if "description_length" in head else []))
    if "description_length" not in p:
        p["description_length"] = np.nan
    c = pd.read_csv(M.CUR / "dim_customers.csv", usecols=["customer_id", "customer_zip_code_prefix"])
    s = pd.read_csv(M.CUR / "dim_sellers.csv", usecols=["seller_id", "seller_zip_code_prefix"])
    g = pd.read_csv(M.CUR / "dim_geolocation.csv")
    g = g[g["lat"].between(-35, 6) & g["lng"].between(-75, -30)].set_index("zip_code_prefix")
    return Extra(products=p.set_index("product_id"),
                 cust_zip=c.set_index("customer_id")["customer_zip_code_prefix"],
                 seller_zip=s.set_index("seller_id")["seller_zip_code_prefix"], geo_zip=g)


def _pct(part, total):
    return round(float(part) / float(total) * 100, 2) if total else None


# --------------------------------------------------------------------------
# Clients et fidélisation
# --------------------------------------------------------------------------
def _first_orders(store: M.Store) -> pd.DataFrame:
    """Première commande du CA de chaque client, sur TOUT l'historique (une cohorte ne dépend pas du filtre)."""
    o = store.orders[store.orders["is_revenue_order"]].dropna(subset=["customer_unique_id"])
    first = o.sort_values("purchase_date").drop_duplicates("customer_unique_id")
    top_cat = (store.sales.sort_values("revenue_brl", ascending=False).drop_duplicates("order_id")
               .set_index("order_id")["category"])
    first = first.assign(first_category=first["order_id"].map(top_cat).fillna("non_defini"))
    return first[["customer_unique_id", "order_id", "m", "customer_state", "first_category"]]


def customers_overview(store: M.Store, sc: M.Scope) -> dict:
    o, _, _ = M.slice_scope(store, sc)
    o = o.dropna(subset=["customer_unique_id"])
    per = o.groupby("customer_unique_id").agg(n=("order_id", "nunique"), ca=("revenue_brl", "sum"),
                                              first=("purchase_date", "min"))
    rep = per[per["n"] > 1]
    gaps = (o[o["customer_unique_id"].isin(rep.index)].sort_values("purchase_date")
            .groupby("customer_unique_id")["purchase_date"].apply(lambda d: (d.iloc[1] - d.iloc[0]).days))
    return {"periode": sc.label(), "clients": int(len(per)), "clients_recurrents": int(len(rep)),
            "taux_recurrents_pct": _pct(len(rep), len(per)),
            "commandes_par_client": round(float(per["n"].mean()), 3) if len(per) else None,
            "part_ca_clients_recurrents_pct": _pct(rep["ca"].sum(), per["ca"].sum()),
            "delai_median_2e_commande_jours": float(gaps.median()) if len(gaps) else None}


def cohort_retention(store: M.Store, sc: M.Scope, horizon: int = 12) -> pd.DataFrame:
    """Part cumulée des clients d'une cohorte (mois de 1re commande) ayant recommandé dans les k mois suivants.
    Les cohortes sont celles dont le premier achat tombe dans la période ; filtre État = État de la 1re commande,
    filtre catégorie = catégorie principale de la 1re commande. NaN quand le recul n'est pas encore disponible."""
    first = _first_orders(store)
    first = first[first["m"].between(sc.start, sc.end)]
    if sc.states:
        first = first[first["customer_state"].isin(sc.states)]
    if sc.cats:
        first = first[first["first_category"].isin(sc.cats)]
    o = store.orders[store.orders["is_revenue_order"] & store.orders["customer_unique_id"].isin(first["customer_unique_id"])]
    o = o.merge(first[["customer_unique_id", "order_id", "m"]].rename(columns={"order_id": "first_id", "m": "cohort"}),
                on="customer_unique_id")
    o = o[o["order_id"] != o["first_id"]]
    to_p = lambda s: pd.PeriodIndex(s, freq="M")
    o["offset"] = (to_p(o["m"]) - to_p(o["cohort"])).map(lambda x: x.n)
    back = o[o["offset"] >= 0].groupby("customer_unique_id").agg(cohort=("cohort", "first"), offset=("offset", "min"))
    size = first.groupby("m").size()
    last = pd.Period(store.last_full_month, freq="M")
    rows = []
    for cohort, n in size.items():
        row = {"cohorte": cohort, "clients": int(n)}
        reach = (last - pd.Period(cohort, freq="M")).n
        b = back[back["cohort"] == cohort]["offset"]
        for k in range(1, horizon + 1):
            row[f"M+{k}"] = round(float((b <= k).sum()) / n * 100, 2) if k <= reach else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


RFM_ORDER = ["Champions", "Fidèles à relancer", "Nouveaux clients", "Gros paniers sans retour", "Endormis", "Perdus"]
RFM_ACTION = {
    "Champions": "programme de fidélité, avant-premières",
    "Fidèles à relancer": "offre de retour personnalisée",
    "Nouveaux clients": "e-mail de bienvenue, 2e commande incitée",
    "Gros paniers sans retour": "relance premium, recommandations",
    "Endormis": "campagne de réactivation ciblée",
    "Perdus": "faible priorité, campagne générique",
}


def rfm_segments(store: M.Store, sc: M.Scope) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Segmentation RFM (récence, fréquence, montant) des clients du périmètre, date de référence = fin de période.
    Règles lisibles (et non des quantiles) car 97 % des clients n'ont qu'une commande."""
    o, _, _ = M.slice_scope(store, sc)
    o = o.dropna(subset=["customer_unique_id"])
    if o.empty:
        return pd.DataFrame(columns=["segment"]), pd.DataFrame()
    ref = min(pd.Period(sc.end, freq="M").end_time.normalize(), o["purchase_date"].max())
    c = o.groupby("customer_unique_id").agg(derniere=("purchase_date", "max"), frequence=("order_id", "nunique"),
                                            montant=("revenue_brl", "sum"))
    c["recence"] = (ref - c["derniere"]).dt.days
    big = c["montant"].quantile(0.8)
    c["segment"] = np.select(
        [(c["frequence"] >= 2) & (c["recence"] <= 180), c["frequence"] >= 2, c["recence"] <= 90,
         c["montant"] >= big, c["recence"] <= 365],
        ["Champions", "Fidèles à relancer", "Nouveaux clients", "Gros paniers sans retour", "Endormis"], default="Perdus")
    seg = c.groupby("segment").agg(clients=("frequence", "size"), ca=("montant", "sum"),
                                   recence_moyenne=("recence", "mean"), frequence_moyenne=("frequence", "mean"),
                                   panier_moyen=("montant", "mean")).reindex(RFM_ORDER).dropna(subset=["clients"])
    seg["part_clients_pct"] = seg["clients"] / seg["clients"].sum() * 100
    seg["part_ca_pct"] = seg["ca"] / seg["ca"].sum() * 100
    seg["action"] = seg.index.map(RFM_ACTION)
    seg["clients"] = seg["clients"].astype(int)
    return seg.reset_index(), c.reset_index()


def repeat_by(store: M.Store, sc: M.Scope, dim: str, min_customers: int = 200) -> pd.DataFrame:
    """Taux de clients récurrents par État client ('etat') ou par catégorie de la 1re commande ('categorie')."""
    o, _, _ = M.slice_scope(store, sc)
    o = o.dropna(subset=["customer_unique_id"])
    per = o.groupby("customer_unique_id")["order_id"].nunique().rename("n").to_frame()
    first = _first_orders(store).set_index("customer_unique_id")
    col = "customer_state" if dim == "etat" else "first_category"
    per[dim] = per.index.map(first[col])
    g = per.dropna(subset=[dim]).groupby(dim).agg(clients=("n", "size"), recurrents=("n", lambda x: (x > 1).sum()))
    g = g[g["clients"] >= min_customers]
    g["taux_recurrents_pct"] = g["recurrents"] / g["clients"] * 100
    return g.sort_values("taux_recurrents_pct", ascending=False).reset_index()


# --------------------------------------------------------------------------
# Paiements
# --------------------------------------------------------------------------
def payments_mix(store: M.Store, sc: M.Scope) -> pd.DataFrame:
    """Par moyen de paiement principal : commandes, CA, panier, mensualités, note, avis négatifs, annulations."""
    o, _, o_all = M.slice_scope(store, sc)
    o, o_all = o.dropna(subset=["main_payment_type"]), o_all.dropna(subset=["main_payment_type"])
    g = o.groupby("main_payment_type").agg(commandes=("order_id", "nunique"), ca=("revenue_brl", "sum"),
                                           mensualites=("max_installments", "mean"), note=("review_score", "mean"),
                                           avis_negatifs=("review_score", lambda x: x.dropna().isin([1, 2]).mean() * 100))
    lost = o_all.groupby("main_payment_type")["order_status"].agg(lambda x: x.isin(M.LOST_STATUSES).mean() * 100)
    g["commandes_perdues_pct"] = lost
    g["panier_moyen"] = g["ca"] / g["commandes"]
    g["part_commandes_pct"] = g["commandes"] / g["commandes"].sum() * 100
    g["part_ca_pct"] = g["ca"] / g["ca"].sum() * 100
    g["moyen"] = g.index.map(lambda k: PAYMENT_FR.get(k, k))
    return g.sort_values("commandes", ascending=False).reset_index()


def payments_monthly(store: M.Store, sc: M.Scope) -> pd.DataFrame:
    o, _, _ = M.slice_scope(store, sc)
    o = o.dropna(subset=["main_payment_type"])
    o = o[o["m"].map(store.reliable).fillna(False).astype(bool)]
    t = o.groupby(["m", "main_payment_type"])["order_id"].nunique().unstack(fill_value=0)
    share = t.div(t.sum(axis=1), axis=0) * 100
    out = share.stack().rename("part_pct").reset_index()
    out["moyen"] = out["main_payment_type"].map(lambda k: PAYMENT_FR.get(k, k))
    out["mois"] = out["m"].map(M.month_label)
    return out


INSTALLMENT_BINS = [0, 1, 3, 6, 10, np.inf]
INSTALLMENT_LABELS = ["1 fois", "2-3 fois", "4-6 fois", "7-10 fois", "plus de 10 fois"]


def installments_vs_basket(store: M.Store, sc: M.Scope) -> pd.DataFrame:
    """Paiements par carte de crédit : panier moyen et note selon le nombre de mensualités."""
    o, _, _ = M.slice_scope(store, sc)
    o = o[(o["main_payment_type"] == "credit_card") & (o["max_installments"] >= 1)]
    o = o.assign(tranche=pd.cut(o["max_installments"], INSTALLMENT_BINS, labels=INSTALLMENT_LABELS))
    g = o.groupby("tranche", observed=False).agg(commandes=("order_id", "nunique"), panier_moyen=("revenue_brl", "mean"),
                                                 note=("review_score", "mean"))
    g["part_pct"] = g["commandes"] / g["commandes"].sum() * 100
    return g.reset_index()


def boleto_by_state(store: M.Store, sc: M.Scope, min_orders: int = 300) -> pd.DataFrame:
    o, _, _ = M.slice_scope(store, M.Scope(sc.start, sc.end, (), sc.cats))
    o = o.dropna(subset=["main_payment_type"])
    g = o.groupby("customer_state").agg(commandes=("order_id", "nunique"),
                                        boleto_pct=("main_payment_type", lambda x: (x == "boleto").mean() * 100))
    g = g[g["commandes"] >= min_orders].join(store.geo, how="left")
    return g.sort_values("boleto_pct", ascending=False).reset_index().rename(columns={"customer_state": "etat"})


# --------------------------------------------------------------------------
# Produits et catalogue
# --------------------------------------------------------------------------
def _product_table(store: M.Store, extra: Extra, sc: M.Scope) -> pd.DataFrame:
    o, s, _ = M.slice_scope(store, sc)
    s = s.merge(o[["order_id", "review_score"]], on="order_id", how="left")
    p = s.groupby("product_id").agg(produit=("product_name", "first"), categorie=("category", "first"),
                                    unites=("order_id", "size"), ca=("revenue_brl", "sum"), note=("review_score", "mean"),
                                    fret_moyen=("freight_value", "mean"), prix_moyen=("price", "mean"))
    return p.join(extra.products, how="left")


def pareto(store: M.Store, extra: Extra, sc: M.Scope) -> dict:
    p = _product_table(store, extra, sc).sort_values("ca", ascending=False)
    if p.empty:
        return {"produits": 0, "curve": pd.DataFrame()}
    cum = p["ca"].cumsum() / p["ca"].sum() * 100
    n = len(p)
    top20 = max(1, int(round(n * 0.2)))
    curve = pd.DataFrame({"part_produits_pct": np.arange(1, n + 1) / n * 100, "part_ca_pct": cum.to_numpy()})
    step = max(1, n // 400)                       # courbe allégée pour l'affichage
    curve = pd.concat([curve.iloc[::step], curve.tail(1)]).drop_duplicates()
    return {"produits": n, "part_ca_top20_pct": round(float(cum.iloc[top20 - 1]), 1),
            "produits_pour_80pct": int((cum < 80).sum() + 1),
            "part_produits_pour_80pct": round(float((cum < 80).sum() + 1) / n * 100, 1), "curve": curve}


def best_sellers(store: M.Store, extra: Extra, sc: M.Scope, n: int = 15) -> pd.DataFrame:
    p = _product_table(store, extra, sc)
    top = p.sort_values(["unites", "ca"], ascending=False).head(n).copy()
    rank_ca = p["ca"].rank(ascending=False, method="min")
    top["rang_ca"] = top.index.map(rank_ca).astype(int)
    top["categorie"] = top["categorie"].map(M.nice_cat)
    return top.reset_index()


PHOTO_BINS, PHOTO_LABELS = [0, 1, 2, 3, 5, np.inf], ["1 photo", "2 photos", "3 photos", "4-5 photos", "6 photos et +"]


def listing_quality(store: M.Store, extra: Extra, sc: M.Scope) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ventes moyennes par produit et note selon le nombre de photos et la longueur de la description."""
    p = _product_table(store, extra, sc)
    ph = p.dropna(subset=["photos_qty"]).assign(
        tranche=lambda d: pd.cut(d["photos_qty"], PHOTO_BINS, labels=PHOTO_LABELS))
    by_photo = ph.groupby("tranche", observed=False).agg(produits=("unites", "size"), unites_par_produit=("unites", "mean"),
                                                        ca_par_produit=("ca", "mean"), note=("note", "mean"))
    d = p.dropna(subset=["description_length"])
    if len(d) >= 50:
        # quartiles de longueur : quatre groupes de taille comparable
        cut = pd.qcut(d["description_length"], 4, duplicates="drop")
        d = d.assign(tranche=cut.cat.rename_categories(
            [f"{max(0, int(np.ceil(iv.left)))}-{int(iv.right)} car." for iv in cut.cat.categories]))
        by_desc = d.groupby("tranche", observed=False).agg(produits=("unites", "size"),
                                                          unites_par_produit=("unites", "mean"), note=("note", "mean"))
    else:
        by_desc = pd.DataFrame()
    return by_photo.reset_index(), by_desc.reset_index()


WEIGHT_BINS = [0, 500, 1000, 2000, 5000, 10000, np.inf]
WEIGHT_LABELS = ["< 0,5 kg", "0,5-1 kg", "1-2 kg", "2-5 kg", "5-10 kg", "> 10 kg"]


def weight_vs_freight(store: M.Store, extra: Extra, sc: M.Scope) -> pd.DataFrame:
    _, s, _ = M.slice_scope(store, sc)
    s = s.join(extra.products["weight_g"], on="product_id").dropna(subset=["weight_g"])
    s = s.assign(tranche=pd.cut(s["weight_g"], WEIGHT_BINS, labels=WEIGHT_LABELS))
    g = s.groupby("tranche", observed=False).agg(articles=("order_id", "size"), fret_median=("freight_value", "median"),
                                                 prix_median=("price", "median"), fret=("freight_value", "sum"),
                                                 prix=("price", "sum"))
    g["fret_pct_prix"] = g["fret"] / g["prix"] * 100
    return g.drop(columns=["fret", "prix"]).reset_index()


# --------------------------------------------------------------------------
# Carte des ventes et flux entre États
# --------------------------------------------------------------------------
def sales_points(store: M.Store, extra: Extra, sc: M.Scope, side: str = "clients") -> pd.DataFrame:
    """CA par code postal (des clients ou des vendeurs), avec coordonnées, pour la carte de densité."""
    o, s, _ = M.slice_scope(store, sc)
    if side == "vendeurs":
        zip_ = s["seller_id"].map(extra.seller_zip)
        g = s.assign(zip=zip_).groupby("zip").agg(ca=("revenue_brl", "sum"), n=("seller_id", "nunique"))
    else:
        zip_ = o["customer_id"].map(extra.cust_zip)
        g = o.assign(zip=zip_).groupby("zip").agg(ca=("revenue_brl", "sum"), n=("customer_unique_id", "nunique"))
    g = g.join(extra.geo_zip[["lat", "lng", "city", "state"]], how="inner")
    return g.reset_index()


def city_ranking(points: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    g = points.groupby(["city", "state"]).agg(ca=("ca", "sum"), n=("n", "sum")).sort_values("ca", ascending=False)
    g["part_ca_pct"] = g["ca"] / points["ca"].sum() * 100
    return g.head(n).reset_index()


def state_flows(store: M.Store, sc: M.Scope, min_orders: int = 50) -> pd.DataFrame:
    """Flux vendeur -> client entre États : commandes, délai médian, retard, fret."""
    o, s, _ = M.slice_scope(store, sc)
    link = s[["order_id", "seller_state"]].drop_duplicates("order_id")
    f = o.merge(link, on="order_id", how="inner")
    g = f.groupby(["seller_state", "customer_state"]).agg(commandes=("order_id", "nunique"),
                                                          delai_median=("delivery_days", "median"),
                                                          taux_retard=("late", "mean"),
                                                          fret_moyen=("freight_total", "mean"), note=("review_score", "mean"))
    g["taux_retard"] = g["taux_retard"] * 100
    g = g[g["commandes"] >= min_orders].reset_index()
    g["meme_etat"] = g["seller_state"] == g["customer_state"]
    return g.sort_values("commandes", ascending=False)


def same_vs_other_state(store: M.Store, sc: M.Scope) -> pd.DataFrame:
    o, s, _ = M.slice_scope(store, sc)
    link = s[["order_id", "seller_state"]].drop_duplicates("order_id")
    f = o.merge(link, on="order_id", how="inner")
    f["flux"] = np.where(f["seller_state"] == f["customer_state"], "Même État", "Autre État")
    g = f.groupby("flux").agg(commandes=("order_id", "nunique"), delai_median=("delivery_days", "median"),
                              taux_retard=("late", "mean"), fret_moyen=("freight_total", "mean"),
                              note=("review_score", "mean"))
    g["taux_retard"] = g["taux_retard"] * 100
    return g.reindex(["Même État", "Autre État"]).dropna(how="all").reset_index()
