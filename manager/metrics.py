"""Couche de calcul de l'app manager (Olist Pilot).

Tous les chiffres affichés ou cités par l'assistant viennent d'ici : calculs pandas déterministes sur
data/curated/, avec les mêmes définitions que l'agent KPI (commandes du CA = is_revenue_order).
Le LLM ne calcule jamais rien : il reformule ce que ces fonctions renvoient.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CUR = DATA / "curated"
LOST_STATUSES = ["canceled", "unavailable"]
MOIS = ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."]
MOIS_LONG = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre",
             "octobre", "novembre", "décembre"]


# --------------------------------------------------------------------------
# Formatage (français)
# --------------------------------------------------------------------------
def fmt_num(x, d=0) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    s = f"{x:,.{d}f}".replace(",", " ").replace(".", ",")
    return s.replace(" ", " ")


def fmt_brl(x, compact=True) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    if compact and abs(x) >= 1e6:
        return f"{fmt_num(x / 1e6, 2)} M BRL"
    if compact and abs(x) >= 1e4:
        return f"{fmt_num(x / 1e3, 0)} k BRL"
    return f"{fmt_num(x, 0)} BRL"


def fmt_pct(x, d=1, sign=False) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return ("+" if sign and x > 0 else "") + fmt_num(x, d) + " %"


def month_label(m: str, long=False) -> str:
    y, mm = m.split("-")
    return f"{(MOIS_LONG if long else MOIS)[int(mm) - 1]} {y}"


def shift_month(m: str, k: int) -> str:
    p = pd.Period(m, freq="M") + k
    return str(p)


def nice_cat(c) -> str:
    return str(c).replace("_", " ")


# --------------------------------------------------------------------------
# Chargement
# --------------------------------------------------------------------------
def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().eq("true")


@dataclass
class Store:
    orders: pd.DataFrame          # toutes les commandes
    sales: pd.DataFrame           # lignes d'articles des commandes du CA
    months: pd.DataFrame          # m, reliable
    events: pd.DataFrame          # name, start, type
    geo: pd.DataFrame             # state -> lat, lng
    kpi: dict = field(default_factory=dict)
    analyst: dict = field(default_factory=dict)
    audit: dict = field(default_factory=dict)
    profile: dict = field(default_factory=dict)

    @property
    def all_months(self) -> list[str]:
        return self.months["m"].tolist()

    @property
    def reliable(self) -> dict:
        return dict(zip(self.months["m"], self.months["reliable"]))

    @property
    def last_full_month(self) -> str:
        r = self.months[self.months["reliable"]]
        return r["m"].iloc[-1]

    @property
    def default_start(self) -> str:
        # 2016 = lancement de la plateforme (quelques commandes) : on démarre en janvier 2017
        return "2017-01" if "2017-01" in self.all_months else self.all_months[0]


def load_store() -> Store:
    o = pd.read_csv(CUR / "fact_orders.csv")
    o["purchase_date"] = pd.to_datetime(o["purchase_date"], errors="coerce")
    o = o.dropna(subset=["purchase_date"])
    o["m"] = o["purchase_date"].dt.strftime("%Y-%m")
    o["is_revenue_order"] = _bool(o["is_revenue_order"])
    o["late"] = o["is_late"].astype(str).map({"True": 1.0, "False": 0.0})
    o["customer_state"] = o["customer_state"].fillna("NA")

    s = pd.read_csv(CUR / "fact_sales.csv", usecols=[
        "order_id", "product_name", "seller_id", "seller_name", "customer_state", "seller_state", "category",
        "purchase_date", "is_revenue_order", "price", "freight_value", "revenue_brl", "margin_brl"])
    s = s[_bool(s["is_revenue_order"])].copy()
    s["purchase_date"] = pd.to_datetime(s["purchase_date"], errors="coerce")
    s = s.dropna(subset=["purchase_date"])
    s["m"] = s["purchase_date"].dt.strftime("%Y-%m")
    s["category"] = s["category"].fillna("non_defini")
    s["customer_state"] = s["customer_state"].fillna("NA")

    dd = pd.read_csv(CUR / "dim_date.csv", usecols=["year_month", "is_reliable_month"])
    months = (dd.groupby("year_month")["is_reliable_month"].first().reset_index()
                .rename(columns={"year_month": "m", "is_reliable_month": "reliable"}))
    months["reliable"] = _bool(months["reliable"])
    months = months[months["m"].isin(set(o["m"]))].sort_values("m").reset_index(drop=True)

    try:
        ev = pd.read_csv(CUR / "dim_events.csv")
        ev["start"] = pd.to_datetime(ev["start"], errors="coerce")
    except (OSError, ValueError):
        ev = pd.DataFrame(columns=["name", "start", "type"])

    geo = pd.read_csv(CUR / "dim_geolocation.csv", usecols=["lat", "lng", "state"])
    geo = geo[(geo["lat"].between(-35, 6)) & (geo["lng"].between(-75, -30))]
    geo = geo.groupby("state")[["lat", "lng"]].median()

    return Store(orders=o, sales=s, months=months, events=ev, geo=geo,
                 kpi=_json(DATA / "kpi" / "kpi_results.json"),
                 analyst=_json(DATA / "analyst" / "analyst_insights.json"),
                 audit=_json(DATA / "audit" / "audit_report.json"),
                 profile=_json(DATA / "profile" / "profile.json"))


# --------------------------------------------------------------------------
# Périmètre (période x États x catégories)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Scope:
    start: str
    end: str
    states: tuple = ()
    cats: tuple = ()

    def shifted(self, k: int) -> "Scope":
        return Scope(shift_month(self.start, k), shift_month(self.end, k), self.states, self.cats)

    def label(self) -> str:
        p = month_label(self.start) if self.start == self.end else f"{month_label(self.start)} → {month_label(self.end)}"
        extra = []
        if self.states:
            extra.append("États : " + ", ".join(self.states))
        if self.cats:
            extra.append("catégories : " + ", ".join(nice_cat(c) for c in self.cats))
        return p + (" · " + " · ".join(extra) if extra else "")


def slice_scope(store: Store, sc: Scope):
    """Renvoie (commandes du CA, lignes de ventes, toutes les commandes) du périmètre."""
    o_all = store.orders[store.orders["m"].between(sc.start, sc.end)]
    s = store.sales[store.sales["m"].between(sc.start, sc.end)]
    if sc.states:
        o_all = o_all[o_all["customer_state"].isin(sc.states)]
        s = s[s["customer_state"].isin(sc.states)]
    if sc.cats:
        s = s[s["category"].isin(sc.cats)]
        o_all = o_all[o_all["order_id"].isin(set(s["order_id"]))]
    o = o_all[o_all["is_revenue_order"]]
    return o, s, o_all


def summary(store: Store, sc: Scope) -> dict:
    o, s, o_all = slice_scope(store, sc)
    rev = float(s["revenue_brl"].sum())
    n = int(o["order_id"].nunique())
    known = o.dropna(subset=["late"])
    rv = o["review_score"].dropna()
    per = o.dropna(subset=["customer_unique_id"]).groupby("customer_unique_id").size()
    late_scores = known.groupby("late")["review_score"].mean()
    late_neg = known.dropna(subset=["review_score"]).groupby("late")["review_score"].agg(lambda x: x.isin([1, 2]).mean() * 100)
    mar = float(s["margin_brl"].sum())
    return {
        "periode": sc.label(),
        "chiffre_affaires_brl": round(rev, 2),
        "commandes": n,
        "panier_moyen_brl": round(rev / n, 2) if n else None,
        "marge_simulee_brl": round(mar, 2),
        "marge_simulee_pct": round(mar / rev * 100, 1) if rev else None,
        "taux_retard_pct": round(known["late"].mean() * 100, 2) if len(known) else None,
        "delai_median_jours": float(o["delivery_days"].median()) if o["delivery_days"].notna().any() else None,
        "note_moyenne": round(rv.mean(), 2) if len(rv) else None,
        "avis_negatifs_pct": round(rv.isin([1, 2]).mean() * 100, 1) if len(rv) else None,
        "clients_uniques": int(len(per)),
        "clients_recurrents_pct": round((per > 1).mean() * 100, 2) if len(per) else None,
        "commandes_perdues_pct": round(o_all["order_status"].isin(LOST_STATUSES).mean() * 100, 2) if len(o_all) else None,
        "commandes_perdues": int(o_all["order_status"].isin(LOST_STATUSES).sum()),
        "fret_pct_ca": round(o["freight_total"].sum() / o["items_value"].sum() * 100, 1) if o["items_value"].sum() else None,
        "note_si_retard": round(late_scores.get(1.0), 2) if 1.0 in late_scores else None,
        "note_si_a_l_heure": round(late_scores.get(0.0), 2) if 0.0 in late_scores else None,
        "avis_negatifs_si_retard_pct": round(late_neg.get(1.0), 1) if 1.0 in late_neg else None,
        "avis_negatifs_si_a_l_heure_pct": round(late_neg.get(0.0), 1) if 0.0 in late_neg else None,
    }


# --------------------------------------------------------------------------
# Scénarios « et si ? » : ordres de grandeur, hypothèses explicites
# --------------------------------------------------------------------------
SCENARIOS = {
    "retard": "réduire le taux de retard de X points",
    "fidelisation": "porter le taux de clients récurrents à X %",
    "commandes_perdues": "récupérer X % des commandes perdues (annulées / indisponibles)",
}


def simulate(store: Store, sc: Scope, scenario: str, value: float) -> dict:
    b = summary(store, sc)
    aov = b["panier_moyen_brl"] or 0
    if scenario == "retard":
        x = max(0.0, min(float(value), b["taux_retard_pct"] or 0)) / 100
        gap = (b["note_si_a_l_heure"] or 0) - (b["note_si_retard"] or 0)
        neg_gap = (b["avis_negatifs_si_retard_pct"] or 0) - (b["avis_negatifs_si_a_l_heure_pct"] or 0)
        return {"scenario": SCENARIOS["retard"].replace("X", fmt_num(x * 100, 1)), "periode": b["periode"],
                "taux_retard_actuel_pct": b["taux_retard_pct"], "taux_retard_cible_pct": round((b["taux_retard_pct"] or 0) - x * 100, 2),
                "commandes_qui_ne_seraient_plus_en_retard": int(round(x * b["commandes"])),
                "note_actuelle": b["note_moyenne"], "note_estimee": round((b["note_moyenne"] or 0) + x * gap, 2),
                "avis_negatifs_actuels_pct": b["avis_negatifs_pct"],
                "avis_negatifs_estimes_pct": round((b["avis_negatifs_pct"] or 0) - x * neg_gap, 1),
                "hypothese": (f"Une commande qui passe de « en retard » à « à l'heure » prend la note moyenne des commandes à l'heure "
                              f"({fmt_num(b['note_si_a_l_heure'], 2)} au lieu de {fmt_num(b['note_si_retard'], 2)}). "
                              "C'est une association observée, pas un effet causal garanti.")}
    if scenario == "fidelisation":
        target = float(value)
        cur = b["clients_recurrents_pct"] or 0
        extra_cust = max(0.0, (target - cur) / 100 * b["clients_uniques"])
        return {"scenario": SCENARIOS["fidelisation"].replace("X", fmt_num(target, 1)), "periode": b["periode"],
                "taux_actuel_pct": cur, "taux_cible_pct": target, "clients_uniques": b["clients_uniques"],
                "clients_fideles_supplementaires": int(round(extra_cust)),
                "ca_supplementaire_brl": round(extra_cust * aov, 2),
                "ca_supplementaire_pct": round(extra_cust * aov / b["chiffre_affaires_brl"] * 100, 2) if b["chiffre_affaires_brl"] else None,
                "hypothese": f"Chaque client fidélisé en plus passe une commande supplémentaire au panier moyen actuel ({fmt_brl(aov, False)})."}
    if scenario == "commandes_perdues":
        r = max(0.0, min(float(value), 100.0)) / 100
        n = b["commandes_perdues"] * r
        return {"scenario": SCENARIOS["commandes_perdues"].replace("X", fmt_num(r * 100, 0)), "periode": b["periode"],
                "commandes_perdues": b["commandes_perdues"], "taux_commandes_perdues_pct": b["commandes_perdues_pct"],
                "commandes_recuperees": int(round(n)), "ca_recupere_brl": round(n * aov, 2),
                "hypothese": f"Une commande récupérée vaut le panier moyen actuel ({fmt_brl(aov, False)})."}
    raise ValueError(f"Scénario inconnu : {scenario}. Choix : {', '.join(SCENARIOS)}")


def growth(cur, prev):
    if cur is None or prev in (None, 0) or (isinstance(prev, float) and np.isnan(prev)):
        return None
    return round((cur - prev) / abs(prev) * 100, 1)


def compare(store: Store, a: Scope, b: Scope) -> dict:
    """Compare deux périmètres : valeurs et écarts (en % pour les montants, en points pour les taux)."""
    sa, sb = summary(store, a), summary(store, b)
    out = {"periode_a": sa["periode"], "periode_b": sb["periode"], "indicateurs": {}}
    for k, v in sa.items():
        if k == "periode" or not isinstance(v, (int, float)):
            continue
        w = sb.get(k)
        rate = k.endswith("_pct") or k.startswith("note")
        out["indicateurs"][k] = {"a": v, "b": w,
                                 ("ecart_points" if rate else "evolution_pct"):
                                     (round(v - w, 2) if (rate and w is not None and v is not None) else growth(v, w))}
    return out


# --------------------------------------------------------------------------
# Séries et découpages
# --------------------------------------------------------------------------
def monthly(store: Store, sc: Scope) -> pd.DataFrame:
    o, s, o_all = slice_scope(store, sc)
    ms = [m for m in store.all_months if sc.start <= m <= sc.end]
    rev = s.groupby("m")["revenue_brl"].sum()
    g = o.groupby("m")
    df = pd.DataFrame({"m": ms})
    df["chiffre_affaires"] = df["m"].map(rev).fillna(0.0)
    df["commandes"] = df["m"].map(g["order_id"].nunique()).fillna(0).astype(int)
    df["panier_moyen"] = df["chiffre_affaires"] / df["commandes"].replace(0, np.nan)
    df["taux_retard"] = df["m"].map(g["late"].mean() * 100)
    df["note_moyenne"] = df["m"].map(g["review_score"].mean())
    df["fiable"] = df["m"].map(store.reliable).fillna(False)
    df["mois"] = df["m"].map(month_label)
    return df


DIMENSIONS = {"categorie": "category", "etat": "customer_state", "vendeur": "seller_name"}


def comparable_months(store: Store, sc: Scope) -> list[str]:
    rel = store.reliable
    return [m for m in store.all_months if sc.start <= m <= sc.end and rel.get(m) and rel.get(shift_month(m, -12))]


def growth_label(store: Store, sc: Scope) -> str:
    c = comparable_months(store, sc)
    return f"{month_label(c[0])} → {month_label(c[-1])} vs même période N-1" if c else "aucun mois comparable"


def breakdown(store: Store, sc: Scope, dim: str, with_growth: bool = True) -> pd.DataFrame:
    """CA, commandes, panier, retard, note et croissance sur un an par catégorie / État client / vendeur."""
    col = DIMENSIONS[dim]
    o, s, _ = slice_scope(store, sc)
    rev = s.groupby(col).agg(chiffre_affaires=("revenue_brl", "sum"), commandes=("order_id", "nunique"),
                             articles=("order_id", "size"))
    if dim == "etat":
        om = o[["order_id", "customer_state", "late", "review_score", "delivery_days", "freight_total", "items_value"]]
        q = om.groupby("customer_state").agg(taux_retard=("late", "mean"), note_moyenne=("review_score", "mean"),
                                             delai_median=("delivery_days", "median"),
                                             fret=("freight_total", "sum"), valeur=("items_value", "sum"))
        q["fret_pct_ca"] = q["fret"] / q["valeur"] * 100
        q = q.drop(columns=["fret", "valeur"])
    else:
        link = s[["order_id", col]].drop_duplicates()
        om = link.merge(o[["order_id", "late", "review_score", "delivery_days"]], on="order_id", how="left")
        q = om.groupby(col).agg(taux_retard=("late", "mean"), note_moyenne=("review_score", "mean"),
                                delai_median=("delivery_days", "median"))
    q["taux_retard"] = q["taux_retard"] * 100
    df = rev.join(q, how="left")
    df["part_ca_pct"] = df["chiffre_affaires"] / df["chiffre_affaires"].sum() * 100
    df["panier_moyen"] = df["chiffre_affaires"] / df["commandes"]
    if dim == "vendeur":
        df = df.join(s.groupby(col)["seller_state"].agg(lambda x: x.mode().iat[0] if len(x.mode()) else ""))
        df = df.join(s.groupby(col)["category"].agg(lambda x: nice_cat(x.value_counts().index[0])).rename("categorie_principale"))
    if with_growth:
        # croissance sur un an calculée uniquement sur les mois comparables (mois fiable ET même mois N-1 fiable),
        # sinon la comparaison avec 2016 (lancement, quelques commandes) gonfle tout
        comp = comparable_months(store, sc)
        prev = store.sales[store.sales["m"].isin({shift_month(m, -12) for m in comp})]
        if sc.states:
            prev = prev[prev["customer_state"].isin(sc.states)]
        if sc.cats:
            prev = prev[prev["category"].isin(sc.cats)]
        cur = s[s["m"].isin(comp)].groupby(col)["revenue_brl"].sum()
        pr = prev.groupby(col)["revenue_brl"].sum()
        df["ca_annee_precedente"] = df.index.map(pr)
        df["croissance_1an_pct"] = np.where(df["ca_annee_precedente"] >= 1000,
                                            (df.index.map(cur).fillna(0) / df["ca_annee_precedente"] - 1) * 100, np.nan)
    if dim == "etat":
        df = df.join(store.geo, how="left")
    df.index.name = dim
    return df.sort_values("chiffre_affaires", ascending=False).reset_index()


def delay_buckets(store: Store, sc: Scope) -> pd.DataFrame:
    """Note moyenne selon l'avance / le retard de livraison par rapport à la date promise."""
    o, _, _ = slice_scope(store, sc)
    d = o.dropna(subset=["delay_days", "review_score"])
    bins = [-np.inf, -14, -7, 0, 3, 7, 14, np.inf]
    labels = ["> 2 sem. d'avance", "1-2 sem. d'avance", "0-7 j d'avance", "1-3 j de retard",
              "4-7 j de retard", "1-2 sem. de retard", "> 2 sem. de retard"]
    d = d.assign(tranche=pd.cut(d["delay_days"], bins=bins, labels=labels, right=True))
    g = d.groupby("tranche", observed=False).agg(commandes=("order_id", "size"), note_moyenne=("review_score", "mean"),
                                                 avis_negatifs=("review_score", lambda x: x.isin([1, 2]).mean() * 100))
    return g.reset_index()


def seller_scorecard(store: Store, sc: Scope, min_orders: int = 30) -> pd.DataFrame:
    df = breakdown(store, sc, "vendeur")
    nat = summary(store, sc)
    late_n, rv_n = nat["taux_retard_pct"] or 0, nat["note_moyenne"] or 0
    df = df[df["commandes"] >= min_orders].copy()

    def flags(r):
        f = []
        if r["taux_retard"] >= max(late_n * 1.5, late_n + 3):
            f.append("Retards élevés")
        if r["note_moyenne"] <= rv_n - 0.3:
            f.append("Note faible")
        if pd.notna(r.get("croissance_1an_pct")) and r["croissance_1an_pct"] <= -20:
            f.append("CA en baisse")
        return f

    df["alertes"] = df.apply(flags, axis=1)
    df["statut"] = df["alertes"].map(lambda f: "À risque" if len(f) >= 2 else ("À surveiller" if f else "OK"))
    df["rang_ca"] = np.arange(1, len(df) + 1)
    return df


def seller_profile(store: Store, sc: Scope, name: str) -> dict | None:
    s = store.sales[store.sales["seller_name"] == name]
    if s.empty:
        return None
    sc2 = Scope(sc.start, sc.end, sc.states, sc.cats)
    s_p = s[s["m"].between(sc2.start, sc2.end)]
    ids = set(s_p["order_id"])
    o = store.orders[store.orders["order_id"].isin(ids) & store.orders["is_revenue_order"]]
    cats = (s_p.groupby("category")["revenue_brl"].sum().sort_values(ascending=False).head(5))
    mon = s_p.groupby("m")["revenue_brl"].sum()
    return {
        "vendeur": name,
        "etat": s["seller_state"].mode().iat[0] if len(s["seller_state"].mode()) else "",
        "chiffre_affaires_brl": round(float(s_p["revenue_brl"].sum()), 2),
        "commandes": int(len(ids)),
        "taux_retard_pct": round(o["late"].mean() * 100, 2) if o["late"].notna().any() else None,
        "note_moyenne": round(o["review_score"].mean(), 2) if o["review_score"].notna().any() else None,
        "delai_median_jours": float(o["delivery_days"].median()) if o["delivery_days"].notna().any() else None,
        "etats_clients_principaux": o["customer_state"].value_counts().head(5).to_dict(),
        "categories": {nice_cat(k): round(v, 2) for k, v in cats.items()},
        "ca_mensuel": {k: round(v, 2) for k, v in mon.items()},
    }


def find_sellers(store: Store, query: str, n: int = 5) -> list[str]:
    q = query.strip().lower()
    names = store.sales.groupby("seller_name")["revenue_brl"].sum().sort_values(ascending=False)
    hits = [x for x in names.index if q in str(x).lower()]
    if not hits:
        words = [w for w in q.split() if len(w) > 2]
        hits = [x for x in names.index if any(w in str(x).lower() for w in words)]
    return hits[:n]


def events_impact(store: Store) -> list[dict]:
    return (store.kpi.get("breakdowns") or {}).get("events", [])


# --------------------------------------------------------------------------
# Briefing : signaux du dernier mois complet (règles simples et transparentes)
# --------------------------------------------------------------------------
def month_signals(store: Store, m: str) -> list[dict]:
    cur, prev, ly = Scope(m, m), Scope(shift_month(m, -1), shift_month(m, -1)), Scope(shift_month(m, -12), shift_month(m, -12))
    a, b, c = summary(store, cur), summary(store, prev), summary(store, ly)
    out = []
    g_m, g_y = growth(a["chiffre_affaires_brl"], b["chiffre_affaires_brl"]), growth(a["chiffre_affaires_brl"], c["chiffre_affaires_brl"])
    tone = "bad" if (g_m or 0) < -5 else ("good" if (g_m or 0) > 5 else "neutral")
    out.append({"tone": tone, "titre": "Chiffre d'affaires",
                "texte": f"{fmt_brl(a['chiffre_affaires_brl'])} en {month_label(m, True)} : {fmt_pct(g_m, 1, True)} vs mois précédent, "
                         f"{fmt_pct(g_y, 0, True)} sur un an."})
    dl = (a["taux_retard_pct"] or 0) - (b["taux_retard_pct"] or 0)
    out.append({"tone": "bad" if dl > 1 else ("good" if dl < -1 else "neutral"), "titre": "Livraison",
                "texte": f"{fmt_pct(a['taux_retard_pct'])} de commandes en retard ({'+' if dl > 0 else ''}{fmt_num(dl, 1)} pt vs mois précédent), "
                         f"délai médian {fmt_num(a['delai_median_jours'], 0)} jours."})
    dr = (a["note_moyenne"] or 0) - (b["note_moyenne"] or 0)
    out.append({"tone": "bad" if dr < -0.1 else ("good" if dr > 0.1 else "neutral"), "titre": "Satisfaction",
                "texte": f"Note moyenne {fmt_num(a['note_moyenne'], 2)} / 5 ({'+' if dr > 0 else ''}{fmt_num(dr, 2)} vs mois précédent) ; "
                         f"{fmt_pct(a['avis_negatifs_pct'])} d'avis à 1-2 étoiles."})
    cat = breakdown(store, Scope(shift_month(m, -2), m), "categorie")
    cat = cat[(cat["chiffre_affaires"] > 20000) & (cat["ca_annee_precedente"] >= 20000)].dropna(subset=["croissance_1an_pct"])
    if len(cat):
        up, dn = cat.sort_values("croissance_1an_pct").iloc[-1], cat.sort_values("croissance_1an_pct").iloc[0]
        out.append({"tone": "good", "titre": "Catégorie en forte hausse",
                    "texte": f"{nice_cat(up['categorie'])} : {fmt_pct(up['croissance_1an_pct'], 0, True)} sur un an (3 derniers mois)."})
        if dn["croissance_1an_pct"] < 0:
            out.append({"tone": "bad", "titre": "Catégorie en recul",
                        "texte": f"{nice_cat(dn['categorie'])} : {fmt_pct(dn['croissance_1an_pct'], 0, True)} sur un an (3 derniers mois)."})
    st_ = breakdown(store, Scope(shift_month(m, -2), m), "etat", with_growth=False)
    st_ = st_[st_["commandes"] >= 300].dropna(subset=["taux_retard"])
    if len(st_):
        nat = summary(store, Scope(shift_month(m, -2), m))["taux_retard_pct"] or 0
        w = st_.sort_values("taux_retard").iloc[-1]
        out.append({"tone": "bad" if w["taux_retard"] > nat + 2 else "neutral", "titre": "État le plus en retard",
                    "texte": f"{w['etat']} : {fmt_pct(w['taux_retard'])} de retards (moyenne {fmt_pct(nat)}) et note "
                             f"{fmt_num(w['note_moyenne'], 2)}, sur les 3 derniers mois (États ≥ 300 commandes)."})
    return out
