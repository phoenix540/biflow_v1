"""Dashboard Agent BIFlow : génère un dashboard interactif (HTML autonome).

Deux parties :
  1. "Analyses BI" : visualise les résultats du BI Analyst (synthèse, insights priorisés avec leurs preuves,
     tendance, anomalies, baisses, régions, événements, vendeurs) à partir de
     data/analyst/analyst_insights.json et data/kpi/kpi_results.json.
  2. "Exploration" : cubes filtrables (période, État, catégorie, devise) construits depuis data/curated/.

Principes (comme pour les autres agents) :
  - les calculs sont déterministes : le LLM ne touche jamais aux chiffres ;
  - chaque KPI affiché est documenté (formule + table source) pour l'Auditor ;
  - les contrôles vérifient que les totaux du dashboard = totaux des tables ETL et des KPI ;
  - les mois non fiables (is_reliable_month = False) sont signalés et peuvent être exclus.

Sortie : data/dashboard/dashboard.html + data/dashboard/dashboard_agent_decision.json
"""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CURATED = ROOT / "data" / "curated"
ETL_DECISION = ROOT / "data" / "etl" / "olist_model_decision.json"   # configuration métier (taux BRL->TND)
PROFILE_PATH = ROOT / "data" / "profile" / "profile.json"
KPI_PATH = ROOT / "data" / "kpi" / "kpi_results.json"
ANALYST_PATH = ROOT / "data" / "analyst" / "analyst_insights.json"
OUT = ROOT / "data" / "dashboard"

# Définition des KPI affichés : traçabilité pour l'Auditor
KPI_DEFINITIONS = [
    {"kpi": "Chiffre d'affaires", "formula": "somme(price) des articles des commandes is_revenue_order=True (hors livraison)",
     "source": "fact_sales.revenue_brl"},
    {"kpi": "Nombre de commandes", "formula": "nombre de commandes distinctes comptées dans le CA",
     "source": "fact_sales.order_id (is_revenue_order=True)"},
    {"kpi": "Panier moyen", "formula": "Chiffre d'affaires / Nombre de commandes", "source": "fact_sales"},
    {"kpi": "Marge (simulée)", "formula": "CA - coût ; coût = production_cost_brl, SIMULÉ à partir du prix (~70 %)",
     "source": "fact_sales.margin_brl"},
    {"kpi": "Taux de retard", "formula": "commandes livrées après la date estimée (par jour) / commandes avec les deux dates",
     "source": "fact_orders.is_late"},
    {"kpi": "Délai moyen de livraison", "formula": "moyenne de delivery_days (achat -> livraison client)",
     "source": "fact_orders.delivery_days"},
    {"kpi": "Note moyenne", "formula": "moyenne de review_score (un avis par commande)",
     "source": "fact_orders.review_score"},
    {"kpi": "Croissance sur un an", "formula": "CA des mois fiables vs mêmes mois de l'année précédente",
     "source": "kpi_results.json (breakdowns.monthly)"},
    {"kpi": "Insights", "formula": "signaux détectés par Python, rédigés par le LLM, nombres vérifiés",
     "source": "analyst_insights.json"},
]


# --------------------------------------------------------------------------
# Lecture des données
# --------------------------------------------------------------------------
def _read(name: str, **kw) -> pd.DataFrame:
    path = CURATED / name
    if not path.exists():
        raise FileNotFoundError(f"Fichier manquant : {path} (le modèle métier Olist, olist_model_agent, a-t-il tourné ?)")
    return pd.read_csv(path, **kw)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower() == "true"


# --------------------------------------------------------------------------
# Cubes agrégés (embarqués dans le HTML : pas besoin de serveur)
# --------------------------------------------------------------------------
def build_sales_cube(sales: pd.DataFrame) -> pd.DataFrame:
    """CA, marge, articles et commandes par mois x catégorie x état client (commandes du CA uniquement)."""
    s = sales[sales["is_revenue_order"]].copy()
    s["m"] = s["purchase_date"].dt.to_period("M").astype(str)
    cube = (s.groupby(["m", "category", "customer_state"], dropna=False)
              .agg(rev=("revenue_brl", "sum"), mar=("margin_brl", "sum"),
                   items=("order_id", "size"), orders=("order_id", "nunique"))
              .reset_index().rename(columns={"category": "c", "customer_state": "st"}))
    cube["st"] = cube["st"].fillna("NA")
    return cube.round({"rev": 2, "mar": 2})


def build_orders_cube(fo: pd.DataFrame) -> pd.DataFrame:
    """Livraison et satisfaction par mois x état client (toutes les commandes)."""
    o = fo.copy()
    o["m"] = o["purchase_date"].dt.to_period("M").astype(str)
    o["late"] = o["is_late"].astype(str).map({"True": 1.0, "False": 0.0})
    o["customer_state"] = o["customer_state"].fillna("NA")
    g = o.groupby(["m", "customer_state"])
    cube = pd.DataFrame({
        "n": g.size(),
        "dd_sum": g["delivery_days"].sum(), "dd_n": g["delivery_days"].count(),
        "late": g["late"].sum(), "late_n": g["late"].count(),
        "rv_sum": g["review_score"].sum(), "rv_n": g["review_score"].count(),
    }).reset_index().rename(columns={"customer_state": "st"})
    return cube


def build_daily(fo: pd.DataFrame, dim_date: pd.DataFrame) -> list:
    """CA quotidien des mois fiables + événement du jour (pour visualiser les anomalies de l'Analyste)."""
    rev = fo[fo["is_revenue_order"]].groupby("purchase_date")["revenue_brl"].sum()
    cal = dim_date.copy()
    cal["date"] = pd.to_datetime(cal["date"])
    cal = cal[_bool(cal["is_reliable_month"])].set_index("date")
    cal["rev"] = rev.reindex(cal.index).fillna(0.0)
    cal["event"] = cal["event"].fillna("") if "event" in cal else ""
    return [{"d": f"{d:%Y-%m-%d}", "rev": round(float(r["rev"]), 2), "ev": r["event"]} for d, r in cal.iterrows()]


# --------------------------------------------------------------------------
# Résultats du BI Analyst (insights + preuves) et découpages des KPI
# --------------------------------------------------------------------------
def build_analyst_view(analyst: dict, kpi: dict) -> dict:
    if not analyst:
        return {}
    signals = {s["id"]: s for s in analyst.get("signals", [])}
    rank = {"high": 0, "medium": 1, "low": 2}
    insights = []
    for i in sorted(analyst.get("insights", []), key=lambda x: rank.get(x.get("priority"), 3)):
        insights.append({
            "id": i.get("id"), "title": i.get("title"), "type": i.get("type"), "priority": i.get("priority"),
            "finding": i.get("finding"), "explanation": i.get("explanation"),
            "recommendation": i.get("recommendation"), "verified": bool(i.get("verified")),
            "added_by": i.get("added_by"),
            "evidence": [f"{sid} : {signals[sid]['summary']}" for sid in i.get("signal_ids", []) if sid in signals],
        })
    alert_states = []
    for s in analyst.get("signals", []):
        if s["kind"] == "region_delivery":
            alert_states = [x["state"] for x in s["facts"].get("states", [])]
    anomalies = [{"start": s["facts"]["start"], "end": s["facts"]["end"], "peak": s["facts"]["peak_date"],
                  "label": s["title"], "explained": s["facts"].get("explained_by_event", False)}
                 for s in analyst.get("signals", []) if s["kind"] == "daily_anomaly"]
    b = kpi.get("breakdowns", {})
    try:   # précautions recalculées sur les KPI actuels (même règle que l'Analyste)
        from analyst_agent import caveats
        current_caveats = caveats(kpi)
    except Exception:
        current_caveats = analyst.get("caveats", [])
    return {
        "generated_at": analyst.get("generated_at"), "generated_by": analyst.get("generated_by"),
        "model": analyst.get("model"), "summary": analyst.get("executive_summary", ""),
        "insights": insights, "watchlist": analyst.get("watchlist", []), "caveats": current_caveats,
        "alerts": sum(s["severity"] == "alert" for s in analyst.get("signals", [])),
        "signals_total": len(signals), "alert_states": alert_states, "anomalies": anomalies,
        "headline": kpi.get("headline", {}), "late_vs_ontime": kpi.get("insights", {}),
        "monthly": b.get("monthly", []), "yoy": b.get("yoy_summary", {}),
        "states": b.get("by_customer_state", []), "categories": b.get("categories", {}),
        "sellers": b.get("sellers", {}), "events": b.get("events", []),
        "weekday": b.get("weekday_revenue_share", []),
    }


# --------------------------------------------------------------------------
# Contrôles
# --------------------------------------------------------------------------
def run_checks(sales, fo, sales_cube, orders_cube, kpi, analyst) -> list:
    checks = []

    def check(name, ok, detail):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    rev_src = float(sales["revenue_brl"].sum())
    rev_cube = float(sales_cube["rev"].sum())
    check("CA du dashboard = CA de fact_sales", abs(rev_src - rev_cube) < 1.0,
          f"fact_sales {rev_src:,.2f} / dashboard {rev_cube:,.2f}")
    check("CA fact_orders = CA fact_sales", abs(float(fo["revenue_brl"].sum()) - rev_src) < 1.0,
          f"{float(fo['revenue_brl'].sum()):,.2f} vs {rev_src:,.2f}")
    check("Nb de commandes (cube livraison) = lignes de fact_orders", int(orders_cube["n"].sum()) == len(fo),
          f"{int(orders_cube['n'].sum())} vs {len(fo)}")
    exp_orders = int(sales.loc[sales["is_revenue_order"], "order_id"].nunique())
    # une commande multi-catégories est comptée dans chaque catégorie : le cube peut dépasser légèrement
    check("Commandes du CA : cube >= commandes distinctes (multi-catégories comptées une fois par catégorie)",
          int(sales_cube["orders"].sum()) >= exp_orders,
          f"cube {int(sales_cube['orders'].sum())} / distinctes {exp_orders}")
    check("Aucun CA négatif", bool((sales_cube["rev"] >= 0).all()), "")
    kpi_rev = (kpi.get("headline") or {}).get("total_revenue_brl")
    check("CA du dashboard = CA de l'agent KPI", kpi_rev is not None and abs(rev_cube - kpi_rev) < 1.0,
          f"dashboard {rev_cube:,.2f} / KPI {kpi_rev}")
    check("Analyses du BI Analyst disponibles", bool(analyst.get("insights")),
          f"{len(analyst.get('insights', []))} insights")
    return checks


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------
HTML = r"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BIFlow - Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{
  --bg:#f4f6fa;--card:#fff;--ink:#17202f;--ink2:#3d4757;--mute:#6b7280;--line:#e3e7ef;--soft:#f1f3f8;
  --navy:#0b1b2b;--blue:#2f6fd6;--good:#12976f;--bad:#d14343;--warn:#c97c00;--grey:#c3c8d4;
  --good-bg:#e7f6f0;--bad-bg:#fdeeee;--warn-bg:#fff5e4;--blue-bg:#eaf1fd;
}
*{box-sizing:border-box}
body{margin:0;font-family:system-ui,"Segoe UI",Arial,sans-serif;background:var(--bg);color:var(--ink);font-size:14px}
header{padding:16px 28px 0;background:var(--navy);color:#fff}
.top{display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:6px}
header h1{margin:0;font-size:20px;font-weight:650}header small{color:#9fb3c8;font-size:12px}
nav.tabs{display:flex;gap:4px;margin-top:10px}
nav.tabs button{background:none;border:none;color:#9fb3c8;padding:10px 16px;font-size:14px;cursor:pointer;border-bottom:3px solid transparent}
nav.tabs button.on{color:#fff;border-bottom-color:#2fc7a8;font-weight:600}
.jump{position:sticky;top:0;z-index:5;display:flex;gap:6px;flex-wrap:wrap;padding:10px 28px;background:rgba(244,246,250,.95);backdrop-filter:blur(4px);border-bottom:1px solid var(--line)}
.jump a{font-size:12.5px;color:var(--ink2);text-decoration:none;padding:4px 10px;border-radius:14px;background:var(--card);border:1px solid var(--line)}
.jump a:hover{border-color:var(--blue);color:var(--blue)}
main{padding:8px 28px 18px;max-width:1400px;margin:auto}
section.block{scroll-margin-top:56px}
h2{font-size:17px;margin:26px 0 4px}
.lead{margin:0 0 12px;color:var(--mute);font-size:13px}
/* KPI */
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:14px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;border-top:3px solid var(--grey)}
.kpi.good{border-top-color:var(--good)}.kpi.bad{border-top-color:var(--bad)}.kpi.warn{border-top-color:var(--warn)}.kpi.info{border-top-color:var(--blue)}
.kpi .l{font-size:12px;color:var(--mute);text-transform:uppercase;letter-spacing:.03em}
.kpi .v{font-size:clamp(20px,2vw,26px);font-weight:700;margin-top:4px;font-variant-numeric:tabular-nums;white-space:nowrap}
.kpi .n{font-size:12px;color:var(--mute);margin-top:3px;line-height:1.35}
/* synthèse */
.overview{display:grid;grid-template-columns:2fr 1fr;gap:12px}
@media(max-width:900px){.overview{grid-template-columns:1fr}}
.summary{background:var(--card);border:1px solid var(--line);border-left:5px solid var(--blue);border-radius:10px;padding:14px 18px;line-height:1.6;color:var(--ink2)}
.summary b{color:var(--ink)}
.cav{background:var(--warn-bg);border:1px solid #f1d39b;border-radius:10px;padding:12px 16px;font-size:13px;line-height:1.5}
.cav h4{margin:0 0 6px;font-size:13px}.cav ul{margin:0;padding-left:18px}
/* insights */
.chips{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px}
.chip{border:1px solid var(--line);background:var(--card);border-radius:14px;padding:4px 12px;font-size:12.5px;cursor:pointer;color:var(--ink2)}
.chip.on{background:var(--navy);color:#fff;border-color:var(--navy)}
.insights{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:12px}
@media(max-width:500px){.insights{grid-template-columns:1fr}}
.ins{background:var(--card);border:1px solid var(--line);border-left:5px solid var(--grey);border-radius:10px;padding:12px 16px;font-size:13.5px;line-height:1.5;display:flex;flex-direction:column}
.ins.high{border-left-color:var(--bad)}.ins.medium{border-left-color:var(--warn)}.ins.low{border-left-color:var(--good)}
.ins h3{margin:6px 0;font-size:15px;line-height:1.35}
.ins .finding{margin:0;color:var(--ink2);display:-webkit-box;-webkit-line-clamp:4;-webkit-box-orient:vertical;overflow:hidden}
.ins.open .finding{-webkit-line-clamp:unset}
.ins .act{margin-top:10px;background:var(--soft);border-radius:8px;padding:8px 10px;font-size:13px}
.ins .act b{color:var(--ink)}
.ins .more{margin-top:auto;padding-top:8px}
.ins .more button{background:none;border:none;padding:0;color:var(--blue);cursor:pointer;font-size:12.5px}
.ins .extra{display:none;font-size:12.5px;color:var(--ink2);margin-top:8px}
.ins.open .extra{display:block}
.tag{display:inline-block;font-size:11px;padding:2px 8px;border-radius:10px;margin:0 4px 2px 0;background:#eef0f6;color:#374151}
.tag.high{background:var(--bad-bg);color:#a82a2a}.tag.medium{background:var(--warn-bg);color:#8a5600}.tag.low{background:var(--good-bg);color:#0f6e52}
.tag.ok{background:var(--good-bg);color:#0f6e52}.tag.ko{background:var(--bad-bg);color:#a82a2a}
ul.plain{margin:6px 0;padding-left:20px;line-height:1.6}
/* cartes graphiques */
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:14px}
@media(max-width:520px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px 8px;min-width:0}
.card h3{margin:0;font-size:14.5px}
.card .cap{margin:3px 0 6px;font-size:12.5px;color:var(--mute);line-height:1.4}
.card .cap b{color:var(--ink2)}
.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:12px;color:var(--mute);margin-bottom:4px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
.wide{grid-column:1/-1}
/* tableau */
table{border-collapse:collapse;width:100%}
td,th{border-bottom:1px solid var(--line);padding:7px 8px;text-align:left;vertical-align:middle;font-size:13px}
th{font-size:12px;color:var(--mute);font-weight:600}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.bar{height:6px;background:var(--blue);border-radius:3px;opacity:.75;margin-top:3px}
.badge{display:inline-block;padding:1px 7px;border-radius:9px;font-size:12px}
.badge.bad{background:var(--bad-bg);color:#a82a2a}.badge.ok{background:var(--soft);color:var(--ink2)}
.tablewrap{overflow-x:auto}
/* exploration */
.filters{display:flex;gap:14px;flex-wrap:wrap;padding:14px 28px;background:var(--card);border-bottom:1px solid var(--line);align-items:end}
.filters label{font-size:12px;color:var(--mute);display:flex;flex-direction:column;gap:4px}
select,.btn{padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:#fff;font-size:13px}
.btn{cursor:pointer;color:var(--ink2)}.btn:hover{border-color:var(--blue);color:var(--blue)}
.banner{margin:14px 0 0;padding:10px 14px;border-radius:8px;background:var(--warn-bg);border:1px solid #f1d39b;font-size:13px}
details.box{margin-top:16px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px;font-size:13px}
details.box summary{cursor:pointer;font-weight:600}
.hidden{display:none !important}
</style></head><body>
<header>
  <div class="top"><h1>BIFlow - Dashboard Olist</h1><small id="meta"></small></div>
  <nav class="tabs"><button id="tab_a" class="on">Analyses BI</button><button id="tab_e">Exploration (filtres)</button></nav>
</header>

<section id="sec_a">
<div class="jump" id="jump">
  <a href="#s_over">Vue d'ensemble</a><a href="#s_ins">Insights</a><a href="#s_trend">Tendance</a>
  <a href="#s_deliv">Livraison et satisfaction</a><a href="#s_cat">Catégories</a><a href="#s_opp">Événements et vendeurs</a><a href="#s_watch">À surveiller</a>
</div>
<main>
  <div id="a_empty" class="banner hidden">Aucune analyse du BI Analyst : lancer analyst_agent.py puis régénérer le dashboard.</div>
  <div id="a_body">
    <section class="block" id="s_over">
      <div class="kpis" id="a_kpis"></div>
      <div class="overview">
        <div class="summary" id="a_summary"></div>
        <div class="cav"><h4>À lire avant d'interpréter</h4><ul id="a_cav"></ul></div>
      </div>
    </section>

    <section class="block" id="s_ins">
      <h2>Insights priorisés</h2>
      <p class="lead">Chaque insight est fondé sur des signaux calculés par Python ; les chiffres cités sont vérifiés. Cliquer « Détails » pour l'explication et les preuves.</p>
      <div class="chips" id="a_chips"></div>
      <div class="insights" id="a_insights"></div>
    </section>

    <section class="block" id="s_trend">
      <h2>Tendance du chiffre d'affaires</h2>
      <p class="lead">Le CA progresse-t-il, et quels jours sortent de l'ordinaire ?</p>
      <div class="grid">
        <div class="card"><h3>CA mensuel</h3><p class="cap" id="cap_monthly"></p>
          <div class="legend"><span><i style="background:var(--blue)"></i>mois fiable</span><span><i style="background:var(--grey)"></i>mois incomplet (exclu des tendances)</span></div>
          <div id="a_monthly" style="height:290px"></div></div>
        <div class="card"><h3>Croissance sur un an, par mois</h3><p class="cap" id="cap_yoy"></p>
          <div id="a_yoy" style="height:310px"></div></div>
        <div class="card wide"><h3>CA quotidien et épisodes atypiques</h3><p class="cap" id="cap_daily"></p>
          <div class="legend"><span><i style="background:var(--blue)"></i>CA du jour</span><span><i style="background:var(--warn)"></i>jour d'événement</span>
            <span><i style="background:rgba(201,124,0,.35)"></i>épisode expliqué par un événement</span><span><i style="background:rgba(209,67,67,.35)"></i>épisode inexpliqué</span></div>
          <div id="a_daily" style="height:330px"></div></div>
      </div>
    </section>

    <section class="block" id="s_deliv">
      <h2>Livraison et satisfaction</h2>
      <p class="lead">Les retards de livraison sont le premier facteur associé aux mauvaises notes.</p>
      <div class="grid">
        <div class="card"><h3>Note moyenne : commandes en retard vs à l'heure</h3><p class="cap" id="cap_late"></p>
          <div id="a_late" style="height:230px"></div></div>
        <div class="card"><h3>Par État : taux de retard vs note moyenne</h3><p class="cap">Chaque bulle est un État (taille = nombre de commandes, États ≥ 100 commandes). Les lignes pointillées marquent la moyenne nationale. <b>En bas à droite = plus de retards, moins bonne note.</b></p>
          <div class="legend"><span><i style="background:var(--bad)"></i>État en alerte</span><span><i style="background:var(--blue)"></i>autres États</span></div>
          <div id="a_states" style="height:380px"></div></div>
      </div>
    </section>

    <section class="block" id="s_cat">
      <h2>Catégories de produits</h2>
      <p class="lead">Quelles catégories tirent la croissance, lesquelles reculent ou déçoivent les clients ?</p>
      <div class="grid">
        <div class="card"><h3>Croissance sur un an par catégorie</h3><p class="cap">Les catégories à plus forte croissance (vert) et celles en baisse (rouge).</p>
          <div id="a_cat" style="height:360px"></div></div>
        <div class="card"><h3>Catégories les moins bien notées</h3><p class="cap" id="cap_lowcat"></p>
          <div id="a_lowcat" style="height:360px"></div></div>
      </div>
    </section>

    <section class="block" id="s_opp">
      <h2>Événements, vendeurs et calendrier</h2>
      <p class="lead">Les leviers commerciaux et les dépendances à surveiller.</p>
      <div class="grid">
        <div class="card wide"><h3>Effet des événements sur le CA du jour</h3><p class="cap">Écart de CA par rapport à un jour normal du même mois, trié du plus fort au plus faible.</p>
          <div class="legend"><span><i style="background:var(--good)"></i>événement commercial</span><span><i style="background:var(--blue)"></i>jour férié, hausse</span><span><i style="background:var(--bad)"></i>jour férié, baisse</span></div>
          <div id="a_events"></div></div>
        <div class="card"><h3>Concentration du CA chez les vendeurs</h3><p class="cap" id="cap_conc"></p>
          <div id="a_conc" style="height:240px"></div></div>
        <div class="card"><h3>Part du CA par jour de la semaine</h3><p class="cap" id="cap_week"></p>
          <div id="a_week" style="height:260px"></div></div>
        <div class="card wide"><h3>Meilleurs vendeurs</h3><p class="cap" id="cap_sellers"></p>
          <div class="tablewrap"><table id="a_sellers"></table></div></div>
      </div>
    </section>

    <section class="block" id="s_watch">
      <h2>À surveiller</h2>
      <div class="card"><ul class="plain" id="a_watch"></ul></div>
    </section>
  </div>
</main></section>

<section id="sec_e" class="hidden">
<div class="filters">
  <label>Du mois<select id="from"></select></label>
  <label>Au mois<select id="to"></select></label>
  <label>État client<select id="state"></select></label>
  <label>Catégorie<select id="cat"></select></label>
  <label>Devise<select id="cur"><option value="BRL">BRL</option><option value="TND">TND</option></select></label>
  <label><span>Mois incomplets</span><select id="rel"><option value="keep">Inclure (signalés)</option><option value="drop">Exclure</option></select></label>
  <button class="btn" id="reset">Réinitialiser</button>
</div>
<main>
  <div class="banner" id="banner" style="display:none"></div>
  <div class="kpis" id="kpis"></div>
  <div class="grid">
    <div class="card wide"><h3>Chiffre d'affaires mensuel</h3><p class="cap">Points orange = mois incomplet ; lignes pointillées = événements commerciaux.</p><div id="c_rev" style="height:320px"></div></div>
    <div class="card"><h3>Top 10 catégories par CA</h3><p class="cap">Selon les filtres actifs.</p><div id="c_cat" style="height:360px"></div></div>
    <div class="card"><h3>Top 10 États clients par CA</h3><p class="cap">Selon les filtres actifs.</p><div id="c_state" style="height:360px"></div></div>
    <div class="card"><h3>Taux de retard de livraison</h3><p class="cap">Par mois d'achat (filtre catégorie non appliqué).</p><div id="c_late" style="height:280px"></div></div>
    <div class="card"><h3>Note moyenne des avis</h3><p class="cap">Par mois d'achat (filtre catégorie non appliqué).</p><div id="c_rev_score" style="height:280px"></div></div>
  </div>
</main>
</section>
<main style="padding-top:0">
  <details class="box"><summary>Définition des KPI et hypothèses (traçabilité)</summary>
    <div class="tablewrap"><table id="defs"></table></div><p id="assump"></p></details>
</main>
<script>
const D = __DATA__;
const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const fmt = (x, d=0) => x==null||isNaN(x) ? "n/a" : x.toLocaleString("fr-FR",{maximumFractionDigits:d,minimumFractionDigits:d});
const big = x => x==null||isNaN(x) ? "n/a" : Math.abs(x)>=1e6 ? fmt(x/1e6,2)+" M" : Math.abs(x)>=1e4 ? fmt(x/1e3,0)+" k" : fmt(x);
const pct = (x,d=1) => (x>0?"+":"")+fmt(x,d)+" %";
const nice = s => String(s ?? "").replace(/_/g," ");
const MOIS = ["janv.","févr.","mars","avr.","mai","juin","juil.","août","sept.","oct.","nov.","déc."];
const mlabel = m => { const [y,mm] = String(m).split("-"); return MOIS[+mm-1]+" "+y; };
const C = {blue:"#2f6fd6",good:"#12976f",bad:"#d14343",warn:"#c97c00",grey:"#c3c8d4",ink:"#17202f",mute:"#6b7280",line:"#e3e7ef"};
const AX = {gridcolor:"#eef1f6",zerolinecolor:"#cfd5e0",linecolor:C.line,tickfont:{size:11,color:"#4b5563"},titlefont:{size:12,color:C.mute}};
const L = (extra={}) => {
  const base = {margin:{t:10,l:56,r:16,b:40},font:{family:"system-ui,Segoe UI,Arial",size:12,color:C.ink},
    paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"rgba(0,0,0,0)",showlegend:false,hoverlabel:{bgcolor:"#fff",bordercolor:C.line,font:{color:C.ink}},
    separators:", "};
  const o = Object.assign(base, extra);
  o.xaxis = Object.assign({}, AX, extra.xaxis||{}); o.yaxis = Object.assign({}, AX, extra.yaxis||{});
  return o;
};
const CFG = {responsive:true, displaylogo:false, displayModeBar:false};
const relMap = Object.fromEntries(D.months.map(m=>[m.m,m.reliable]));
const months = D.months.map(m=>m.m);
const kpiCard = (l,v,n,tone="",title="") => `<div class="kpi ${tone}" ${title?`title="${esc(title)}"`:""}><div class="l">${esc(l)}</div><div class="v">${esc(v)}</div><div class="n">${esc(n)}</div></div>`;

$("meta").textContent = `Généré le ${D.generated_at} · Score qualité des données : ${D.quality_score ?? "n/a"}/100`;
$("defs").innerHTML = "<tr><th>KPI</th><th>Formule</th><th>Source</th></tr>" +
  D.kpi_definitions.map(k=>`<tr><td>${esc(k.kpi)}</td><td>${esc(k.formula)}</td><td>${esc(k.source)}</td></tr>`).join("");
$("assump").innerHTML = "<b>Hypothèses :</b><br>" + D.assumptions.map(a=>"- "+esc(a)).join("<br>");

// ---------------------------------------------------------------- onglets
let exploreDrawn = false;
function show(tab){
  $("sec_a").classList.toggle("hidden", tab!=="a"); $("sec_e").classList.toggle("hidden", tab!=="e");
  $("tab_a").classList.toggle("on", tab==="a"); $("tab_e").classList.toggle("on", tab==="e");
  if(tab==="e" && !exploreDrawn){ render(); exploreDrawn = true; }
  window.dispatchEvent(new Event("resize"));
}
$("tab_a").onclick = () => show("a"); $("tab_e").onclick = () => show("e");
// liens d'ancrage : défilement doux (fonctionne aussi dans l'iframe Streamlit)
document.querySelectorAll("#jump a").forEach(a=>a.onclick = e => { e.preventDefault(); const t=document.querySelector(a.getAttribute("href")); if(t) t.scrollIntoView({behavior:"smooth"}); });

// ---------------------------------------------------------------- Analyses BI
function renderAnalyst(){
  const A = D.analyst;
  if(!A || !A.insights){ $("a_empty").classList.remove("hidden"); $("a_body").classList.add("hidden"); $("jump").classList.add("hidden"); return; }
  const h = A.headline || {}, lv = A.late_vs_ontime || {};

  // ---- KPI
  const yoyP = A.yoy && A.yoy.period ? `${A.yoy.period.replace("->","à")} vs même période ${A.yoy.compared_to ? A.yoy.compared_to.slice(0,4) : "N-1"}` : "";
  $("a_kpis").innerHTML = [
    kpiCard("Chiffre d'affaires", big(h.total_revenue_brl)+" BRL", `${fmt(h.delivered_orders)} commandes livrées`, "info", fmt(h.total_revenue_brl,2)+" BRL"),
    kpiCard("Croissance sur un an", pct(h.yoy_revenue_growth_pct), yoyP, h.yoy_revenue_growth_pct>=0?"good":"bad"),
    kpiCard("Panier moyen", fmt(h.average_order_value_brl,2)+" BRL", `fret = ${fmt(h.freight_to_revenue_pct,1)} % du CA`, ""),
    kpiCard("Livraisons en retard", fmt(h.late_delivery_rate_pct,1)+" %", `délai médian ${fmt(h.median_delivery_days,0)} jours`, "warn"),
    kpiCard("Note moyenne", fmt(h.average_review_score,2)+" / 5", `${fmt(h.low_rating_rate_pct,1)} % d'avis à 1-2 étoiles`, ""),
    kpiCard("Clients récurrents", fmt(h.repeat_customer_rate_pct,1)+" %", `sur ${fmt(h.unique_customers)} clients uniques`, "bad"),
  ].join("");

  const src = A.generated_by==="llm" ? `rédigée par LLM (${esc(A.model)}), chiffres vérifiés` : "générée par règles (sans LLM)";
  $("a_summary").innerHTML = `<b>Synthèse</b> <span class="tag">${src}</span><span class="tag">${fmt(A.alerts)} alertes · ${A.signals_total} signaux</span><br>${esc(A.summary)}`;
  $("a_cav").innerHTML = A.caveats.map(w=>`<li>${esc(w)}</li>`).join("");

  // ---- Insights avec filtre de priorité
  const PRIO = {high:"Priorité haute", medium:"Priorité moyenne", low:"Priorité basse"};
  const TYPE = {trend:"tendance",alert:"alerte",risk:"risque",opportunity:"opportunité",anomaly:"anomalie"};
  const cnt = p => A.insights.filter(i=>i.priority===p).length;
  const chips = [["all",`Toutes (${A.insights.length})`], ...["high","medium","low"].filter(cnt).map(p=>[p,`${PRIO[p]} (${cnt(p)})`])];
  let cur = "all";
  const drawIns = () => {
    $("a_chips").innerHTML = chips.map(([k,l])=>`<button class="chip ${k===cur?"on":""}" data-k="${k}">${esc(l)}</button>`).join("");
    $("a_chips").querySelectorAll(".chip").forEach(b=>b.onclick=()=>{cur=b.dataset.k; drawIns();});
    $("a_insights").innerHTML = A.insights.filter(i=>cur==="all"||i.priority===cur).map(i=>`
      <article class="ins ${esc(i.priority)}">
        <div><span class="tag ${esc(i.priority)}">${esc(PRIO[i.priority]||i.priority)}</span><span class="tag">${esc(TYPE[i.type]||i.type)}</span><span class="tag ${i.verified?"ok":"ko"}">${i.verified?"✓ chiffres vérifiés":"à contrôler"}</span></div>
        <h3>${esc(i.title)}</h3>
        <p class="finding">${esc(i.finding)}</p>
        <div class="act"><b>→ Action :</b> ${esc(i.recommendation)}</div>
        <div class="extra">
          ${i.explanation?`<p><b>Explication :</b> ${esc(i.explanation)}</p>`:""}
          <b>Preuves (${i.evidence.length} signal${i.evidence.length>1?"s":""}) :</b><ul class="plain">${i.evidence.map(e=>`<li>${esc(e)}</li>`).join("")}</ul>
          ${i.added_by?`<p class="tag">ajouté par ${esc(i.added_by)}</p>`:""}
        </div>
        <div class="more"><button>Détails ▾</button></div>
      </article>`).join("");
    $("a_insights").querySelectorAll(".ins").forEach(c=>c.querySelector(".more button").onclick=e=>{
      c.classList.toggle("open"); e.target.textContent = c.classList.contains("open")?"Masquer ▴":"Détails ▾"; });
  };
  drawIns();

  // ---- CA mensuel
  const mo = A.monthly, rel = mo.filter(m=>m.is_reliable_month);
  const best = rel.reduce((a,b)=>b.revenue_brl>a.revenue_brl?b:a, rel[0]||{});
  const last = rel[rel.length-1]||{};
  $("cap_monthly").innerHTML = `Pic en <b>${mlabel(best.month)}</b> (${big(best.revenue_brl)} BRL) ; dernier mois fiable <b>${mlabel(last.month)}</b> : ${big(last.revenue_brl)} BRL.`;
  const moC = mo.filter(m=>m.month>="2017-01");   // 2016 : quelques commandes seulement, masqué
  Plotly.newPlot("a_monthly",[{x:moC.map(m=>m.month+"-01"),y:moC.map(m=>m.revenue_brl),type:"bar",
    marker:{color:moC.map(m=>m.is_reliable_month?C.blue:C.grey)},hovertemplate:"%{x|%b %Y}<br>%{y:,.0f} BRL<extra></extra>"}],
    L({yaxis:{title:"BRL",tickformat:"~s"},xaxis:{type:"date",tickformat:"%b %Y"}}),CFG);

  // ---- Croissance sur un an
  const yo = mo.filter(m=>m.yoy_growth_pct!=null);
  if(yo.length){
    const f = yo[0], l = yo[yo.length-1];
    $("cap_yoy").innerHTML = l.yoy_growth_pct < f.yoy_growth_pct
      ? `Toujours positive mais <b>en ralentissement</b> : ${pct(f.yoy_growth_pct,0)} en ${mlabel(f.month)} → ${pct(l.yoy_growth_pct,0)} en ${mlabel(l.month)}.`
      : `De ${pct(f.yoy_growth_pct,0)} en ${mlabel(f.month)} à ${pct(l.yoy_growth_pct,0)} en ${mlabel(l.month)}.`;
  }
  const ymax = Math.max(0,...yo.map(m=>m.yoy_growth_pct)), ymin = Math.min(0,...yo.map(m=>m.yoy_growth_pct));
  Plotly.newPlot("a_yoy",[{x:yo.map(m=>mlabel(m.month)),y:yo.map(m=>m.yoy_growth_pct),type:"bar",
    marker:{color:yo.map(m=>m.yoy_growth_pct<0?C.bad:C.good)},text:yo.map(m=>pct(m.yoy_growth_pct,0)),textposition:"outside",
    cliponaxis:false,hovertemplate:"%{x}<br>%{y:.1f} % vs même mois N-1<extra></extra>"}],
    L({yaxis:{title:"% vs N-1",range:[ymin*1.25, ymax*1.18]}}),CFG);

  // ---- CA quotidien
  const shapes = A.anomalies.map(a=>({type:"rect",xref:"x",yref:"paper",x0:a.start,x1:a.end+"T23:59",y0:0,y1:1,
    fillcolor:a.explained?"rgba(201,124,0,.28)":"rgba(209,67,67,.28)",line:{width:0},layer:"below"}));
  const evd = D.daily.filter(d=>d.ev);
  const pk = D.daily.reduce((a,b)=>b.rev>a.rev?b:a, D.daily[0]||{});
  $("cap_daily").innerHTML = `${A.anomalies.length} épisode(s) atypique(s) détecté(s). Jour record : <b>${esc(pk.d)}</b>${pk.ev?` (${esc(pk.ev)})`:""} avec ${big(pk.rev)} BRL. Glisser pour zoomer, double-clic pour revenir.`;
  Plotly.newPlot("a_daily",[
    {x:D.daily.map(d=>d.d),y:D.daily.map(d=>d.rev),type:"scatter",mode:"lines",line:{color:C.blue,width:1.3},name:"CA du jour",
     hovertemplate:"%{x|%d %b %Y}<br>%{y:,.0f} BRL<extra></extra>"},
    {x:evd.map(d=>d.d),y:evd.map(d=>d.rev),type:"scatter",mode:"markers",marker:{color:C.warn,size:7,symbol:"diamond",line:{color:"#fff",width:1}},
     name:"Événement",text:evd.map(d=>d.ev),hovertemplate:"<b>%{text}</b><br>%{x|%d %b %Y}<br>%{y:,.0f} BRL<extra></extra>"}],
    L({shapes,yaxis:{title:"BRL",tickformat:"~s"},xaxis:{type:"date"},
       annotations:pk.d?[{x:pk.d,y:pk.rev,text:pk.ev||"record",showarrow:true,arrowhead:0,ax:60,ay:10,font:{size:11,color:C.ink},bgcolor:"rgba(255,255,255,.85)"}]:[]}),CFG);

  // ---- Retard vs note
  const gap = (lv.review_score_on_time_orders||0)-(lv.review_score_late_orders||0);
  $("cap_late").innerHTML = `Une commande en retard est notée <b>${fmt(gap,2)} point(s) de moins</b> sur 5. Association observée, pas une preuve de causalité.`;
  Plotly.newPlot("a_late",[{type:"bar",orientation:"h",y:["En retard","À l'heure"],x:[lv.review_score_late_orders,lv.review_score_on_time_orders],
    marker:{color:[C.bad,C.good]},text:[fmt(lv.review_score_late_orders,2)+" / 5",fmt(lv.review_score_on_time_orders,2)+" / 5"],textposition:"outside",cliponaxis:false,
    hovertemplate:"%{y} : %{x:.2f} / 5<extra></extra>"}],
    L({margin:{t:10,l:80,r:50,b:36},xaxis:{range:[0,5],dtick:1,title:"Note moyenne / 5"},bargap:.45}),CFG);

  // ---- États
  const st = A.states.filter(s=>s.orders>=100), al = new Set(A.alert_states);
  const bigSt = new Set(st.slice().sort((a,b)=>b.orders-a.orders).slice(0,3).map(s=>s.state));
  const avgR = h.average_review_score, natL = h.late_delivery_rate_pct;
  Plotly.newPlot("a_states",[{x:st.map(s=>s.late_delivery_rate_pct),y:st.map(s=>s.avg_review_score),
    text:st.map(s=>al.has(s.state)||bigSt.has(s.state)?s.state:""),hovertext:st.map(s=>s.state),
    mode:"markers+text",textposition:"top center",textfont:{size:11,color:C.ink},type:"scatter",cliponaxis:false,
    marker:{size:st.map(s=>Math.max(9,Math.sqrt(s.orders)/3)),color:st.map(s=>al.has(s.state)?C.bad:C.blue),opacity:.7,line:{color:"#fff",width:1}},
    customdata:st.map(s=>[s.orders,s.median_delivery_days,s.freight_to_revenue_pct]),
    hovertemplate:"<b>%{hovertext}</b><br>retard %{x:.1f} %<br>note %{y:.2f} / 5<br>%{customdata[0]:,} commandes<br>délai médian %{customdata[1]} j<br>fret %{customdata[2]:.1f} % du CA<extra></extra>"}],
    L({xaxis:{title:"Taux de retard (%)",zeroline:false},yaxis:{title:"Note moyenne / 5"},
      shapes:[{type:"line",xref:"x",yref:"paper",x0:natL,x1:natL,y0:0,y1:1,line:{dash:"dot",color:C.mute,width:1}},
              {type:"line",xref:"paper",yref:"y",x0:0,x1:1,y0:avgR,y1:avgR,line:{dash:"dot",color:C.mute,width:1}}],
      annotations:[{xref:"x",yref:"paper",x:natL,y:1,text:"moyenne nationale",showarrow:false,xanchor:"left",font:{size:10,color:C.mute}}]}),CFG);

  // ---- Catégories : croissance vs baisse
  const c = A.categories || {};
  const cg = [...(c.declining||[]), ...(c.fastest_growing||[]).slice().reverse()];
  const cmax = Math.max(0,...cg.map(x=>x.yoy_growth_pct)), cmin = Math.min(0,...cg.map(x=>x.yoy_growth_pct));
  Plotly.newPlot("a_cat",[{type:"bar",orientation:"h",y:cg.map(x=>nice(x.category)),x:cg.map(x=>x.yoy_growth_pct),
    marker:{color:cg.map(x=>x.yoy_growth_pct<0?C.bad:C.good)},text:cg.map(x=>pct(x.yoy_growth_pct,0)),
    textposition:"outside",cliponaxis:false,hovertemplate:"%{y}<br>%{x:.1f} % sur un an<extra></extra>"}],
    L({margin:{t:10,l:150,r:50,b:40},xaxis:{title:"% vs N-1",range:[cmin-(cmax-cmin)*.18, cmax*1.2]}}),CFG);

  // ---- Catégories les moins bien notées (axe démarrant à 0 + repère moyenne)
  const lc = (c.lowest_rated||[]).slice().reverse();
  $("cap_lowcat").innerHTML = `Note moyenne par catégorie ; la ligne pointillée = moyenne globale (${fmt(avgR,2)} / 5).`;
  Plotly.newPlot("a_lowcat",[{type:"bar",orientation:"h",y:lc.map(x=>nice(x.category)),x:lc.map(x=>x.avg_review_score),marker:{color:C.warn},
    text:lc.map(x=>fmt(x.avg_review_score,2)),textposition:"outside",cliponaxis:false,hovertemplate:"%{y}<br>%{x:.2f} / 5<extra></extra>"}],
    L({margin:{t:10,l:150,r:40,b:40},xaxis:{range:[0,5],dtick:1,title:"Note moyenne / 5"},
      shapes:[{type:"line",xref:"x",yref:"paper",x0:avgR,x1:avgR,y0:0,y1:1,line:{dash:"dot",color:C.ink,width:1}}]}),CFG);

  // ---- Événements : barres horizontales triées
  const ev = A.events.slice().sort((a,b)=>a.vs_normal_day_pct-b.vs_normal_day_pct);
  const evCol = e => String(e.type).includes("commercial") ? C.good : (e.vs_normal_day_pct<0 ? C.bad : C.blue);
  $("a_events").style.height = Math.max(260, ev.length*22+60)+"px";
  Plotly.newPlot("a_events",[{type:"bar",orientation:"h",y:ev.map(e=>`${e.event} (${e.date})`),x:ev.map(e=>e.vs_normal_day_pct),
    marker:{color:ev.map(evCol)},text:ev.map(e=>pct(e.vs_normal_day_pct,0)),textposition:"outside",cliponaxis:false,
    customdata:ev.map(e=>[e.revenue_brl,e.orders]),
    hovertemplate:"<b>%{y}</b><br>%{x:.1f} % vs jour normal<br>%{customdata[0]:,.0f} BRL · %{customdata[1]} commandes<extra></extra>"}],
    L({margin:{t:10,l:230,r:60,b:40},xaxis:{title:"% vs jour normal du mois",zeroline:true},yaxis:{automargin:true}}),CFG);

  // ---- Concentration vendeurs (barre 100 %)
  const cc = (A.sellers||{}).concentration || {};
  const top = cc.top_10pct_sellers_revenue_share_pct || 0;
  $("cap_conc").innerHTML = `<b>10 % des ${fmt(cc.active_sellers)} vendeurs réalisent ${fmt(top,0)} % du CA.</b> Les vendeurs de ${esc(cc.top_seller_state)} pèsent ${fmt(cc.top_seller_state_revenue_share_pct,0)} % du CA.`;
  Plotly.newPlot("a_conc",[
    {type:"bar",orientation:"h",y:["Vendeurs"],x:[top],name:"10 % meilleurs vendeurs",marker:{color:C.blue},text:[`10 % meilleurs : ${fmt(top,0)} %`],textposition:"inside",insidetextanchor:"middle",hovertemplate:"10 % meilleurs vendeurs : %{x:.1f} % du CA<extra></extra>"},
    {type:"bar",orientation:"h",y:["Vendeurs"],x:[100-top],name:"90 % restants",marker:{color:C.grey},text:[`90 % restants : ${fmt(100-top,0)} %`],textposition:"inside",insidetextanchor:"middle",textfont:{color:C.ink},hovertemplate:"90 % restants : %{x:.1f} % du CA<extra></extra>"},
    {type:"bar",orientation:"h",y:["État vendeur"],x:[cc.top_seller_state_revenue_share_pct||0],marker:{color:C.blue},text:[`${esc(cc.top_seller_state)} : ${fmt(cc.top_seller_state_revenue_share_pct,0)} %`],textposition:"inside",insidetextanchor:"middle",hovertemplate:"%{x:.1f} % du CA<extra></extra>"},
    {type:"bar",orientation:"h",y:["État vendeur"],x:[100-(cc.top_seller_state_revenue_share_pct||0)],marker:{color:C.grey},text:["autres États"],textposition:"inside",insidetextanchor:"middle",textfont:{color:C.ink},hovertemplate:"%{x:.1f} % du CA<extra></extra>"}],
    L({barmode:"stack",margin:{t:10,l:90,r:16,b:40},xaxis:{range:[0,100],ticksuffix:" %",title:"Part du CA"},yaxis:{autorange:"reversed"},bargap:.35}),CFG);

  // ---- Jours de la semaine
  const DAYS = {Monday:"Lundi",Tuesday:"Mardi",Wednesday:"Mercredi",Thursday:"Jeudi",Friday:"Vendredi",Saturday:"Samedi",Sunday:"Dimanche"};
  const wk = A.weekday, we = wk.filter(w=>["Saturday","Sunday"].includes(w.weekday)).reduce((t,w)=>t+w.revenue_share_pct,0);
  $("cap_week").innerHTML = `Le week-end ne représente que <b>${fmt(we,0)} %</b> du CA ; l'activité se concentre en semaine.`;
  Plotly.newPlot("a_week",[{type:"bar",x:wk.map(w=>DAYS[w.weekday]||w.weekday),y:wk.map(w=>w.revenue_share_pct),
    marker:{color:wk.map(w=>["Saturday","Sunday"].includes(w.weekday)?C.grey:C.blue)},
    text:wk.map(w=>fmt(w.revenue_share_pct,1)+" %"),textposition:"outside",cliponaxis:false,hovertemplate:"%{x} : %{y:.1f} % du CA<extra></extra>"}],
    L({yaxis:{title:"% du CA",range:[0,Math.max(...wk.map(w=>w.revenue_share_pct))*1.2]}}),CFG);

  // ---- Meilleurs vendeurs
  const ts = (A.sellers||{}).top_sellers || [], nat = h.late_delivery_rate_pct;
  const maxRev = Math.max(1,...ts.map(s=>s.revenue_brl||0));
  const nLate = ts.filter(s=>s.late_delivery_rate_pct>nat).length;
  $("cap_sellers").innerHTML = `Retards en rouge = au-dessus de la moyenne nationale (${fmt(nat,1)} %). <b>${nLate} sur ${ts.length}</b> des meilleurs vendeurs sont concernés.`;
  $("a_sellers").innerHTML = "<tr><th>#</th><th>Vendeur</th><th>État</th><th class='num'>CA (BRL)</th><th class='num'>Commandes</th><th class='num'>Retards</th><th class='num'>Note</th></tr>" +
    ts.map((s,k)=>`<tr><td>${k+1}</td><td>${esc(s.seller_name)}</td><td>${esc(s.state)}</td>
      <td class="num">${fmt(s.revenue_brl)}<div class="bar" style="width:${(s.revenue_brl/maxRev*100).toFixed(0)}%;margin-left:auto"></div></td>
      <td class="num">${fmt(s.orders)}</td>
      <td class="num"><span class="badge ${s.late_delivery_rate_pct>nat?"bad":"ok"}">${fmt(s.late_delivery_rate_pct,1)} %</span></td>
      <td class="num">${fmt(s.avg_review_score,2)}</td></tr>`).join("");

  $("a_watch").innerHTML = A.watchlist.map(w=>`<li>${esc(w)}</li>`).join("");
}

// ---------------------------------------------------------------- Exploration (filtres)
function opt(sel, values, first, label=v=>v){ sel.innerHTML = (first?`<option value="">${first}</option>`:"") + values.map(v=>`<option value="${esc(v)}">${esc(label(v))}</option>`).join(""); }
opt($("from"), months, null, mlabel); opt($("to"), months, null, mlabel);
opt($("state"), [...new Set(D.sales.map(r=>r.st))].sort(), "Tous");
opt($("cat"), [...new Set(D.sales.map(r=>r.c))].sort(), "Toutes", nice);
function resetFilters(){
  $("from").value = months[0]; $("to").value = months[months.length-1];
  $("state").value = ""; $("cat").value = ""; $("cur").value = "BRL"; $("rel").value = "keep";
}
resetFilters();

function render(){
  const cur = $("cur").value, rate = cur==="TND" ? D.brl_to_tnd : 1;
  let a = months.indexOf($("from").value), b = months.indexOf($("to").value);
  if(a > b){ [a,b] = [b,a]; }
  const drop = $("rel").value==="drop";
  const sel = new Set(months.filter((m,i)=>i>=a && i<=b && (!drop || relMap[m])));
  const st = $("state").value, cat = $("cat").value;
  const S = D.sales.filter(r=>sel.has(r.m) && (!st||r.st===st) && (!cat||r.c===cat));
  const O = D.orders.filter(r=>sel.has(r.m) && (!st||r.st===st));
  const sum = (arr,k)=>arr.reduce((t,r)=>t+r[k],0);

  const rev=sum(S,"rev")*rate, mar=sum(S,"mar")*rate, ord=sum(S,"orders");
  const lateN=sum(O,"late_n"), dN=sum(O,"dd_n"), rN=sum(O,"rv_n");
  $("kpis").innerHTML = [
    kpiCard("Chiffre d'affaires", big(rev)+" "+cur, "", "info", fmt(rev,2)+" "+cur),
    kpiCard("Commandes", fmt(ord), cat?"multi-catégories : comptée par catégorie":""),
    kpiCard("Panier moyen", fmt(ord? rev/ord : null,2)+" "+cur, ""),
    kpiCard("Marge (simulée)", big(mar)+" "+cur, `${fmt(rev? mar/rev*100 : null,1)} % du CA · coût simulé`, "warn"),
    kpiCard("Taux de retard", fmt(lateN? sum(O,"late")/lateN*100 : null,1)+" %", cat?"non filtré par catégorie":""),
    kpiCard("Délai moyen", fmt(dN? sum(O,"dd_sum")/dN : null,1)+" jours", cat?"non filtré par catégorie":""),
    kpiCard("Note moyenne", fmt(rN? sum(O,"rv_sum")/rN : null,2)+" / 5", cat?"non filtrée par catégorie":""),
  ].join("");

  const bad = [...sel].filter(m=>!relMap[m]);
  const banner = $("banner");
  if(bad.length){ banner.style.display="block";
    banner.textContent = "Attention : mois à volume très faible inclus (période incomplète, risque de fausses baisses) : "+bad.map(mlabel).join(", ")+". Choisir « Exclure » pour les retirer.";
  } else banner.style.display="none";

  const ms = months.filter(m=>sel.has(m));
  const byM = Object.fromEntries(ms.map(m=>[m,0]));
  S.forEach(r=>byM[r.m]+=r.rev*rate);
  const X = ms.map(mlabel);
  const evm = D.events.filter(e=>ms.includes(e.m) && e.type==="commercial");
  const shapes = evm.map(e=>({type:"line",x0:mlabel(e.m),x1:mlabel(e.m),yref:"paper",y0:0,y1:1,line:{dash:"dot",color:C.warn}}));
  const ann = evm.map(e=>({x:mlabel(e.m),y:1,yref:"paper",text:esc(e.name),showarrow:false,textangle:-90,xanchor:"left",yanchor:"top",font:{size:10,color:C.warn}}));
  Plotly.react("c_rev",[{x:X,y:ms.map(m=>byM[m]),type:"scatter",mode:"lines+markers",line:{color:C.blue,width:2},
    marker:{color:ms.map(m=>relMap[m]?C.blue:C.warn),size:8},
    hovertemplate:"%{x}<br>%{y:,.0f} "+cur+"<extra></extra>"}],
    L({shapes,annotations:ann,yaxis:{title:cur,tickformat:"~s"},xaxis:{type:"category"}}),CFG);

  const agg=(arr,key)=>{const o={};arr.forEach(r=>o[r[key]]=(o[r[key]]||0)+r.rev*rate);return Object.entries(o).sort((x,y)=>y[1]-x[1]).slice(0,10).reverse();};
  const tc=agg(S,"c"), tst=agg(S,"st");
  Plotly.react("c_cat",[{type:"bar",orientation:"h",x:tc.map(x=>x[1]),y:tc.map(x=>nice(x[0])),marker:{color:C.blue},
    hovertemplate:"%{y}<br>%{x:,.0f} "+cur+"<extra></extra>"}],
    L({margin:{t:10,l:170,r:20,b:40},xaxis:{title:cur,tickformat:"~s"}}),CFG);
  Plotly.react("c_state",[{type:"bar",orientation:"h",x:tst.map(x=>x[1]),y:tst.map(x=>x[0]),marker:{color:C.blue},
    hovertemplate:"%{y}<br>%{x:,.0f} "+cur+"<extra></extra>"}],
    L({margin:{t:10,l:50,r:20,b:40},xaxis:{title:cur,tickformat:"~s"}}),CFG);

  const perM=(num,den)=>ms.map(m=>{const rows=O.filter(r=>r.m===m);const d=sum(rows,den);return d? sum(rows,num)/d : null;});
  Plotly.react("c_late",[{x:X,y:perM("late","late_n").map(v=>v==null?null:v*100),type:"scatter",mode:"lines+markers",line:{color:C.bad},
    hovertemplate:"%{x}<br>%{y:.1f} % en retard<extra></extra>"}],
    L({yaxis:{title:"% en retard",rangemode:"tozero"},xaxis:{type:"category"}}),CFG);
  Plotly.react("c_rev_score",[{x:X,y:perM("rv_sum","rv_n"),type:"scatter",mode:"lines+markers",line:{color:C.good},
    hovertemplate:"%{x}<br>%{y:.2f} / 5<extra></extra>"}],
    L({yaxis:{title:"Note / 5",range:[1,5]},xaxis:{type:"category"}}),CFG);
}
["from","to","state","cat","cur","rel"].forEach(id=>$(id).addEventListener("change",render));
$("reset").onclick = () => { resetFilters(); render(); };
renderAnalyst();
</script></body></html>"""


# --------------------------------------------------------------------------
# Point d'entrée
# --------------------------------------------------------------------------
def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)

    sales = _read("fact_sales.csv", parse_dates=["purchase_date"])
    fo = _read("fact_orders.csv", parse_dates=["purchase_date"])
    for frame in (sales, fo):
        frame["is_revenue_order"] = _bool(frame["is_revenue_order"])
    dim_date = _read("dim_date.csv")
    try:
        events = _read("dim_events.csv")
    except Exception:
        events = pd.DataFrame(columns=["name", "start", "type"])

    etl = _read_json(ETL_DECISION)
    profile = _read_json(PROFILE_PATH)
    kpi = _read_json(KPI_PATH)
    analyst = _read_json(ANALYST_PATH)
    cfg = etl.get("config_used", {})
    brl_to_tnd = float(cfg.get("brl_to_tnd", 0.62))

    print("1) Construction des cubes agrégés et des vues d'analyse")
    sales_cube = build_sales_cube(sales)
    orders_cube = build_orders_cube(fo)
    daily = build_daily(fo, dim_date)
    analyst_view = build_analyst_view(analyst, kpi)

    print("2) Contrôles")
    checks = run_checks(sales, fo, sales_cube, orders_cube, kpi, analyst)
    for c in checks:
        print(f"  {'OK ' if c['passed'] else 'ECHEC'} {c['check']} ({c['detail']})")
    # les analyses sont attendues mais leur absence ne bloque pas le dashboard
    all_ok = all(c["passed"] for c in checks if not c["check"].startswith("Analyses"))

    months = (dim_date.groupby("year_month")["is_reliable_month"].first().reset_index()
              .rename(columns={"year_month": "m", "is_reliable_month": "reliable"}))
    months["reliable"] = months["reliable"].astype(str).eq("True")
    ev = [{"m": str(pd.Timestamp(r["start"]).to_period("M")), "name": r["name"], "type": r.get("type", "")}
          for _, r in events.iterrows() if pd.notna(r.get("start"))]

    data = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "quality_score": profile.get("global_score"),
        "brl_to_tnd": brl_to_tnd,
        "kpi_definitions": KPI_DEFINITIONS,
        "assumptions": etl.get("assumptions", []),
        "months": months.to_dict("records"),
        "events": ev,
        "sales": sales_cube.to_dict("records"),
        "orders": orders_cube.to_dict("records"),
        "daily": daily,
        "analyst": analyst_view,
    }

    print("3) Écriture du dashboard")
    payload = json.dumps(data, ensure_ascii=False, default=str).replace("</", "<\\/")   # pas de </script> parasite
    (OUT / "dashboard.html").write_text(HTML.replace("__DATA__", payload), encoding="utf-8")

    charts = ["Croissance sur un an", "CA mensuel (fiabilité)", "CA quotidien + anomalies + événements",
              "Catégories en croissance / en baisse", "États : retard vs note", "Note selon la ponctualité",
              "Catégories les moins bien notées", "Événements vs jour normal", "Concentration des vendeurs",
              "CA par jour de la semaine", "CA mensuel filtrable", "Top catégories", "Top états",
              "Taux de retard mensuel", "Note moyenne mensuelle"]
    decision = {
        "agent": "dashboard_agent", "run_at": datetime.now().isoformat(timespec="seconds"),
        "status": "success" if all_ok else "failed_validation",
        "sources": ["fact_sales.csv", "fact_orders.csv", "dim_date.csv", "dim_events.csv",
                    "kpi_results.json", "analyst_insights.json"],
        "kpis": KPI_DEFINITIONS, "filters": ["période", "état client", "catégorie", "devise", "mois incomplets"],
        "charts": charts,
        "insights_displayed": len(analyst_view.get("insights", [])),
        "analyst_generated_at": analyst_view.get("generated_at"),
        "unreliable_months": months.loc[~months["reliable"], "m"].tolist(),
        "rows": {"sales_cube": len(sales_cube), "orders_cube": len(orders_cube), "daily": len(daily)},
        "checks": checks,
        "assumptions": etl.get("assumptions", []),
        "outputs": ["data/dashboard/dashboard.html", "data/dashboard/dashboard_agent_decision.json"],
        "next_agent": "auditor" if all_ok else "STOP - corriger le dashboard",
    }
    (OUT / "dashboard_agent_decision.json").write_text(
        json.dumps(decision, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print(f"\nTerminé : {len(charts)} graphiques, {decision['insights_displayed']} insights. "
          "Ouvrir data/dashboard/dashboard.html")
    return {"ok": all_ok, "status": decision["status"], "charts": len(charts),
            "insights": decision["insights_displayed"],
            "unreliable_months": decision["unreliable_months"],
            "checks_failed": [c["check"] for c in checks if not c["passed"]],
            "outputs": decision["outputs"]}


if __name__ == "__main__":
    run()
