"""Olist Pilot : l'application des managers Olist (Streamlit).

Lancer :  .venv\\Scripts\\python.exe -m streamlit run manager_app.py --server.port 8502

Différence avec app.py : app.py sert à piloter et contrôler le pipeline (profiling, ETL, audit...).
Olist Pilot s'adresse aux équipes métier : ventes, livraison, vendeurs, scénarios « et si ? », avec un
chatbot (Pilot) affiché à côté du dashboard. Pilot répond en appelant les agents du pipeline :
agent KPI, BI Analyst et XAI (voir manager/agents_bridge.py).
"""
from __future__ import annotations

import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from manager import agents_bridge as B
from manager import assistant as A
from manager import insights as I
from manager import metrics as M

warnings.filterwarnings("ignore", message=".*fixed sampling defaults.*")
load_dotenv(M.ROOT / ".env")

st.set_page_config(page_title="Olist Pilot", page_icon=":material/insights:", layout="wide")

# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------
BLUE, GOOD, BAD, WARN, GREY, INK, MUTE = "#1f4e8c", "#1b8a5a", "#c53b3b", "#c27c0e", "#c3c8d4", "#14213d", "#6b7488"
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html,body,[class*="css"],.stApp{font-family:Inter,system-ui,sans-serif}
.stApp{background:#f5f6f8}
.block-container{padding-top:4.2rem;max-width:1680px}
section[data-testid="stSidebar"]{background:#fff;border-right:1px solid #e3e7ee}
h1,h2,h3{color:#14213d;letter-spacing:-.01em}
div[data-testid="stMetric"]{background:#fff;border-radius:10px}
div[data-testid="stMetricValue"]{font-size:clamp(1.05rem,1.3vw,1.6rem);font-weight:650;color:#14213d}
div[data-testid="stMetricLabel"] p{white-space:normal!important;overflow:visible!important}
div[data-testid="stMetric"]{padding:12px 12px!important}
div[data-testid="stMetricValue"] > div{overflow:visible;text-overflow:clip}
.ph .k{font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;color:#1f4e8c;font-weight:600}
.ph h1{margin:2px 0 4px;font-size:1.6rem}.ph p{margin:0 0 10px;color:#5b6577}
.callout{background:#fff;border:1px solid #e3e7ee;border-left:4px solid #1f4e8c;border-radius:8px;padding:12px 16px;margin:6px 0 12px;line-height:1.55}
.callout.warn{border-left-color:#c27c0e;background:#fffaf0}.callout.bad{border-left-color:#c53b3b;background:#fff6f6}
.small{color:#6b7488;font-size:.85rem}
.brand{font-size:1.25rem;font-weight:700;color:#14213d}.brand span{color:#1f4e8c}
div[data-testid="stColumn"]:has(.st-key-pilot_chat){position:sticky;top:3.6rem;align-self:flex-start}
.st-key-pilot_chat{background:#fff;border:1px solid #e3e7ee;border-radius:14px;padding:14px 14px 8px;
  box-shadow:0 6px 24px rgba(20,33,61,.07)}
.st-key-pilot_chat [data-testid="stChatMessage"]{padding:.35rem .2rem;font-size:.93rem}
.st-key-pilot_chat [data-testid="stChatMessage"] p{margin-bottom:.35rem}
.chat-h{display:flex;align-items:center;gap:10px;margin-bottom:2px}
.chat-h .dot{width:34px;height:34px;border-radius:10px;background:linear-gradient(135deg,#1f4e8c,#3b82a0);color:#fff;
  display:flex;align-items:center;justify-content:center;font-weight:700}
.chat-h b{font-size:1.05rem;color:#14213d}.chat-h span{display:block;font-size:.78rem;color:#6b7488}
</style>""", unsafe_allow_html=True)


def style(fig, h=340, legend=False):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, system-ui", size=12, color=INK),
                      showlegend=legend, separators=", ",
                      legend=dict(orientation="h", y=1.08, x=0, title=None),
                      hoverlabel=dict(bgcolor="white", font_size=12))
    fig.update_xaxes(gridcolor="#eef1f6", linecolor="#dde2ea", title=None)
    fig.update_yaxes(gridcolor="#eef1f6", zerolinecolor="#cfd5e0", title=None)
    return fig


def show(fig, key=None):
    st.plotly_chart(fig, config={"displayModeBar": False}, key=key)


def page_head(kicker, title, sub):
    st.markdown(f'<div class="ph"><div class="k">{kicker}</div><h1>{title}</h1><p>{sub}</p></div>', unsafe_allow_html=True)


def callout(html_text, kind=""):
    st.markdown(f'<div class="callout {kind}">{html_text}</div>', unsafe_allow_html=True)


def ask_assistant(question: str):
    """Envoie une question au chatbot (traitée dans ce même affichage par le panneau de droite)."""
    st.session_state["pending_q"] = question


# --------------------------------------------------------------------------
# Données (chargées une fois) + filtres globaux
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Chargement des données Olist…")
def get_store():
    return M.load_store()


@st.cache_data(show_spinner=False)
def c_summary(sc):
    return M.summary(store, sc)


@st.cache_data(show_spinner=False)
def c_monthly(sc):
    return M.monthly(store, sc)


@st.cache_data(show_spinner=False)
def c_breakdown(sc, dim):
    return M.breakdown(store, sc, dim)


@st.cache_data(show_spinner=False)
def c_scorecard(sc):
    return M.seller_scorecard(store, sc)


@st.cache_resource(show_spinner="Chargement des produits et de la géolocalisation…")
def get_extra():
    return I.load_extra()


@st.cache_data(show_spinner=False)
def c_insight(name, sc, *args):
    """Appel mis en cache d'une analyse de manager/insights.py (les fonctions « produits » et « carte »
    reçoivent en plus les dimensions chargées par get_extra)."""
    fn = getattr(I, name)
    needs_extra = name in {"pareto", "best_sellers", "listing_quality", "weight_vs_freight", "sales_points"}
    return fn(store, get_extra(), sc, *args) if needs_extra else fn(store, sc, *args)


store = get_store()
MONTHS = store.all_months

with st.sidebar:
    st.markdown('<div class="brand">Olist <span>Pilot</span></div>', unsafe_allow_html=True)
    st.caption("Le cockpit des équipes Olist")
    st.markdown("**Périmètre d'analyse**")
    per = st.session_state.get("period")
    if per is not None and not (isinstance(per, (tuple, list)) and len(per) == 2 and all(p in MONTHS for p in per)):
        # état abîmé (données rechargées avec d'autres mois) : on l'efface, le curseur reprend sa valeur par défaut.
        # Ne pas écrire la valeur par défaut dans session_state : le curseur resterait bloqué sur cette période.
        del st.session_state["period"]
    start, end = st.select_slider("Période", options=MONTHS, key="period", format_func=M.month_label,
                                  value=(store.default_start, store.last_full_month))
    states_all = sorted(store.orders["customer_state"].unique())
    cats_all = sorted(store.sales["category"].unique())
    sel_states = st.multiselect("États clients", states_all, key="states", placeholder="Tous les États")
    sel_cats = st.multiselect("Catégories", cats_all, key="cats", format_func=M.nice_cat, placeholder="Toutes les catégories")
    if st.button("Réinitialiser les filtres", width="stretch", icon=":material/restart_alt:"):
        for k in ("period", "states", "cats"):
            st.session_state.pop(k, None)
        st.rerun()
    unrel = [m for m in MONTHS if start <= m <= end and not store.reliable.get(m)]
    if unrel:
        st.caption(":material/info: Mois incomplet(s) inclus, signalés en gris : " + ", ".join(M.month_label(m) for m in unrel))
    st.divider()
    st.toggle("Afficher l'assistant Pilot", value=True, key="show_chat")
    q = store.profile.get("global_score")
    a_ = store.audit or {}
    au = (("conforme" if a_["passed"] else "**non conforme**") + f" ({a_.get('checks_total', '?')} contrôles)") if "passed" in a_ else None
    st.caption(f"Données : {M.month_label(MONTHS[0])} → {M.month_label(MONTHS[-1])}  \n"
               f"Dernier mois complet : **{M.month_label(store.last_full_month, True)}**  \n"
               f"Qualité des données : **{q}/100**" + (f"  \nAudit : {au}" if au else ""))

SCOPE = M.Scope(start, end, tuple(sel_states), tuple(sel_cats))


# --------------------------------------------------------------------------
# PAGE : Performance commerciale
# --------------------------------------------------------------------------
def page_performance():
    page_head("Performance commerciale", "Ventes", f"Périmètre : {SCOPE.label()}")
    s = c_summary(SCOPE)
    comp = M.comparable_months(store, SCOPE)
    if comp:
        a, b = M.Scope(comp[0], comp[-1], SCOPE.states, SCOPE.cats), M.Scope(comp[0], comp[-1], SCOPE.states, SCOPE.cats).shifted(-12)
        sa, sb = c_summary(a), c_summary(b)
        d = lambda k: M.fmt_pct(M.growth(sa[k], sb[k]), 1, True)
        desc = "vs N-1*"
    else:
        d, desc = (lambda k: None), None
    c = st.columns(4)
    c[0].metric("Chiffre d'affaires", M.fmt_brl(s["chiffre_affaires_brl"]), d("chiffre_affaires_brl"), border=True)
    c[1].metric("Commandes", M.fmt_num(s["commandes"]), d("commandes"), border=True)
    c[2].metric("Panier moyen", M.fmt_brl(s["panier_moyen_brl"], False), d("panier_moyen_brl"), border=True)
    c[3].metric("Clients uniques", M.fmt_num(s["clients_uniques"]), d("clients_uniques"), border=True)
    if comp:
        st.caption(f"Évolutions sur un an, calculées sur les mois comparables : {M.growth_label(store, SCOPE)}. "
                   f"Fret payé par les clients : {M.fmt_pct(s['fret_pct_ca'])} du CA.")

    st.subheader("Chiffre d'affaires mois par mois")
    mo = c_monthly(SCOPE)
    mo["moy3"] = mo["chiffre_affaires"].where(mo["fiable"]).rolling(3, min_periods=2).mean()
    fig = go.Figure()
    fig.add_bar(x=mo["mois"], y=mo["chiffre_affaires"], marker_color=[BLUE if f else GREY for f in mo["fiable"]], name="CA du mois",
                hovertemplate="%{x}<br>%{y:,.0f} BRL<extra></extra>")
    fig.add_scatter(x=mo["mois"], y=mo["moy3"], mode="lines", line=dict(color=WARN, width=2.5), name="Moyenne 3 mois",
                    hovertemplate="moyenne 3 mois : %{y:,.0f} BRL<extra></extra>")
    bf = mo[mo["m"].str.endswith("-11")]
    for _, r in bf.iterrows():
        fig.add_annotation(x=r["mois"], y=r["chiffre_affaires"], text="Black Friday", showarrow=True, arrowhead=0, ay=-28, font=dict(size=11))
    fig.update_yaxes(tickformat="~s", title="BRL")
    show(style(fig, 330, legend=True))
    st.caption("Barres grises : mois incomplets (peu de commandes), à ne pas interpréter comme une baisse.")

    st.subheader("Catégories : qui porte la croissance ?")
    cat = c_breakdown(SCOPE, "categorie")
    cat = cat[cat["commandes"] >= 30].copy()
    cat["nom"] = cat["categorie"].map(M.nice_cat)
    l, r = st.columns([1.25, 1], gap="large")
    with l:
        top = cat.head(20).copy()
        top["croissance"] = top["croissance_1an_pct"].clip(-60, 300)
        fig = px.treemap(top, path=["nom"], values="chiffre_affaires", color="croissance",
                         color_continuous_scale=[[0, BAD], [0.17, "#f3d9d9"], [0.2, "#f4f4f4"], [1, GOOD]],
                         range_color=(-60, 300), custom_data=["croissance_1an_pct", "part_ca_pct"])
        fig.update_traces(hovertemplate="<b>%{label}</b><br>%{value:,.0f} BRL (%{customdata[1]:.1f} % du CA)<br>"
                                        "croissance sur un an : %{customdata[0]:+.0f} %<extra></extra>",
                          texttemplate="<b>%{label}</b><br>%{customdata[1]:.1f} %", marker_line_color="white", root_color="rgba(0,0,0,0)")
        fig.update_layout(coloraxis_colorbar=dict(title="Croiss. %", thickness=10))
        show(style(fig, 420))
        st.caption("Taille = poids dans le CA ; couleur = croissance sur un an (rouge = recul, vert = forte hausse). 20 premières catégories.")
    with r:
        st.dataframe(cat[["nom", "chiffre_affaires", "part_ca_pct", "croissance_1an_pct", "note_moyenne"]], hide_index=True, height=420,
                     column_config={
                         "nom": st.column_config.TextColumn("Catégorie"),
                         "chiffre_affaires": st.column_config.NumberColumn("CA (BRL)", format="compact"),
                         "part_ca_pct": st.column_config.ProgressColumn("Part du CA", format="%.1f %%", min_value=0,
                                                                        max_value=float(cat["part_ca_pct"].max())),
                         "croissance_1an_pct": st.column_config.NumberColumn("Croiss. 1 an", format="%+.0f %%"),
                         "note_moyenne": st.column_config.NumberColumn("Note", format="%.2f")})

    with st.expander("Effet des événements commerciaux et jours fériés"):
        ev = pd.DataFrame(M.events_impact(store))
        if len(ev):
            ev = ev.sort_values("vs_normal_day_pct")
            ev["label"] = ev["event"] + " (" + ev["date"] + ")"
            fig = px.bar(ev, x="vs_normal_day_pct", y="label", orientation="h",
                         color=np.where(ev["type"].str.contains("commercial"), "commercial",
                                        np.where(ev["vs_normal_day_pct"] < 0, "férié, baisse", "férié, hausse")),
                         color_discrete_map={"commercial": GOOD, "férié, baisse": BAD, "férié, hausse": BLUE})
            fig.update_traces(hovertemplate="%{y}<br>%{x:+.0f} % vs jour normal<extra></extra>")
            fig.update_xaxes(title="Écart de CA vs un jour normal du mois (%)")
            show(style(fig, max(300, 22 * len(ev)), legend=True))
            callout("<b>À retenir :</b> le Black Friday multiplie le CA du jour ; les jours fériés font plutôt baisser l'activité. "
                    "Préparer stocks, vendeurs et transporteurs plusieurs semaines avant les grands événements.")


# --------------------------------------------------------------------------
# PAGE : Clients et livraison
# --------------------------------------------------------------------------
def page_delivery():
    page_head("Expérience client", "Livraison et satisfaction", f"Périmètre : {SCOPE.label()}")
    s = c_summary(SCOPE)
    c = st.columns(5)
    c[0].metric("Livraisons en retard", M.fmt_pct(s["taux_retard_pct"]), border=True)
    c[1].metric("Délai médian", f"{M.fmt_num(s['delai_median_jours'])} jours", border=True)
    c[2].metric("Note moyenne", f"{M.fmt_num(s['note_moyenne'], 2)} / 5", border=True)
    c[3].metric("Avis à 1-2 étoiles", M.fmt_pct(s["avis_negatifs_pct"]), border=True)
    c[4].metric("Clients récurrents", M.fmt_pct(s["clients_recurrents_pct"]), border=True,
                help="Part des clients uniques ayant passé plus d'une commande sur la période.")

    db = M.delay_buckets(store, SCOPE)
    if len(db) and db["commandes"].sum():
        on = db.iloc[:3]
        on_note = (on["note_moyenne"] * on["commandes"]).sum() / on["commandes"].sum()
        worst = db.iloc[4]
        callout(f"<b>Le retard est le premier ennemi de la note.</b> Livrée à temps, une commande obtient en moyenne "
                f"<b>{M.fmt_num(on_note, 2)} / 5</b> ; avec 4 à 7 jours de retard elle tombe à <b>{M.fmt_num(worst['note_moyenne'], 2)} / 5</b> "
                f"et {M.fmt_pct(worst['avis_negatifs'], 0)} des clients laissent un avis négatif. "
                "<span class='small'>(Association observée, pas une preuve de causalité.)</span>", "bad")
    l, r = st.columns(2, gap="large")
    with l:
        st.subheader("Note selon l'écart à la date promise")
        colors = [GOOD, GOOD, "#7cc3a0", "#e9a23b", BAD, BAD, "#8f2323"]
        fig = go.Figure()
        fig.add_bar(x=db["tranche"].astype(str), y=db["note_moyenne"], marker_color=colors, customdata=db[["commandes", "avis_negatifs"]],
                    text=db["note_moyenne"].map(lambda v: M.fmt_num(v, 2)), textposition="outside", cliponaxis=False,
                    hovertemplate="%{x}<br>note %{y:.2f} / 5<br>%{customdata[0]:,} commandes<br>%{customdata[1]:.0f} % d'avis négatifs<extra></extra>")
        fig.update_yaxes(range=[0, 5.2], title="Note / 5")
        show(style(fig, 340))
    with r:
        st.subheader("Où sont les retards ?")
        stt = c_breakdown(M.Scope(SCOPE.start, SCOPE.end, (), SCOPE.cats), "etat")
        stt = stt.dropna(subset=["lat"])
        fig = px.scatter_geo(stt, lat="lat", lon="lng", size="commandes", color="taux_retard", hover_name="etat", text="etat",
                             color_continuous_scale=[[0, GOOD], [0.35, "#f2d16b"], [1, BAD]], size_max=55,
                             custom_data=["commandes", "taux_retard", "note_moyenne", "delai_median"])
        fig.update_traces(hovertemplate="<b>%{hovertext}</b><br>%{customdata[0]:,} commandes<br>retard %{customdata[1]:.1f} %<br>"
                                        "note %{customdata[2]:.2f}<br>délai médian %{customdata[3]:.0f} j<extra></extra>",
                          textfont=dict(size=9, color=INK), marker_line_color="white")
        fig.update_geos(projection_type="mercator", lataxis_range=[-34.5, 5.5], lonaxis_range=[-74.5, -33.5],
                        showcountries=True, countrycolor="#c9d1de", showland=True, landcolor="#eef1f6",
                        showocean=False, bgcolor="rgba(0,0,0,0)", showframe=False, showcoastlines=True, coastlinecolor="#c9d1de")
        fig.update_layout(coloraxis_colorbar=dict(title="Retard %", thickness=10, len=.8))
        show(style(fig, 420))
        st.caption("Taille = nombre de commandes ; couleur = taux de retard (filtre État ignoré ici).")

    st.subheader("Évolution mensuelle")
    mo = c_monthly(SCOPE)
    mo = mo[mo["fiable"]]
    fig = go.Figure()
    fig.add_scatter(x=mo["mois"], y=mo["taux_retard"], name="Taux de retard (%)", line=dict(color=BAD, width=2.5), mode="lines+markers",
                    hovertemplate="%{x}<br>retard %{y:.1f} %<extra></extra>")
    fig.add_scatter(x=mo["mois"], y=mo["note_moyenne"], name="Note moyenne", yaxis="y2", line=dict(color=BLUE, width=2.5), mode="lines+markers",
                    hovertemplate="%{x}<br>note %{y:.2f}<extra></extra>")
    fig.update_layout(yaxis=dict(title="Retard (%)", rangemode="tozero"),
                      yaxis2=dict(title="Note / 5", overlaying="y", side="right", range=[3, 5], dtick=0.5, tickformat=".1f", showgrid=False))
    show(style(fig, 320, legend=True))
    st.caption("À lire ensemble : les mois où le retard (rouge) grimpe sont aussi ceux où la note (bleu) décroche.")

    with st.expander("Détail par État"):
        t = c_breakdown(SCOPE, "etat")
        st.dataframe(t[["etat", "commandes", "chiffre_affaires", "taux_retard", "delai_median", "note_moyenne", "fret_pct_ca"]],
                     hide_index=True, column_config={
                         "etat": "État", "commandes": st.column_config.NumberColumn("Commandes", format="localized"),
                         "chiffre_affaires": st.column_config.NumberColumn("CA (BRL)", format="compact"),
                         "taux_retard": st.column_config.ProgressColumn("Retard", format="%.1f %%", min_value=0, max_value=25),
                         "delai_median": st.column_config.NumberColumn("Délai médian (j)", format="%.0f"),
                         "note_moyenne": st.column_config.NumberColumn("Note", format="%.2f"),
                         "fret_pct_ca": st.column_config.NumberColumn("Fret / CA", format="%.1f %%")})


# --------------------------------------------------------------------------
# PAGE : Vendeurs
# --------------------------------------------------------------------------
def page_sellers():
    page_head("Partenaires", "Vendeurs", f"Vendeurs ayant au moins 30 commandes · {SCOPE.label()}")
    sc = c_scorecard(SCOPE)
    nat = c_summary(SCOPE)
    tot = c_breakdown(SCOPE, "vendeur")
    n10 = max(1, int(len(tot) * 0.1))
    share10 = tot.head(n10)["chiffre_affaires"].sum() / tot["chiffre_affaires"].sum() * 100 if len(tot) else 0
    risk = sc[sc["statut"] == "À risque"]
    c = st.columns(4)
    c[0].metric("Vendeurs suivis", M.fmt_num(len(sc)), border=True, help=f"{M.fmt_num(len(tot))} vendeurs actifs au total")
    c[1].metric("À risque", M.fmt_num(len(risk)), border=True, help="Au moins 2 alertes : retards élevés, note faible, CA en baisse")
    c[2].metric("À surveiller", M.fmt_num((sc["statut"] == "À surveiller").sum()), border=True)
    c[3].metric("Poids des 10 % meilleurs", M.fmt_pct(share10, 0), border=True, help="Part du CA réalisée par les 10 % de vendeurs au plus fort CA")
    callout(f"<b>Comment lire :</b> un vendeur est en alerte si son taux de retard dépasse "
            f"{M.fmt_pct(max(nat['taux_retard_pct'] * 1.5, nat['taux_retard_pct'] + 3))} (moyenne {M.fmt_pct(nat['taux_retard_pct'])}), "
            f"si sa note est inférieure à {M.fmt_num(nat['note_moyenne'] - 0.3, 2)} (moyenne {M.fmt_num(nat['note_moyenne'], 2)}) "
            "ou si son CA recule de plus de 20 % sur un an. <b>Cliquez sur une ligne</b> pour ouvrir la fiche du vendeur.")
    f1, f2 = st.columns([1, 1.4])
    statut = f1.segmented_control("Statut", ["Tous", "À risque", "À surveiller", "OK"], default="Tous", key="sel_statut")
    qtxt = f2.text_input("Rechercher un vendeur", placeholder="Nom du vendeur…", key="sel_q")
    view = sc.copy()
    if statut and statut != "Tous":
        view = view[view["statut"] == statut]
    if qtxt:
        view = view[view["vendeur"].str.contains(qtxt, case=False, na=False)]
    view = view[["rang_ca", "vendeur", "seller_state", "statut", "alertes", "chiffre_affaires", "commandes", "taux_retard", "note_moyenne",
                 "croissance_1an_pct", "categorie_principale"]].reset_index(drop=True)
    ev = st.dataframe(view, hide_index=True, on_select="rerun", selection_mode="single-row", height=380, key="sel_tab",
                      column_config={
                          "rang_ca": st.column_config.NumberColumn("#", width="small"),
                          "vendeur": "Vendeur", "seller_state": st.column_config.TextColumn("État", width="small"),
                          "statut": "Statut", "alertes": st.column_config.ListColumn("Alertes"),
                          "chiffre_affaires": st.column_config.NumberColumn("CA (BRL)", format="compact"),
                          "commandes": st.column_config.NumberColumn("Cmd", format="localized"),
                          "taux_retard": st.column_config.NumberColumn("Retard", format="%.1f %%"),
                          "note_moyenne": st.column_config.NumberColumn("Note", format="%.2f"),
                          "croissance_1an_pct": st.column_config.NumberColumn("Croiss. 1 an", format="%+.0f %%"),
                          "categorie_principale": "Catégorie principale"})
    rows = ev.selection.rows if ev and hasattr(ev, "selection") else []
    if not rows:
        st.caption("Sélectionnez un vendeur dans le tableau pour afficher sa fiche.")
        return
    name = view.iloc[rows[0]]["vendeur"]
    p = M.seller_profile(store, SCOPE, name)
    st.markdown(f"### Fiche vendeur · {name} ({p['etat']})")
    c = st.columns(4)
    c[0].metric("CA", M.fmt_brl(p["chiffre_affaires_brl"]), border=True)
    c[1].metric("Commandes", M.fmt_num(p["commandes"]), border=True)
    dl = (p["taux_retard_pct"] or 0) - (nat["taux_retard_pct"] or 0)
    c[2].metric("Taux de retard", M.fmt_pct(p["taux_retard_pct"]), f"{dl:+.1f} pt vs moyenne".replace(".", ","), delta_color="inverse", border=True)
    dr = (p["note_moyenne"] or 0) - (nat["note_moyenne"] or 0)
    c[3].metric("Note moyenne", f"{M.fmt_num(p['note_moyenne'], 2)} / 5", f"{dr:+.2f} vs moyenne".replace(".", ","), border=True)
    l, r = st.columns([1.5, 1], gap="large")
    with l:
        mon = pd.DataFrame({"mois": [M.month_label(k) for k in p["ca_mensuel"]], "ca": list(p["ca_mensuel"].values())})
        fig = px.area(mon, x="mois", y="ca")
        fig.update_traces(line_color=BLUE, fillcolor="rgba(31,78,140,.15)", hovertemplate="%{x}<br>%{y:,.0f} BRL<extra></extra>")
        fig.update_yaxes(tickformat="~s", title="CA (BRL)")
        show(style(fig, 260))
    with r:
        cats = pd.DataFrame({"categorie": list(p["categories"]), "ca": list(p["categories"].values())})
        fig = px.bar(cats.iloc[::-1], x="ca", y="categorie", orientation="h")
        fig.update_traces(marker_color=BLUE, hovertemplate="%{y}<br>%{x:,.0f} BRL<extra></extra>")
        fig.update_xaxes(tickformat="~s")
        show(style(fig, 260))
    if st.button(f"Préparer un plan d'action pour {name}", icon=":material/forum:", type="primary"):
        ask_assistant(f"Analyse le vendeur « {name} » (retards, note, évolution du CA) et propose un plan d'action concret pour l'équipe partenaires.")


# --------------------------------------------------------------------------
# PAGE : Scénarios
# --------------------------------------------------------------------------
def page_scenarios():
    page_head("Aide à la décision", "Scénarios « et si ? »",
              f"Ordres de grandeur calculés sur la période {SCOPE.label()}. Chaque estimation affiche son hypothèse.")
    b = c_summary(SCOPE)
    t1, t2, t3 = st.tabs([":material/local_shipping: Réduire les retards", ":material/favorite: Fidéliser",
                          ":material/undo: Récupérer les commandes perdues"])
    with t1:
        x = st.slider("Réduction du taux de retard (points)", 0.0, float(round(b["taux_retard_pct"] or 0, 1)),
                      min(3.0, float(b["taux_retard_pct"] or 0)), 0.5, format="%.1f pt")
        r = M.simulate(store, SCOPE, "retard", x)
        c = st.columns(3)
        c[0].metric("Taux de retard", M.fmt_pct(r["taux_retard_cible_pct"]), f"{-x:+.1f} pt".replace(".", ","), delta_color="inverse", border=True)
        c[1].metric("Note moyenne estimée", f"{M.fmt_num(r['note_estimee'], 2)} / 5",
                    f"{r['note_estimee'] - r['note_actuelle']:+.2f}".replace(".", ","), border=True)
        c[2].metric("Avis négatifs estimés", M.fmt_pct(r["avis_negatifs_estimes_pct"]),
                    f"{r['avis_negatifs_estimes_pct'] - r['avis_negatifs_actuels_pct']:+.1f} pt".replace(".", ","), delta_color="inverse", border=True)
        callout(f"<b>{M.fmt_num(r['commandes_qui_ne_seraient_plus_en_retard'])} commandes</b> arriveraient à l'heure au lieu d'être en retard. "
                f"<br><span class='small'>Hypothèse : {r['hypothese']}</span>", "warn")
    with t2:
        cur = float(b["clients_recurrents_pct"] or 0)
        x = st.slider("Taux de clients récurrents visé (%)", round(cur, 1), 15.0, min(15.0, round(cur * 2, 1)), 0.5, format="%.1f %%")
        r = M.simulate(store, SCOPE, "fidelisation", x)
        c = st.columns(3)
        c[0].metric("Clients fidèles en plus", M.fmt_num(r["clients_fideles_supplementaires"]), border=True)
        c[1].metric("CA supplémentaire", M.fmt_brl(r["ca_supplementaire_brl"]), M.fmt_pct(r["ca_supplementaire_pct"], 1, True), border=True)
        c[2].metric("Taux actuel → cible", f"{M.fmt_pct(cur)} → {M.fmt_pct(x)}", border=True)
        callout(f"Avec seulement {M.fmt_pct(cur)} de clients qui reviennent, la fidélisation est le levier de croissance le plus sous-exploité. "
                f"<br><span class='small'>Hypothèse : {r['hypothese']}</span>", "warn")
    with t3:
        x = st.slider("Part des commandes perdues récupérées (%)", 0, 100, 30, 5, format="%d %%")
        r = M.simulate(store, SCOPE, "commandes_perdues", x)
        c = st.columns(3)
        c[0].metric("Commandes perdues", M.fmt_num(r["commandes_perdues"]), help="Commandes annulées ou indisponibles", border=True)
        c[1].metric("Commandes récupérées", M.fmt_num(r["commandes_recuperees"]), border=True)
        c[2].metric("CA récupéré", M.fmt_brl(r["ca_recupere_brl"]), border=True)
        callout(f"<span class='small'>Hypothèse : {r['hypothese']} Les commandes perdues représentent "
                f"{M.fmt_pct(r['taux_commandes_perdues_pct'], 2)} de toutes les commandes.</span>", "warn")
    st.caption("Ces scénarios sont des ordres de grandeur pour prioriser, pas des prévisions. Les liens retard → note sont des associations observées.")


# --------------------------------------------------------------------------
# PAGE : Clients et fidélisation
# --------------------------------------------------------------------------
SEG_COLORS = {"Champions": GOOD, "Fidèles à relancer": "#7cc3a0", "Nouveaux clients": BLUE,
              "Gros paniers sans retour": WARN, "Endormis": "#a9b1c2", "Perdus": BAD}


def page_customers():
    page_head("Clients", "Clients et fidélisation", f"Périmètre : {SCOPE.label()}")
    ov = c_insight("customers_overview", SCOPE)
    c = st.columns(4)
    c[0].metric("Clients uniques", M.fmt_num(ov["clients"]), border=True)
    c[1].metric("Clients récurrents", M.fmt_pct(ov["taux_recurrents_pct"], 2), border=True,
                help="Part des clients ayant passé au moins 2 commandes sur la période.")
    c[2].metric("Part du CA des récurrents", M.fmt_pct(ov["part_ca_clients_recurrents_pct"]), border=True)
    c[3].metric("Délai médian avant la 2e commande", f"{M.fmt_num(ov['delai_median_2e_commande_jours'])} jours", border=True)
    callout(f"<b>Les clients reviennent très peu :</b> seulement {M.fmt_pct(ov['taux_recurrents_pct'], 1)} passent une "
            f"2e commande, pour {M.fmt_pct(ov['part_ca_clients_recurrents_pct'])} du CA. La croissance d'Olist repose presque "
            "entièrement sur l'acquisition : chaque point de fidélisation gagné est un levier majeur "
            "(voir la page Scénarios).", "warn")

    st.subheader("Rétention par cohorte")
    co = c_insight("cohort_retention", SCOPE)
    if co.empty:
        st.info("Aucune cohorte sur ce périmètre.")
    else:
        cols = [k for k in co.columns if k.startswith("M+")]
        z = co.set_index("cohorte")[cols]
        z.index = [f"{M.month_label(m)} ({M.fmt_num(n)})" for m, n in zip(co["cohorte"], co["clients"])]
        fig = px.imshow(z, color_continuous_scale=[[0, "#f4f6fa"], [1, BLUE]], aspect="auto", text_auto=".1f",
                        labels=dict(x="Mois après le 1er achat", y="Cohorte (clients)", color="% revenus"))
        fig.update_traces(hovertemplate="%{y}<br>%{x} : %{z:.2f} % ont recommandé<extra></extra>")
        fig.update_layout(coloraxis_colorbar=dict(title="%", thickness=10))
        show(style(fig, max(320, 24 * len(z))))
        st.caption("Lecture : part cumulée des clients d'une cohorte (mois de leur 1er achat) ayant recommandé dans les "
                   "k mois suivants. Cases vides : pas encore assez de recul. Filtres État et catégorie = ceux de la 1re commande.")

    st.subheader("Segmentation RFM (récence, fréquence, montant)")
    seg, _ = c_insight("rfm_segments", SCOPE)
    if len(seg):
        l, r = st.columns([1, 1.25], gap="large")
        with l:
            fig = px.bar(seg, x="part_clients_pct", y="segment", orientation="h", color="segment",
                         color_discrete_map=SEG_COLORS, custom_data=["clients", "part_ca_pct"])
            fig.update_traces(hovertemplate="<b>%{y}</b><br>%{customdata[0]:,} clients (%{x:.1f} %)<br>"
                                            "%{customdata[1]:.1f} % du CA<extra></extra>")
            fig.update_yaxes(autorange="reversed")
            fig.update_xaxes(title="% des clients", ticksuffix=" %")
            show(style(fig, 300))
        with r:
            st.dataframe(seg[["segment", "clients", "part_ca_pct", "panier_moyen", "recence_moyenne", "action"]],
                         hide_index=True, column_config={
                             "segment": "Segment", "clients": st.column_config.NumberColumn("Clients", format="localized"),
                             "part_ca_pct": st.column_config.ProgressColumn("Part du CA", format="%.1f %%", min_value=0,
                                                                            max_value=100),
                             "panier_moyen": st.column_config.NumberColumn("Dépense moy.", format="%.0f BRL"),
                             "recence_moyenne": st.column_config.NumberColumn("Dernier achat (j)", format="%.0f"),
                             "action": "Action conseillée"})
        st.caption("Règles : Champions = au moins 2 commandes et achat il y a moins de 6 mois ; Fidèles à relancer = "
                   "2 commandes ou plus, plus anciennes ; Nouveaux = 1 commande de moins de 90 jours ; Gros paniers = "
                   "1 commande dans les 20 % de dépenses les plus élevées ; Endormis = 1 commande de 3 à 12 mois ; "
                   "Perdus = plus d'un an. Date de référence : fin de la période.")
        big = seg.set_index("segment")
        if "Gros paniers sans retour" in big.index:
            g = big.loc["Gros paniers sans retour"]
            callout(f"<b>Cible prioritaire :</b> les « gros paniers sans retour » sont {M.fmt_pct(g['part_clients_pct'], 0)} "
                    f"des clients mais {M.fmt_pct(g['part_ca_pct'], 0)} du CA. Les faire revenir une seule fois pèserait plus "
                    "que n'importe quel autre segment.")
        if st.button("Quels segments de clients relancer en priorité ?", icon=":material/forum:"):
            ask_assistant("Quels segments de clients relancer en priorité, et avec quelle action ?")

    st.subheader("Fidélité selon l'État et la première catégorie achetée")
    l, r = st.columns(2, gap="large")
    for col, dim, title in ((l, "etat", "État du client"), (r, "categorie", "Catégorie de la 1re commande")):
        with col:
            d = c_insight("repeat_by", SCOPE, dim).head(12)
            if d.empty:
                st.info("Pas assez de clients (minimum 200 par groupe).")
                continue
            d["nom"] = d[dim].map(M.nice_cat)
            fig = px.bar(d.iloc[::-1], x="taux_recurrents_pct", y="nom", orientation="h", custom_data=["clients"])
            fig.update_traces(marker_color=BLUE, hovertemplate="<b>%{y}</b><br>%{x:.2f} % de récurrents<br>"
                                                               "%{customdata[0]:,} clients<extra></extra>")
            fig.update_xaxes(title=f"% de clients récurrents · {title}", ticksuffix=" %")
            show(style(fig, 360), key=f"rep_{dim}")
    st.caption("Groupes d'au moins 200 clients. Une catégorie qui fidélise mieux est un bon produit d'appel pour la 1re commande.")


# --------------------------------------------------------------------------
# PAGE : Paiements
# --------------------------------------------------------------------------
PAY_COLORS = {"Carte de crédit": BLUE, "Boleto": WARN, "Bon d'achat": GOOD, "Carte de débit": "#3b82a0",
              "Non défini": GREY}


def page_payments():
    page_head("Paiements", "Moyens de paiement", f"Périmètre : {SCOPE.label()}")
    mix = c_insight("payments_mix", SCOPE)
    if mix.empty:
        st.info("Aucune commande sur ce périmètre.")
        return
    by = mix.set_index("main_payment_type")
    cc = by.loc["credit_card"] if "credit_card" in by.index else None
    bo = by.loc["boleto"] if "boleto" in by.index else None
    c = st.columns(4)
    c[0].metric("Payé par carte de crédit", M.fmt_pct(cc["part_commandes_pct"]) if cc is not None else "n/a",
                help="Part des commandes (moyen de paiement principal)", border=True)
    c[1].metric("Payé par boleto", M.fmt_pct(bo["part_commandes_pct"]) if bo is not None else "n/a", border=True,
                help="Boleto bancário : bordereau payé en agence ou en ligne, sans carte bancaire.")
    c[2].metric("Mensualités moyennes (carte)", M.fmt_num(cc["mensualites"], 1) if cc is not None else "n/a", border=True)
    c[3].metric("Panier moyen carte vs boleto",
                f"{M.fmt_brl(cc['panier_moyen'], False)} / {M.fmt_brl(bo['panier_moyen'], False)}"
                if cc is not None and bo is not None else "n/a", border=True)

    l, r = st.columns([1, 1.3], gap="large")
    with l:
        st.subheader("Répartition du CA")
        fig = px.pie(mix, names="moyen", values="ca", hole=.55, color="moyen", color_discrete_map=PAY_COLORS)
        fig.update_traces(textinfo="percent", hovertemplate="<b>%{label}</b><br>%{value:,.0f} BRL (%{percent})<extra></extra>",
                          sort=False)
        show(style(fig, 300, legend=True))
    with r:
        st.subheader("Évolution de la part des commandes")
        pm = c_insight("payments_monthly", SCOPE)
        fig = px.area(pm, x="mois", y="part_pct", color="moyen", color_discrete_map=PAY_COLORS,
                      category_orders={"moyen": list(PAY_COLORS)})
        fig.update_traces(hovertemplate="%{x}<br>%{y:.1f} %<extra></extra>")
        fig.update_yaxes(range=[0, 100], ticksuffix=" %")
        show(style(fig, 300, legend=True))

    st.subheader("Comparaison des moyens de paiement")
    st.dataframe(mix[["moyen", "commandes", "part_ca_pct", "panier_moyen", "mensualites", "note", "avis_negatifs",
                      "commandes_perdues_pct"]], hide_index=True, column_config={
        "moyen": "Moyen", "commandes": st.column_config.NumberColumn("Commandes", format="localized"),
        "part_ca_pct": st.column_config.ProgressColumn("Part du CA", format="%.1f %%", min_value=0, max_value=100),
        "panier_moyen": st.column_config.NumberColumn("Panier moyen", format="%.0f BRL"),
        "mensualites": st.column_config.NumberColumn("Mensualités", format="%.1f"),
        "note": st.column_config.NumberColumn("Note", format="%.2f"),
        "avis_negatifs": st.column_config.NumberColumn("Avis 1-2 ★", format="%.1f %%"),
        "commandes_perdues_pct": st.column_config.NumberColumn("Annulées / indispo.", format="%.2f %%")})
    st.caption("Annulées / indisponibles : part de toutes les commandes du moyen de paiement (avec un filtre catégorie, "
               "seules les commandes du CA sont connues : ce taux devient alors partiel).")

    l, r = st.columns(2, gap="large")
    with l:
        st.subheader("Mensualités et panier (carte de crédit)")
        iv = c_insight("installments_vs_basket", SCOPE)
        fig = go.Figure()
        fig.add_bar(x=iv["tranche"].astype(str), y=iv["panier_moyen"], marker_color=BLUE, customdata=iv[["commandes", "part_pct"]],
                    text=iv["panier_moyen"].map(lambda v: M.fmt_num(v)), textposition="outside", cliponaxis=False,
                    hovertemplate="%{x}<br>panier moyen %{y:,.0f} BRL<br>%{customdata[0]:,} commandes "
                                  "(%{customdata[1]:.1f} %)<extra></extra>")
        fig.update_yaxes(title="Panier moyen (BRL)")
        show(style(fig, 320))
        if len(iv) and iv["panier_moyen"].notna().all():
            callout(f"Payer en plusieurs fois accompagne des paniers bien plus élevés : "
                    f"{M.fmt_brl(iv['panier_moyen'].iloc[0], False)} en 1 fois contre "
                    f"{M.fmt_brl(iv['panier_moyen'].iloc[3], False)} en 7 à 10 fois. Mettre en avant le paiement en "
                    "plusieurs fois sur les produits chers peut soutenir le panier moyen.")
    with r:
        st.subheader("Part du boleto par État")
        bs = c_insight("boleto_by_state", SCOPE).dropna(subset=["lat"])
        fig = px.scatter_geo(bs, lat="lat", lon="lng", size="commandes", color="boleto_pct", hover_name="etat", text="etat",
                             color_continuous_scale=[[0, "#f4f6fa"], [1, WARN]], size_max=45,
                             custom_data=["commandes", "boleto_pct"])
        fig.update_traces(hovertemplate="<b>%{hovertext}</b><br>%{customdata[0]:,} commandes<br>boleto %{customdata[1]:.1f} %"
                                        "<extra></extra>", textfont=dict(size=9, color=INK), marker_line_color="white")
        fig.update_geos(projection_type="mercator", lataxis_range=[-34.5, 5.5], lonaxis_range=[-74.5, -33.5],
                        showcountries=True, countrycolor="#c9d1de", showland=True, landcolor="#eef1f6",
                        showocean=False, bgcolor="rgba(0,0,0,0)", showframe=False, showcoastlines=True, coastlinecolor="#c9d1de")
        fig.update_layout(coloraxis_colorbar=dict(title="Boleto %", thickness=10, len=.8))
        show(style(fig, 360))
        st.caption("États d'au moins 300 commandes (filtre État ignoré ici). Le boleto sert les clients sans carte bancaire : "
                   "là où il pèse lourd, un paiement plus simple peut élargir la clientèle.")


# --------------------------------------------------------------------------
# PAGE : Produits et catalogue
# --------------------------------------------------------------------------
def page_products():
    page_head("Catalogue", "Produits et catalogue", f"Périmètre : {SCOPE.label()}")
    pa = c_insight("pareto", SCOPE)
    if not pa["produits"]:
        st.info("Aucune vente sur ce périmètre.")
        return
    c = st.columns(3)
    c[0].metric("Produits vendus", M.fmt_num(pa["produits"]), border=True)
    c[1].metric("CA des 20 % meilleurs produits", M.fmt_pct(pa["part_ca_top20_pct"]), border=True)
    c[2].metric("Produits pour 80 % du CA", f"{M.fmt_num(pa['produits_pour_80pct'])} ({M.fmt_pct(pa['part_produits_pour_80pct'], 0)})",
                border=True)

    l, r = st.columns([1, 1.2], gap="large")
    with l:
        st.subheader("Concentration du CA (Pareto)")
        cv = pa["curve"]
        fig = go.Figure()
        fig.add_scatter(x=cv["part_produits_pct"], y=cv["part_ca_pct"], mode="lines", line=dict(color=BLUE, width=3),
                        fill="tozeroy", fillcolor="rgba(31,78,140,.12)", name="CA cumulé",
                        hovertemplate="%{x:.1f} % des produits<br>%{y:.1f} % du CA<extra></extra>")
        fig.add_scatter(x=[0, 100], y=[0, 100], mode="lines", line=dict(color=GREY, dash="dot"), name="Répartition égale",
                        hoverinfo="skip")
        fig.add_vline(x=20, line_dash="dash", line_color=WARN)
        fig.add_annotation(x=20, y=pa["part_ca_top20_pct"], text=f"20 % des produits = {M.fmt_pct(pa['part_ca_top20_pct'], 0)} du CA",
                           showarrow=True, arrowhead=0, ax=70, ay=30, font=dict(size=11))
        fig.update_xaxes(title="% des produits (du plus vendu au moins vendu)", ticksuffix=" %", range=[0, 100])
        fig.update_yaxes(title="% du CA cumulé", ticksuffix=" %", range=[0, 101])
        show(style(fig, 340))
    with r:
        st.subheader("Best-sellers : unités vs CA")
        bs = c_insight("best_sellers", SCOPE, 15)
        st.dataframe(bs[["produit", "categorie", "unites", "ca", "rang_ca", "note"]], hide_index=True, height=340,
                     column_config={
                         "produit": "Produit", "categorie": "Catégorie",
                         "unites": st.column_config.NumberColumn("Unités", format="localized"),
                         "ca": st.column_config.NumberColumn("CA (BRL)", format="compact"),
                         "rang_ca": st.column_config.NumberColumn("Rang en CA", format="%d",
                                                                  help="Un best-seller en volume peut être loin en CA (petit prix)."),
                         "note": st.column_config.NumberColumn("Note", format="%.2f")})
        st.caption("Classés par unités vendues. Le rang en CA montre si un produit très vendu rapporte vraiment.")

    st.subheader("Qualité des fiches produits")
    ph, de = c_insight("listing_quality", SCOPE)
    l, r = st.columns(2, gap="large")
    with l:
        fig = go.Figure()
        fig.add_bar(x=ph["tranche"].astype(str), y=ph["ca_par_produit"], marker_color=BLUE, customdata=ph[["produits", "note"]],
                    text=ph["ca_par_produit"].map(lambda v: M.fmt_num(v)), textposition="outside", cliponaxis=False,
                    hovertemplate="%{x}<br>%{y:,.0f} BRL de CA par produit<br>%{customdata[0]:,} produits<br>"
                                  "note %{customdata[1]:.2f}<extra></extra>")
        fig.update_yaxes(title="CA moyen par produit (BRL)")
        show(style(fig, 300))
        st.caption("CA moyen par produit selon le nombre de photos de la fiche.")
    with r:
        if len(de):
            fig = go.Figure()
            fig.add_bar(x=de["tranche"].astype(str), y=de["note"], marker_color="#3b82a0", customdata=de[["produits", "unites_par_produit"]],
                        text=de["note"].map(lambda v: M.fmt_num(v, 2)), textposition="outside", cliponaxis=False,
                        hovertemplate="%{x}<br>note %{y:.2f}<br>%{customdata[0]:,} produits<br>"
                                      "%{customdata[1]:.2f} unités par produit<extra></extra>")
            fig.update_yaxes(title="Note moyenne / 5", range=[3.8, 4.3])
            show(style(fig, 300))
            st.caption("Note moyenne selon la longueur de la description (quartiles). Axe resserré pour lire les écarts.")
    if len(ph) >= 5 and ph["ca_par_produit"].notna().all():
        callout(f"Les fiches avec 6 photos ou plus génèrent en moyenne {M.fmt_brl(ph['ca_par_produit'].iloc[4], False)} de CA par "
                f"produit contre {M.fmt_brl(ph['ca_par_produit'].iloc[0], False)} avec une seule photo. "
                "<span class='small'>Association observée : les produits chers ont aussi souvent plus de photos.</span>")

    st.subheader("Poids et coût du fret")
    wf = c_insight("weight_vs_freight", SCOPE)
    fig = go.Figure()
    fig.add_bar(x=wf["tranche"].astype(str), y=wf["fret_median"], name="Fret médian (BRL)", marker_color=BLUE,
                hovertemplate="%{x}<br>fret médian %{y:.1f} BRL<extra></extra>")
    fig.add_scatter(x=wf["tranche"].astype(str), y=wf["fret_pct_prix"], name="Fret / prix (%)", yaxis="y2",
                    mode="lines+markers", line=dict(color=WARN, width=2.5),
                    hovertemplate="%{x}<br>fret = %{y:.1f} % du prix<extra></extra>")
    fig.update_layout(yaxis=dict(title="Fret médian par article (BRL)"),
                      yaxis2=dict(title="Fret / prix (%)", overlaying="y", side="right", rangemode="tozero", showgrid=False))
    show(style(fig, 320, legend=True))
    st.caption("Le fret augmente avec le poids, mais pèse le plus lourd (en % du prix) sur les petits articles bon marché.")


# --------------------------------------------------------------------------
# PAGE : Carte des ventes
# --------------------------------------------------------------------------
def page_map():
    page_head("Géographie", "Carte des ventes", f"Périmètre : {SCOPE.label()}")
    side = st.segmented_control("Afficher", ["Clients", "Vendeurs"], default="Clients", key="map_side") or "Clients"
    pts = c_insight("sales_points", SCOPE, side.lower())
    total = c_summary(SCOPE)["chiffre_affaires_brl"]
    if pts.empty:
        st.info("Aucune vente géolocalisée sur ce périmètre.")
        return
    l, r = st.columns([1.6, 1], gap="large")
    with l:
        fig = px.density_map(pts, lat="lat", lon="lng", z="ca", radius=9, zoom=3.2, center=dict(lat=-15.5, lon=-51),
                             map_style="carto-positron", color_continuous_scale=[[0, "#c8d7ee"], [.5, BLUE], [1, "#0d2b55"]],
                             hover_name="city", custom_data=["state", "ca", "n"])
        fig.update_traces(hovertemplate="<b>%{hovertext}</b> (%{customdata[0]})<br>%{customdata[1]:,.0f} BRL<br>"
                                        f"%{{customdata[2]:,}} {side.lower()}<extra></extra>")
        fig.update_layout(coloraxis_showscale=False)
        show(style(fig, 520))
        st.caption(f"Densité du CA par code postal des {side.lower()} · {M.fmt_pct(pts['ca'].sum() / total * 100 if total else None)} "
                   "du CA géolocalisé (codes postaux sans coordonnées exclus).")
    with r:
        st.markdown(f"**Villes qui pèsent le plus ({side.lower()})**")
        cr = I.city_ranking(pts, 12)
        cr["ville"] = cr["city"].str.title() + " (" + cr["state"] + ")"
        st.dataframe(cr[["ville", "ca", "part_ca_pct", "n"]], hide_index=True, height=480, column_config={
            "ville": "Ville", "ca": st.column_config.NumberColumn("CA (BRL)", format="compact"),
            "part_ca_pct": st.column_config.ProgressColumn("Part", format="%.1f %%", min_value=0,
                                                           max_value=float(cr["part_ca_pct"].max())),
            "n": st.column_config.NumberColumn(side, format="localized")})

    st.subheader("Flux vendeur → client entre États")
    sv = c_insight("same_vs_other_state", SCOPE)
    if len(sv) == 2:
        a, b = sv.set_index("flux").loc["Même État"], sv.set_index("flux").loc["Autre État"]
        c = st.columns(4)
        c[0].metric("Commandes hors de l'État du vendeur", M.fmt_pct(b["commandes"] / sv["commandes"].sum() * 100, 0), border=True)
        c[1].metric("Délai médian : même État / autre", f"{M.fmt_num(a['delai_median'])} j / {M.fmt_num(b['delai_median'])} j", border=True)
        c[2].metric("Retard : même État / autre", f"{M.fmt_pct(a['taux_retard'])} / {M.fmt_pct(b['taux_retard'])}", border=True)
        c[3].metric("Note : même État / autre", f"{M.fmt_num(a['note'], 2)} / {M.fmt_num(b['note'], 2)}", border=True)
        callout(f"<b>La distance coûte cher :</b> livrée dans un autre État, une commande met {M.fmt_num(b['delai_median'])} jours "
                f"au lieu de {M.fmt_num(a['delai_median'])}, arrive en retard {M.fmt_pct(b['taux_retard'])} du temps "
                f"(contre {M.fmt_pct(a['taux_retard'])}) et coûte {M.fmt_brl(b['fret_moyen'], False)} de fret "
                f"(contre {M.fmt_brl(a['fret_moyen'], False)}). Recruter des vendeurs hors de São Paulo, près des grands "
                "bassins de clients, réduirait délais et retards.", "bad")
    fl = c_insight("state_flows", SCOPE)
    if len(fl):
        top_s = fl.groupby("seller_state")["commandes"].sum().nlargest(8).index
        top_c = fl.groupby("customer_state")["commandes"].sum().nlargest(12).index
        m = fl[fl["seller_state"].isin(top_s) & fl["customer_state"].isin(top_c)]
        z = m.pivot(index="seller_state", columns="customer_state", values="taux_retard").reindex(index=top_s, columns=top_c)
        n = m.pivot(index="seller_state", columns="customer_state", values="commandes").reindex(index=top_s, columns=top_c)
        fig = go.Figure(go.Heatmap(z=z.values, x=list(z.columns), y=list(z.index), customdata=n.values,
                                   colorscale=[[0, GOOD], [.35, "#f2d16b"], [1, BAD]], zmin=0, zmax=25,
                                   text=[["" if np.isnan(v) else f"{v:.0f}" for v in row] for row in z.values],
                                   texttemplate="%{text}", colorbar=dict(title="Retard %", thickness=10),
                                   hovertemplate="vendeur %{y} → client %{x}<br>retard %{z:.1f} %<br>"
                                                 "%{customdata:,} commandes<extra></extra>"))
        fig.update_xaxes(title="État du client", side="top")
        fig.update_yaxes(title="État du vendeur", autorange="reversed")
        show(style(fig, 380))
        st.caption("Taux de retard (%) par flux, pour les principaux États vendeurs et clients (flux d'au moins 50 commandes). "
                   "Filtre État client appliqué.")
        with st.expander("Flux les plus problématiques (au moins 200 commandes)"):
            w = fl[(fl["commandes"] >= 200) & ~fl["meme_etat"]].sort_values("taux_retard", ascending=False).head(15)
            w = w.assign(flux=w["seller_state"] + " → " + w["customer_state"])
            st.dataframe(w[["flux", "commandes", "delai_median", "taux_retard", "fret_moyen", "note"]], hide_index=True,
                         column_config={
                             "flux": "Vendeur → client", "commandes": st.column_config.NumberColumn("Commandes", format="localized"),
                             "delai_median": st.column_config.NumberColumn("Délai médian (j)", format="%.0f"),
                             "taux_retard": st.column_config.ProgressColumn("Retard", format="%.1f %%", min_value=0, max_value=30),
                             "fret_moyen": st.column_config.NumberColumn("Fret moyen", format="%.1f BRL"),
                             "note": st.column_config.NumberColumn("Note", format="%.2f")})


# --------------------------------------------------------------------------
# Chatbot Pilot (panneau à droite du dashboard)
# --------------------------------------------------------------------------
AGENT_BADGE = {"Agent KPI": ":blue-badge[Agent KPI]", "BI Analyst": ":violet-badge[BI Analyst]",
               "XAI": ":orange-badge[XAI]", "Calculs Pilot": ":gray-badge[Calculs]",
               "Garde-fou": ":red-badge[Hors périmètre]"}
PAGE_SUGGESTIONS = {
    "Ventes": ["Pourquoi le CA a-t-il baissé le dernier mois ?", "Quelles catégories reculent ?",
               "Compare août 2018 à août 2017"],
    "Livraison & clients": ["Pourquoi les clients donnent-ils de mauvaises notes ?",
                            "Quel risque de mauvaise note avec 5 jours de retard ?",
                            "Où les retards sont-ils les plus graves ?"],
    "Vendeurs": ["Quels vendeurs dois-je appeler en priorité ?", "Quelles alertes sur les vendeurs ?"],
    "Scénarios": ["Et si on réduisait les retards de 3 points ?", "Que vaut une fidélisation à 6 % ?"],
    "Clients": ["Quels segments de clients relancer en priorité ?", "Combien de clients reviennent ?"],
    "Paiements": ["Quel moyen de paiement a le panier moyen le plus élevé ?", "Le boleto augmente-t-il les annulations ?"],
    "Produits": ["Quels produits se vendent le plus ?", "Quelles catégories reculent ?"],
    "Carte": ["Où les retards sont-ils les plus graves ?", "Pourquoi les clients donnent-ils de mauvaises notes ?"],
}
COMMON_SUGGESTIONS = ["Quelles sont les alertes sur ce périmètre ?", "Que faire en priorité ?"]


@st.cache_data(show_spinner=False)
def analyst_overview(sc):
    """Signaux du BI Analyst sur le périmètre affiché (pour le message d'accueil)."""
    try:
        r = B.analyst_signals((sc.start, sc.end), sc.states, sc.cats)
        return [s for s in r["signaux"] if s["gravite"] == "alerte"], len(r["signaux"])
    except Exception:  # noqa: BLE001
        return [], 0


def render_artifacts(arts, key):
    for k, a in enumerate(arts):
        ch = a.chart
        if ch and ch["y"] in a.df and ch["x"] in a.df and len(a.df) > 1:
            d = a.df.copy()
            if ch["type"] == "line":
                fig = px.line(d, x=ch["x"], y=ch["y"], markers=True)
                fig.update_traces(line_color=BLUE, line_width=2.2)
            elif ch["type"] == "barh":
                d = d.iloc[::-1].copy()
                d[ch["x"]] = d[ch["x"]].astype(str).map(lambda t: t if len(t) <= 22 else t[:21] + "…")
                fig = px.bar(d, x=ch["y"], y=ch["x"], orientation="h")
                big = pd.to_numeric(d[ch["y"]], errors="coerce").abs().max() >= 1e4
                fig.update_xaxes(nticks=4, tickformat="~s" if big else None, tickangle=0)
                fig.update_traces(marker_color=[BAD if (isinstance(v, (int, float)) and v < 0) else BLUE for v in d[ch["y"]]])
            else:
                fig = px.bar(d, x=ch["x"], y=ch["y"])
                fig.update_traces(marker_color=BLUE)
            st.caption(f"**{a.title}**")
            fig.update_layout(font_size=11)
            fig.update_xaxes(tickangle=0, nticks=5)
            show(style(fig, 200 if ch["type"] != "barh" else max(170, 24 * len(d))), key=f"{key}_{k}")
        with st.expander(f"Données : {a.title}"):
            st.dataframe(a.df, hide_index=True)


def render_answer(msg, key):
    agents = list(dict.fromkeys(t.get("agent", "Calculs Pilot") for t in msg.get("trace", [])))
    if agents:
        st.markdown("Sources : " + " ".join(AGENT_BADGE.get(a, a) for a in agents))
    if msg.get("warning"):
        st.caption(f":material/info: {msg['warning']}")
    st.markdown(msg["content"])
    render_artifacts(msg.get("artifacts", []), key)
    if msg.get("trace") and agents == ["Garde-fou"]:
        p = msg["trace"][0]["parametres"]
        with st.expander("Contrôle du périmètre"):
            st.markdown(f"- Motif : {p.get('motif')} (décidé par : {p.get('decide_par')})")
            st.caption("Aucun agent n'a été appelé : la question ne porte pas sur l'activité d'Olist.")
    elif msg.get("trace"):
        with st.expander(f"Comment j'ai calculé ({len(msg['trace'])})"):
            for t in msg["trace"]:
                params = " · ".join(f"{k} = {v}" for k, v in t["parametres"].items()) or "périmètre du dashboard"
                st.markdown(f"- {AGENT_BADGE.get(t.get('agent'), '')} `{t['outil']}` : {params}")
            st.caption("Les chiffres viennent des agents du pipeline (KPI, BI Analyst, XAI) et de calculs pandas ; "
                       "le modèle de langage ne fait que rédiger.")


def chat_panel(page_title: str):
    llm_ok = bool(os.environ.get("LLM_MODEL"))
    chat = st.session_state.setdefault("chat", [])
    with st.container(key="pilot_chat"):
        h1, h2 = st.columns([4, 1], vertical_alignment="center")
        h1.markdown('<div class="chat-h"><div class="dot">P</div><div><b>Pilot</b>'
                    '<span>Assistant BI · agents KPI, BI Analyst et XAI</span></div></div>', unsafe_allow_html=True)
        with h2.popover("", icon=":material/settings:", help="Options de l'assistant"):
            use_llm = st.toggle("Réponses rédigées par IA", value=llm_ok, disabled=not llm_ok, key="use_llm",
                                help="Désactivé : réponses par modèles de texte, sans appel au LLM.")
            st.caption(f"Modèle : {os.environ.get('LLM_MODEL', 'aucun').split(':')[-1]}" if llm_ok
                       else "Aucun LLM configuré (.env) : mode règles.")
            if chat:
                md = "\n\n".join(("**Vous :** " if m["role"] == "user" else "**Pilot :** ") + m["content"] for m in chat)
                st.download_button("Exporter la conversation", md.replace(" ", " "), file_name="conversation_pilot.md",
                                   icon=":material/download:", width="stretch")
                if st.button("Nouvelle conversation", icon=":material/add_comment:", width="stretch"):
                    st.session_state["chat"] = []
                    st.rerun()
        st.caption(f":material/filter_alt: Périmètre : {SCOPE.label()}")

        box = st.container(height=540, border=False, autoscroll=True)
        with box:
            if not chat:
                alerts, n_sig = analyst_overview(SCOPE)
                with st.chat_message("assistant", avatar=":material/insights:"):
                    txt = ("Bonjour ! Je suis **Pilot**. Je réponds avec le travail des agents du pipeline : "
                           "**KPI** (chiffres officiels), **BI Analyst** (alertes et recommandations) et "
                           "**XAI** (pourquoi un chiffre bouge, risque de mauvaise note).")
                    if n_sig:
                        txt += (f"\n\nSur ce périmètre, le BI Analyst relève **{n_sig} signaux** dont **{len(alerts)} alertes** :\n"
                                + "\n".join(f"- {a['titre']}" for a in alerts[:3]))
                    st.markdown(txt)
            for k, msg in enumerate(chat):
                with st.chat_message(msg["role"], avatar=":material/person:" if msg["role"] == "user" else ":material/insights:"):
                    if msg["role"] == "user":
                        st.markdown(msg["content"])
                    else:
                        render_answer(msg, f"h{k}")

        sugg = PAGE_SUGGESTIONS.get(page_title, []) + COMMON_SUGGESTIONS
        pick = st.pills("Suggestions", sugg, key=f"sugg_{len(chat)}", label_visibility="collapsed")
        typed = st.chat_input("Posez votre question…", key="pilot_input")
        question = st.session_state.pop("pending_q", None) or typed or pick
        if question:
            chat.append({"role": "user", "content": question})
            with box:
                with st.chat_message("user", avatar=":material/person:"):
                    st.markdown(question)
                with st.chat_message("assistant", avatar=":material/insights:"):
                    with st.spinner("Pilot interroge les agents…"):
                        hist = [{"role": m["role"], "content": m["content"]} for m in chat[:-1]]
                        ans = A.ask(store, question, hist, use_llm=use_llm, scope=SCOPE,
                                    context={"page": page_title, "filtres": SCOPE.label()})
            chat.append({"role": "assistant", "content": ans.text, "artifacts": ans.artifacts,
                         "trace": ans.trace, "warning": ans.warning})
            st.rerun()


# --------------------------------------------------------------------------
# Navigation : dashboard à gauche, chatbot à droite
# --------------------------------------------------------------------------
PAGES = {
    "perf": st.Page(page_performance, title="Ventes", icon=":material/trending_up:", url_path="ventes", default=True),
    "deliv": st.Page(page_delivery, title="Livraison & clients", icon=":material/local_shipping:", url_path="livraison"),
    "customers": st.Page(page_customers, title="Clients", icon=":material/group:", url_path="clients"),
    "sellers": st.Page(page_sellers, title="Vendeurs", icon=":material/storefront:", url_path="vendeurs"),
    "products": st.Page(page_products, title="Produits", icon=":material/inventory_2:", url_path="produits"),
    "payments": st.Page(page_payments, title="Paiements", icon=":material/payments:", url_path="paiements"),
    "map": st.Page(page_map, title="Carte", icon=":material/map:", url_path="carte"),
    "scen": st.Page(page_scenarios, title="Scénarios", icon=":material/tune:", url_path="scenarios"),
}
current = st.navigation(list(PAGES.values()), position="top")
if st.session_state.get("show_chat", True):
    main_col, chat_col = st.columns([2.45, 1], gap="medium")
    with main_col:
        current.run()
    with chat_col:
        chat_panel(current.title)
else:
    current.run()
