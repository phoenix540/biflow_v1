import csv, html, json, os, subprocess, sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw" / "olist"
CURATED, CLEAN = DATA / "curated", DATA / "clean"
PROFILE_DIR, ETL_DIR = DATA / "profile", DATA / "etl"
VALIDATION_FILE, LOG_FILE = DATA / "validated_translations.json", DATA / "human_review_log.json"
KPI_RESULTS_FILE = DATA / "kpi" / "kpi_results.json"
ANALYST_FILE = DATA / "analyst" / "analyst_insights.json"
AUDIT_FILE = DATA / "audit" / "audit_report.json"
DASHBOARD_HTML = DATA / "dashboard" / "dashboard.html"

st.set_page_config(page_title="Olist BI", layout="wide", page_icon="■")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:Inter,sans-serif;color:#1a2233}.stApp{background:#f5f6f8}
.block-container{padding-top:1.6rem;max-width:1280px}section[data-testid="stSidebar"]{background:#fff;border-right:1px solid #e3e7ee}
h2,h3{color:#14213d}.ph{padding:4px 0 14px;margin-bottom:18px;border-bottom:1px solid #dde2ea}
.ph .tag{font-size:.72rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:#1f4e8c}
.ph h1{margin:4px 0;font-size:1.75rem;font-weight:700;color:#14213d}.ph p{margin:0;color:#5b6577}
.kpi{background:#fff;border:1px solid #e3e7ee;border-top:3px solid #1f4e8c;border-radius:8px;padding:14px 16px;height:100%}
.kpi .l{color:#6b7488;font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;font-weight:600}
.kpi .v{font-size:1.65rem;font-weight:700;color:#14213d;margin-top:2px}.kpi .s{color:#7a8397;font-size:.78rem;margin-top:2px}
.note{padding:11px 15px;border-radius:6px;margin:8px 0;background:#fff;border:1px solid #e3e7ee;border-left:4px solid #1f4e8c}
.note.warn{border-left-color:#b7791f;background:#fffaf0}.note.ok{border-left-color:#2f855a;background:#f4fbf7}.note.bad{border-left-color:#c53030;background:#fff5f5}
.brand{font-size:1.2rem;font-weight:700;color:#14213d}.brand span{color:#1f4e8c}
.stButton>button{border-radius:6px;border:1px solid #cdd3de;font-weight:600;background:#fff;color:#14213d}
.stButton>button[kind="primary"]{background:#1f4e8c;border-color:#1f4e8c;color:#fff}
</style>""", unsafe_allow_html=True)

COLORS = ["#1f4e8c", "#3b82a0", "#7aa6c8", "#b7791f", "#2f855a", "#8a94a6"]
SEV_FR = {"high": "Critique", "medium": "Moyen", "low": "Mineur"}
STATUS_FR = {"success": "terminé", "failed": "échec", "skipped": "à venir", "info": "info"}
AGENT_FR = {
    "profiler": "Profiler",
    "etl": "ETL",
    "kpi": "KPI",
    "analyst": "Analyste",
    "dashboard": "Dashboard",
    "auditor": "Auditeur",
}
ORDER = ["profiler", "etl", "kpi", "analyst", "dashboard", "auditor"]

TABLE_INFO = {
    "fact_sales": ("Ventes par article", "Une ligne par article vendu (CA, coût, marge)."),
    "fact_orders": ("Commandes", "Une ligne par commande."),
    "dim_products": ("Produits", "Nom, catégorie, coût de production, poids et dimensions."),
    "dim_customers": ("Clients", "Identifiant, ville et État."),
    "dim_sellers": ("Vendeurs", "Identifiant, nom, ville et État."),
    "dim_geolocation": ("Géolocalisation", "Positions par code postal."),
    "dim_date": ("Calendrier", "Jours, mois, trimestres et jours fériés."),
    "dim_events": ("Événements", "Événements commerciaux et jours fériés."),
    "reviews": ("Avis clients", "Notes et commentaires."),
}


def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None


def header(title, sub, tag="Olist BI"):
    st.markdown(
        f'<div class="ph"><div class="tag">{tag}</div>'
        f'<h1>{title}</h1><p>{sub}</p></div>',
        unsafe_allow_html=True,
    )


def kpi(col, label, value, sub=""):
    col.markdown(
        f'<div class="kpi"><div class="l">{label}</div>'
        f'<div class="v">{value}</div><div class="s">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def note(text, kind=""):
    st.markdown(f'<div class="note {kind}">{text}</div>', unsafe_allow_html=True)


def help_box(text):
    with st.expander("Comment lire cette page"):
        st.markdown(text)


def style_fig(fig, height=350):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#fff",
        colorway=COLORS,
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(family="Inter", color="#1a2233"),
    )
    fig.update_xaxes(gridcolor="#eef0f4")
    fig.update_yaxes(gridcolor="#eef0f4")
    return fig


def fmt(n):
    try:
        return f"{int(n):,}".replace(",", " ")
    except (ValueError, TypeError):
        return "n/a"


def money(n):
    try:
        return f"R$ {float(n):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except (ValueError, TypeError):
        return "n/a"


def num(n, digits=2):
    try:
        return f"{float(n):.{digits}f}".replace(".", ",")
    except (ValueError, TypeError):
        return "n/a"


def pct(n):
    try:
        return f"{float(n):.2f}%".replace(".", ",")
    except (ValueError, TypeError):
        return "n/a"


def run_command(args, box):
    env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
    }
    proc = subprocess.Popen(
        [sys.executable, *args],
        cwd=ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    lines = []
    for line in proc.stdout:
        lines.append(line.rstrip())
        box.code("\n".join(lines[-40:]), language="text")
    return proc.wait()


@st.cache_data(show_spinner=False)
def untranslated_from_raw(stamp):
    products = pd.read_csv(
        RAW / "olist_products_dataset.csv",
        usecols=["product_category_name"],
    )
    translations = pd.read_csv(
        RAW / "product_category_name_translation.csv",
        encoding="utf-8-sig",
    )
    known = set(translations["product_category_name"])
    counts = products["product_category_name"].dropna().value_counts()
    return {name: int(n) for name, n in counts.items() if name not in known}


@st.cache_data(show_spinner=False)
def count_rows(path, stamp):
    # csv.reader compte des ENREGISTREMENTS : un commentaire d'avis sur plusieurs lignes reste 1 ligne
    with open(path, encoding="utf-8", errors="ignore", newline="") as f:
        return sum(1 for _ in csv.reader(f)) - 1


def raw_stamp():
    f = RAW / "olist_products_dataset.csv"
    return f.stat().st_mtime if f.exists() else 0


orch = load_json(PROFILE_DIR / "orchestrator_log.json")
profile = load_json(PROFILE_DIR / "profile.json")
decision = load_json(PROFILE_DIR / "profiler_agent_decision.json")
etl_dec = load_json(ETL_DIR / "etl_agent_decision.json")
etl_log = load_json(ETL_DIR / "correction_log.json")
kpi_data = load_json(KPI_RESULTS_FILE) or {}
analyst = load_json(ANALYST_FILE) or {}
audit = load_json(AUDIT_FILE) or {}
saved = load_json(VALIDATION_FILE) or {}


def pending_translations():
    if not (RAW / "olist_products_dataset.csv").exists():
        return {}
    missing = untranslated_from_raw(raw_stamp())
    decided = set(saved.get("translations", {})) | set(saved.get("rejected", []))
    return {k: v for k, v in missing.items() if k not in decided}


PAGES = [
    "Accueil",
    "Qualité des données",
    "Validation humaine",
    "Données préparées",
    "KPI Dashboard",
    "Analyses",
    "Dashboard BI",
    "Audit & XAI",
    "Lancer et suivre",
]

pend = pending_translations()

st.sidebar.markdown(
    '<div class="brand"><span>■</span> Olist BI</div>',
    unsafe_allow_html=True,
)
st.sidebar.caption("Pipeline BI multi-agents")
page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")

if pend:
    st.sidebar.warning(f"{len(pend)} catégories à valider")

st.sidebar.divider()
st.sidebar.caption("Marketplace e-commerce Olist · toutes catégories.")

if st.sidebar.button("Rafraîchir", width="stretch"):
    st.cache_data.clear()
    st.rerun()


# ============================================================
# PAGE 1 — ACCUEIL
# ============================================================

if page == PAGES[0]:
    header(
        "Suivi du projet",
        "De la donnée brute à des tables fiables : avancement et décisions en attente",
        "Vue d'ensemble",
    )

    help_box(
        "Le parcours suit le pipeline. Les indicateurs de vente sont disponibles "
        "dans KPI Dashboard après calcul. Les corrections et validations sont tracées."
    )

    raw_ok = (RAW / "olist_products_dataset.csv").exists()

    for label, exists, desc in [
        ("Données brutes", raw_ok, "fichiers Olist"),
        ("Profiler", bool(decision), "diagnostic et score"),
        ("Validation humaine", bool(saved) and not pend, "traductions"),
        ("ETL", bool(etl_dec), "nettoyage tracé"),
        (
            "Tables préparées",
            (CURATED / "fact_sales.csv").exists(),
            "ventes et commandes",
        ),
        ("KPI", bool(kpi_data), "indicateurs calculés"),
        ("Analyste", bool(analyst), "tendances, alertes et recommandations"),
        ("Dashboard", DASHBOARD_HTML.exists(), "dashboard HTML interactif (KPI + analyses)"),
        ("Audit", bool(audit) and audit.get("passed", False), "recalcul indépendant et traçabilité"),
    ]:
        st.write(("✅ " if exists else "⏳ ") + f"**{label}** — {desc}")

    st.subheader("Indicateurs clés")
    cols = st.columns(4)

    kpi(
        cols[0],
        "Score de fiabilité",
        f'{profile.get("global_score", "?")} / 100' if profile else "n/a",
    )
    kpi(
        cols[1],
        "Tables analysées",
        len(profile.get("tables", {})) if profile else "n/a",
    )
    kpi(
        cols[2],
        "Articles vendus",
        fmt(etl_dec.get("rows_fact_sales", 0)) if etl_dec else "n/a",
    )
    kpi(
        cols[3],
        "Commandes",
        fmt(etl_dec.get("rows_fact_orders", 0)) if etl_dec else "n/a",
    )

    if pend:
        note(
            f"{len(pend)} catégories de produits restent à valider. "
            "Ouvre Validation humaine.",
            "warn",
        )

    if not kpi_data:
        note(
            "Les KPI ne sont pas encore disponibles. "
            "Exécute le pipeline complet puis rafraîchis.",
            "warn",
        )

    if audit:
        note(
            f'<b>Audit {"conforme" if audit.get("passed") else "NON conforme"}</b> · '
            f'{audit.get("checks_total", 0)} contrôles, {audit.get("critical_failures", 0)} échec(s) critique(s), '
            f'{audit.get("warnings", 0)} avertissement(s) · {html.escape(str(audit.get("audited_at", "")))}',
            "ok" if audit.get("passed") else "bad",
        )

    if decision:
        fit = decision.get("fit_for_project", {})
        note(
            f'<b>Adéquation au projet : {html.escape(str(fit.get("verdict", "?")))}.</b> '
            f'{html.escape(str(fit.get("reasoning", "")))}'
        )
        st.write(decision.get("summary", ""))


# ============================================================
# PAGE 2 — QUALITÉ DES DONNÉES
# ============================================================

elif page == PAGES[1]:
    header(
        "Qualité des données",
        "Peut-on se fier aux données Olist brutes ? Constats du Profiler",
        "Profiler",
    )

    help_box(
        "Le score combine les valeurs manquantes, doublons et valeurs atypiques. "
        "Les problèmes affichent leur gravité, leur preuve et l'action prévue."
    )

    if not profile:
        st.warning("Aucun profil. Lance le Profiler dans « Lancer et suivre ».")
        st.stop()

    tables = profile.get("tables", {})
    cols = st.columns(3)

    kpi(cols[0], "Score global", f'{profile.get("global_score", "?")} / 100')
    kpi(cols[1], "Tables analysées", len(tables))

    if decision:
        kpi(
            cols[2],
            "Adéquation",
            decision.get("fit_for_project", {}).get("verdict", "?"),
        )

    df = pd.DataFrame(
        [
            {
                "Table": n,
                "Lignes": t.get("rows"),
                "Doublons": t.get("duplicates"),
                "% manquants": t.get("avg_missing_pct"),
                "Score": t.get("reliability_score"),
            }
            for n, t in tables.items()
        ]
    )

    a, b, c, d = st.tabs(
        [
            "Scores par table",
            "Détail d'une table",
            "Problèmes détectés",
            "Ce qui manque",
        ]
    )

    with a:
        fig = px.bar(
            df.sort_values("Score"),
            x="Score",
            y="Table",
            orientation="h",
            range_x=[0, 100],
            title="Score de fiabilité par table",
        )
        st.plotly_chart(style_fig(fig, 380), width="stretch")
        st.dataframe(df, width="stretch", hide_index=True)

    with b:
        choice = st.selectbox("Table", list(tables))
        columns = tables[choice]["columns"]

        cdf = pd.DataFrame(
            [
                {
                    "Colonne": k,
                    "Type": v.get("dtype"),
                    "% manquants": v.get("missing_pct"),
                    "Distincts": v.get("unique"),
                    "Alertes": " ; ".join(v.get("hints", [])),
                }
                for k, v in columns.items()
            ]
        )

        if st.checkbox("Seulement les colonnes avec un point d'attention"):
            cdf = cdf[(cdf["% manquants"] > 1) | (cdf["Alertes"] != "")]

        st.dataframe(cdf, width="stretch", hide_index=True)

    with c:
        issues = (decision or {}).get("issues", [])

        if not issues:
            st.info("Aucune décision de l'agent disponible.")

        for item in issues:
            sev = item.get("severity", "low")

            with st.expander(
                f'{SEV_FR.get(sev, sev)} · '
                f'{item.get("table")} · {item.get("column")}'
            ):
                st.write("Diagnostic :", item.get("diagnosis"))
                st.write("Preuve :", item.get("evidence"))
                st.write("Action ETL :", item.get("action_for_etl"))

    with d:
        for gap in (decision or {}).get("data_gaps", []):
            with st.expander(gap.get("need", "Besoin")):
                st.write(
                    "Indicateurs touchés :",
                    ", ".join(gap.get("kpis_affected", [])),
                )
                st.write("Options :", " ; ".join(gap.get("options", [])))
                st.write("Recommandation :", gap.get("recommendation", ""))


# ============================================================
# PAGE 3 — VALIDATION HUMAINE
# ============================================================

elif page == PAGES[2]:
    header(
        "Validation humaine",
        "L'agent propose, une personne décide : rien n'est appliqué sans accord",
        "Contrôle humain",
    )

    help_box(
        "Choisis Approuvé ou Rejeté, corrige éventuellement la traduction, "
        "puis enregistre. Relance ensuite l'ETL."
    )

    if not (RAW / "olist_products_dataset.csv").exists():
        st.error(f"Fichier introuvable : {RAW / 'olist_products_dataset.csv'}")
        st.stop()

    props = {
        x.get("original"): x
        for x in (decision or {}).get("proposed_translations", [])
    }

    missing = untranslated_from_raw(raw_stamp())
    rows = []

    for name, n in missing.items():
        state = (
            "Approuvé"
            if name in saved.get("translations", {})
            else "Rejeté"
            if name in saved.get("rejected", [])
            else "À valider"
        )

        rows.append(
            {
                "Décision": state,
                "Catégorie originale": name,
                "Traduction": saved.get("translations", {}).get(
                    name,
                    props.get(name, {}).get("english", ""),
                ),
                "Produits": n,
                "Confiance LLM": props.get(name, {}).get(
                    "confidence", "aucune"
                ),
            }
        )

    if not rows:
        note(
            "Toutes les catégories disposent d'une traduction ou ont été traitées.",
            "ok",
        )

    else:
        if "tr_df" not in st.session_state:
            st.session_state.tr_df = pd.DataFrame(rows)

        x, y, z, _ = st.columns([1.5, 1, 1, 3])

        if x.button("Approuver confiance haute"):
            d = st.session_state.tr_df
            d.loc[
                (d["Confiance LLM"] == "high")
                & (d["Décision"] == "À valider"),
                "Décision",
            ] = "Approuvé"
            st.session_state.pop("tr_editor", None)
            st.rerun()

        if y.button("Tout approuver"):
            st.session_state.tr_df["Décision"] = "Approuvé"
            st.session_state.pop("tr_editor", None)
            st.rerun()

        if z.button("Réinitialiser"):
            st.session_state.pop("tr_df", None)
            st.session_state.pop("tr_editor", None)
            st.rerun()

        edited = st.data_editor(
            st.session_state.tr_df,
            key="tr_editor",
            width="stretch",
            hide_index=True,
            disabled=[
                "Catégorie originale",
                "Produits",
                "Confiance LLM",
            ],
            column_config={
                "Décision": st.column_config.SelectboxColumn(
                    options=["À valider", "Approuvé", "Rejeté"],
                    required=True,
                ),
                "Traduction": st.column_config.TextColumn(
                    "Traduction (modifiable)"
                ),
            },
        )

        cc = st.columns(3)

        for col, label, value in zip(
            cc,
            ["Approuvées", "Rejetées", "À valider"],
            [
                (edited["Décision"] == "Approuvé").sum(),
                (edited["Décision"] == "Rejeté").sum(),
                (edited["Décision"] == "À valider").sum(),
            ],
        ):
            kpi(col, label, int(value))

        reviewer = st.text_input(
            "Nom du validateur",
            value=saved.get("validated_by", ""),
        )

        if st.button("Enregistrer la validation", type="primary"):
            if not reviewer.strip():
                st.error("Indique ton nom avant d'enregistrer.")

            else:
                now = datetime.now().isoformat(timespec="seconds")

                out = {
                    "validated_by": reviewer.strip(),
                    "validated_at": now,
                    "translations": {
                        r["Catégorie originale"]: str(
                            r["Traduction"]
                        ).strip()
                        for _, r in edited.iterrows()
                        if r["Décision"] == "Approuvé"
                        and str(r["Traduction"] or "").strip()
                    },
                    "rejected": [
                        r["Catégorie originale"]
                        for _, r in edited.iterrows()
                        if r["Décision"] == "Rejeté"
                    ],
                    "pending": [
                        r["Catégorie originale"]
                        for _, r in edited.iterrows()
                        if r["Décision"] == "À valider"
                    ],
                }

                VALIDATION_FILE.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                VALIDATION_FILE.write_text(
                    json.dumps(out, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

                history = load_json(LOG_FILE) or []
                history.append(
                    {
                        "reviewer": reviewer.strip(),
                        "at": now,
                        "changes": [
                            {
                                "category": r["Catégorie originale"],
                                "decision": r["Décision"],
                                "final_translation": str(
                                    r["Traduction"] or ""
                                ).strip(),
                            }
                            for _, r in edited.iterrows()
                        ],
                    }
                )

                LOG_FILE.write_text(
                    json.dumps(history, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

                st.success("Validation enregistrée. Relance l'ETL.")

    st.subheader("Historique des validations")
    history = load_json(LOG_FILE) or []

    if history:
        st.dataframe(
            pd.json_normalize(
                history,
                record_path="changes",
                meta=["reviewer", "at"],
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("Aucune validation enregistrée.")


# ============================================================
# PAGE 4 — DONNÉES PRÉPARÉES
# ============================================================

elif page == PAGES[3]:
    header(
        "Données préparées",
        "Corrections appliquées par l'ETL et tables propres prêtes pour l'analyse",
        "ETL",
    )

    help_box(
        "Consulte les contrôles, hypothèses, journal de corrections et tables préparées."
    )

    if not etl_dec:
        st.warning("L'ETL n'a pas encore tourné.")
        st.stop()

    note(
        f'Dernière exécution : {etl_dec.get("run_at", "?")} · '
        f'statut : {etl_dec.get("status", "?")}',
        "ok" if etl_dec.get("status") == "success" else "warn",
    )

    a, b, c, d = st.tabs(
        [
            "Avant / après",
            "Contrôles et hypothèses",
            "Corrections",
            "Explorer les tables",
        ]
    )

    with a:
        cols = st.columns(2)
        kpi(
            cols[0],
            "Table des ventes",
            fmt(etl_dec.get("rows_fact_sales", 0)),
        )
        kpi(
            cols[1],
            "Table des commandes",
            fmt(etl_dec.get("rows_fact_orders", 0)),
        )

        rb = etl_dec.get("rows_before", {})
        df = pd.DataFrame(
            [{"Table brute": k, "Lignes": v} for k, v in rb.items()]
        )

        if not df.empty:
            st.plotly_chart(
                style_fig(
                    px.bar(
                        df,
                        x="Lignes",
                        y="Table brute",
                        orientation="h",
                        title="Fichiers bruts Olist",
                    )
                ),
                width="stretch",
            )

    with b:
        checks = etl_dec.get("checks", [])

        if checks:
            st.dataframe(
                pd.DataFrame(checks),
                width="stretch",
                hide_index=True,
            )

        for item in etl_dec.get("assumptions", []):
            note(html.escape(str(item)), "warn")

        if etl_dec.get("untranslated_categories_left"):
            note(
                "Catégories non traduites : "
                + html.escape(", ".join(etl_dec["untranslated_categories_left"])),
                "warn",
            )

    with c:
        if etl_log:
            st.dataframe(
                pd.DataFrame(etl_log),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("Pas de journal de corrections.")

    with d:
        files = [
            f
            for folder in (CURATED, CLEAN)
            if folder.exists()
            for f in sorted(folder.glob("*.csv"))
        ]

        if not files:
            st.warning("Aucun fichier dans data/curated ou data/clean.")

        else:
            overview = []

            for f in files:
                title, desc = TABLE_INFO.get(
                    f.stem,
                    (f.stem, ""),
                )

                overview.append(
                    {
                        "Table": title,
                        "Contenu": desc,
                        "Lignes": count_rows(f, f.stat().st_mtime),
                        "Fichier": f"{f.parent.name}/{f.name}",
                    }
                )

            st.dataframe(
                pd.DataFrame(overview),
                width="stretch",
                hide_index=True,
            )

            pick = st.selectbox(
                "Voir le contenu d'une table",
                files,
                format_func=lambda f: (
                    f"{TABLE_INFO.get(f.stem, (f.stem, ''))[0]} "
                    f"({f.parent.name}/{f.name})"
                ),
            )

            n = st.slider("Lignes affichées", 10, 500, 50)
            st.dataframe(
                pd.read_csv(pick, nrows=n),
                width="stretch",
            )

            st.download_button(
                "Télécharger ce CSV",
                pick.read_bytes(),
                file_name=pick.name,
            )


# ============================================================
# PAGE 5 — KPI DASHBOARD
# ============================================================

elif page == PAGES[4]:
    header(
        "KPI Dashboard",
        "Indicateurs clés et analyses calculés à partir des tables préparées",
        "Performance",
    )

    help_box(
        "Les indicateurs sont lus depuis data/kpi/kpi_results.json. "
        "Relance le pipeline complet pour les actualiser. "
        "Le chiffre d'affaires n'est pas le bénéfice ; la marge est simulée."
    )

    if not kpi_data or "headline" not in kpi_data:
        st.warning("Résultats KPI absents ou anciens. Lance le pipeline complet puis rafraîchis.")
        st.stop()

    # ---------------- Filtres : les KPI sont recalculés par l'agent KPI lui-même (kpi_agent.compute)
    sys.path.insert(0, str(ROOT / "agents"))
    import kpi_agent as KA

    @st.cache_resource(show_spinner="Chargement des tables pour les filtres…", max_entries=1)
    def kpi_tables(stamp):
        return KA.load()

    @st.cache_data(show_spinner="Recalcul des KPI sur le périmètre choisi…", max_entries=50)
    def kpi_filtered(stamp, period, states, cats):
        o, s_, d = kpi_tables(stamp)
        o, s_ = KA.filter_frames(o, s_, states, cats)
        return KA.compute(o, s_, d, period)

    MOIS_FR = ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."]
    mlabel = lambda m: f"{MOIS_FR[int(m[5:7]) - 1]} {m[:4]}"
    all_months = [m["month"] for m in kpi_data.get("breakdowns", {}).get("monthly", [])]
    reliable_m = {m["month"]: m["is_reliable_month"] for m in kpi_data.get("breakdowns", {}).get("monthly", [])}
    stamp = tuple(p_.stat().st_mtime for p_ in (CURATED / "fact_orders.csv", CURATED / "fact_sales.csv",
                                                 CURATED / "dim_date.csv") if p_.exists())
    kd = kpi_data
    if all_months:
        try:
            _, sales_t, _ = kpi_tables(stamp)
            cat_options = sorted(sales_t["category"].dropna().unique())
        except Exception as e:   # tables absentes : on affiche quand même les KPI de l'agent
            cat_options = []
            st.warning(f"Filtres indisponibles (tables préparées illisibles) : {e}")
        # Valeurs choisies gardées hors des widgets ; un raccourci (2017, 2018...) ou la réinitialisation
        # change kd_ver, ce qui recrée les widgets avec la nouvelle valeur (sinon le navigateur renvoie l'ancienne).
        ss = st.session_state
        per = ss.get("kd_range")
        if not (isinstance(per, (tuple, list)) and len(per) == 2 and all(x in all_months for x in per)):
            ss["kd_range"] = (all_months[0], all_months[-1])
        ss.setdefault("kd_ver", 0)
        ss.setdefault("kd_sel_states", [])
        ss.setdefault("kd_sel_cats", [])
        v = ss["kd_ver"]
        with st.container(border=True):
            st.markdown("**Filtres**")
            f1, f2, f3 = st.columns([2.2, 1.3, 1.9])
            p0, p1 = f1.select_slider("Période", options=all_months, value=tuple(ss["kd_range"]),
                                      key=f"kd_period_{v}", format_func=mlabel)
            state_options = sorted(x["state"] for x in kpi_data["breakdowns"].get("by_customer_state", []))
            sel_states = f2.multiselect("États clients", state_options, key=f"kd_states_{v}", placeholder="Tous les États",
                                        default=[x for x in ss["kd_sel_states"] if x in state_options])
            sel_cats = f3.multiselect("Catégories", cat_options, key=f"kd_cats_{v}", placeholder="Toutes les catégories",
                                      default=[x for x in ss["kd_sel_cats"] if x in cat_options],
                                      format_func=lambda c: c.replace("_", " "))
            ss["kd_range"], ss["kd_sel_states"], ss["kd_sel_cats"] = (p0, p1), list(sel_states), list(sel_cats)
            q1, q2, q3, q4, _ = st.columns([1, 1, 1, 1.2, 2.2])

            def set_period(a, z):
                ss["kd_range"] = (a, z)
                ss["kd_ver"] += 1

            last_ok = [m for m in all_months if reliable_m.get(m)][-1]
            q1.button("Tout", on_click=set_period, args=(all_months[0], all_months[-1]), width="stretch")
            q2.button("2017", on_click=set_period, args=("2017-01", "2017-12"), width="stretch",
                      disabled=not {"2017-01", "2017-12"} <= set(all_months))
            q3.button("2018", on_click=set_period, args=("2018-01", all_months[-1]), width="stretch",
                      disabled="2018-01" not in all_months)
            q4.button("12 derniers mois", on_click=set_period, width="stretch",
                      args=(all_months[max(0, all_months.index(last_ok) - 11)], last_ok),
                      help="Les 12 derniers mois complets")

            def reset_filters():
                ss["kd_range"] = (all_months[0], all_months[-1])
                ss["kd_sel_states"], ss["kd_sel_cats"] = [], []
                ss["kd_ver"] += 1

            st.button("Réinitialiser les filtres", on_click=reset_filters, icon=":material/restart_alt:")

        period = None if (p0, p1) == (all_months[0], all_months[-1]) else (p0, p1)
        if period or sel_states or sel_cats:
            try:
                kd = kpi_filtered(stamp, period, tuple(sel_states), tuple(sel_cats))
            except Exception as e:
                st.error(f"Recalcul impossible : {e}")
                st.stop()
            parts = [f"{mlabel(p0)} → {mlabel(p1)}"]
            if sel_states:
                parts.append("États : " + ", ".join(sel_states))
            if sel_cats:
                parts.append("catégories : " + ", ".join(c.replace("_", " ") for c in sel_cats))
            note("<b>Filtres actifs :</b> " + html.escape(" · ".join(parts)) + ". Indicateurs recalculés par l'agent KPI "
                 "sur ce périmètre (mêmes formules)."
                 + (" Avec un filtre catégorie, le CA ne compte que les articles de ces catégories." if sel_cats else ""), "warn")
            unrel = [m for m in all_months if p0 <= m <= p1 and not reliable_m.get(m)]
            if unrel:
                st.caption("Mois peu fiables inclus (exclus des croissances) : " + ", ".join(mlabel(m) for m in unrel))
            if not kd.get("headline", {}).get("delivered_orders"):
                st.info("Aucune commande sur ce périmètre : élargis les filtres.")
                st.stop()
        else:
            note("Aucun filtre : chiffres officiels de l'agent KPI (data/kpi/kpi_results.json).", "ok")

    h = kd["headline"]
    b = kd.get("breakdowns", {})
    ins = kd.get("insights", {})
    yoy = b.get("yoy_summary", {})

    cols = st.columns(4)
    kpi(cols[0], "Chiffre d'affaires", money(h.get("total_revenue_brl")), "commandes livrées, hors fret")
    kpi(cols[1], "Commandes livrées", fmt(h.get("delivered_orders")))
    kpi(cols[2], "Panier moyen", money(h.get("average_order_value_brl")))
    kpi(cols[3], "Croissance sur un an", pct(h.get("yoy_revenue_growth_pct")),
        f'{yoy.get("period", "")} vs {yoy.get("compared_to", "")}')

    cols = st.columns(4)
    kpi(cols[0], "Clients récurrents", pct(h.get("repeat_customer_rate_pct")),
        f'{fmt(h.get("unique_customers"))} clients uniques')
    kpi(cols[1], "Commandes perdues", pct(h.get("lost_order_rate_pct")), "annulées ou indisponibles")
    kpi(cols[2], "Livraisons en retard", pct(h.get("late_delivery_rate_pct")),
        f'délai médian {num(h.get("median_delivery_days"), 1)} jours')
    kpi(cols[3], "Note moyenne", f'{num(h.get("average_review_score"))} / 5',
        f'avis faibles (1-2) : {pct(h.get("low_rating_rate_pct"))}')

    if ins.get("review_score_late_orders") is not None:
        note(
            f'<b>Insight :</b> une commande livrée en retard reçoit en moyenne '
            f'<b>{num(ins["review_score_late_orders"])} / 5</b>, contre '
            f'<b>{num(ins["review_score_on_time_orders"])} / 5</b> à l\'heure. '
            f'Le fret représente {pct(h.get("freight_to_revenue_pct"))} du chiffre d\'affaires.',
            "warn",
        )

    t1, t2, t3, t4, t5 = st.tabs(
        ["Évolution", "Régions", "Catégories", "Vendeurs et produits", "Événements et jours"]
    )

    with t1:
        monthly = pd.DataFrame(b.get("monthly", []))
        if not monthly.empty:
            monthly["Fiabilité"] = monthly["is_reliable_month"].map(
                {True: "Mois fiable", False: "Mois peu fiable"})
            st.plotly_chart(style_fig(px.bar(
                monthly, x="month", y="revenue_brl", color="Fiabilité",
                title="Chiffre d'affaires mensuel")), width="stretch")
            g = monthly.dropna(subset=["yoy_growth_pct"])
            if not g.empty:
                st.plotly_chart(style_fig(px.line(
                    g, x="month", y="yoy_growth_pct", markers=True,
                    title="Croissance sur un an (%, même mois de l'année précédente)"), 300),
                    width="stretch")

    with t2:
        states = pd.DataFrame(b.get("by_customer_state", []))
        if not states.empty:
            st.plotly_chart(style_fig(px.bar(
                states.sort_values("late_delivery_rate_pct"), x="late_delivery_rate_pct", y="state",
                orientation="h", color="avg_review_score", color_continuous_scale="RdYlGn",
                hover_data=["orders", "median_delivery_days", "freight_to_revenue_pct"],
                title="Taux de retard par État client (couleur = note moyenne)"), 650),
                width="stretch")
            st.dataframe(states.rename(columns={
                "state": "État", "orders": "Commandes", "revenue_brl": "CA (R$)",
                "late_delivery_rate_pct": "Retards (%)", "median_delivery_days": "Délai médian (j)",
                "freight_to_revenue_pct": "Fret / CA (%)", "avg_review_score": "Note"}),
                width="stretch", hide_index=True)

    cat_cols = {"category": "Catégorie", "revenue_brl": "CA (R$)", "units": "Unités",
                "avg_review_score": "Note", "yoy_growth_pct": "Croissance 1 an (%)"}
    with t3:
        cats = b.get("categories", {})
        top = pd.DataFrame(cats.get("top_by_revenue", []))
        if not top.empty:
            st.plotly_chart(style_fig(px.bar(
                top.sort_values("revenue_brl"), x="revenue_brl", y="category", orientation="h",
                title="Catégories principales (CA)")), width="stretch")
        left, right = st.columns(2)
        with left:
            st.markdown("**Plus forte croissance**")
            st.dataframe(pd.DataFrame(cats.get("fastest_growing", [])).rename(columns=cat_cols),
                         width="stretch", hide_index=True)
        with right:
            st.markdown("**En baisse**")
            dec = pd.DataFrame(cats.get("declining", []))
            if dec.empty:
                st.info("Aucune catégorie en baisse.")
            else:
                st.dataframe(dec.rename(columns=cat_cols), width="stretch", hide_index=True)
        st.markdown("**Catégories les moins bien notées** (au moins 1 000 articles)")
        st.dataframe(pd.DataFrame(cats.get("lowest_rated", [])).rename(columns=cat_cols),
                     width="stretch", hide_index=True)

    with t4:
        sel = b.get("sellers", {})
        conc = sel.get("concentration", {})
        cols = st.columns(3)
        kpi(cols[0], "Vendeurs actifs", fmt(conc.get("active_sellers")))
        kpi(cols[1], "Part des 10 % meilleurs vendeurs", pct(conc.get("top_10pct_sellers_revenue_share_pct")))
        kpi(cols[2], f'Part des vendeurs {conc.get("top_seller_state", "")}',
            pct(conc.get("top_seller_state_revenue_share_pct")))
        st.markdown("**Meilleurs vendeurs**")
        st.dataframe(pd.DataFrame(sel.get("top_sellers", [])).rename(columns={
            "seller_name": "Vendeur", "seller_id": "Identifiant", "state": "État", "revenue_brl": "CA (R$)",
            "orders": "Commandes", "late_delivery_rate_pct": "Retards (%)", "avg_review_score": "Note"}),
            width="stretch", hide_index=True)
        st.markdown("**Best-sellers (unités vendues)**")
        st.dataframe(pd.DataFrame(b.get("best_sellers", [])).rename(columns={
            "product_name": "Produit", "product_id": "Identifiant", "category": "Catégorie",
            "units": "Unités", "revenue_brl": "CA (R$)"}),
            width="stretch", hide_index=True)

    with t5:
        ev = pd.DataFrame(b.get("events", []))
        if not ev.empty:
            st.plotly_chart(style_fig(px.bar(
                ev, x="date", y="vs_normal_day_pct", color="type", hover_data=["event", "revenue_brl", "orders"],
                title="Écart de CA vs un jour normal du même mois (%)"), 330), width="stretch")
            st.dataframe(ev.rename(columns={
                "date": "Date", "event": "Événement", "type": "Type", "revenue_brl": "CA (R$)",
                "orders": "Commandes", "vs_normal_day_pct": "Écart vs jour normal (%)"}),
                width="stretch", hide_index=True)
        wd = pd.DataFrame(b.get("weekday_revenue_share", []))
        if not wd.empty:
            st.plotly_chart(style_fig(px.bar(
                wd, x="weekday", y="revenue_share_pct", title="Part du CA par jour de la semaine (%)"), 300),
                width="stretch")

    m = kd.get("margin_simulated", {})
    if m:
        with st.expander("Marge (simulée)"):
            note(f'Marge brute : <b>{money(m.get("gross_margin_brl"))}</b> '
                 f'({pct(m.get("gross_margin_rate_pct"))}). {html.escape(str(m.get("warning", "")))}', "bad")

    with st.expander("Définitions et limites"):
        for name, value in kd.get("definitions", {}).items():
            st.markdown(f"**{name.replace('_', ' ').capitalize()}** : {value}")
        for item in kd.get("limitations", []):
            st.markdown("- " + str(item))

    with st.expander("Couverture des données (technique)"):
        st.json(kd.get("coverage", {}))

    with st.expander("Données KPI brutes (JSON)"):
        st.json(kd)


# ============================================================
# PAGE — ANALYSES (BI ANALYST AGENT)
# ============================================================

elif page == "Analyses":
    header(
        "Analyses",
        "Tendances, baisses, anomalies et recommandations produites par le BI Analyst Agent",
        "BI Analyst",
    )

    help_box(
        "Python détecte les faits (signaux chiffrés S1, S2...). Le LLM les transforme en insights "
        "en citant ses signaux ; chaque nombre écrit est vérifié. Sans LLM, les insights sont générés "
        "par règles. Un insight marqué « à contrôler » contient un chiffre non retrouvé dans les faits."
    )

    if not analyst:
        st.warning("Aucune analyse. Lance l'Analyste ou le pipeline complet dans « Lancer et suivre ».")
        st.stop()

    by_llm = analyst.get("generated_by") == "llm"
    note(
        f'Généré le {html.escape(str(analyst.get("generated_at", "?")))} · '
        + (f'LLM {html.escape(str(analyst.get("model")))}' if by_llm else "règles (sans LLM)"),
        "ok" if by_llm else "warn",
    )

    signals = analyst.get("signals", [])
    insights = analyst.get("insights", [])
    cols = st.columns(4)
    kpi(cols[0], "Insights", len(insights))
    kpi(cols[1], "Priorité haute", sum(i.get("priority") == "high" for i in insights))
    kpi(cols[2], "Alertes détectées", sum(s.get("severity") == "alert" for s in signals))
    kpi(cols[3], "Insights à contrôler", sum(not i.get("verified") for i in insights))

    st.subheader("Synthèse")
    st.write(analyst.get("executive_summary", ""))

    st.subheader("Insights")
    prio_fr = {"high": "Haute", "medium": "Moyenne", "low": "Basse"}
    by_id = {s["id"]: s for s in signals}
    for i in insights:
        flag = "" if i.get("verified") else " · ⚠ à contrôler"
        with st.expander(
            f'[{prio_fr.get(i.get("priority"), i.get("priority"))}] {i.get("title")}{flag}',
            expanded=i.get("priority") == "high",
        ):
            st.markdown(f'**Constat :** {i.get("finding", "")}')
            if i.get("explanation"):
                st.markdown(f'**Explication :** {i.get("explanation")}')
            st.markdown(f'**Recommandation :** {i.get("recommendation", "")}')
            if i.get("ungrounded_numbers"):
                st.caption("Nombres non retrouvés dans les faits : " + ", ".join(i["ungrounded_numbers"]))
            if i.get("added_by"):
                st.caption(f'Ajouté par {i["added_by"]}')
            st.caption("Preuves : " + " | ".join(
                f'{sid} : {by_id[sid]["summary"]}' for sid in i.get("signal_ids", []) if sid in by_id))

    if analyst.get("watchlist"):
        st.subheader("À surveiller")
        for w in analyst["watchlist"]:
            st.markdown(f"- {w}")

    if analyst.get("caveats"):
        with st.expander("Précautions"):
            for c in analyst["caveats"]:
                st.markdown(f"- {c}")

    with st.expander(f"Signaux détectés ({len(signals)}) : traçabilité"):
        st.dataframe(
            pd.DataFrame([{"Id": s["id"], "Gravité": s["severity"], "Type": s["kind"], "Constat": s["summary"]}
                          for s in signals]),
            width="stretch", hide_index=True,
        )


# ============================================================
# PAGE — DASHBOARD BI (HTML généré par le Dashboard Agent)
# ============================================================

elif page == "Dashboard BI":
    header(
        "Dashboard BI",
        "Dashboard interactif généré par le Dashboard Agent : analyses du BI Analyst et exploration filtrable",
        "Dashboard",
    )

    help_box(
        "Ce dashboard est un fichier HTML autonome (data/dashboard/dashboard.html), régénéré à chaque "
        "pipeline. Onglet « Analyses BI » : synthèse, insights priorisés et graphiques des alertes. "
        "Onglet « Exploration » : filtres par période, État, catégorie et devise. "
        "Les graphiques nécessitent une connexion internet (bibliothèque Plotly)."
    )

    if not DASHBOARD_HTML.exists():
        st.warning("Dashboard absent. Clique sur « Générer le dashboard » dans « Lancer et suivre ».")
        st.stop()

    dash_html = DASHBOARD_HTML.read_text(encoding="utf-8")
    note(
        "Généré le "
        + html.escape(datetime.fromtimestamp(DASHBOARD_HTML.stat().st_mtime).strftime("%Y-%m-%d %H:%M"))
        + " · ouvrable aussi directement dans un navigateur (bouton ci-dessous).",
        "ok",
    )
    st.download_button("Télécharger le dashboard (HTML)", dash_html.encode("utf-8"),
                       file_name="biflow_dashboard.html", mime="text/html")
    components.html(dash_html, height=5400, scrolling=True)


# ============================================================
# PAGE 6 — LANCER ET SUIVRE
# ============================================================

elif page == "Lancer et suivre":
    header(
        "Lancer et suivre",
        "Exécuter les agents et consulter les décisions de l'Orchestrator",
        "Orchestrator",
    )

    help_box(
        "Ordre conseillé : Profiler, validation humaine, ETL. "
        "« Pipeline complet » enchaîne les agents. "
        "Les agents non construits peuvent être ignorés."
    )

    if orch:
        status = {
            x["agent"]: x["status"]
            for x in orch.get("steps", [])
        }

        st.write(
            "**Dernière exécution :**",
            orch.get("run_id", "n/a"),
            "· **Statut :**",
            orch.get("status", "n/a"),
        )

        st.write(
            "**Agents terminés :**",
            sum(
                1
                for x in orch.get("steps", [])
                if x.get("status") == "success"
            ),
        )

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Agent": AGENT_FR.get(
                            x.get("agent"),
                            x.get("agent"),
                        ),
                        "Statut": STATUS_FR.get(
                            x.get("status"),
                            x.get("status"),
                        ),
                        "Durée (s)": x.get("seconds"),
                        "Heure": x.get("at"),
                        "Raison / erreur": (
                            x.get("reason") or x.get("error", "")
                        ),
                    }
                    for x in orch.get("steps", [])
                ]
            ),
            width="stretch",
            hide_index=True,
        )

        if orch.get("decisions"):
            st.subheader("Décisions d'aiguillage")

            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Après": AGENT_FR.get(
                                d.get("agent"),
                                d.get("agent"),
                            ),
                            "Vers": d.get("next"),
                            "Décidé par": (
                                "LLM"
                                if d.get("by") == "llm"
                                else "Règle"
                            ),
                            "Raison": d.get("reason"),
                        }
                        for d in orch["decisions"]
                    ]
                ),
                width="stretch",
                hide_index=True,
            )

    else:
        note("Aucune exécution enregistrée. Lance un agent ci-dessous.")

    if audit:
        with st.expander(
            f'Dernier audit : {"conforme" if audit.get("passed") else "NON conforme"} '
            f'({audit.get("checks_total", 0)} contrôles)',
            expanded=not audit.get("passed", True),
        ):
            st.dataframe(
                pd.DataFrame(audit.get("checks", [])).rename(columns={
                    "area": "Domaine", "check": "Contrôle", "passed": "OK", "severity": "Gravité",
                    "detail": "Détail", "blame": "Étape en cause"}),
                width="stretch", hide_index=True,
            )

    reuse = st.checkbox(
        "Réutiliser la dernière décision du profiler (économise le quota LLM)",
        value=True,
    )

    no_llm = st.checkbox(
        "Orchestrator sans LLM (règles seulement)",
        value=False,
    )

    a, b, d, e, f, c = st.columns(6)
    task = None

    if f.button("Générer le dashboard", width="stretch"):
        task = ("Dashboard", ["agents/dashboard_agent.py"])

    if e.button("Lancer l'Analyste", width="stretch"):
        task = ("Analyste", ["agents/analyst_agent.py"] + (["--no-llm"] if no_llm else []))

    if d.button("Lancer les KPI", width="stretch"):
        task = ("KPI", ["agents/kpi_agent.py"])

    if a.button("Lancer le Profiler", width="stretch"):
        task = (
            "Profiler",
            ["agents/profiler_agent.py"]
            + (["--reuse"] if reuse else []),
        )

    if b.button("Lancer l'ETL", width="stretch"):
        task = ("ETL", ["agents/etl_agent.py"])

    if c.button(
        "Pipeline complet",
        type="primary",
        width="stretch",
    ):
        task = (
            "Orchestrator",
            ["agents/orchestrator.py"]
            + (["--reuse"] if reuse else [])
            + (["--no-llm"] if no_llm else []),
        )

    if task:
        label, args = task

        st.write(
            f"**{label}** en cours : `python {' '.join(args)}`"
        )

        box = st.empty()

        with st.spinner("Exécution..."):
            result = run_command(args, box)

        st.cache_data.clear()

        if result == 0:
            st.success(f"{label} terminé.")
            st.rerun()
        else:
            st.error(
                f"{label} a échoué (code {result}). "
                "Lis la sortie ci-dessus."
            )

    st.caption(
        "Les règles Python fixent les options autorisées ; "
        "le LLM choisit parmi elles dans les cas ambigus."
    )


# ============================================================
# PAGE — AUDIT & XAI (auditeur, test d'injection, explications, évaluation)
# ============================================================

elif page == "Audit & XAI":
    import plotly.graph_objects as go

    sys.path.insert(0, str(ROOT / "agents"))
    import auditor_agent
    import xai_engine

    AUDIT_DIR, EVAL_DIR = DATA / "audit", DATA / "eval"

    def facts_stamp():
        f = CURATED / "fact_sales.csv"
        return f.stat().st_mtime if f.exists() else 0

    # cache_resource : tables lues une seule fois (≈50 Mo), jamais modifiées par l'auditeur/XAI
    @st.cache_resource(show_spinner="Chargement des tables de faits...")
    def audit_facts(stamp):
        return auditor_agent.load_facts()

    @st.cache_resource(show_spinner=False)
    def xai_sales(stamp):
        return xai_engine.with_period(audit_facts(stamp)["sales"])

    header(
        "Audit & XAI",
        "Les KPI sont-ils justes, pourquoi ont-ils changé, et comment chaque agent s'en sort",
        "Contrôle",
    )

    help_box(
        "*Auditeur* : recalcule chaque KPI directement depuis les tables de faits "
        "et le compare à la valeur publiée par l'agent KPI (tolérance : arrondi à 0,01).\n\n"
        "*Test d'injection* : modifie une valeur KPI en mémoire pour vérifier que l'auditeur "
        "détecte l'erreur (aucun fichier n'est modifié).\n\n"
        "*XAI* : décompose la variation du chiffre d'affaires entre deux mois en effet volume, "
        "effet prix et effet combiné. La somme des effets égale exactement la variation totale.\n\n"
        "*Évaluation* : note chaque agent à partir de sa propre sortie après chaque run."
    )

    # ---------------- Lancement ----------------
    no_llm = st.checkbox("Sans LLM (explications par gabarit, plus rapide)", value=True)
    a, b = st.columns(2)
    task = None
    if a.button("Relancer l'auditeur", width="stretch"):
        task = ("Auditeur", ["agents/auditor_agent.py"] + (["--no-llm"] if no_llm else []))
    if b.button("Relancer l'évaluation", width="stretch"):
        task = ("Évaluation", ["agents/evaluation_agent.py"])
    if task:
        label, args = task
        box = st.empty()
        with st.spinner(f"{label} en cours..."):
            result = run_command(args, box)
        if result == 0:
            st.success(f"{label} terminé.")
            st.rerun()
        else:
            st.error(f"{label} a échoué (code {result}). Lis la sortie ci-dessus.")

    audit = load_json(AUDIT_DIR / "lineage_cards.json")
    xai = load_json(AUDIT_DIR / "xai_explanations.json") or {}
    evaluation = load_json(EVAL_DIR / "evaluation.json")

    if not audit:
        st.warning("Aucun audit disponible. Lance le pipeline complet ou l'auditeur ci-dessus.")
        st.stop()

    tab_audit, tab_test, tab_xai, tab_shap, tab_eval = st.tabs(
        ["1. Auditeur", "2. Test d'injection", "3. XAI : variation du CA",
         "4. XAI : modèle ML (SHAP)", "5. Évaluation"]
    )

    status_fr = {"ok": "✅ OK", "ecart": "❌ Écart", "non_rapporte": "⚠️ Non rapporté",
                 "erreur_recalcul": "⚠️ Erreur"}

    def cards_table(cards):
        return pd.DataFrame([{
            "Statut": status_fr.get(c["status"], c["status"]),
            "KPI": c.get("label", c["id"]),
            "Rapporté (agent KPI)": str(c.get("reported")),
            "Recalculé (auditeur)": str(c.get("recomputed", "—")),
            "Écart": str(c.get("delta") if c.get("delta") is not None else "—"),
            "Formule": c.get("formula", ""),
        } for c in cards])

    # ---------------- 1. Auditeur ----------------
    with tab_audit:
        verdict = audit.get("verdict", {})
        recon = audit.get("reconciliations", [])
        cards = audit.get("cards", [])
        st.caption(f"Audit généré le {audit.get('generated_at', '—')}")

        if verdict.get("passed"):
            note(f"<b>CONFORME</b> : {verdict.get('audited')} KPI recalculés, aucun écart.", "ok")
        else:
            note(
                f"<b>NON CONFORME</b> : {verdict.get('mismatches', 0)} écart(s), "
                f"{verdict.get('audited', 0)} KPI audités. "
                f"Cause probable : {verdict.get('cause') or '—'}.",
                "bad",
            )

        cols = st.columns(4)
        kpi(cols[0], "KPI audités", fmt(verdict.get("audited", 0)))
        kpi(cols[1], "Écarts", fmt(verdict.get("mismatches", 0)))
        kpi(cols[2], "Non rapportés", fmt(verdict.get("not_audited", 0)))
        kpi(cols[3], "Réconciliations", f"{sum(r['passed'] for r in recon)}/{len(recon)}",
            "contrôles niveau ETL")

        st.dataframe(cards_table(cards), width="stretch", hide_index=True)

        st.markdown("*Réconciliations des tables de faits*")
        for r in recon:
            note(f"{'OK' if r['passed'] else 'ÉCHEC'} — {r['check']} ({r['detail']})",
                 "ok" if r["passed"] else "bad")

        st.markdown("*Carte de lignage par KPI*")
        for c in cards:
            with st.expander(f"{status_fr.get(c['status'], c['status'])} · {c.get('label', c['id'])}"):
                if c.get("explanation"):
                    st.write(c["explanation"])
                st.markdown(f"- Chemin dans kpi_results.json : `{c['id']}`\n"
                            f"- Formule : `{c.get('formula', '—')}`\n"
                            f"- Table source : {c.get('source', '—')}")
                if c.get("lineage"):
                    st.json(c["lineage"])
                if c.get("reason") or c.get("error"):
                    note(c.get("reason") or c.get("error"), "warn")

        with st.expander("Rapport brut (audit_report.md)"):
            report = AUDIT_DIR / "audit_report.md"
            st.markdown(report.read_text(encoding="utf-8") if report.exists() else "—")

    # ---------------- 2. Test d'injection ----------------
    with tab_test:
        st.markdown(
            "Fausse une valeur publiée par l'agent KPI et vérifie que l'auditeur la détecte. "
            "Le test se fait *en mémoire* : kpi_results.json n'est pas modifié."
        )
        if not kpi_data:
            st.warning("Résultats KPI absents. Lance le pipeline complet.")
        else:
            labels = {path: spec[0] for path, spec in auditor_agent.REGISTRY.items()}
            path = st.selectbox("KPI à fausser", list(labels), format_func=labels.get)
            current = auditor_agent.get_path(kpi_data, path)
            if isinstance(current, str):
                fake = st.text_input("Valeur falsifiée", value="toys" if current != "toys" else "auto")
            else:
                fake = st.number_input("Valeur falsifiée", value=float(current or 0) * 1.05,
                                       format="%.4f")
            st.caption(f"Valeur réellement publiée : {current}")

            if st.button("Lancer l'audit sur la valeur falsifiée", type="primary"):
                tampered = json.loads(json.dumps(kpi_data))   # copie profonde
                node = tampered
                keys = path.split(".")
                for k in keys[:-1]:
                    node = node[int(k)] if isinstance(node, list) else node[k]
                node[keys[-1]] = fake
                result = auditor_agent.build_cards(tampered, audit_facts(facts_stamp()))
                card = next(c for c in result if c["id"] == path)
                if card["status"] == "ecart":
                    note(f"✅ Erreur détectée : l'agent KPI annonce <b>{card['reported']}</b>, "
                         f"l'auditeur recalcule <b>{card['recomputed']}</b> "
                         f"(écart {card.get('delta', '—')}). Verdict : NON CONFORME, cause = kpi.", "ok")
                else:
                    note(f"La valeur {fake} reste dans la tolérance (±{auditor_agent.ABS_TOL}) : "
                         "pas d'écart détecté.", "warn")
                st.dataframe(cards_table(result), width="stretch", hide_index=True)

    # ---------------- 3. XAI ----------------
    with tab_xai:
        sales_p = xai_sales(facts_stamp())
        periods = sorted(sales_p["period"].unique())
        if len(periods) < 2:
            note("Moins de deux mois exploitables dans fact_sales.", "warn")
        else:
            dims = {"category": "Catégorie", "customer_state": "État du client",
                    "seller_state": "État du vendeur"}
            c1, c2, c3 = st.columns(3)
            prev = c1.selectbox("Mois de référence", periods, index=len(periods) - 2)
            curr = c2.selectbox("Mois comparé", periods, index=len(periods) - 1)
            dim = c3.selectbox("Expliquer par", list(dims), format_func=dims.get)

            if prev == curr:
                note("Choisis deux mois différents.", "warn")
            else:
                bridge = xai_engine.pvm_bridge(sales_p, prev, curr, dim)
                eff = bridge["effects"]
                note(xai_engine.explain_sentence(bridge))

                cols = st.columns(4)
                kpi(cols[0], f"CA {prev}", money(bridge["revenue_prev"]))
                kpi(cols[1], f"CA {curr}", money(bridge["revenue_curr"]))
                kpi(cols[2], "Variation", money(bridge["total_delta"]), pct(bridge["pct_change"]))
                kpi(cols[3], "Réconciliation", "OK" if bridge["reconciles"] else "ÉCHEC",
                    "somme des effets = variation")

                fig = go.Figure(go.Waterfall(
                    x=[f"CA {prev}", "Effet volume", "Effet prix", "Effet combiné", f"CA {curr}"],
                    measure=["absolute", "relative", "relative", "relative", "total"],
                    y=[bridge["revenue_prev"], eff["volume"], eff["price"], eff["interaction"], 0],
                    increasing={"marker": {"color": "#2f855a"}},
                    decreasing={"marker": {"color": "#c53030"}},
                    totals={"marker": {"color": "#1f4e8c"}},
                    connector={"line": {"color": "#cdd3de"}},
                ))
                fig.update_layout(title="Pont prix-volume-mix du chiffre d'affaires (BRL)",
                                  showlegend=False)
                st.plotly_chart(style_fig(fig, 380), width="stretch")

                movers = xai_engine.top_movers(bridge, n=5)
                mv = pd.DataFrame(movers["up"] + movers["down"]).drop_duplicates(dim).sort_values("delta")
                fig = px.bar(
                    mv, x="delta", y=dim, orientation="h", color=mv["delta"] > 0,
                    color_discrete_map={True: "#2f855a", False: "#c53030"},
                    hover_data=["volume_effect", "price_effect", "interaction_effect",
                                "units_prev", "units_curr", "avg_price_prev", "avg_price_curr"],
                    title=f"{dims[dim]} : segments qui expliquent le plus la variation (BRL)",
                    labels={"delta": "Variation (BRL)", dim: ""},
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(style_fig(fig), width="stretch")

                with st.expander(f"Détail complet par {dims[dim].lower()} ({len(bridge['by_segment'])} lignes)"):
                    st.dataframe(
                        pd.DataFrame(bridge["by_segment"]).sort_values("delta").rename(columns={
                            dim: dims[dim], "rev_prev": f"CA {prev}", "rev_curr": f"CA {curr}",
                            "delta": "Variation", "volume_effect": "Effet volume",
                            "price_effect": "Effet prix", "interaction_effect": "Effet combiné",
                            "units_prev": f"Unités {prev}", "units_curr": f"Unités {curr}",
                            "avg_price_prev": f"Prix moyen {prev}", "avg_price_curr": f"Prix moyen {curr}",
                        }),
                        width="stretch", hide_index=True,
                    )
                st.latex(r"\Delta CA = \underbrace{(u_1-u_0)\,p_0}_{volume}"
                         r" + \underbrace{(p_1-p_0)\,u_0}_{prix}"
                         r" + \underbrace{(p_1-p_0)(u_1-u_0)}_{combin\acute{e}}")

        with st.expander("Sortie XAI brute de l'auditeur (xai_explanations.json)"):
            st.json(xai)

    # ---------------- 4. XAI : modèle ML (SHAP) ----------------
    with tab_shap:
        st.markdown(
            "*Question métier :* quelles commandes risquent de recevoir une **mauvaise note "
            "(1 ou 2 étoiles)**, et *pourquoi* ? Un modèle de gradient boosting est entraîné sur "
            "les commandes livrées ; *SHAP* mesure la contribution de chaque variable à chaque "
            "prédiction (contributions additives : base + somme des contributions = prédiction)."
        )

        if not xai_engine.MODEL_PATH.exists():
            note("Modèle pas encore entraîné.", "warn")
            if st.button("Entraîner le modèle", type="primary"):
                with st.spinner("Entraînement (~10 s)..."):
                    xai_engine.train_review_model(force=True)
                st.rerun()
        else:
            @st.cache_resource(show_spinner="Calcul des valeurs SHAP...")
            def shap_model(stamp):
                import joblib
                import numpy as np
                import shap
                model = joblib.load(xai_engine.MODEL_PATH)
                sample = pd.read_csv(xai_engine.SAMPLE_PATH)
                feats = list(model.feature_names_in_)
                explainer = shap.TreeExplainer(model)
                sv = np.asarray(explainer.shap_values(sample[feats]))
                return sample, feats, sv, float(np.ravel(explainer.expected_value)[0])

            import numpy as np
            sample, feats, sv, base = shap_model(xai_engine.MODEL_PATH.stat().st_mtime)
            metrics = load_json(xai_engine.METRICS_PATH) or {}
            names = [xai_engine.FEATURES_FR.get(f, f) for f in feats]
            proba = lambda logodds: 1 / (1 + np.exp(-logodds))

            cols = st.columns(4)
            kpi(cols[0], "AUC (test)", f"{metrics.get('auc_test', 0):.3f}".replace(".", ","),
                "0,5 = hasard · 1 = parfait")
            kpi(cols[1], "Mauvaises notes", pct(metrics.get("base_rate", 0) * 100), "taux réel")
            kpi(cols[2], "Commandes d'entraînement", fmt(metrics.get("n_train")),
                f"{fmt(metrics.get('n_test'))} en test")
            kpi(cols[3], "Commandes expliquées", fmt(len(sample)), "échantillon de test")

            shap_info = xai.get("model_explanations", {})
            if shap_info.get("sentence"):
                note(shap_info["sentence"])

            # --- explication globale
            imp = pd.DataFrame({"Variable": names, "Importance": np.abs(sv).mean(axis=0)}) \
                .sort_values("Importance")
            fig = px.bar(imp, x="Importance", y="Variable", orientation="h",
                         title="Importance globale : impact moyen |SHAP| sur le risque (log-odds)",
                         labels={"Variable": ""})
            st.plotly_chart(style_fig(fig, 360), width="stretch")

            # --- beeswarm : chaque point = une commande ; couleur = valeur de la variable
            order = list(imp["Variable"])
            rng = np.random.default_rng(0)
            swarm = pd.concat([pd.DataFrame({
                "SHAP": sv[:, i],
                "y": order.index(names[i]) + rng.uniform(-0.3, 0.3, len(sample)),
                "Valeur": sample[f],
                "Valeur relative": sample[f].rank(pct=True),
                "Variable": names[i],
            }) for i, f in enumerate(feats)])
            fig = px.scatter(swarm, x="SHAP", y="y", color="Valeur relative",
                             color_continuous_scale=["#1f4e8c", "#cdd3de", "#c53030"],
                             hover_data={"Variable": True, "Valeur": True, "y": False,
                                         "Valeur relative": False},
                             title="Effet de chaque variable par commande "
                                   "(rouge = valeur élevée, bleu = faible ; à droite = plus de risque)")
            fig.update_traces(marker={"size": 4, "opacity": 0.6})
            fig.update_yaxes(tickvals=list(range(len(order))), ticktext=order, title="")
            fig.add_vline(x=0, line_color="#8a94a6")
            st.plotly_chart(style_fig(fig, 420), width="stretch")

            # --- explication locale
            st.markdown("*Expliquer une commande précise*")
            ranked = sample.reset_index(drop=True).sort_values("proba_bad", ascending=False)
            pick = st.selectbox(
                "Commande (triées par risque prédit)", ranked.index.tolist(),
                format_func=lambda i: (f"{ranked.at[i, 'order_id'][:12]}… · risque "
                                       f"{ranked.at[i, 'proba_bad']:.0%} · note réelle "
                                       f"{'mauvaise' if ranked.at[i, 'bad_review'] else 'bonne'}"),
            )
            row = sample.iloc[pick]
            contrib = pd.DataFrame({
                "Variable": [f"{n} = {row[f]:g}" for n, f in zip(names, feats)],
                "Contribution": sv[pick],
            }).sort_values("Contribution", key=abs, ascending=True)

            cols = st.columns(3)
            kpi(cols[0], "Risque moyen (base)", pct(proba(base) * 100), "avant de regarder la commande")
            kpi(cols[1], "Risque prédit", pct(proba(base + sv[pick].sum()) * 100), "pour cette commande")
            kpi(cols[2], "Note réelle", "mauvaise (1-2★)" if row["bad_review"] else "bonne (3-5★)")

            fig = px.bar(contrib, x="Contribution", y="Variable", orientation="h",
                         color=contrib["Contribution"] > 0,
                         color_discrete_map={True: "#c53030", False: "#2f855a"},
                         title="Pourquoi ce risque ? (rouge = augmente, vert = diminue ; log-odds)",
                         labels={"Variable": ""})
            fig.update_layout(showlegend=False)
            st.plotly_chart(style_fig(fig, 360), width="stretch")

            top = contrib.sort_values("Contribution", ascending=False).head(2)
            note("Principales raisons du risque pour cette commande : "
                 + " ; ".join(f"<b>{v}</b> ({c:+.2f})" for v, c in zip(top["Variable"], top["Contribution"]))
                 + ". SHAP explique le <i>modèle</i> : ce sont des associations, pas des causes prouvées.")

    # ---------------- 5. Évaluation ----------------
    with tab_eval:
        if not evaluation:
            note("Aucune évaluation disponible. Lance le pipeline ou le bouton « Relancer l'évaluation ».", "warn")
        else:
            o = evaluation.get("orchestrator", {})
            score = evaluation.get("pipeline_score")
            st.caption(f"Évaluation générée le {evaluation.get('generated_at', '—')}")
            cols = st.columns(4)
            kpi(cols[0], "Score global", pct(score * 100) if score is not None else "n/a",
                "moyenne des agents notés")
            kpi(cols[1], "Statut du run", o.get("final_status", "n/a"))
            kpi(cols[2], "Agents réussis", f"{o.get('agents_success', 0)}/{o.get('agents_executed', 0)}",
                f"{o.get('agents_skipped', 0)} ignorés")
            kpi(cols[3], "Durée totale", f"{o.get('total_runtime_s', 0)} s",
                f"{o.get('decisions_by_llm', 0)} décision(s) LLM")

            rows = []
            for name, ag in evaluation.get("per_agent", {}).items():
                step = o.get("per_step", {}).get(name, {})
                q = ag.get("quality_score")
                rows.append({
                    "Agent": AGENT_FR.get(name, name),
                    "Statut": STATUS_FR.get(step.get("status"), step.get("status") or "—"),
                    "Durée (s)": step.get("seconds"),
                    "Qualité": pct(q * 100) if q is not None else "non mesurable",
                })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

            steps = pd.DataFrame([{"Agent": AGENT_FR.get(k, k), "Durée (s)": v.get("seconds") or 0}
                                  for k, v in o.get("per_step", {}).items()])
            if not steps.empty:
                fig = px.bar(steps, x="Agent", y="Durée (s)", title="Temps d'exécution par agent")
                st.plotly_chart(style_fig(fig, 300), width="stretch")

            st.caption("Qualité : part des contrôles réussis (ETL, KPI) ou des KPI sans écart "
                       "(auditeur). Le profiler ne produit pas de score chiffré.")

            for name, ag in evaluation.get("per_agent", {}).items():
                with st.expander(f"Métriques détaillées · {AGENT_FR.get(name, name)}"):
                    st.json(ag)

            if orch and orch.get("decisions"):
                with st.expander("Décisions d'aiguillage de l'orchestrateur"):
                    st.dataframe(pd.DataFrame([{
                        "Après": AGENT_FR.get(d["agent"], d["agent"]),
                        "Vers": d["next"],
                        "Par": "LLM" if d.get("by") == "llm" else "règle",
                        "Raison": d.get("reason", ""),
                    } for d in orch["decisions"]]), width="stretch", hide_index=True)

            with st.expander("Rapport brut (evaluation_report.md)"):
                report = EVAL_DIR / "evaluation_report.md"
                st.markdown(report.read_text(encoding="utf-8") if report.exists() else "—")