# BIFlow — pipeline BI multi-agents (Olist)

BIFlow transforme les données brutes de la marketplace brésilienne **Olist** (2016-2018) en indicateurs,
analyses et recommandations, grâce à une chaîne d'agents orchestrée par **LangGraph**. Chaque étape est
tracée, contrôlée, et un humain valide les décisions sensibles (traductions de catégories).

```
données brutes ─► Profiler ─► ETL ─► KPI ─► BI Analyst ─► Dashboard ─► Auditor ─► fin
 (10 CSV)        (qualité)  (modèle) (indicateurs) (insights)  (HTML)     (contrôle)
                                                                  │
                         ◄──── relance KPI ou ETL si l'audit échoue ────┘
Application Streamlit (app.py) : pilotage, validation humaine, KPI, analyses et dashboard intégré.
```

## Agents

| Agent | Fichier | LLM | Rôle | Sorties |
|---|---|---|---|---|
| Orchestrator | `agents/orchestrator.py` | optionnel | Enchaîne les agents (graphe LangGraph), décide des réessais et des relances après audit. Python fixe les options autorisées, le LLM choisit parmi elles. | `data/profile/orchestrator_log.json`, `orchestrator_report.md` |
| Profiler | `agents/profiler.py` + `profiler_agent.py` | oui | Profilage statistique (manquants, doublons, atypiques, relations, règles métier, score de fiabilité), puis diagnostic du LLM : problèmes priorisés, plan pour l'ETL, traductions proposées, données manquantes. Chaque chiffre cité est vérifié. | `data/profile/` |
| ETL | `agents/etl_agent.py` | revue facultative | Nettoyage déterministe et modèle en étoile : `fact_sales`, `fact_orders`, `dim_*`. Applique les traductions validées par un humain, journalise chaque correction. | `data/curated/`, `data/etl/` |
| KPI | `agents/kpi_agent.py` | non | 12 indicateurs clés + découpages (mois, États, catégories, vendeurs, produits, événements, jours). | `data/kpi/` |
| BI Analyst | `agents/analyst_agent.py` | oui (repli par règles) | Python détecte des signaux chiffrés (tendances, baisses, anomalies, régions…), le LLM les transforme en insights et recommandations en citant ses signaux ; nombres vérifiés, aucune alerte omise. | `data/analyst/` |
| Dashboard | `agents/dashboard_agent.py` | non | Génère un dashboard HTML autonome. Onglet « Analyses BI » : synthèse, insights priorisés avec leurs preuves, croissance sur un an, CA quotidien avec anomalies et événements, catégories en hausse/baisse, États (retard vs note), événements, vendeurs, jours. Onglet « Exploration » : filtres période / État / catégorie / devise. Contrôle que ses totaux = ETL et KPI. | `data/dashboard/dashboard.html` |
| Auditor XAI | `agents/auditor_agent.py` + `xai_engine.py` | explications facultatives | Recalcule 13 KPI depuis les tables de faits (cartes de lignage : formule, source, rapporté vs recalculé), réconcilie avec les **CSV bruts** (cause ETL), contrôle l'Analyste et le Dashboard. XAI : pont prix-volume-mix de la variation du CA et modèle « mauvaise note » expliqué par **SHAP**. `passed=false` déclenche une relance KPI ou ETL. | `data/audit/`, `data/xai/` |
| Evaluation | `agents/evaluation_agent.py` | non | Lancé par l'orchestrateur en fin de run : statut, durée et score de qualité de chaque agent, score global du pipeline. | `data/eval/` |

## Installation

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Créer un fichier `.env` à la racine (jamais versionné) :

```
LLM_MODEL=google_genai:gemini-3.5-flash-lite
GOOGLE_API_KEY=<votre clé>
```

Placer les CSV dans `data/raw/olist/` (9 fichiers Olist + `br_holidays.csv`).

## Utilisation

```powershell
# Application (tableau de bord + lancement des agents)
.venv\Scripts\python.exe -m streamlit run app.py          # http://localhost:8501

# Olist Pilot : application des managers (briefing, ventes, livraison, vendeurs, scénarios, assistant)
.venv\Scripts\python.exe -m streamlit run manager_app.py --server.port 8502   # http://localhost:8502

# Pipeline complet
.venv\Scripts\python.exe agents\orchestrator.py --reuse   # réutilise les décisions LLM encore valides
.venv\Scripts\python.exe agents\orchestrator.py --no-llm  # sans aucun appel LLM

# Un agent seul
.venv\Scripts\python.exe agents\kpi_agent.py
.venv\Scripts\python.exe agents\analyst_agent.py --no-llm

# Tests
.venv\Scripts\python.exe -m pytest tests
```

## Olist Pilot (application manager)

`app.py` sert à piloter le pipeline ; `manager_app.py` s'adresse aux équipes métier d'Olist.

Pages (filtres communs dans la barre latérale : période, États, catégories) :

| Page | Contenu |
|---|---|
| Ventes | CA mensuel, catégories (treemap coloré par croissance), effet des événements |
| Livraison & clients | Note selon le retard, carte des retards par État, évolution mensuelle |
| Vendeurs | Tableau de bord des vendeurs (alertes), fiche vendeur au clic, « plan d'action » envoyé au chatbot |
| Scénarios | « Et si ? » : moins de retards, plus de fidélisation, commandes récupérées (hypothèses affichées) |

**Chatbot Pilot** (panneau à droite de chaque page, masquable) : il répond sur le périmètre filtré à l'écran
en appelant les agents du pipeline via `manager/agents_bridge.py`. Chaque réponse affiche ses sources :

| Agent | Utilisation par le chatbot |
|---|---|
| Agent KPI | `kpi_agent.filter_frames` + `kpi_agent.compute` : indicateurs officiels sur le périmètre |
| BI Analyst | `analyst_agent.detect_*` sur ces KPI : alertes, anomalies, opportunités ; insights publiés |
| XAI | `xai_engine.explain_kpi_change` (pourquoi le CA a changé : prix / volume / mix), `shap_attributions` (facteurs de mauvaise note), modèle `review_model.joblib` (risque d'une commande ou d'un scénario) |

Le LLM ne calcule rien : il choisit les outils et rédige. Sans LLM, un mode règles appelle les mêmes outils.
Tests : `tests/test_manager.py`, `tests/test_agents_bridge.py`, `tests/test_kpi_filters.py`.
