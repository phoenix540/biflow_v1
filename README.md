# BIFlow — pipeline BI multi-agents

BIFlow transforme des fichiers CSV bruts en indicateurs, analyses et recommandations, grâce à une chaîne
d'agents orchestrée par **LangGraph**. Chaque étape est tracée et contrôlée, et un humain valide les décisions
sensibles (traductions de catégories).

Le projet a deux niveaux :

- **Socle générique** : Profiler, ETL et KPI fonctionnent sur **n'importe quel dossier de CSV**. Aucune
  colonne ni règle métier n'y est écrite en dur : l'ETL déduit le rôle de chaque colonne, les clés, les jointures
  et le type de modèle (étoile, flocon, constellation) ; l'agent KPI planifie ses indicateurs sur ce modèle.
- **Couche métier Olist** (marketplace brésilienne, 2016-2018) : ajoute les règles qu'aucun algorithme
  générique ne peut deviner (CA = commandes livrées, retard comparé en jours, coût et marge, calendrier des
  jours fériés) et alimente l'Analyste, le Dashboard, l'Auditeur, le moteur XAI et l'application manager.

```
                 SOCLE GÉNÉRIQUE (tout jeu de CSV)          COUCHE MÉTIER OLIST
données brutes ─► Profiler ─► ETL ─► KPI ─► Modèle Olist ─► KPI Olist ─► Analyste ─► Dashboard ─► Auditeur ─► fin
                                             │ (non applicable si autres données : fin du run)       │
                                             ◄────────── relance du modèle ou des KPI si l'audit échoue ┘
```

## Agents

| Étape | Fichier | LLM | Rôle | Sorties |
|---|---|---|---|---|
| Orchestrator | `agents/orchestrator.py` | optionnel | Enchaîne les agents (graphe LangGraph), décide des réessais et des relances après audit. Python fixe les options autorisées, le LLM choisit parmi elles. | `data/profile/orchestrator_log.json`, `orchestrator_report.md` |
| Profiler | `agents/profiler.py` + `profiler_agent.py` | oui (repli par règles) | Profilage statistique (manquants, doublons, atypiques, relations, score de fiabilité) puis diagnostic du LLM : problèmes priorisés, plan pour l'ETL, traductions proposées, données manquantes. Chaque chiffre cité est vérifié. Sans LLM : décision déterministe tirée du profil. | `data/profile/` |
| ETL générique | `agents/etl_agent.py` | veto facultatif | Analyse le rôle de chaque colonne, construit un plan dans un catalogue fermé d'opérations (dédoublonnage, typage, imputation, valeurs atypiques signalées), découvre clés et jointures, valide (3 niveaux de prudence), déduit le modèle. Le LLM peut seulement opposer un veto au plan et proposer une lecture métier du modèle, à valider. | `data/clean/`, `data/curated/analytical_base.csv`, `data/etl/model.json`, `etl_agent_decision.json`, `etl_report.md` |
| KPI générique | `agents/kpi_agent.py` | planificateur (repli par règles) | Le LLM conçoit un plan de KPI (indicateurs, tendances, découpages, comparaisons) dans un catalogue validé par Pydantic ; Python l'exécute de façon déterministe, valide chaque résultat et redemande un plan en cas d'erreur. | `data/kpi/generic_kpi_results.json`, `generic_kpi_plan.json`, `generic_kpi_report.md` |
| Modèle métier Olist | `agents/olist_model_agent.py` | revue facultative | Modèle en étoile métier : `fact_sales`, `fact_orders`, `dim_*`. Applique les traductions validées par un humain, journalise chaque correction. « Non applicable » si les fichiers Olist sont absents. | `data/curated/`, `data/etl/olist_model_decision.json`, `olist_model_report.md` |
| KPI métier Olist | `agents/olist_kpi_agent.py` | non | 12 indicateurs officiels aux définitions fixées + découpages (mois, États, catégories, vendeurs, produits, événements, jours). | `data/kpi/kpi_results.json`, `kpi_report.md` |
| BI Analyst | `agents/analyst_agent.py` | oui (repli par règles) | Python détecte des signaux chiffrés (tendances, baisses, anomalies, régions…), le LLM les transforme en insights et recommandations en citant ses signaux ; nombres vérifiés, aucune alerte omise. | `data/analyst/` |
| Dashboard | `agents/dashboard_agent.py` | non | Dashboard HTML autonome : onglet « Analyses BI » (synthèse, insights et preuves, croissance, anomalies, catégories, États, événements, vendeurs) et onglet « Exploration » (filtres période / État / catégorie / devise). Contrôle que ses totaux = modèle et KPI. | `data/dashboard/dashboard.html` |
| Auditor XAI | `agents/auditor_agent.py` + `xai_engine.py` | explications facultatives | Recalcule 13 KPI depuis les tables de faits (cartes de lignage : formule, source, rapporté vs recalculé), réconcilie avec les **CSV bruts**, contrôle l'Analyste et le Dashboard. XAI : pont prix-volume-mix de la variation du CA et modèle « mauvaise note » expliqué par **SHAP**. `passed=false` déclenche une relance du modèle ou des KPI métier. | `data/audit/`, `data/xai/` |
| Evaluation | `agents/evaluation_agent.py` | non | Lancé par l'orchestrateur en fin de run : statut, durée et score de qualité de chaque étape, score global du pipeline. | `data/eval/` |

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

Placer les CSV dans un sous-dossier de `data/raw/` (pour Olist : `data/raw/olist/`, 9 fichiers Olist +
`br_holidays.csv`). Le Profiler et l'ETL lisent le sous-dossier qui contient le plus de CSV (les dossiers
`backup`, `archive`, `old` sont ignorés) ; on peut aussi l'indiquer avec `--data-dir=chemin` ou la variable
`BIFLOW_RAW_DIR`.

Dans VS Code, choisir l'interpréteur `.venv\Scripts\python.exe` (sinon l'éditeur signale des imports introuvables).

## Utilisation

```powershell
# Application de pilotage (avancement, validation humaine, KPI, analyses, dashboard, audit)
.venv\Scripts\python.exe -m streamlit run app.py          # http://localhost:8501

# Olist Pilot : application des managers (ventes, livraison, vendeurs, scénarios, assistant)
.venv\Scripts\python.exe -m streamlit run manager_app.py --server.port 8502   # http://localhost:8502

# Pipeline complet
.venv\Scripts\python.exe agents\orchestrator.py --reuse   # réutilise la décision du Profiler si elle est à jour
.venv\Scripts\python.exe agents\orchestrator.py --no-llm  # sans aucun appel LLM (replis déterministes)

# Une ou plusieurs étapes, dans l'ordre
.venv\Scripts\python.exe agents\run_steps.py etl olist_model
.venv\Scripts\python.exe agents\run_steps.py kpi olist_kpi --no-llm

# Socle générique sur un autre jeu de données
.venv\Scripts\python.exe agents\run_steps.py etl kpi --data-dir=C:\chemin\vers\mes_csv

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
| Clients | Fidélisation, rétention par cohorte, segmentation RFM avec actions conseillées, fidélité par État et par 1re catégorie |
| Vendeurs | Tableau de bord des vendeurs (alertes), fiche vendeur au clic, « plan d'action » envoyé au chatbot |
| Produits | Concentration du CA (Pareto), best-sellers unités vs CA, qualité des fiches (photos, description), poids vs fret |
| Paiements | Mix des moyens de paiement et son évolution, mensualités vs panier, annulations et notes par moyen, boleto par État |
| Carte | Densité du CA par code postal (clients ou vendeurs), villes clés, flux vendeur → client entre États (délai, retard) |
| Scénarios | « Et si ? » : moins de retards, plus de fidélisation, commandes récupérées (hypothèses affichées) |

**Chatbot Pilot** (panneau à droite de chaque page, masquable) : il répond sur le périmètre filtré à l'écran
en appelant les agents du pipeline via `manager/agents_bridge.py`. Chaque réponse affiche ses sources :

| Agent | Utilisation par le chatbot |
|---|---|
| KPI métier | `olist_kpi_agent.filter_frames` + `olist_kpi_agent.compute` : indicateurs officiels sur le périmètre |
| BI Analyst | `analyst_agent.detect_*` sur ces KPI : alertes, anomalies, opportunités ; insights publiés |
| XAI | `xai_engine.explain_kpi_change` (pourquoi le CA a changé : prix / volume / mix), `shap_attributions` (facteurs de mauvaise note), modèle `review_model.joblib` (risque d'une commande ou d'un scénario) |

Le LLM ne calcule rien : il choisit les outils et rédige. Sans LLM, un mode règles appelle les mêmes outils.

**Périmètre** (`manager/scope_guard.py`) : Pilot ne répond qu'aux questions sur l'activité d'Olist. Chaque
question passe d'abord par un classement déterministe : tentative de détournement des consignes et sujets
étrangers (météo, recettes, code, politique…) → refus poli avec exemples de questions, sans appeler aucun agent ;
salutations → réponse courte ; vocabulaire métier → traitement normal. Les questions ambiguës sont confiées au
LLM, dont la consigne définit le périmètre (il répond `HORS_PERIMETRE` s'il les juge hors sujet) ; sans LLM,
elles sont refusées.

Calculs des pages Clients, Produits, Paiements et Carte : `manager/insights.py` (aussi exposés au chatbot :
`segments_clients`, `analyse_paiements`, `concentration_produits`, `flux_livraison`).

Tests : `tests/test_manager.py`, `tests/test_agents_bridge.py`, `tests/test_kpi_filters.py`,
`tests/test_scope_guard.py`, `tests/test_manager_insights.py` ; socle générique : `tests/test_generic_etl_kpi.py` (boutique fictive, sans
aucune colonne Olist).

## Limites connues

- Le coût de production (`production_cost_brl`) est **simulé** (~70 % du prix) : la marge n'est pas exploitable.
- Le taux BRL → TND (`etl_config.json`) est indicatif : sa date de référence reste à renseigner (`brl_to_tnd_date`).
- Les règles métier du Profiler et toute la couche métier sont propres à Olist ; sur un autre jeu de données,
  le pipeline s'arrête proprement après le socle générique (profil, ETL, KPI).
