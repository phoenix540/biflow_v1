# BIFlow — pipeline BI multi-agents

BIFlow transforme des fichiers CSV bruts en indicateurs, analyses et recommandations, grâce à une chaîne
d'agents orchestrée par **LangGraph**. Chaque étape est tracée et contrôlée, les chiffres sont toujours calculés
par Python (jamais par le LLM), et un humain valide les décisions sensibles (traductions de catégories).

Deux applications Streamlit l'accompagnent :

- **`app.py`** : pilotage du pipeline (qualité des données, validation humaine, KPI, analyses, audit, lancement des agents) ;
- **`manager_app.py` (Olist Pilot)** : tableaux de bord pour les équipes métier, avec un assistant conversationnel.

## Architecture

Le projet a deux niveaux :

- **Socle générique** : l'ETL et l'agent KPI fonctionnent sur **n'importe quel dossier de CSV**, sans aucun nom
  de colonne écrit en dur. L'ETL déduit le rôle de chaque colonne, les clés, les jointures et le type de modèle
  (étoile, flocon, constellation) ; l'agent KPI planifie ses indicateurs sur ce modèle. Le Profiler calcule ses
  statistiques sur tout jeu de CSV (ses règles métier, elles, sont propres à Olist).
- **Couche métier Olist** (marketplace brésilienne, 2016-2018) : ajoute les règles qu'aucun algorithme générique
  ne peut deviner (CA = commandes livrées, retard comparé en jours, coût et marge, calendrier des jours fériés) et
  alimente le BI Analyst, le Dashboard, l'Auditeur, le moteur XAI et l'application manager.

```
SOCLE GÉNÉRIQUE      Profiler ──► ETL ──► KPI
                                            │
COUCHE MÉTIER OLIST                         ▼
                     Modèle Olist ──► KPI Olist ──► BI Analyst ──► Dashboard ──► Auditeur ──► Évaluation
                          ▲               ▲                                          │
                          └───────────────┴──── relance si l'audit n'est pas conforme ┘
```

Si les fichiers Olist sont absents, le modèle métier se déclare « non applicable » et le run se termine
proprement après le socle générique.

## Agents

| Étape | Fichier | LLM | Rôle | Sorties |
|---|---|---|---|---|
| Orchestrator | `agents/orchestrator.py` | optionnel | Enchaîne les agents (graphe LangGraph), décide des réessais et des relances après audit. Python fixe les options autorisées, le LLM choisit parmi elles. | `data/profile/orchestrator_log.json`, `orchestrator_report.md` |
| Profiler | `agents/profiler.py` + `profiler_agent.py` | oui (repli par règles) | Profilage statistique (manquants, doublons, atypiques, relations, règles métier, score de fiabilité), puis diagnostic du LLM : problèmes priorisés, plan pour l'ETL, traductions proposées, données manquantes. Chaque chiffre cité est vérifié. Sans LLM : décision déterministe tirée du profil. | `data/profile/` |
| ETL générique | `agents/etl_agent.py` | veto facultatif | Analyse le rôle de chaque colonne, construit un plan dans un catalogue fermé d'opérations (dédoublonnage, typage, imputation, atypiques signalés), découvre clés et jointures, valide le résultat (3 niveaux de prudence) et déduit le modèle. Le LLM peut seulement opposer un veto au plan et proposer une lecture métier du modèle, à valider. | `data/clean/`, `data/curated/analytical_base.csv`, `data/etl/model.json`, `etl_agent_decision.json`, `etl_report.md` |
| KPI générique | `agents/kpi_agent.py` | planificateur (repli par règles) | Le LLM conçoit un plan de KPI (indicateurs, tendances, découpages, comparaisons) dans un catalogue validé par Pydantic ; Python l'exécute, valide chaque résultat et redemande un plan en cas d'erreur. | `data/kpi/generic_kpi_results.json`, `generic_kpi_plan.json`, `generic_kpi_report.md` |
| Modèle métier Olist | `agents/olist_model_agent.py` | revue facultative | Modèle en étoile métier : `fact_sales`, `fact_orders`, `dim_*`. Applique les traductions validées par un humain, journalise chaque correction. | `data/curated/`, `data/etl/olist_model_decision.json`, `olist_model_report.md` |
| KPI métier Olist | `agents/olist_kpi_agent.py` | non | 12 indicateurs officiels aux définitions fixées + découpages (mois, États, catégories, vendeurs, produits, événements, jours). | `data/kpi/kpi_results.json`, `kpi_report.md` |
| BI Analyst | `agents/analyst_agent.py` | oui (repli par règles) | Python détecte des signaux chiffrés (tendances, baisses, anomalies, régions…), le LLM les transforme en insights et recommandations en citant ses signaux ; nombres vérifiés, aucune alerte omise. | `data/analyst/` |
| Dashboard | `agents/dashboard_agent.py` | non | Dashboard HTML autonome : onglet « Analyses BI » (synthèse, insights et preuves, croissance, anomalies, catégories, États, événements, vendeurs) et onglet « Exploration » (filtres période / État / catégorie / devise). Contrôle que ses totaux sont égaux à ceux du modèle et des KPI. | `data/dashboard/dashboard.html` |
| Auditeur XAI | `agents/auditor_agent.py` + `xai_engine.py` | explications facultatives | Recalcule 13 KPI depuis les tables de faits (cartes de lignage : formule, source, rapporté vs recalculé), réconcilie avec les **CSV bruts**, contrôle l'Analyste et le Dashboard. XAI : pont prix-volume-mix de la variation du CA et modèle « mauvaise note » expliqué par **SHAP**. Un audit non conforme déclenche une relance du modèle ou des KPI métier. | `data/audit/`, `data/xai/` |
| Évaluation | `agents/evaluation_agent.py` | non | Lancée en fin de run : statut, durée et score de qualité de chaque étape, score global du pipeline. | `data/eval/` |

## Installation

Prérequis : Python 3.12.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Créer un fichier `.env` à la racine (jamais versionné) :

```
LLM_MODEL=google_genai:gemini-3.5-flash-lite
GOOGLE_API_KEY=<votre clé>
```

Sans `.env`, tout fonctionne en mode déterministe (option `--no-llm`, mode règles de l'assistant).

Dans VS Code, choisir l'interpréteur `.venv\Scripts\python.exe` (sinon l'éditeur signale des imports introuvables).

### Données

Le dossier `data/` n'est pas versionné (210 Mo), à l'exception des décisions humaines
(`data/validated_translations.json`, `data/human_review_log.json`). Placer les CSV dans un sous-dossier de
`data/raw/` ; pour Olist, `data/raw/olist/` contient :

- les 9 fichiers du jeu public *Brazilian E-Commerce Public Dataset by Olist* (Kaggle) ;
- `br_holidays.csv` (`date, localName, name`) : jours fériés brésiliens ;
- trois colonnes **ajoutées** au jeu d'origine : `product_name` et `production_cost_brl` (coût **simulé**,
  ~70 % du prix) dans `olist_products_dataset.csv`, `seller_name` dans `olist_sellers_dataset.csv`.

Le Profiler et l'ETL lisent le sous-dossier de `data/raw/` qui contient le plus de CSV (les dossiers `backup`,
`archive`, `old` et `tmp` sont ignorés), ou celui indiqué par `--data-dir=chemin` ou par la variable `BIFLOW_RAW_DIR`.

Les réglages métier (statuts comptés dans le CA, coût par défaut, taux BRL → TND, événements commerciaux,
traductions) sont dans `etl_config.json`.

## Utilisation

```powershell
# Application de pilotage
.venv\Scripts\python.exe -m streamlit run app.py                              # http://localhost:8501

# Olist Pilot : tableaux de bord métier et assistant
.venv\Scripts\python.exe -m streamlit run manager_app.py --server.port 8502   # http://localhost:8502

# Pipeline complet
.venv\Scripts\python.exe agents\orchestrator.py --reuse    # réutilise la décision du Profiler si elle est à jour
.venv\Scripts\python.exe agents\orchestrator.py --no-llm   # sans aucun appel LLM (replis déterministes)

# Une ou plusieurs étapes, dans l'ordre
.venv\Scripts\python.exe agents\run_steps.py etl olist_model
.venv\Scripts\python.exe agents\run_steps.py kpi olist_kpi --no-llm

# Socle générique sur un autre jeu de données
.venv\Scripts\python.exe agents\run_steps.py etl kpi --data-dir=C:\chemin\vers\mes_csv

# Tests (sans appel LLM)
.venv\Scripts\python.exe -m pytest tests
```

Ordre conseillé au premier lancement : pipeline complet, puis validation humaine des traductions dans `app.py`
(page « Validation humaine »), puis nouveau lancement du pipeline pour les appliquer.

## Application de pilotage (`app.py`)

| Page | Contenu |
|---|---|
| Accueil | Avancement du pipeline, indicateurs clés, décisions en attente |
| Qualité des données | Score de fiabilité par table, détail d'une table, problèmes détectés, données manquantes pour le projet |
| Validation humaine | Traductions de catégories proposées par le Profiler, à accepter ou corriger |
| Données préparées | Socle générique (schéma déduit, contrôles, KPI génériques) et modèle métier (contrôles, hypothèses, journal des corrections, tables) |
| KPI Dashboard | Indicateurs officiels recalculés selon les filtres : évolution, régions, catégories, vendeurs et produits, événements et jours |
| Analyses | Insights et recommandations du BI Analyst, avec leurs preuves |
| Dashboard BI | Dashboard HTML généré par l'agent Dashboard |
| Audit & XAI | Auditeur (cartes de lignage), test d'injection d'erreur, XAI de la variation du CA, modèle ML expliqué par SHAP, évaluation des agents |
| Lancer et suivre | Lancement des agents, journal et décisions de l'Orchestrator |

## Olist Pilot (`manager_app.py`)

Filtres communs dans la barre latérale : période, États clients, catégories.

| Page | Contenu |
|---|---|
| Ventes | CA mensuel, catégories (treemap coloré par croissance), effet des événements |
| Livraison & clients | Note selon le retard, carte des retards par État, évolution mensuelle |
| Clients | Fidélisation, rétention par cohorte, segmentation RFM avec actions conseillées, fidélité par État et par 1re catégorie |
| Vendeurs | Tableau de bord des vendeurs (alertes), fiche vendeur au clic, « plan d'action » envoyé à l'assistant |
| Produits | Concentration du CA (Pareto), best-sellers unités vs CA, qualité des fiches (photos, description), poids vs fret |
| Paiements | Mix des moyens de paiement et son évolution, mensualités vs panier, annulations et notes par moyen, boleto par État |
| Carte | Densité du CA par code postal (clients ou vendeurs), villes clés, flux vendeur → client entre États (délai, retard) |
| Scénarios | « Et si ? » : moins de retards, plus de fidélisation, commandes récupérées (hypothèses affichées) |

La carte de densité charge son fond de carte en ligne (connexion internet nécessaire).

### Assistant Pilot

Panneau à droite de chaque page (masquable). Il répond sur le périmètre filtré à l'écran en appelant des outils ;
chaque réponse affiche ses sources et le détail des calculs.

| Source | Outils |
|---|---|
| KPI métier | `olist_kpi_agent.filter_frames` + `compute` : indicateurs officiels sur le périmètre |
| BI Analyst | `analyst_agent.detect_*` sur ces KPI : alertes, anomalies, opportunités ; insights publiés |
| XAI | `xai_engine.explain_kpi_change` (pourquoi le CA a changé : prix / volume / mix), `shap_attributions` (facteurs de mauvaise note), modèle `review_model.joblib` (risque d'une commande ou d'un scénario) |
| Calculs Pilot | `manager/metrics.py` (classements, séries, vendeurs, scénarios) et `manager/insights.py` (`segments_clients`, `analyse_paiements`, `concentration_produits`, `flux_livraison`) |

Le LLM ne calcule rien : il choisit les outils et rédige. Sans LLM, un mode règles appelle les mêmes outils.

**Périmètre** (`manager/scope_guard.py`) : Pilot ne répond qu'aux questions sur l'activité d'Olist. Chaque
question passe d'abord par un classement déterministe :

- tentative de détournement des consignes, ou sujet étranger (météo, recettes, code, politique…) → refus poli
  avec des exemples de questions, sans appeler aucun outil ;
- salutation → réponse courte ;
- vocabulaire métier → traitement normal ;
- question ambiguë → confiée au LLM, dont la consigne définit le périmètre (il répond `HORS_PERIMETRE` s'il la
  juge hors sujet) ; sans LLM, elle est refusée.

## Structure du projet

```
agents/                 agents du pipeline, orchestrateur, run_steps.py
manager/                Olist Pilot : metrics.py, insights.py, assistant.py, scope_guard.py, agents_bridge.py
app.py                  application de pilotage
manager_app.py          application manager (Olist Pilot)
etl_config.json         réglages métier Olist
tests/                  tests pytest (sans appel LLM)
data/                   données brutes et sorties des agents (non versionné, sauf décisions humaines)
```

## Tests

76 tests, tous sans appel au LLM (`tests/conftest.py` force `--no-llm`) :

| Fichier | Ce qui est testé |
|---|---|
| `test_generic_etl_kpi.py` | Socle générique sur une boutique fictive sans aucune colonne Olist : nettoyage, modèle en flocon déduit, KPI calculés |
| `test_profiler_etl.py` | Profilage et règles du modèle métier (retards au jour près, traductions humaines, coût) |
| `test_kpi_analyst_auditor.py` | KPI métier, garde-fous de l'Analyste, Auditeur, aiguillage de l'Orchestrator |
| `test_kpi_filters.py` | Filtres des KPI : sans filtre = résultats publiés, partition par État |
| `test_manager.py`, `test_agents_bridge.py` | Chiffres d'Olist Pilot identiques à l'agent KPI ; routage de l'assistant vers les agents |
| `test_manager_insights.py` | Pages Clients, Produits, Paiements, Carte : cohérence avec les KPI officiels, périmètres presque vides |
| `test_scope_guard.py` | Questions métier acceptées, hors sujet et détournements refusés |

Les tests qui lisent `data/curated/` sont ignorés si le pipeline n'a pas encore tourné.

## Limites connues

- Le coût de production (`production_cost_brl`) est **simulé** (~70 % du prix) : la marge n'est pas exploitable.
- Le taux BRL → TND (`etl_config.json`) est indicatif : sa date de référence reste à renseigner (`brl_to_tnd_date`).
- Stock et trafic web absents du jeu Olist : rotation du stock et taux de conversion non calculables.
- Les liens observés (retard → note, photos → ventes) sont des associations, pas des causalités prouvées.
- Les règles métier du Profiler et toute la couche métier sont propres à Olist ; sur un autre jeu de données,
  le pipeline s'arrête proprement après le socle générique.
