"""BIFlow KPI Agent (générique, multi-tables / constellation, piloté par LLM + exécution déterministe).

Fonctionne sur n'importe quel jeu de données préparé par etl_agent.py :
  1. Lit model.json, semantic_metadata.json, model_annotations.json (optionnel),
     data/curated/analytical_base.csv ainsi que les tables nettoyées de data/clean/*.csv
     (indispensable pour les schémas en constellation où certaines tables de faits ne sont
     pas fusionnées dans analytical_base afin d'éviter la duplication de lignes).
  2. Extrait un profil statistique léger des colonnes par table (sans exposer de lignes brutes au LLM).
  3. Demande au LLM de concevoir un PLAN de KPI (KpiPlan) dans un catalogue fermé d'opérations
     pouvant cibler `"analytical_base"` ou n'importe quelle table de `data/clean/`.
  4. Exécute les calculs de façon 100 % déterministe en Python/Pandas (y compris `date_diff` pour les délais).
  5. Valide chaque KPI calculé ; en cas d'échec, boucle d'auto-correction (re-planification LLM).
  6. Génère data/kpi/generic_kpi_results.json, generic_kpi_plan.json et generic_kpi_report.md
     (les indicateurs métier Olist sont produits à part par olist_kpi_agent.py dans kpi_results.json).

Point d'entrée : run(), appelé par l'orchestrateur.
"""

import json
import math
import os
import re
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
warnings.filterwarnings("ignore", message=".*fixed sampling defaults.*")

CLEAN = ROOT / "data" / "clean"
CURATED = ROOT / "data" / "curated"
ETL_DIR = ROOT / "data" / "etl"
OUT = ROOT / "data" / "kpi"

MAX_PLAN_ATTEMPTS = 2
TOP_N_DEFAULT = 10
LOW_VOLUME_PERIOD_RATIO = 0.05   # période exclue des tendances si < 5 % du volume médian


# ---------------------------------------------------------------------------
# 1. Catalogue fermé de spécifications KPI (multi-tables, validé par Pydantic)
# ---------------------------------------------------------------------------
def _slugify(text: str, prefix: str = "kpi") -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s[:40] or prefix


class HeadlineSpec(BaseModel):
    id: str = Field(default="", description="Identifiant snake_case unique")
    label: str = Field(..., description="Libellé métier clair en français")
    table: str = Field(default="analytical_base", description="Table cible ('analytical_base' ou nom d'une table de data/clean)")
    agg: Literal["sum", "mean", "median", "min", "max", "count_rows", "nunique", "ratio", "rate_where", "date_diff"]
    column: Optional[str] = Field(None, description="Colonne cible (ou date de fin pour date_diff ; null pour count_rows)")
    denominator_column: Optional[str] = Field(None, description="Dénominateur pour 'ratio', ou date de début pour 'date_diff'")
    filter_column: Optional[str] = Field(None, description="Colonne de filtre optionnelle")
    filter_operator: Optional[Literal["eq", "neq", "in", "gt", "lt", "is_true", "is_false"]] = None
    filter_value: Optional[str | float | int | bool | list] = None
    unit: str = Field("", description="Unité d'affichage (ex: '€', 'R$', '%', 'jours', '/ 5', '')")
    definition: str = Field("", description="Définition métier concise")

    def model_post_init(self, __context) -> None:
        if not self.id:
            self.id = _slugify(self.label, "headline")
        if not self.table:
            self.table = "analytical_base"


class TimeTrendSpec(BaseModel):
    id: str = ""
    label: str
    table: str = "analytical_base"
    date_column: str
    metric_column: Optional[str] = None
    agg: Literal["sum", "mean", "median", "count_rows", "nunique"] = "sum"
    freq: Literal["M", "W", "Q"] = "M"
    unit: str = ""

    def model_post_init(self, __context) -> None:
        if not self.id:
            self.id = _slugify(self.label, "trend")
        if not self.table:
            self.table = "analytical_base"


class BreakdownSpec(BaseModel):
    id: str = ""
    label: str
    table: str = "analytical_base"
    dimension_column: str
    metric_column: Optional[str] = None
    agg: Literal["sum", "mean", "median", "count_rows", "nunique"] = "sum"
    top_n: int = Field(default=TOP_N_DEFAULT, ge=2, le=30)
    unit: str = ""

    def model_post_init(self, __context) -> None:
        if not self.id:
            self.id = _slugify(self.label, "breakdown")
        if not self.table:
            self.table = "analytical_base"


class SegmentComparisonSpec(BaseModel):
    id: str = ""
    label: str
    table: str = "analytical_base"
    segment_column: str
    metric_column: str
    agg: Literal["mean", "median", "sum"] = "mean"
    unit: str = ""
    insight_question: str = ""

    def model_post_init(self, __context) -> None:
        if not self.id:
            self.id = _slugify(self.label, "segment")
        if not self.table:
            self.table = "analytical_base"


class KpiPlan(BaseModel):
    domain_detected: str = Field("Données tabulaires génériques", description="Domaine métier inféré")
    headline: list[HeadlineSpec] = Field(default_factory=list)
    time_trends: list[TimeTrendSpec] = Field(default_factory=list)
    breakdowns: list[BreakdownSpec] = Field(default_factory=list)
    segment_comparisons: list[SegmentComparisonSpec] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 2. Helpers de formatage & parsing
# ---------------------------------------------------------------------------
def rounded(value, digits: int = 2):
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return None
    try:
        v = float(value)
        return None if (math.isnan(v) or math.isinf(v)) else round(v, digits)
    except (TypeError, ValueError):
        return None


def fmt_val(value, unit: str = "", digits: int = 2) -> str:
    v = rounded(value, digits)
    if v is None:
        return "N/A"
    if unit == "%":
        return f"{v:.2f}%"
    num_str = f"{int(v):,}" if float(v).is_integer() and digits == 0 else f"{v:,.{digits}f}"
    return f"{num_str} {unit}".strip()


def _text_of(msg) -> str:
    c = msg.content
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return str(c)


def _extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Aucun objet JSON trouvé dans la réponse LLM.")
    return json.loads(text[start:end + 1])


# ---------------------------------------------------------------------------
# 3. Chargement des artefacts de etl_agent.py (base + tables nettoyées)
# ---------------------------------------------------------------------------
def _read_json_optional(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _apply_column_roles(df: pd.DataFrame, table_name: str, metadata: dict) -> pd.DataFrame:
    """Re-type proprement les colonnes d'un DataFrame selon les rôles inférés par etl_agent."""
    tables_meta = metadata.get("tables", {})
    base_col_origins = metadata.get("base_columns", {})
    fact_table = metadata.get("fact_table", "")

    for col in df.columns:
        role = None
        if table_name == "analytical_base":
            origin = base_col_origins.get(col, "")
            orig_t, orig_c = origin.split(".", 1) if "." in origin else (fact_table, col)
            role = tables_meta.get(orig_t, {}).get("columns", {}).get(orig_c)
        else:
            role = tables_meta.get(table_name, {}).get("columns", {}).get(col)

        if role == "numeric":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif role == "date":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df[col] = pd.to_datetime(df[col], errors="coerce")
        elif role == "boolean":
            s = df[col].astype("string").str.strip().str.lower()
            df[col] = s.map({
                "true": True, "1": True, "yes": True, "oui": True,
                "false": False, "0": False, "no": False, "non": False,
            })
    return df


def load_etl_context():
    base_path = CURATED / "analytical_base.csv"
    if not base_path.exists():
        raise FileNotFoundError(f"Sortie ETL manquante : {base_path} (exécuter etl_agent d'abord)")

    model = _read_json_optional(ETL_DIR / "model.json")
    metadata = _read_json_optional(ETL_DIR / "semantic_metadata.json")
    annotations = _read_json_optional(ETL_DIR / "model_annotations.json")
    etl_decision = _read_json_optional(ETL_DIR / "etl_agent_decision.json")

    tables: dict[str, pd.DataFrame] = {}
    base_df = pd.read_csv(base_path, low_memory=False)
    tables["analytical_base"] = _apply_column_roles(base_df, "analytical_base", metadata)

    if CLEAN.exists():
        for csv_path in sorted(CLEAN.glob("*.csv")):
            t_name = csv_path.stem
            df_t = pd.read_csv(csv_path, low_memory=False)
            tables[t_name] = _apply_column_roles(df_t, t_name, metadata)

    return tables, model, metadata, annotations, etl_decision


def _summarize_single_table(df: pd.DataFrame, table_name: str, metadata: dict) -> list[dict]:
    base_origins = metadata.get("base_columns", {})
    tables_meta = metadata.get("tables", {})
    cols_summary = []

    for col in df.columns:
        if col.endswith("_was_missing") or col.endswith("_is_outlier"):
            continue
        s = df[col]
        nn = s.dropna()

        if table_name == "analytical_base":
            origin = base_origins.get(col, "")
            t, c = origin.split(".", 1) if "." in origin else (metadata.get("fact_table", ""), col)
            role = tables_meta.get(t, {}).get("columns", {}).get(c, "unknown")
        else:
            origin = f"{table_name}.{col}"
            role = tables_meta.get(table_name, {}).get("columns", {}).get(col, "unknown")

        info = {
            "column": col,
            "origin": origin or col,
            "role": role,
            "missing_pct": round(100 * float(s.isna().mean()), 1) if len(s) else 0.0,
            "distinct": int(nn.nunique()),
        }
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) and len(nn):
            info["min"] = rounded(nn.min())
            info["median"] = rounded(nn.median())
            info["max"] = rounded(nn.max())
        elif pd.api.types.is_datetime64_any_dtype(s) and len(nn):
            info["date_min"] = str(nn.min())[:10]
            info["date_max"] = str(nn.max())[:10]
        elif 1 <= info["distinct"] <= 15:
            info["sample_categories"] = [str(v) for v in nn.value_counts().head(8).index.tolist()]

        cols_summary.append(info)
    return cols_summary


def summarize_schema_for_llm(tables: dict[str, pd.DataFrame], model: dict, metadata: dict, annotations: dict) -> dict:
    """Construit un résumé multi-tables compact (aucune donnée brute sensible) pour le LLM."""
    not_merged = set(metadata.get("tables_not_merged", []))
    fact_tables = {f["table"] for f in model.get("facts", [])}
    dim_tables = {d["table"] for d in model.get("dimensions", [])}

    # Inclut analytical_base, les faits non fusionnés ET les dimensions/tables mères nettoyées
    # afin de permettre des calculs au bon niveau de grain (ex: commande vs article).
    relevant_tables = ["analytical_base"] + [
        t for t in tables
        if t != "analytical_base" and (t in not_merged or t in fact_tables or t in dim_tables)
    ]

    tables_schema = {}
    for t in relevant_tables:
        if t not in tables:
            continue
        df = tables[t]
        if t == "analytical_base":
            note = "Table analytique principale fusionnée (grain le plus fin : fait principal + dimensions jointes)."
        elif t in not_merged or t in fact_tables:
            note = "Table de faits séparée (non fusionnée dans analytical_base pour éviter les doublons)."
        else:
            note = "Table parente / dimension nettoyée (utile pour calculer des métriques sans duplication liée au grain fin de analytical_base)."
        tables_schema[t] = {
            "rows_total": int(len(df)),
            "note": note,
            "columns": _summarize_single_table(df, t, metadata),
        }

    return {
        "schema_type": model.get("schema_type", "unknown"),
        "facts": model.get("facts", []),
        "dimensions": model.get("dimensions", []),
        "tables_not_merged": sorted(not_merged),
        "llm_annotations": annotations if annotations else None,
        "available_tables": tables_schema,
    }

# ---------------------------------------------------------------------------
# 4. Planification : LLM Planner + Fallback déterministe multi-tables
# ---------------------------------------------------------------------------
PLANNER_SYSTEM_PROMPT = """Tu es le KPI Agent d'un pipeline BI autonome (BIFlow).
On te donne le schéma nettoyé de plusieurs tables analytiques produites par l'ETL Agent dans `available_tables` :
- `"analytical_base"` : la table principale fusionnée (fait principal + dimensions).
- D'autres tables séparées (ex: paiements, avis/reviews, etc.) présentes dans `tables_not_merged` ou `facts` parce que les fusionner aurait dupliqué les lignes de `analytical_base`.

Tu ne vois jamais les données brutes, uniquement les noms de tables/colonnes, rôles, cardinalités et bornes statistiques.

RÈGLES IMPORTANTES :
1. Chaque KPI, tendance, découpage ou comparaison DOIT préciser `"table"` (choisie parmi les clés de `available_tables`, ex: `"analytical_base"` ou `"reviews"`).
2. Utilise EXACTEMENT les noms de colonnes listés dans `available_tables[table].columns`.
3. Si tu calcules un chiffre d'affaires réalisé et qu'une colonne de statut existe (ex: `status` avec `'delivered'` ou `'paid'`), filtre sur le statut livré/validé pour le CA net réalisé (et calcule séparément le taux d'annulation).
4. Exploite AUSSI les tables de faits non fusionnées (ex: note moyenne `review_score` sur la table des avis, montant/mensualités moyennes sur la table des paiements) si elles existent dans `available_tables`.
5. Si deux colonnes de dates permettent de mesurer un délai métier pertinent (ex: livraison - achat, résolution - ouverture), utilise `agg="date_diff"` (`column` = date de fin, `denominator_column` = date de début, `unit="jours"`).
6. RESPECT DU GRAIN (ANTI-DUPLICATION) : Vérifie toujours le grain (1 ligne = quoi ?) de `analytical_base` (souvent le niveau ligne/article) par rapport aux autres tables de `available_tables`.
   - Ne calcule JAMAIS de moyenne, médiane ou taux d'une métrique appartenant à une entité parente (ex: délai de livraison d'une commande, statut d'une commande) sur `analytical_base` si cette entité parente possède sa propre table dans `available_tables` (ex: `orders`) : cible directement cette table parente dans `"table"` !
   - Réserve `analytical_base` aux métriques du grain fin (ex: chiffre d'affaires par article, quantité, découpages par catégorie produit) ou aux comptes distincts (`agg="nunique"`).
Réponds UNIQUEMENT par un objet JSON valide suivant STRICTEMENT ce format :
{
  "domain_detected": "E-commerce / Marketplace",
  "headline": [
    {
      "id": "net_revenue",
      "label": "Chiffre d'affaires livré",
      "table": "analytical_base",
      "agg": "sum",
      "column": "<nom_colonne_numerique>",
      "denominator_column": null,
      "filter_column": "<colonne_filtre_ou_null>",
      "filter_operator": "eq",
      "filter_value": "<valeur_ou_null>",
      "unit": "R$",
      "definition": "Description courte du calcul"
    }
  ],
  "time_trends": [
    {
      "id": "monthly_revenue",
      "label": "Évolution mensuelle du CA",
      "table": "analytical_base",
      "date_column": "<nom_colonne_date>",
      "metric_column": "<nom_colonne_numerique_ou_null>",
      "agg": "sum",
      "freq": "M",
      "unit": "R$"
    }
  ],
  "breakdowns": [
    {
      "id": "revenue_by_category",
      "label": "Répartition par catégorie",
      "table": "analytical_base",
      "dimension_column": "<nom_colonne_categorie>",
      "metric_column": "<nom_colonne_numerique_ou_null>",
      "agg": "sum",
      "top_n": 10,
      "unit": "R$"
    }
  ],
  "segment_comparisons": [
    {
      "id": "compare_segment",
      "label": "Comparaison selon segment",
      "table": "analytical_base",
      "segment_column": "<nom_colonne_faible_cardinalite>",
      "metric_column": "<nom_colonne_numerique>",
      "agg": "mean",
      "unit": "",
      "insight_question": "Quel est l'impact de ce segment ?"
    }
  ],
  "limitations": ["Limite méthodologique 1", "Limite méthodologique 2"]
}

Contraintes du catalogue :
- `headline` (8 à 12 KPI couvrant `analytical_base` ET les tables de faits non fusionnées) :
  * `agg` parmi `["sum", "mean", "median", "min", "max", "count_rows", "nunique", "ratio", "rate_where", "date_diff"]`.
  * `count_rows` : `"column": null`.
  * `ratio` : `column` (numérateur) et `denominator_column` (dénominateur).
  * `date_diff` : `column` (date de fin) et `denominator_column` (date de début), résultat en jours (`unit="jours"`).
  * `rate_where` : `filter_column`, `filter_operator` (`"eq"`, `"neq"`, `"in"`, `"gt"`, `"lt"`, `"is_true"`, `"is_false"`) et `filter_value`.
  * Ne fais JAMAIS `sum` ou `mean` sur une colonne de rôle `id`.
- `time_trends` (1 à 3) : `date_column` de rôle `date`, `freq` parmi `["M", "W", "Q"]`.
- `breakdowns` (3 à 6) : `dimension_column` avec `2 <= distinct <= 500` (jamais un ID unique).
- `segment_comparisons` (1 à 3) : `segment_column` avec `2 <= distinct <= 8`."""


# Colonnes numériques sans sens additif (coordonnées, rangs) : jamais sommées ni moyennées par le repli
NON_ADDITIVE_TOKENS = {"lat", "lng", "lon", "latitude", "longitude", "sequential", "seq", "rank", "index",
                       "lenght", "length", "zip", "prefix", "year", "month", "day"}
# Mots qui désignent souvent la date « métier » d'un enregistrement (création, achat, transaction)
EVENT_DATE_TOKENS = {"purchase", "order", "created", "creation", "transaction", "sale", "invoice", "date"}


def _tokens(name: str) -> set:
    return {t for t in re.split(r"[^a-z0-9]+", name.lower()) if t}


def _is_additive(col: dict) -> bool:
    return col["role"] == "numeric" and col["distinct"] > 2 and not (_tokens(col["column"]) & NON_ADDITIVE_TOKENS)


def build_deterministic_fallback_plan(schema: dict) -> KpiPlan:
    """Plan 100 % déterministe multi-tables si --no-llm est activé ou si le LLM est indisponible."""
    avail = schema.get("available_tables", {})
    base_info = avail.get("analytical_base", {"columns": []})
    cols = base_info["columns"]
    n_rows = base_info.get("rows_total", 0) or 1
    fact_tables = {f["table"] for f in schema.get("facts", [])}

    num_cols = [c["column"] for c in cols if _is_additive(c)]
    # date « métier » : peu de manquants, nom évocateur d'un événement (achat, création...)
    dated = [c for c in cols if c["role"] == "date"]
    dated.sort(key=lambda c: (c["missing_pct"], -len(_tokens(c["column"]) & EVENT_DATE_TOKENS)))
    date_cols = [c["column"] for c in dated]
    # identifiants d'entités (commande, client...) : un compte distinct n'a de sens qu'au-delà de 100 valeurs
    id_cols = [c["column"] for c in cols if c["role"] == "id" and c["distinct"] >= min(100, n_rows)]
    cat_cols = [c["column"] for c in cols
                if c["role"] in ("category", "boolean") and 2 <= c["distinct"] <= 100]

    headline = [HeadlineSpec(
        id="total_records", label="Volume total d'enregistrements", table="analytical_base",
        agg="count_rows", definition="Nombre total de lignes dans la base analytique."
    )]
    for cid in id_cols[:2]:
        headline.append(HeadlineSpec(
            id=f"unique_{cid}", label=f"Nombre unique de {cid}", table="analytical_base",
            agg="nunique", column=cid, definition=f"Cardinalité distincte de {cid}."
        ))
    for num in num_cols[:3]:
        headline.append(HeadlineSpec(
            id=f"sum_{num}", label=f"Total {num}", table="analytical_base",
            agg="sum", column=num, definition=f"Somme globale de {num}."
        ))
        headline.append(HeadlineSpec(
            id=f"avg_{num}", label=f"Moyenne {num}", table="analytical_base",
            agg="mean", column=num, definition=f"Valeur moyenne de {num}."
        ))

    # tables de faits non fusionnées (ex. paiements, avis) : leurs mesures propres, à leur grain
    for t_name, t_info in avail.items():
        if t_name == "analytical_base" or t_name not in fact_tables:
            continue
        if any(t_name in str(c.get("origin", "")) for c in cols):
            continue   # déjà présente dans analytical_base
        t_nums = [c["column"] for c in t_info["columns"] if _is_additive(c)]
        for num in t_nums[:2]:
            headline.append(HeadlineSpec(
                id=f"{t_name}_avg_{num}", label=f"Moyenne {num} ({t_name})", table=t_name,
                agg="mean", column=num, definition=f"Moyenne de {num} sur la table {t_name}."
            ))

    primary_num = num_cols[0] if num_cols else None
    primary_agg = "sum" if primary_num else "count_rows"

    trends = []
    if date_cols:
        trends.append(TimeTrendSpec(
            id=f"trend_{date_cols[0]}", table="analytical_base",
            label=f"Évolution mensuelle ({primary_num or 'volume'})",
            date_column=date_cols[0], metric_column=primary_num, agg=primary_agg, freq="M"
        ))

    breakdowns = []
    for cat in cat_cols[:4]:
        breakdowns.append(BreakdownSpec(
            id=f"by_{cat}", table="analytical_base",
            label=f"Répartition par {cat}",
            dimension_column=cat, metric_column=primary_num, agg=primary_agg, top_n=10
        ))

    comparisons = []
    low_card = [c["column"] for c in cols if c["role"] in ("boolean", "category") and 2 <= c["distinct"] <= 5]
    if low_card and num_cols:
        comparisons.append(SegmentComparisonSpec(
            id=f"compare_{num_cols[0]}_by_{low_card[0]}", table="analytical_base",
            label=f"{num_cols[0]} selon {low_card[0]}",
            segment_column=low_card[0], metric_column=num_cols[0], agg="mean"
        ))

    return KpiPlan(
        domain_detected="Analyse tabulaire multi-tables (mode déterministe)",
        headline=headline[:12],
        time_trends=trends,
        breakdowns=breakdowns,
        segment_comparisons=comparisons,
        limitations=["Plan généré automatiquement sans LLM (agrégations statistiques standard)."],
    )


def _invoke_llm_with_backoff(llm, messages, max_retries: int = 3):
    """Réessaie sur erreurs transitoires (503 high demand, 429 rate limit) avant fallback."""
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            msg = str(e).lower()
            transient = any(code in msg for code in ("503", "429", "unavailable", "rate", "timeout", "overloaded"))
            if not transient or attempt == max_retries - 1:
                raise
            wait = 2 ** (attempt + 1)
            print(f"  (KPI Agent : API surchargée, nouvelle tentative dans {wait}s...)")
            time.sleep(wait)


def generate_kpi_plan(schema: dict, feedback_errors: list[str] | None = None, _pydantic_retried: bool = False) -> tuple[KpiPlan, str]:
    """Appelle le LLM pour planifier les KPI, avec fallback déterministe si désactivé/indisponible."""
    model_name = os.environ.get("LLM_MODEL")
    if not model_name or "--no-llm" in sys.argv:
        return build_deterministic_fallback_plan(schema), "deterministic_rule"

    try:
        from langchain.chat_models import init_chat_model
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = init_chat_model(model_name, temperature=float(os.environ.get("LLM_TEMPERATURE", "0")))
        user_payload = {"schema": schema}
        if feedback_errors:
            user_payload["previous_attempt_errors"] = feedback_errors
            user_payload["instruction"] = (
                "Ta proposition précédente contient des erreurs de table, de colonne ou de calcul. "
                "Corrige les clés/tables/colonnes fautives et renvoie un JSON complet valide."
            )

        ai = _invoke_llm_with_backoff(llm, [
            SystemMessage(PLANNER_SYSTEM_PROMPT),
            HumanMessage(json.dumps(user_payload, ensure_ascii=False, default=str)),
        ])
        raw_dict = _extract_json(_text_of(ai))
        plan = KpiPlan.model_validate(raw_dict)
        if not plan.headline:
            raise ValueError("Le plan LLM ne contient aucun indicateur headline.")
        return plan, f"llm:{model_name}"
    except ValidationError as ve:
        if not _pydantic_retried:
            err_lines = [f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}" for e in ve.errors()[:8]]
            print(f"  (KPI Agent : schéma LLM à corriger — {err_lines[:2]})")
            return generate_kpi_plan(schema, feedback_errors=err_lines, _pydantic_retried=True)
        print(f"  (KPI Agent : fallback déterministe après échec schéma — {str(ve)[:120]})")
        return build_deterministic_fallback_plan(schema), "deterministic_fallback"
    except Exception as e:
        print(f"  (KPI Agent : fallback déterministe activé — {str(e)[:120]})")
        return build_deterministic_fallback_plan(schema), "deterministic_fallback"


# ---------------------------------------------------------------------------
# 5. Exécution déterministe multi-tables + Validation
# ---------------------------------------------------------------------------
def _get_table(tables: dict[str, pd.DataFrame], table_name: str) -> pd.DataFrame:
    if table_name not in tables:
        raise KeyError(f"Table introuvable : '{table_name}' (disponibles : {sorted(tables.keys())})")
    return tables[table_name]


def _apply_filter(df: pd.DataFrame, col: Optional[str], op: Optional[str], val) -> pd.DataFrame:
    if not col or not op:
        return df
    if col not in df.columns:
        raise KeyError(f"Colonne de filtre introuvable : {col}")
    s = df[col]
    if op == "is_true":
        return df[s.fillna(False).astype(bool)]
    if op == "is_false":
        return df[~s.fillna(True).astype(bool)]
    if op == "eq":
        return df[s.astype(str) == str(val)]
    if op == "neq":
        return df[s.astype(str) != str(val)]
    if op == "in":
        vals = [str(v) for v in (val if isinstance(val, list) else [val])]
        return df[s.astype(str).isin(vals)]
    if op == "gt":
        return df[pd.to_numeric(s, errors="coerce") > float(val)]
    if op == "lt":
        return df[pd.to_numeric(s, errors="coerce") < float(val)]
    raise ValueError(f"Opérateur de filtre inconnu : {op}")


def _aggregate_series(df: pd.DataFrame, col: Optional[str], agg: str):
    if agg == "count_rows":
        return int(len(df))
    if not col or col not in df.columns:
        raise KeyError(f"Colonne cible introuvable : {col}")
    s = df[col].dropna()
    if s.empty:
        return None
    if agg == "nunique":
        return int(s.nunique())
    num = pd.to_numeric(s, errors="coerce").dropna()
    if num.empty:
        raise TypeError(f"La colonne '{col}' n'est pas numérique pour '{agg}'.")
    if agg == "sum":
        return rounded(num.sum())
    if agg == "mean":
        return rounded(num.mean())
    if agg == "median":
        return rounded(num.median())
    if agg == "min":
        return rounded(num.min())
    if agg == "max":
        return rounded(num.max())
    raise ValueError(f"Agrégation inconnue : {agg}")


def execute_and_validate(tables: dict[str, pd.DataFrame], plan: KpiPlan) -> tuple[dict, list[str]]:
    """Exécute le KpiPlan sur les tables ciblées et retourne (résultats_structurés, erreurs)."""
    errors = []
    headline_out = []

    # 1. Headline KPIs
    for spec in plan.headline:
        try:
            df = _get_table(tables, spec.table)
            if spec.agg == "rate_where":
                if len(df) == 0:
                    raise ValueError(f"Table '{spec.table}' vide pour rate_where.")
                sub = _apply_filter(df, spec.filter_column, spec.filter_operator, spec.filter_value)
                val = rounded(100.0 * len(sub) / len(df))
                rows_used = len(df)
            elif spec.agg == "ratio":
                sub = _apply_filter(df, spec.filter_column, spec.filter_operator, spec.filter_value)
                num_v = _aggregate_series(sub, spec.column, "sum")
                den_v = _aggregate_series(sub, spec.denominator_column, "sum")
                if num_v is None or not den_v:
                    raise ValueError(f"Dénominateur nul ou manquant pour ratio {spec.id}")
                val = rounded(100.0 * num_v / den_v if spec.unit == "%" else num_v / den_v)
                rows_used = len(sub)
            elif spec.agg == "date_diff":
                sub = _apply_filter(df, spec.filter_column, spec.filter_operator, spec.filter_value)
                if not spec.column or not spec.denominator_column:
                    raise ValueError(f"date_diff '{spec.id}' requiert column (fin) et denominator_column (début).")
                if spec.column not in sub.columns or spec.denominator_column not in sub.columns:
                    raise KeyError(f"Colonnes date introuvables dans {spec.table} : {spec.column} / {spec.denominator_column}")
                end_dt = pd.to_datetime(sub[spec.column], errors="coerce")
                start_dt = pd.to_datetime(sub[spec.denominator_column], errors="coerce")
                diff_days = (end_dt - start_dt).dt.total_seconds().div(86400.0).dropna()
                if diff_days.empty:
                    raise ValueError(f"Aucune paire de dates valide pour date_diff '{spec.id}'.")
                use_median = "median" in spec.id.lower() or "médian" in spec.label.lower()
                val = rounded(diff_days.median() if use_median else diff_days.mean())
                rows_used = int(len(diff_days))
            else:
                sub = _apply_filter(df, spec.filter_column, spec.filter_operator, spec.filter_value)
                val = _aggregate_series(sub, spec.column, spec.agg)
                rows_used = len(sub)

            if val is None:
                errors.append(f"Headline '{spec.id}' ({spec.table}) a produit une valeur nulle (NaN).")
                continue

            headline_out.append({
                "id": spec.id,
                "label": spec.label,
                "table": spec.table,
                "value": val,
                "formatted": fmt_val(val, spec.unit, digits=0 if spec.agg in ("count_rows", "nunique") else 2),
                "unit": spec.unit,
                "agg": spec.agg,
                "column": spec.column,
                "rows_used": int(rows_used),
                "definition": spec.definition,
            })
        except Exception as e:
            errors.append(f"Headline '{spec.id}' invalide : {e}")

    # 2. Time trends
    trends_out = []
    for spec in plan.time_trends:
        try:
            df = _get_table(tables, spec.table)
            if spec.date_column not in df.columns:
                raise KeyError(f"Colonne date introuvable dans {spec.table} : {spec.date_column}")
            dt = pd.to_datetime(df[spec.date_column], errors="coerce")
            valid = df[dt.notna()].copy()
            if valid.empty:
                raise ValueError(f"Aucune date valide dans {spec.table}.{spec.date_column}")
            valid["_period"] = dt[dt.notna()].dt.to_period(spec.freq).astype(str)

            if spec.agg == "count_rows" or not spec.metric_column:
                grp = valid.groupby("_period").size().rename("value")
            elif spec.agg == "nunique":
                grp = valid.groupby("_period")[spec.metric_column].nunique().rename("value")
            else:
                if spec.metric_column not in valid.columns:
                    raise KeyError(f"Colonne mesure introuvable dans {spec.table} : {spec.metric_column}")
                valid["_m"] = pd.to_numeric(valid[spec.metric_column], errors="coerce")
                grp = valid.groupby("_period")["_m"].agg(spec.agg).rename("value")

            # périodes quasi vides (début/fin de collecte, dates aberrantes) : exclues, sinon des
            # variations de +700 000 % faussent la lecture
            counts = valid.groupby("_period").size()
            reliable = counts >= LOW_VOLUME_PERIOD_RATIO * counts.median()
            dropped = sorted(str(p) for p in counts.index[~reliable])
            grp = grp[grp.index.isin(counts.index[reliable])].sort_index()
            prev = grp.shift(1)
            growth = ((grp / prev) - 1.0) * 100.0

            series = []
            for p_idx in grp.index:
                v = rounded(grp.loc[p_idx])
                g = rounded(growth.loc[p_idx]) if pd.notna(growth.loc[p_idx]) else None
                series.append({"period": str(p_idx), "value": v, "period_growth_pct": g})

            trends_out.append({
                "id": spec.id,
                "label": spec.label,
                "table": spec.table,
                "date_column": spec.date_column,
                "metric_column": spec.metric_column,
                "agg": spec.agg,
                "freq": spec.freq,
                "unit": spec.unit,
                "series": series,
                "low_volume_periods_excluded": dropped,
            })
        except Exception as e:
            errors.append(f"TimeTrend '{spec.id}' invalide : {e}")

    # 3. Dimension breakdowns
    breakdowns_out = []
    for spec in plan.breakdowns:
        try:
            df = _get_table(tables, spec.table)
            if spec.dimension_column not in df.columns:
                raise KeyError(f"Dimension introuvable dans {spec.table} : {spec.dimension_column}")
            tmp = df.copy()
            tmp["_dim"] = tmp[spec.dimension_column].fillna("Non défini").astype(str)

            if spec.agg == "count_rows" or not spec.metric_column:
                agg_s = tmp.groupby("_dim").size()
            elif spec.agg == "nunique":
                if spec.metric_column not in tmp.columns:
                    raise KeyError(f"Colonne mesure introuvable dans {spec.table} : {spec.metric_column}")
                agg_s = tmp.groupby("_dim")[spec.metric_column].nunique()
            else:
                if spec.metric_column not in tmp.columns:
                    raise KeyError(f"Colonne mesure introuvable dans {spec.table} : {spec.metric_column}")
                tmp["_m"] = pd.to_numeric(tmp[spec.metric_column], errors="coerce")
                agg_s = tmp.groupby("_dim")["_m"].agg(spec.agg)

            agg_s = agg_s.dropna().sort_values(ascending=False)
            if agg_s.empty:
                raise ValueError(f"Aucune donnée exploitable pour {spec.table}.{spec.dimension_column}")

            total_sum = float(agg_s.sum()) if spec.agg in ("sum", "count_rows") else None
            rows = []
            for k, v in agg_s.head(spec.top_n).items():
                val = rounded(v)
                share_pct = rounded(100.0 * float(v) / total_sum) if total_sum else None
                rows.append({"segment": str(k), "value": val, "share_pct": share_pct})

            breakdowns_out.append({
                "id": spec.id,
                "label": spec.label,
                "table": spec.table,
                "dimension_column": spec.dimension_column,
                "metric_column": spec.metric_column,
                "agg": spec.agg,
                "unit": spec.unit,
                "total_segments": int(len(agg_s)),
                "rows": rows,
            })
        except Exception as e:
            errors.append(f"Breakdown '{spec.id}' invalide : {e}")

    # 4. Segment comparisons
    comparisons_out = []
    for spec in plan.segment_comparisons:
        try:
            df = _get_table(tables, spec.table)
            if spec.segment_column not in df.columns or spec.metric_column not in df.columns:
                raise KeyError(f"Colonnes introuvables dans {spec.table} : {spec.segment_column} / {spec.metric_column}")
            tmp = df[[spec.segment_column, spec.metric_column]].dropna().copy()
            tmp["_m"] = pd.to_numeric(tmp[spec.metric_column], errors="coerce")
            tmp = tmp.dropna(subset=["_m"])
            if tmp.empty:
                raise ValueError("Aucune paire valide segment/mesure.")

            grp = tmp.groupby(spec.segment_column)["_m"].agg(["count", spec.agg]).sort_values(spec.agg, ascending=False)
            segments = [
                {"segment": str(idx), "count": int(r["count"]), "value": rounded(r[spec.agg])}
                for idx, r in grp.head(8).iterrows()
            ]
            comparisons_out.append({
                "id": spec.id,
                "label": spec.label,
                "table": spec.table,
                "segment_column": spec.segment_column,
                "metric_column": spec.metric_column,
                "agg": spec.agg,
                "unit": spec.unit,
                "insight_question": spec.insight_question,
                "segments": segments,
            })
        except Exception as e:
            errors.append(f"SegmentComparison '{spec.id}' invalide : {e}")

    if not headline_out:
        errors.append("Aucun indicateur headline n'a pu être calculé avec succès.")

    results = {
        "domain_detected": plan.domain_detected,
        "headline": headline_out,
        "time_trends": trends_out,
        "breakdowns": breakdowns_out,
        "segment_comparisons": comparisons_out,
        "limitations": plan.limitations,
    }
    return results, errors


# ---------------------------------------------------------------------------
# 6. Génération du rapport Markdown
# ---------------------------------------------------------------------------
def _md_table(headers: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return out


def write_report(kpis: dict):
    tables_list = ", ".join(f"`{k}` ({v:,} lignes)" for k, v in kpis["coverage"]["rows_by_table"].items())
    L = [
        "# Rapport KPI BIFlow (Générique Multi-Tables)",
        "",
        f"- **Domaine détecté** : {kpis['domain_detected']}",
        f"- **Généré le** : {kpis['generated_at']}",
        f"- **Planificateur** : `{kpis['planner']}` ({kpis['attempts']} essai(s))",
        f"- **Tables effectivement analysées** : {tables_list}",
        "",
        "## Indicateurs Clés (Headline)",
        "",
    ]
    for h in kpis["headline"]:
        src_badge = f" `[table: {h['table']}]`" if h.get("table") and h["table"] != "analytical_base" else ""
        desc = f" — *{h['definition']}*" if h.get("definition") else ""
        L.append(f"- **{h['label']}**{src_badge} : `{h['formatted']}`{desc}")

    if kpis["segment_comparisons"]:
        L += ["", "## Comparaisons & Insights par Segment", ""]
        for comp in kpis["segment_comparisons"]:
            src_badge = f" *(table: `{comp['table']}`)*" if comp.get("table") != "analytical_base" else ""
            L.append(f"### {comp['label']}{src_badge}")
            if comp.get("insight_question"):
                L.append(f"*{comp['insight_question']}*")
            L.append("")
            L += _md_table(
                ["Segment", "Effectif", f"Valeur ({comp['agg']})"],
                [[s["segment"], f"{s['count']:,}", fmt_val(s["value"], comp["unit"])] for s in comp["segments"]],
            )
            L.append("")

    for tr in kpis["time_trends"]:
        src_badge = f" *(table: `{tr['table']}`)*" if tr.get("table") != "analytical_base" else ""
        L += ["", f"## Tendance Temporelle : {tr['label']}{src_badge}", ""]
        L += _md_table(
            ["Période", "Valeur", "Variation période préc."],
            [
                [
                    pt["period"],
                    fmt_val(pt["value"], tr["unit"]),
                    f"{pt['period_growth_pct']:+.2f}%" if pt["period_growth_pct"] is not None else "N/A",
                ]
                for pt in tr["series"][-24:]
            ],
        )

    for bd in kpis["breakdowns"]:
        src_badge = f" *(table: `{bd['table']}`)*" if bd.get("table") != "analytical_base" else ""
        L += ["", f"## Découpage : {bd['label']} (Top {len(bd['rows'])} / {bd['total_segments']}){src_badge}", ""]
        L += _md_table(
            [bd["dimension_column"], f"Valeur ({bd['agg']})", "Part (%)"],
            [
                [
                    r["segment"],
                    fmt_val(r["value"], bd["unit"]),
                    f"{r['share_pct']:.2f}%" if r["share_pct"] is not None else "-",
                ]
                for r in bd["rows"]
            ],
        )

    if kpis["limitations"]:
        L += ["", "## Limites Méthodologiques", ""] + [f"- {lim}" for lim in kpis["limitations"]]

    (OUT / "generic_kpi_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 7. Point d'entrée de l'agent (avec boucle Plan -> Execute -> Validate -> Retry)
# ---------------------------------------------------------------------------
def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    print("1) Chargement du contexte ETL (analytical_base + tables nettoyées + model + metadata)")
    tables, model, metadata, annotations, etl_decision = load_etl_context()
    schema = summarize_schema_for_llm(tables, model, metadata, annotations)

    attempts_log = []
    feedback_errors = None
    final_results = None
    planner_used = "unknown"
    plan_used = None

    for attempt in range(1, MAX_PLAN_ATTEMPTS + 1):
        print(f"2) Planification des KPI (essai {attempt}/{MAX_PLAN_ATTEMPTS})")
        plan, planner_used = generate_kpi_plan(schema, feedback_errors=feedback_errors)
        results, errors = execute_and_validate(tables, plan)
        attempts_log.append({"attempt": attempt, "planner": planner_used, "validation_errors": errors})
        final_results, plan_used = results, plan

        if not errors:
            print("   Validation OK : tous les KPI planifiés ont été calculés sans erreur.")
            break
        print(f"   {len(errors)} erreur(s) de validation détectée(s) : {errors[:2]}")
        feedback_errors = errors

    etl_warnings = etl_decision.get("warnings", [])
    if etl_warnings:
        final_results["limitations"].extend([f"[ETL] {w}" for w in etl_warnings])

    base_df = tables["analytical_base"]
    used_tables = {"analytical_base"} | {
        s.table
        for s in (plan_used.headline + plan_used.time_trends + plan_used.breakdowns + plan_used.segment_comparisons)
    }
    rows_by_table = {t: int(len(tables[t])) for t in sorted(used_tables) if t in tables}

    output_payload = {
        "agent": "kpi_agent",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "planner": planner_used,
        "attempts": len(attempts_log),
        "attempts_log": attempts_log,
        "source": {
            "analytical_base": "data/curated/analytical_base.csv",
            "clean_tables_dir": "data/clean",
            "model": "data/etl/model.json",
            "semantic_metadata": "data/etl/semantic_metadata.json",
            "fact_table": metadata.get("fact_table"),
            "schema_type": model.get("schema_type", "unknown"),
        },
        "coverage": {
            "rows_total": int(len(base_df)),
            "columns_total": int(len(base_df.columns)),
            "rows_by_table": rows_by_table,
            "headline_computed": len(final_results["headline"]),
            "trends_computed": len(final_results["time_trends"]),
            "breakdowns_computed": len(final_results["breakdowns"]),
        },
        **final_results,
    }

    (OUT / "generic_kpi_results.json").write_text(
        json.dumps(output_payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    (OUT / "generic_kpi_plan.json").write_text(
        json.dumps(plan_used.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(output_payload)

    tables_touched = sorted({h["table"] for h in final_results["headline"]})
    print(f"KPI Agent terminé ({planner_used}) : {len(final_results['headline'])} KPI clés "
          f"(sur {', '.join(tables_touched)}), {len(final_results['breakdowns'])} découpages.")
    return {
        "ok": True,
        "status": "success",
        "domain_detected": final_results["domain_detected"],
        "planner": planner_used,
        "tables_analyzed": tables_touched,
        "headline_count": len(final_results["headline"]),
        "breakdowns_count": len(final_results["breakdowns"]),
        "outputs": ["data/kpi/generic_kpi_results.json", "data/kpi/generic_kpi_plan.json",
                    "data/kpi/generic_kpi_report.md"],
    }


if __name__ == "__main__":
    run()