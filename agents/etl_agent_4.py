"""ETL Agent BIFlow (générique) : fonctionne sur n'importe quel dossier de CSV.

Principe : aucune colonne, aucune table, aucune règle métier n'est écrite en dur.
  1. analyse automatique de chaque colonne (rôle : id, numérique, date, catégorie, texte...)
  2. construction d'un PLAN d'opérations choisies dans un catalogue fermé (CATALOG)
  3. (optionnel) le LLM relit le plan et peut seulement mettre un VETO ; il ne touche jamais aux données
  4. exécution déterministe du plan, journalisée (lineage)
  5. découverte automatique des clés et des jointures (inclusion de valeurs + similarité de noms)
  6. validation générique ; en cas d'échec, nouvel essai plus prudent (3 niveaux)
  7. inférence du modèle (étoile / constellation / flocon) -> model.json (code déterministe)
  8. (optionnel) le LLM PROPOSE une lecture métier du modèle -> model_annotations.json (à valider par l'humain)
Sorties : data/clean/<table>.csv, data/curated/analytical_base.csv, data/etl/*.json|md

Options de lancement :
  --data-dir=chemin   dossier contenant les CSV
  --no-llm            désactive tout appel au LLM (100 % déterministe)
  --no-review         désactive seulement la relecture du plan (llm_review), garde l'annotation du modèle
"""
import copy
import json
import os
import re
import sys
import unicodedata
import warnings
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
warnings.filterwarnings("ignore", message=".*fixed sampling defaults.*")

CLEAN = ROOT / "data" / "clean"
CURATED = ROOT / "data" / "curated"
OUT = ROOT / "data" / "etl"
PROFILER_DECISION = ROOT / "data" / "profile" / "profiler_agent_decision.json"
CONFIG_PATH = ROOT / "etl_config.json"

# Réglages GÉNÉRIQUES uniquement (aucun réglage métier). Modifiables via etl_config.json.
SETTINGS = {
    "numeric_parse_min_ratio": 0.95,   # % de valeurs lisibles pour convertir une colonne en nombre
    "date_parse_min_ratio": 0.95,      # idem pour les dates
    "impute_max_missing_pct": 40.0,    # au-delà, on n'impute pas (colonne signalée peu fiable)
    "flag_missing_min_pct": 1.0,       # colonne indicateur <col>_was_missing si manquants >= ce seuil
    "outlier_iqr_k": 1.5,
    "fk_min_inclusion": 0.95,          # valeurs de la clé étrangère retrouvées dans la clé parente
    "category_max_distinct": 5000,     # normalisation de casse/accents seulement sous ce seuil
    "max_attempts": 3,
}

# ---------------------------------------------------------------------------
# Catalogue fermé d'opérations + schéma du plan
# ---------------------------------------------------------------------------
PRIORITY = {"deduplicate": 0, "drop_column": 1, "strip_text": 2, "cast_numeric": 3, "parse_date": 3,
            "standardize_categories": 4, "flag_missing": 5, "impute": 6, "flag_outliers": 7}


class Op(BaseModel):
    table: str
    op: Literal["deduplicate", "drop_column", "strip_text", "cast_numeric", "parse_date",
                "standardize_categories", "flag_missing", "impute", "flag_outliers"]
    column: Optional[str] = None
    params: dict = Field(default_factory=dict)
    reason: str = ""


def load_settings() -> dict:
    cfg = dict(SETTINGS)
    if CONFIG_PATH.exists():
        try:
            saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            cfg.update({k: v for k, v in saved.items() if k in SETTINGS})   # clés métier ignorées
        except Exception:
            pass
    return cfg


class CorrectionLog:
    def __init__(self):
        self.entries = []

    def add(self, step, table, action, rows_affected, reason):
        self.entries.append({"step": step, "table": table, "action": action,
                             "rows_affected": int(rows_affected), "reason": reason,
                             "timestamp": datetime.now().isoformat(timespec="seconds")})
        print(f"  [{step}] {table}: {action} -> {int(rows_affected)}")


# ---------------------------------------------------------------------------
# 1. Chargement : n'importe quel dossier de CSV
# ---------------------------------------------------------------------------
def find_raw_dir() -> Path:
    for a in sys.argv:
        if a.startswith("--data-dir="):
            return Path(a.split("=", 1)[1])
    if os.environ.get("BIFLOW_RAW_DIR"):
        return Path(os.environ["BIFLOW_RAW_DIR"])
    base = ROOT / "data" / "raw"
    if base.exists():
        subs = [d for d in sorted(base.iterdir()) if d.is_dir() and any(d.glob("*.csv"))]
        if subs:
            return subs[0]
        if any(base.glob("*.csv")):
            return base
    raise FileNotFoundError("Aucun CSV trouvé : utiliser --data-dir=chemin ou BIFLOW_RAW_DIR")


def read_csv_safe(path: Path) -> pd.DataFrame:
    """Tout est lu en texte (les identifiants gardent leurs zéros) ; encodage et séparateur détectés."""
    last = None
    for enc in ("utf-8", "latin-1"):
        for sep in (",", ";", "\t", "|"):
            try:
                df = pd.read_csv(path, dtype=str, encoding=enc, sep=sep,
                                 na_values=["", "NA", "N/A", "null", "NULL", "None", "nan", "NaN"])
                if df.shape[1] > 1:
                    return df
                last = df
            except Exception as e:
                last = e
    if isinstance(last, pd.DataFrame):
        return last
    raise ValueError(f"Lecture impossible : {path} ({last})")


def load_tables(raw_dir: Path) -> dict:
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"Aucun CSV dans {raw_dir}")
    return {f.stem: read_csv_safe(f) for f in files}


# ---------------------------------------------------------------------------
# 2. Analyse des colonnes (rôle inféré à partir des données, jamais du nom de dataset)
# ---------------------------------------------------------------------------
ID_TOKENS = {"id", "uuid", "key", "code", "zip", "sku", "ref", "num", "number", "no", "prefix", "phone", "postal"}
DATE_TOKENS = {"date", "time", "timestamp", "datetime", "at"}
GENERIC_TOKENS = {"id", "key", "code", "no", "num", "number", "ref", "uuid", "unique"}
BOOL_VALUES = {"true", "false", "yes", "no", "y", "n", "t", "f", "0", "1", "oui", "non"}
DATE_RE = re.compile(r"^\s*\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}([ T]\d{1,2}:\d{2}(:\d{2})?)?")


def tokens(name: str) -> set:
    s = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)
    return {t for t in re.split(r"[^a-z0-9]+", s.lower()) if t}


def date_ok(nn: pd.Series, min_ratio: float) -> bool:
    sample = nn.head(500)
    if len(sample) == 0 or not sample.str.match(DATE_RE).mean() >= min_ratio:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(sample, errors="coerce").notna().mean() >= min_ratio


def iqr_mask(num: pd.Series, k: float):
    q1, q3 = num.quantile(0.25), num.quantile(0.75)
    iqr = q3 - q1
    if not iqr or pd.isna(iqr):
        return None
    return (num < q1 - k * iqr) | (num > q3 + k * iqr)


def analyze_column(s: pd.Series, name: str, S: dict) -> dict:
    nn = s.dropna()
    n, m = len(s), len(nn)
    d = int(nn.nunique())
    info = {"missing": int(n - m), "missing_pct": round(100 * (n - m) / n, 2) if n else 0.0, "distinct": d}
    if m == 0:
        return {**info, "role": "empty"}
    if d == 1:
        return {**info, "role": "constant"}
    toks, uniq = tokens(name), d / m
    if d <= 2 and set(nn.str.strip().str.lower().unique()) <= BOOL_VALUES:
        return {**info, "role": "boolean"}
    if (toks & DATE_TOKENS) and date_ok(nn, S["date_parse_min_ratio"]):
        return {**info, "role": "date"}
    if toks & ID_TOKENS:
        return {**info, "role": "id"}
    num = pd.to_numeric(nn, errors="coerce")
    if num.notna().mean() >= S["numeric_parse_min_ratio"]:
        x = num.dropna()
        if bool((x % 1 == 0).all()) and uniq >= 0.99 and m > 20:
            return {**info, "role": "id"}
        mask = iqr_mask(x, S["outlier_iqr_k"]) if d > 10 else None
        return {**info, "role": "numeric",
                "outlier_pct": round(100 * float(mask.mean()), 2) if mask is not None else 0.0}
    if date_ok(nn, S["date_parse_min_ratio"]):
        return {**info, "role": "date"}
    if nn.str.len().mean() > 40:
        return {**info, "role": "text"}
    if uniq >= 0.95 and m > 20:
        return {**info, "role": "id"}
    return {**info, "role": "category"}


def analyze_tables(tables: dict, S: dict) -> dict:
    return {t: {c: analyze_column(df[c], c, S) for c in df.columns} for t, df in tables.items()}


# ---------------------------------------------------------------------------
# 3. Plan d'opérations (règles déterministes ; le LLM peut seulement opposer un veto)
# ---------------------------------------------------------------------------
def build_plan(tables: dict, analysis: dict, S: dict) -> list:
    ops = []
    for t, df in tables.items():
        dups = int(df.duplicated().sum())
        if dups:
            ops.append(Op(table=t, op="deduplicate", reason=f"{dups} lignes entièrement dupliquées"))
        for c, i in analysis[t].items():
            r, mp = i["role"], i["missing_pct"]
            can_impute = 0 < mp <= S["impute_max_missing_pct"]
            if r == "empty":
                ops.append(Op(table=t, op="drop_column", column=c, reason="colonne 100 % vide"))
            elif r == "numeric":
                ops.append(Op(table=t, op="cast_numeric", column=c, reason="valeurs lisibles comme nombres"))
                if can_impute and mp >= S["flag_missing_min_pct"]:
                    ops.append(Op(table=t, op="flag_missing", column=c, reason=f"{mp} % manquants"))
                if can_impute:
                    ops.append(Op(table=t, op="impute", column=c, params={"strategy": "median"},
                                  reason=f"{mp} % manquants, colonne numérique"))
                if i.get("outlier_pct", 0) > 0:
                    ops.append(Op(table=t, op="flag_outliers", column=c, params={"k": S["outlier_iqr_k"]},
                                  reason=f"{i['outlier_pct']} % atypiques (IQR) : signalés, jamais supprimés"))
            elif r == "date":
                ops.append(Op(table=t, op="parse_date", column=c, reason="valeurs lisibles comme dates"))
            elif r == "category":
                if i["distinct"] <= S["category_max_distinct"]:
                    ops.append(Op(table=t, op="standardize_categories", column=c,
                                  reason="unifie casse, accents et espaces"))
                if can_impute:
                    ops.append(Op(table=t, op="impute", column=c,
                                  params={"strategy": "constant", "value": "unknown"},
                                  reason=f"{mp} % manquants, colonne catégorielle"))
            elif r in ("text", "id"):
                ops.append(Op(table=t, op="strip_text", column=c, reason="supprime espaces superflus"))
    return ops


def llm_review(analysis: dict, plan: list):
    """Le LLM relit le plan et peut seulement opposer un veto à des opérations. Aucune donnée n'est vue."""
    model = os.environ.get("LLM_MODEL")
    if not model or "--no-llm" in sys.argv or "--no-review" in sys.argv:
        return None
    try:
        from langchain.chat_models import init_chat_model
        from langchain_core.messages import HumanMessage, SystemMessage
        from profiler_agent import invoke_with_retry, parse_json, text_of
        payload = {
            "colonnes": {t: {c: {k: i[k] for k in ("role", "missing_pct", "distinct")} for c, i in cols.items()}
                         for t, cols in analysis.items()},
            "plan": [{"table": o.table, "op": o.op, "column": o.column, "reason": o.reason} for o in plan],
        }
        system = ("Tu es l'ETL Agent d'un pipeline BI. On te donne, pour des tables inconnues, le rôle inféré "
                  "de chaque colonne et un plan de nettoyage. Repère les opérations dangereuses ou absurdes "
                  "(ex. imputer une valeur qui a un sens métier quand elle manque, convertir un identifiant). "
                  "Tu peux seulement opposer un VETO. Réponds UNIQUEMENT par un JSON : "
                  '{"veto": [{"table": "...", "column": "...", "op": "...", "reason": "..."}], "notes": "..."}')
        llm = init_chat_model(model, temperature=float(os.environ.get("LLM_TEMPERATURE", "0")))
        ai = invoke_with_retry(llm, [SystemMessage(system),
                                     HumanMessage(json.dumps(payload, ensure_ascii=False, default=str))])
        return parse_json(text_of(ai))
    except Exception as e:
        print(f"  (LLM indisponible pour la revue du plan : {str(e)[:100]})")
        return None


def apply_vetoes(plan: list, review: dict, log: CorrectionLog) -> list:
    vetoes = (review or {}).get("veto", []) if isinstance(review, dict) else []
    for v in vetoes:
        if not isinstance(v, dict):
            continue
        before = len(plan)
        plan = [o for o in plan if not (o.table == v.get("table") and o.column == v.get("column")
                                        and o.op == v.get("op"))]
        if len(plan) < before:
            log.add("veto_llm", v.get("table"), f"{v.get('op')} sur {v.get('column')} retiré du plan", 0,
                    str(v.get("reason", ""))[:200])
    return plan


# ---------------------------------------------------------------------------
# 4. Outils génériques : aucune connaissance d'un nom de colonne
# ---------------------------------------------------------------------------
def _fold(v) -> str:
    v = unicodedata.normalize("NFKD", str(v))
    v = "".join(ch for ch in v if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", v).strip().lower()


def op_deduplicate(df, op, S, log):
    out = df.drop_duplicates()
    log.add("doublons", op.table, "lignes entièrement dupliquées supprimées", len(df) - len(out), op.reason)
    return out.reset_index(drop=True)


def op_drop_column(df, op, S, log):
    log.add("colonnes", op.table, f"colonne vide supprimée : {op.column}", len(df), op.reason)
    return df.drop(columns=[op.column])


def op_strip_text(df, op, S, log):
    s = df[op.column]
    if not pd.api.types.is_string_dtype(s):
        return df
    new = s.str.strip().str.replace(r"\s+", " ", regex=True)
    changed = int(((new != s) & s.notna()).sum())
    if changed:
        df[op.column] = new
        log.add("texte", op.table, f"espaces nettoyés : {op.column}", changed, op.reason)
    return df


def op_cast_numeric(df, op, S, log):
    s = df[op.column]
    before = int(s.isna().sum())
    out = pd.to_numeric(s, errors="coerce")
    new_null = int(out.isna().sum()) - before
    if new_null > 0.05 * max(1, len(s) - before):
        log.add("types", op.table, f"conversion numérique ANNULÉE : {op.column}", new_null,
                "trop de valeurs illisibles : colonne conservée en texte")
        return df
    df[op.column] = out
    log.add("types", op.table, f"texte -> nombre : {op.column}", len(df), f"{new_null} valeurs illisibles mises à vide")
    return df


def op_parse_date(df, op, S, log):
    s = df[op.column]
    before = int(s.isna().sum())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = pd.to_datetime(s, errors="coerce")
    new_null = int(out.isna().sum()) - before
    if new_null > 0.05 * max(1, len(s) - before):
        log.add("types", op.table, f"conversion date ANNULÉE : {op.column}", new_null,
                "trop de valeurs illisibles : colonne conservée en texte")
        return df
    df[op.column] = out
    log.add("types", op.table, f"texte -> date : {op.column}", len(df), f"{new_null} valeurs illisibles mises à vide")
    return df


def op_standardize_categories(df, op, S, log):
    s = df[op.column]
    nn = s.dropna()
    if nn.empty or not pd.api.types.is_string_dtype(s):
        return df
    keys = nn.map(_fold)
    canon = nn.groupby(keys).agg(lambda x: x.value_counts().idxmax())
    mapped = keys.map(canon)
    changed = int((mapped != nn).sum())
    if changed:
        df.loc[nn.index, op.column] = mapped
        log.add("categories", op.table, f"variantes de casse/accents unifiées : {op.column}", changed, op.reason)
    return df


def op_flag_missing(df, op, S, log):
    df[f"{op.column}_was_missing"] = df[op.column].isna()
    return df


def op_impute(df, op, S, log):
    s, strat = df[op.column], op.params.get("strategy", "median")
    n = int(s.isna().sum())
    if n == 0:
        return df
    if strat == "median" and pd.api.types.is_numeric_dtype(s):
        fill = s.median()
    elif strat == "mode" and s.notna().any():
        fill = s.mode().iloc[0]
    elif strat == "constant":
        fill = op.params.get("value", "unknown")
    else:
        return df
    df[op.column] = s.fillna(fill)
    log.add("imputation", op.table, f"{strat} appliqué : {op.column}", n, op.reason)
    return df


def op_flag_outliers(df, op, S, log):
    s = df[op.column]
    if not pd.api.types.is_numeric_dtype(s):
        return df
    mask = iqr_mask(s, float(op.params.get("k", S["outlier_iqr_k"])))
    if mask is None:
        return df
    df[f"{op.column}_is_outlier"] = mask.fillna(False)
    log.add("atypiques", op.table, f"valeurs atypiques signalées : {op.column}", int(mask.sum()), op.reason)
    return df


TOOLS = {"deduplicate": op_deduplicate, "drop_column": op_drop_column, "strip_text": op_strip_text,
         "cast_numeric": op_cast_numeric, "parse_date": op_parse_date,
         "standardize_categories": op_standardize_categories, "flag_missing": op_flag_missing,
         "impute": op_impute, "flag_outliers": op_flag_outliers}

MODES = {0: "standard", 1: "prudent (sans imputation ni suppression de colonne)",
         2: "minimal (types et espaces seulement)"}
MODE_ALLOWED = {0: set(PRIORITY), 1: set(PRIORITY) - {"impute", "drop_column", "flag_missing"},
                2: {"strip_text", "cast_numeric", "parse_date"}}


def execute_plan(raw: dict, plan: list, mode: int, S: dict, log: CorrectionLog):
    tables = {t: df.copy() for t, df in raw.items()}
    applied = []
    for op in sorted(plan, key=lambda o: PRIORITY[o.op]):
        if op.op not in MODE_ALLOWED[mode] or op.table not in tables:
            continue
        df = tables[op.table]
        if op.column is not None and op.column not in df.columns:
            continue
        tables[op.table] = TOOLS[op.op](df, op, S, log)
        applied.append(op)
    return tables, applied


# ---------------------------------------------------------------------------
# 5. Découverte automatique des clés et des jointures
# ---------------------------------------------------------------------------
def name_similarity(a: str, b: str) -> float:
    na, nb = re.sub(r"[^a-z0-9]", "", a.lower()), re.sub(r"[^a-z0-9]", "", b.lower())
    if na == nb:
        return 1.0
    if (tokens(a) - GENERIC_TOKENS) & (tokens(b) - GENERIC_TOKENS):
        return 0.8
    return SequenceMatcher(None, na, nb).ratio()


def discover_relations(tables: dict, analysis: dict, S: dict):
    keys = {}
    for t, df in tables.items():
        keys[t] = [c for c, i in analysis[t].items()
                   if i["role"] == "id" and c in df.columns and len(df) > 1
                   and df[c].notna().all() and df[c].is_unique]
    edges = {}
    for ta, dfa in tables.items():
        for ca, i in analysis[ta].items():
            if i["role"] != "id" or ca not in dfa.columns:
                continue
            col = dfa[ca].dropna()
            va = set(col.unique())
            if len(va) < 2:
                continue
            for tb, kb in keys.items():
                if tb == ta:
                    continue
                for cb in kb:
                    inc = len(va & set(tables[tb][cb])) / len(va)
                    if inc < S["fk_min_inclusion"]:
                        continue
                    sim = name_similarity(ca, cb)
                    if sim < 0.5 and not (len(va) >= 100 and inc >= 0.999):
                        continue
                    e = {"child": ta, "child_col": ca, "parent": tb, "parent_col": cb,
                         "inclusion": round(inc, 4), "name_similarity": round(sim, 2),
                         "cardinality": "1-1" if ca in keys[ta] else "N-1",
                         "orphan_rows_pct": round(100 * float((~col.isin(tables[tb][cb])).mean()), 2)}
                    k = (ta, tb)
                    if k not in edges or (sim, inc) > (edges[k]["name_similarity"], edges[k]["inclusion"]):
                        edges[k] = e
    return keys, list(edges.values())


def choose_fact(tables: dict, edges: list) -> str:
    parents = {t: len({e["parent"] for e in edges if e["child"] == t}) for t in tables}
    return max(tables, key=lambda t: (parents[t], len(tables[t])))


def build_base(tables: dict, edges: list, fact: str, log: CorrectionLog):
    """Jointure N-1 depuis la table de faits : le nombre de lignes ne peut pas augmenter."""
    base = tables[fact].copy()
    colmap = {(fact, c): c for c in base.columns}
    joined, path, progress = {fact}, [], True
    while progress:
        progress = False
        for e in edges:
            if e["child"] in joined and e["parent"] not in joined and (e["child"], e["child_col"]) in colmap:
                p, pk = e["parent"], e["parent_col"]
                right = tables[p].rename(columns={c: f"{p}__{c}" for c in tables[p].columns})
                n0 = len(base)
                base = base.merge(right, left_on=colmap[(e["child"], e["child_col"])],
                                  right_on=f"{p}__{pk}", how="left").drop(columns=[f"{p}__{pk}"])
                for c in tables[p].columns:
                    colmap[(p, c)] = f"{p}__{c}" if c != pk else colmap[(e["child"], e["child_col"])]
                joined.add(p)
                path.append({"from": f"{e['child']}.{e['child_col']}", "to": f"{p}.{pk}",
                             "rows_before": n0, "rows_after": len(base)})
                log.add("jointure", p, f"jointure {e['child']}.{e['child_col']} -> {p}.{pk}", len(base),
                        f"inclusion {e['inclusion']}, cardinalité {e['cardinality']}")
                progress = True
    not_merged = sorted(set(tables) - joined)
    return base, path, not_merged, colmap


# ---------------------------------------------------------------------------
# 6. Validation générique
# ---------------------------------------------------------------------------
def validate(raw, clean, analysis, applied, base, fact, edges, keys, S):
    checks, warnings_list = [], []

    def check(name, ok, detail=""):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    for t in raw:
        dedup = any(o.op == "deduplicate" and o.table == t for o in applied)
        expected = len(raw[t].drop_duplicates()) if dedup else len(raw[t])
        check(f"{t} : lignes conservées (hors doublons supprimés)", len(clean[t]) == expected,
              f"{len(raw[t])} -> {len(clean[t])}")
        for c, i in analysis[t].items():
            if c in clean[t].columns and i["role"] in ("numeric", "date") and len(clean[t]) and len(raw[t]):
                rate_before, rate_after = raw[t][c].isna().mean(), clean[t][c].isna().mean()
                if rate_after - rate_before > 0.05:
                    check(f"{t}.{c} : pas de perte de valeurs à la conversion", False,
                          f"manquants {rate_before:.1%} -> {rate_after:.1%}")
    for o in applied:
        if o.op == "impute" and o.column in clean[o.table].columns:
            left = int(clean[o.table][o.column].isna().sum())
            check(f"{o.table}.{o.column} : plus de manquants après imputation", left == 0, f"{left} restants")
    check("table d'analyse : même nombre de lignes que la table de faits", len(base) == len(clean[fact]),
          f"{len(clean[fact])} vs {len(base)}")
    check("table d'analyse : noms de colonnes uniques", base.columns.is_unique, "")
    for e in edges:
        if e["orphan_rows_pct"] > 5:
            warnings_list.append(f"{e['child']}.{e['child_col']} : {e['orphan_rows_pct']} % de lignes orphelines")
    return checks, warnings_list


# ---------------------------------------------------------------------------
# 7. Inférence du modèle (étoile / constellation / flocon), sans rien câbler
# ---------------------------------------------------------------------------
def infer_model(tables: dict, analysis: dict, keys: dict, edges: list) -> dict:
    names = list(tables)

    # relations : on garde les N-1, et une seule fois chaque lien 1-1
    seen, rel = set(), []
    for e in edges:
        if e["cardinality"] == "1-1":
            k = frozenset([(e["child"], e["child_col"]), (e["parent"], e["parent_col"])])
            if k in seen:
                continue
            seen.add(k)
        rel.append(e)

    out_n1 = {t: set() for t in names}   # tables que t référence
    in_n1 = {t: set() for t in names}    # tables qui référencent t
    for e in rel:
        if e["cardinality"] == "N-1":
            out_n1[e["child"]].add(e["parent"])
            in_n1[e["parent"]].add(e["child"])

    measures = {t: [c for c, i in analysis[t].items()
                    if i["role"] == "numeric" and c in tables[t].columns] for t in names}
    times = {t: [c for c, i in analysis[t].items()
                 if i["role"] == "date" and c in tables[t].columns] for t in names}

    # fait = référence d'autres tables, n'est référencé par personne, a des mesures numériques
    facts = [t for t in names if out_n1[t] and not in_n1[t] and measures[t]]
    linked = {t for e in rel for t in (e["child"], e["parent"])}
    dims = [t for t in names if t not in facts and t in linked]
    isolated = [t for t in names if t not in linked]

    dim_links = [e for e in rel if e["child"] in dims and e["parent"] in dims]
    if not facts:
        schema = "indéterminé"
    elif len(facts) == 1:
        schema = "flocon" if dim_links else "étoile"
    else:
        schema = "constellation"

    return {
        "schema_type": schema,
        "facts": [{"table": t,
                   "grain": f"1 ligne par {keys[t][0] if keys.get(t) else 'enregistrement'}",
                   "measures": measures[t], "time": times[t]} for t in facts],
        "dimensions": [{"table": t, "key": (keys.get(t) or [None])[0],
                        "referenced_by": sorted(in_n1[t])} for t in dims],
        "links": [{"fact": e["child"], "dimension": e["parent"], "on": e["child_col"]}
                  for e in rel if e["child"] in facts],
        "dimension_links": [{"from": e["child"], "to": e["parent"], "on": e["child_col"]} for e in dim_links],
        "isolated_tables": isolated,
    }


# ---------------------------------------------------------------------------
# 8. Annotation du modèle par le LLM (propose seulement ; l'humain valide)
# ---------------------------------------------------------------------------
def llm_annotate_model(model: dict):
    """Le LLM PROPOSE une lecture métier de tout le modèle (schéma, faits, dimensions, isolées, mesures).
    Il ne voit que des noms de tables/colonnes (jamais de données) et n'écrit jamais dans model.json."""
    llm_name = os.environ.get("LLM_MODEL")
    if not llm_name or "--no-llm" in sys.argv:
        return None
    try:
        from langchain.chat_models import init_chat_model
        from langchain_core.messages import HumanMessage, SystemMessage
        from profiler_agent import invoke_with_retry, parse_json, text_of
        system = ("Tu es analyste BI. On te donne un modèle de données déduit automatiquement par du code "
                  "(type de schéma, tables de faits avec mesures et dates, dimensions, liens, tables isolées). "
                  "Propose une lecture métier, sans rien inventer au-delà des noms fournis. Réponds UNIQUEMENT "
                  "par un JSON : "
                  '{"schema_review": {"agrees": true, "comment": "..."}, '
                  '"facts": [{"table": "...", "meaning": "...", "grain": "..."}], '
                  '"dimensions": [{"table": "...", "meaning": "..."}], '
                  '"isolated_tables": [{"table": "...", "possible_role": "..."}], '
                  '"measures": [{"table": "...", "column": "...", "meaning": "...", '
                  '"aggregation": "sum|avg|count|min|max|none"}]}')
        llm = init_chat_model(llm_name, temperature=float(os.environ.get("LLM_TEMPERATURE", "0")))
        ai = invoke_with_retry(llm, [SystemMessage(system),
                                     HumanMessage(json.dumps(model, ensure_ascii=False))])
        data = parse_json(text_of(ai))
        if not isinstance(data, dict):
            return None

        fact_tables = {f["table"] for f in model["facts"]}
        dim_tables = {d["table"] for d in model["dimensions"]}
        iso_tables = set(model["isolated_tables"])
        allowed_m = {(f["table"], c) for f in model["facts"] for c in f["measures"]}
        aggs = {"sum", "avg", "count", "min", "max", "none"}

        def rows(key, tables, fields):
            out = []
            for x in data.get(key, []):
                if isinstance(x, dict) and x.get("table") in tables:
                    row = {"table": x["table"]}
                    row.update({f: str(x.get(f, ""))[:200] for f in fields})
                    row["validated"] = False
                    out.append(row)
            return out

        sr = data.get("schema_review", {}) if isinstance(data.get("schema_review"), dict) else {}
        measures = []
        for m in data.get("measures", []):
            if isinstance(m, dict) and (m.get("table"), m.get("column")) in allowed_m \
                    and m.get("aggregation") in aggs:
                measures.append({"table": m["table"], "column": m["column"],
                                 "meaning": str(m.get("meaning", ""))[:150],
                                 "aggregation": m["aggregation"], "validated": False})
        return {"proposed_by": llm_name,
                "schema_review": {"code_says": model["schema_type"], "llm_agrees": bool(sr.get("agrees")),
                                  "comment": str(sr.get("comment", ""))[:300], "validated": False},
                "facts": rows("facts", fact_tables, ("meaning", "grain")),
                "dimensions": rows("dimensions", dim_tables, ("meaning",)),
                "isolated_tables": rows("isolated_tables", iso_tables, ("possible_role",)),
                "measures": measures}
    except Exception as e:
        print(f"  (LLM indisponible pour l'annotation du modèle : {str(e)[:100]})")
        return None


# ---------------------------------------------------------------------------
# Rapports
# ---------------------------------------------------------------------------
def snapshot(df) -> dict:
    return {"rows": int(len(df)), "missing_cells": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum())}


def baseline_naive(raw: dict) -> dict:
    return {t: round(100 * len(df.dropna().drop_duplicates()) / max(1, len(df)), 1) for t, df in raw.items()}


def read_profiler_plan() -> dict:
    if PROFILER_DECISION.exists():
        try:
            return json.loads(PROFILER_DECISION.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def run_etl() -> dict:
    for d in (CLEAN, CURATED, OUT):
        d.mkdir(parents=True, exist_ok=True)
    S = load_settings()
    raw_dir = find_raw_dir()
    print(f"1) Chargement : {raw_dir}")
    raw = load_tables(raw_dir)
    before = {t: snapshot(df) for t, df in raw.items()}

    print("2) Analyse des colonnes")
    analysis = analyze_tables(raw, S)

    print("3) Plan de nettoyage")
    plan = build_plan(raw, analysis, S)
    review_log = CorrectionLog()
    review = llm_review(analysis, plan)
    plan = apply_vetoes(plan, review, review_log)
    try:
        plan = [Op.model_validate(o.model_dump()) for o in plan]   # plan valide ou rejeté
    except ValidationError as e:
        raise RuntimeError(f"Plan invalide : {e}")

    attempts, result = [], None
    for mode in range(min(S["max_attempts"], 3)):
        print(f"4) Exécution : essai {mode + 1} ({MODES[mode]})")
        log = CorrectionLog()
        clean, applied = execute_plan(raw, plan, mode, S, log)
        keys, edges = discover_relations(clean, analysis, S)
        fact = choose_fact(clean, edges)
        base, path, not_merged, colmap = build_base(clean, edges, fact, log)
        checks, warn = validate(raw, clean, analysis, applied, base, fact, edges, keys, S)
        failed = [c["check"] for c in checks if not c["passed"]]
        attempts.append({"attempt": mode + 1, "mode": MODES[mode], "failed_checks": failed})
        result = (clean, applied, keys, edges, fact, base, path, not_merged, colmap, checks, warn, log)
        if not failed:
            break
        print(f"   {len(failed)} contrôle(s) en échec -> essai plus prudent")

    clean, applied, keys, edges, fact, base, path, not_merged, colmap, checks, warn, log = result
    model = infer_model(clean, analysis, keys, edges)
    all_ok = all(c["passed"] for c in checks)
    log.entries = review_log.entries + log.entries

    print("5) Écriture des fichiers")
    for t, df in clean.items():
        df.to_csv(CLEAN / f"{t}.csv", index=False)
    base.to_csv(CURATED / "analytical_base.csv", index=False)
    after = {t: snapshot(df) for t, df in clean.items()}
    naive = baseline_naive(raw)
    kept = {t: round(100 * len(clean[t]) / max(1, len(raw[t])), 1) for t in raw}

    metadata = {
        "source_dir": str(raw_dir), "fact_table": fact,
        "grain": f"1 ligne par ligne de {fact}", "analytical_base": "data/curated/analytical_base.csv",
        "tables": {t: {"rows": len(clean[t]), "key_candidates": keys.get(t, []),
                       "columns": {c: analysis[t][c]["role"] for c in analysis[t] if c in clean[t].columns}}
                   for t in clean},
        "relations": edges, "join_path": path, "tables_not_merged": not_merged,
        "base_columns": {v: f"{t}.{c}" for (t, c), v in colmap.items()},
    }
    (OUT / "semantic_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "model.json").write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"   Modèle déduit : {model['schema_type']} ({len(model['facts'])} table(s) de faits)")

    # Annotation LLM (optionnelle) : propose le sens métier des mesures, fichier séparé, à valider
    annotations = llm_annotate_model(model)
    if annotations:
        (OUT / "model_annotations.json").write_text(json.dumps(annotations, indent=2, ensure_ascii=False),
                                                    encoding="utf-8")
        print(f"   Annotations LLM : schéma, {len(annotations['facts'])} fait(s), "
              f"{len(annotations['dimensions'])} dimension(s), {len(annotations['measures'])} mesure(s) "
              f"proposés, à valider")

    (OUT / "correction_log.json").write_text(json.dumps(log.entries, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "plan.json").write_text(json.dumps([o.model_dump() for o in plan], indent=2, ensure_ascii=False),
                                   encoding="utf-8")

    profiler = read_profiler_plan()
    covered = {(o.table, o.column) for o in applied}
    uncovered = [f"{i.get('table')}.{i.get('column')} : {i.get('diagnosis', '')[:120]}"
                 for i in profiler.get("issues", [])
                 if i.get("column") and (i.get("table"), i.get("column")) not in covered]

    decision = {
        "agent": "etl_agent", "run_at": datetime.now().isoformat(timespec="seconds"),
        "status": "success" if all_ok else "failed_validation", "settings_used": S,
        "attempts": attempts, "fact_table": fact, "rows_analytical_base": len(base),
        "schema_type": model["schema_type"], "model_file": "data/etl/model.json",
        "rows_before": {t: v["rows"] for t, v in before.items()},
        "quality_before": before, "quality_after": after,
        "rows_kept_pct": kept, "baseline_naive_rows_kept_pct": naive,
        "checks": checks, "warnings": warn,
        "profiler_issues_without_operation": uncovered,
        "llm_review_notes": (review or {}).get("notes") if isinstance(review, dict) else None,
        "next_agent": "kpi" if all_ok else "STOP - corriger l'ETL",
    }
    (OUT / "etl_agent_decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False, default=str),
                                                 encoding="utf-8")

    L = ["# ETL Report", "", f"Statut : **{decision['status']}**", f"Source : `{raw_dir}`",
         f"Modèle déduit : **{model['schema_type']}** ({len(model['facts'])} table(s) de faits)",
         f"Table de faits de la table d'analyse : **{fact}** ({len(base)} lignes)", "",
         "## Qualité avant / après", "", "| Table | Lignes | Manquants | Doublons | Lignes gardées | Baseline naïve |",
         "|---|---|---|---|---|---|"]
    for t in raw:
        b, a = before[t], after[t]
        L.append(f"| {t} | {b['rows']} -> {a['rows']} | {b['missing_cells']} -> {a['missing_cells']} | "
                 f"{b['duplicate_rows']} -> {a['duplicate_rows']} | {kept[t]} % | {naive[t]} % |")
    L += ["", "## Contrôles", ""] + [f"- {'OK' if c['passed'] else 'ECHEC'} : {c['check']} {c['detail']}" for c in checks]
    if warn:
        L += ["", "## Avertissements", ""] + [f"- {w}" for w in warn]
    L += ["", "## Relations découvertes", ""] + [
        f"- {e['child']}.{e['child_col']} -> {e['parent']}.{e['parent_col']} "
        f"({e['cardinality']}, inclusion {e['inclusion']}, orphelins {e['orphan_rows_pct']} %)" for e in edges]
    if not_merged:
        L += ["", "Tables non fusionnées dans la table à plat (elles dupliqueraient les lignes de la table de faits) : "
              + ", ".join(not_merged)]
    L += ["", "## Modèle déduit", "",
          f"- Type de schéma : {model['schema_type']}",
          "- Faits : " + (", ".join(f["table"] for f in model["facts"]) or "aucun"),
          "- Dimensions : " + (", ".join(d["table"] for d in model["dimensions"]) or "aucune"),
          "- Tables isolées : " + (", ".join(model["isolated_tables"]) or "aucune")]
    if annotations:
        sr = annotations["schema_review"]
        L += ["", "## Annotations LLM (à valider)", "",
              f"- Schéma : code = {sr['code_says']}, LLM d'accord = {sr['llm_agrees']} - {sr['comment']}"] + [
            f"- {m['table']}.{m['column']} : {m['meaning']} ({m['aggregation']})" for m in annotations["measures"]]
    L += ["", "## Essais", ""] + [f"- essai {a['attempt']} ({a['mode']}) : {len(a['failed_checks'])} échec(s)"
                                  for a in attempts]
    L += ["", "## Journal des corrections", ""] + [
        f"- [{e['step']}] {e['table']} : {e['action']} ({e['rows_affected']}) - {e['reason']}" for e in log.entries]
    if uncovered:
        L += ["", "## Problèmes du Profiler sans opération ETL", ""] + [f"- {u}" for u in uncovered]
    (OUT / "etl_report.md").write_text("\n".join(L), encoding="utf-8")

    print(f"\nTerminé. Statut : {decision['status']}. Voir data/etl/etl_report.md")
    return {"ok": all_ok, "status": decision["status"], "fact_table": fact,
            "schema_type": model["schema_type"],
            "rows_analytical_base": len(base), "attempts": len(attempts),
            "checks_failed": [c["check"] for c in checks if not c["passed"]]}


def run() -> dict:
    """Point d'entrée appelé par l'orchestrateur."""
    return run_etl()


if __name__ == "__main__":
    run()