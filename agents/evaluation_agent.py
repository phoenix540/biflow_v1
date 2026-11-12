"""
Agent Evaluation (module d'observabilité) — projet BIFlow.

Rôle : produire des MÉTRIQUES PAR AGENT après un run du pipeline. Ce n'est pas
une étape de transformation de données : l'évaluation lit le journal de
l'orchestrateur et les fichiers de sortie de chaque agent, puis calcule, pour
chaque agent, son statut, son temps d'exécution et un indicateur de qualité
issu de sa propre sortie.

Lancé automatiquement par l'orchestrateur (finish), juste APRÈS l'écriture de
orchestrator_log.json : il reflète donc le run qui vient de se terminer.
Peut aussi être lancé seul : python agents/evaluation_agent.py

Sortie :
- data/eval/evaluation.json
- data/eval/evaluation_report.md
"""
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data"
OUT_DIR = D / "eval"

SOURCES = {
    "orchestrator": D / "profile" / "orchestrator_log.json",
    "profiler":     D / "profile" / "profiler_agent_decision.json",
    "etl_decision": D / "etl" / "etl_agent_decision.json",   # ETL générique
    "etl_log":      D / "etl" / "correction_log.json",
    "profile":      D / "profile" / "profile.json",          # score de fiabilité global (statistique)
    "kpi":          D / "kpi" / "generic_kpi_results.json",  # KPI générique
    "model":        D / "etl" / "olist_model_decision.json", # couche métier Olist
    "model_log":    D / "etl" / "olist_correction_log.json",
    "olist_kpi":    D / "kpi" / "kpi_results.json",
    "analyst":      D / "analyst" / "analyst_insights.json",
    "dashboard":    D / "dashboard" / "dashboard_agent_decision.json",
    "auditor":      D / "audit" / "lineage_cards.json",
}


def load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# Métriques par agent, extraites de la sortie native de chaque agent.
# quality_score dans [0,1] ; None si non mesurable.
# ---------------------------------------------------------------------------
def eval_profiler(data, profile=None) -> dict:
    if not data:
        return {"present": False}
    issues = data.get("issues", [])
    high = sum(1 for i in issues if i.get("severity") == "high")
    # score de fiabilité global : calculé par profiler.py dans profile.json (ex. 98.1)
    gscore = (profile or {}).get("global_score") or data.get("global_score")
    quality = (gscore / 100.0) if isinstance(gscore, (int, float)) else None
    return {"present": True, "verdict": data.get("fit_for_project", {}).get("verdict"),
            "issues": len(issues), "high_issues": high,
            "data_gaps": len(data.get("data_gaps", [])),
            "quality_score": round(quality, 4) if quality is not None else None}


def eval_etl(decision, log) -> dict:
    """ETL générique ou modèle métier : même logique (part des contrôles réussis)."""
    out = {"present": bool(decision or log)}
    if decision:
        checks = decision.get("checks", [])
        failed = [c for c in checks if not c.get("passed", True)] if checks else decision.get("checks_failed", [])
        total = len(checks) if checks else None
        out.update({"status": decision.get("status"),
                    "schema_type": decision.get("schema_type"),
                    "rows_analytical_base": decision.get("rows_analytical_base"),
                    "rows_fact_sales": decision.get("rows_fact_sales"),
                    "rows_fact_orders": decision.get("rows_fact_orders"),
                    "checks_total": total, "checks_failed": len(failed)})
        if total:
            out["quality_score"] = round((total - len(failed)) / total, 4)
        else:
            out["quality_score"] = 1.0 if not failed else 0.0
    if log:
        entries = log if isinstance(log, list) else log.get("entries", [])
        out["corrections_applied"] = len(entries)
    return out


def eval_generic_kpi(data) -> dict:
    if not data:
        return {"present": False}
    # qualité = part des indicateurs planifiés effectivement calculés au dernier essai
    last = (data.get("attempts_log") or [{}])[-1]
    computed = len(data.get("headline", []))
    errors = len(last.get("validation_errors", []))
    total = computed + errors
    return {"present": True, "planner": data.get("planner"), "domain": data.get("domain_detected"),
            "headline_kpis": computed, "breakdowns": len(data.get("breakdowns", [])),
            "checks_total": total, "checks_failed": errors,
            "quality_score": round(computed / total, 4) if total else None}


def eval_kpi(data) -> dict:
    if not data:
        return {"present": False}
    # qualité = part des 12 indicateurs clés effectivement calculés (non vides) ; leur JUSTESSE
    # est mesurée séparément par l'auditeur (recalcul indépendant).
    headline = data.get("headline", {})
    filled = [v is not None for v in headline.values()]
    sections = list(data.get("breakdowns", {}))
    return {"present": True, "sections": sections, "headline_kpis": len(headline),
            "checks_total": len(filled), "checks_failed": filled.count(False),
            "quality_score": round(filled.count(True) / len(filled), 4) if filled else None}


def eval_analyst(data) -> dict:
    if not data:
        return {"present": False}
    ins = data.get("insights", [])
    verified = sum(1 for i in ins if i.get("verified"))
    alerts = [s["id"] for s in data.get("signals", []) if s.get("severity") == "alert"]
    cited = {sid for i in ins for sid in i.get("signal_ids", [])}
    # qualité = part des insights dont les chiffres sont vérifiés
    return {"present": True, "generated_by": data.get("generated_by"), "insights": len(ins),
            "verified": verified, "signals": len(data.get("signals", [])),
            "alerts_covered": f"{sum(a in cited for a in alerts)}/{len(alerts)}",
            "quality_score": round(verified / len(ins), 4) if ins else None}


def eval_dashboard(data) -> dict:
    if not data:
        return {"present": False}
    checks = data.get("checks", [])
    ok = sum(1 for c in checks if c.get("passed"))
    return {"present": True, "status": data.get("status"), "charts": len(data.get("charts", [])),
            "insights_displayed": data.get("insights_displayed"), "checks_total": len(checks),
            "checks_failed": len(checks) - ok,
            "quality_score": round(ok / len(checks), 4) if checks else None}


def eval_auditor(data) -> dict:
    if not data:
        return {"present": False}
    v = data.get("verdict", {})
    audited = v.get("audited", 0)
    mism = v.get("mismatches", 0)
    quality = ((audited - mism) / audited) if audited else None
    return {"present": True, "passed": v.get("passed"),
            "audited": audited, "mismatches": mism, "cause": v.get("cause"),
            "quality_score": round(quality, 4) if quality is not None else None}


def eval_orchestrator(log) -> dict:
    if not log:
        return {"present": False}
    steps = log.get("steps", [])
    done = [s for s in steps if s.get("status") in ("success", "failed", "stopped")]
    ok = [s for s in steps if s.get("status") == "success"]
    skipped = [s for s in steps if s.get("status") == "skipped"]
    runtime = round(sum(s.get("seconds", 0) for s in steps), 1)
    decisions = log.get("decisions", [])
    by_llm = sum(1 for d in decisions if d.get("by") == "llm")
    per_step = {s["agent"]: {"status": s.get("status"), "seconds": s.get("seconds")} for s in steps}
    return {"present": True, "final_status": log.get("status"),
            "agents_executed": len(done), "agents_success": len(ok),
            "agents_skipped": len(skipped), "total_runtime_s": runtime,
            "decisions_total": len(decisions), "decisions_by_llm": by_llm,
            "per_step": per_step}


def pipeline_score(per_agent: dict) -> float:
    """Score global = moyenne des quality_score disponibles, pondérée par présence."""
    scores = [a["quality_score"] for a in per_agent.values()
              if isinstance(a, dict) and isinstance(a.get("quality_score"), (int, float))]
    return round(sum(scores) / len(scores), 4) if scores else None


# ---------------------------------------------------------------------------
# Rapport
# ---------------------------------------------------------------------------
def write_reports(report: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "evaluation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    pa = report["per_agent"]
    L = ["# Évaluation du pipeline BIFlow", "",
         f"- Généré : {report['generated_at']}",
         f"- Statut final : **{report['orchestrator'].get('final_status', '—')}**",
         f"- Score global : {report['pipeline_score'] if report['pipeline_score'] is not None else '—'}",
         f"- Durée totale : {report['orchestrator'].get('total_runtime_s', '—')} s",
         f"- Agents : {report['orchestrator'].get('agents_success', 0)} ok / "
         f"{report['orchestrator'].get('agents_skipped', 0)} ignorés", "",
         "## Métriques par agent", "",
         "| Agent | Présent | Qualité | Détail |", "|---|---|---|---|"]

    def q(a):
        s = a.get("quality_score")
        return f"{s:.1%}" if isinstance(s, (int, float)) else "—"

    rows = [
        ("profiler", pa["profiler"], f"verdict={pa['profiler'].get('verdict')} issues={pa['profiler'].get('issues')} (high={pa['profiler'].get('high_issues')})"),
        ("etl", pa["etl"], f"schéma {pa['etl'].get('schema_type')}, {pa['etl'].get('rows_analytical_base')} lignes analysées — échecs={pa['etl'].get('checks_failed')}"),
        ("kpi", pa["kpi"], f"{pa['kpi'].get('headline_kpis')} indicateurs ({pa['kpi'].get('planner')}), {pa['kpi'].get('breakdowns')} découpages — erreurs={pa['kpi'].get('checks_failed')}"),
        ("olist_model", pa["olist_model"], f"{pa['olist_model'].get('rows_fact_sales')} ventes / {pa['olist_model'].get('rows_fact_orders')} commandes — échecs={pa['olist_model'].get('checks_failed')}"),
        ("olist_kpi", pa["olist_kpi"], f"{pa['olist_kpi'].get('headline_kpis')} indicateurs clés, {len(pa['olist_kpi'].get('sections', []))} découpages — vides={pa['olist_kpi'].get('checks_failed')}"),
        ("analyst", pa["analyst"], f"{pa['analyst'].get('insights')} insights ({pa['analyst'].get('generated_by')}), vérifiés={pa['analyst'].get('verified')}, alertes couvertes={pa['analyst'].get('alerts_covered')}"),
        ("dashboard", pa["dashboard"], f"{pa['dashboard'].get('charts')} graphiques, {pa['dashboard'].get('insights_displayed')} insights — contrôles échoués={pa['dashboard'].get('checks_failed')}"),
        ("auditor", pa["auditor"], f"passed={pa['auditor'].get('passed')} écarts={pa['auditor'].get('mismatches')} cause={pa['auditor'].get('cause')}"),
    ]
    for name, a, detail in rows:
        L.append(f"| {name} | {'oui' if a.get('present') else 'non'} | {q(a)} | {detail if a.get('present') else '—'} |")

    L += ["", "## Temps par étape (orchestrateur)", ""]
    for agent, st in report["orchestrator"].get("per_step", {}).items():
        L.append(f"- **{agent}** : {st.get('status')} ({st.get('seconds')} s)")
    (OUT_DIR / "evaluation_report.md").write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------
def run() -> dict:
    orch = eval_orchestrator(load(SOURCES["orchestrator"]))
    per_agent = {
        "profiler":  eval_profiler(load(SOURCES["profiler"]), load(SOURCES["profile"])),
        "etl":       eval_etl(load(SOURCES["etl_decision"]), load(SOURCES["etl_log"])),
        "kpi":       eval_generic_kpi(load(SOURCES["kpi"])),
        "olist_model": eval_etl(load(SOURCES["model"]), load(SOURCES["model_log"])),
        "olist_kpi": eval_kpi(load(SOURCES["olist_kpi"])),
        "analyst":   eval_analyst(load(SOURCES["analyst"])),
        "dashboard": eval_dashboard(load(SOURCES["dashboard"])),
        "auditor":   eval_auditor(load(SOURCES["auditor"])),
    }
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pipeline_score": pipeline_score(per_agent),
        "orchestrator": orch,
        "per_agent": per_agent,
    }
    write_reports(report)
    return {"ok": True, "pipeline_score": report["pipeline_score"],
            "agents_present": sum(1 for a in per_agent.values() if a.get("present")),
            "final_status": orch.get("final_status"),
            "outputs": [str(OUT_DIR / "evaluation.json"), str(OUT_DIR / "evaluation_report.md")]}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False, default=str))
