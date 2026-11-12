"""Lance une ou plusieurs étapes du pipeline, dans l'ordre donné, sans passer par l'orchestrateur.

Exemples :
  python agents/run_steps.py etl olist_model          # ETL générique puis modèle métier Olist
  python agents/run_steps.py kpi olist_kpi --no-llm   # KPI générique (plan sans LLM) puis KPI métier

Les noms d'étapes sont ceux de l'orchestrateur (orchestrator.MODULES). Code de sortie 1 si une étape échoue.
"""
import importlib
import sys

from orchestrator import MODULES


def main(steps: list) -> int:
    unknown = [s for s in steps if s not in MODULES]
    if unknown or not steps:
        print(f"Étapes inconnues : {unknown}. Disponibles : {', '.join(MODULES)}")
        return 2
    for name in steps:
        print(f"\n=== {name} ({MODULES[name]}) ===")
        result = importlib.import_module(MODULES[name]).run() or {}
        if not result.get("ok", True):
            print(f"Étape {name} en échec : {result.get('status')}")
            return 1
        if result.get("applicable") is False:
            print(f"Étape {name} non applicable : {result.get('reason')}")
            break
    return 0


if __name__ == "__main__":
    sys.exit(main([a for a in sys.argv[1:] if not a.startswith("--")]))
