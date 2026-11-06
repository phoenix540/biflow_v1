"""Configuration commune : rend importables les modules de agents/ et coupe le LLM pendant les tests."""
import sys
from pathlib import Path

AGENTS = Path(__file__).resolve().parent.parent / "agents"
sys.path.insert(0, str(AGENTS))

# Aucun test ne doit appeler le LLM (quota, réseau, résultats non déterministes)
if "--no-llm" not in sys.argv:
    sys.argv.append("--no-llm")
