"""Olist Pilot : les chiffres de l'app manager doivent être identiques à ceux de l'agent KPI."""
import json

import pytest

from manager import assistant as A
from manager import metrics as M

KPI = M.DATA / "kpi" / "kpi_results.json"
pytestmark = pytest.mark.skipif(not (M.CUR / "fact_orders.csv").exists() or not KPI.exists(),
                                reason="données curated / KPI absentes")


@pytest.fixture(scope="module")
def store():
    return M.load_store()


def test_summary_matches_kpi_agent(store):
    h = json.loads(KPI.read_text(encoding="utf-8"))["headline"]
    s = M.summary(store, M.Scope(store.all_months[0], store.all_months[-1]))
    assert s["chiffre_affaires_brl"] == pytest.approx(h["total_revenue_brl"], abs=1)
    assert s["commandes"] == h["delivered_orders"]
    assert s["taux_retard_pct"] == pytest.approx(h["late_delivery_rate_pct"], abs=0.01)
    assert s["note_moyenne"] == pytest.approx(h["average_review_score"], abs=0.01)
    assert s["clients_recurrents_pct"] == pytest.approx(h["repeat_customer_rate_pct"], abs=0.01)


def test_category_growth_uses_comparable_months(store):
    declining = json.loads(KPI.read_text(encoding="utf-8"))["breakdowns"]["categories"]["declining"]
    b = M.breakdown(store, M.Scope(store.default_start, store.last_full_month), "categorie").set_index("categorie")
    for d in declining:
        assert b.loc[d["category"], "croissance_1an_pct"] == pytest.approx(d["yoy_growth_pct"], abs=0.1)


def test_rules_assistant_answers_without_llm(store):
    for q in ["Quels vendeurs sont à risque ?", "Où sont les retards ?", "Et si on réduisait les retards ?"]:
        ans = A.ask(store, q, [], use_llm=False)
        assert ans.mode == "regles" and ans.text and ans.trace
