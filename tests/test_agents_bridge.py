"""Chatbot Olist Pilot : il doit s'appuyer sur les agents du pipeline (KPI, BI Analyst, XAI)."""
import json

import pytest

from manager import agents_bridge as B
from manager import assistant as A
from manager import metrics as M

pytestmark = pytest.mark.skipif(not (B.KA.CURATED / "fact_orders.csv").exists() or not B.AA.INSIGHTS_FILE.exists(),
                                reason="données curated / analyses absentes")


def test_analyst_detection_reproduces_published_signals():
    published = json.loads(B.AA.INSIGHTS_FILE.read_text(encoding="utf-8"))["signals"]
    mine = B.analyst_signals()["signaux"]
    assert {s["titre"] for s in mine} == {s["title"] for s in published}


def test_analyst_detection_on_filtered_scope_does_not_crash():
    r = B.analyst_signals(period=("2018-01", "2018-08"), states=("RJ",))
    assert r["signaux"] and not r["detecteurs_non_applicables"]


def test_xai_revenue_bridge_reconciles():
    r = B.explain_revenue_change("2018-08", "2018-07")
    assert r["disponible"] and r["reconcilie"]
    assert sum(r["effets"].values()) == pytest.approx(r["variation_brl"], abs=1)


@pytest.mark.skipif(not B.XE.MODEL_PATH.exists(), reason="modèle XAI absent")
def test_xai_risk_increases_with_delay():
    on_time = B.order_risk(retard_jours=-10)["probabilite_mauvaise_note_pct"]
    late = B.order_risk(retard_jours=7)["probabilite_mauvaise_note_pct"]
    assert late > on_time


def test_chatbot_routes_to_agents_and_uses_dashboard_scope():
    store = M.load_store()
    sc = M.Scope("2018-01", "2018-08", ("RJ",))
    cases = {"Quelles sont les alertes ?": "BI Analyst", "Pourquoi le CA a baissé ?": "XAI",
             "Pourquoi les clients donnent de mauvaises notes ?": "XAI", "Donne-moi un bilan": "Agent KPI"}
    for q, agent in cases.items():
        ans = A.ask(store, q, [], use_llm=False, scope=sc)
        assert agent in {t["agent"] for t in ans.trace}, q
    kpi = A.ask(store, "Donne-moi un bilan", [], use_llm=False, scope=sc)
    assert "RJ" in kpi.text
