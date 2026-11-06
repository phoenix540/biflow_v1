"""Tests des KPI, des garde-fous de l'Analyste, de l'Auditor et des règles de l'orchestrateur."""
import pandas as pd

import analyst_agent
import auditor_agent
import kpi_agent
import orchestrator


# --------------------------------------------------------------------------- KPI
def _dates(months_reliable):
    rows = []
    for ym, ok in months_reliable.items():
        rows.append({"date": pd.Timestamp(f"{ym}-01"), "is_reliable_month": ok, "year_month": ym})
    return pd.DataFrame(rows)


def test_yoy_growth_only_between_reliable_months():
    orders = pd.DataFrame({
        "order_id": ["a", "b", "c", "d"],
        "purchase_date": pd.to_datetime(["2017-03-10", "2018-03-10", "2017-04-10", "2018-04-10"]),
        "revenue_brl": [100.0, 150.0, 100.0, 300.0],
    })
    dates = _dates({"2017-03": True, "2018-03": True, "2017-04": False, "2018-04": True})
    rows, summary, comparable = kpi_agent.monthly_trend(orders, dates)
    by_month = {r["month"]: r["yoy_growth_pct"] for r in rows}
    assert by_month["2018-03"] == 50.0
    assert by_month["2018-04"] is None            # 2017-04 peu fiable : pas de comparaison
    assert summary["growth_pct"] == 50.0 and comparable == [("2018-03", "2017-03")]


def test_state_breakdown_rates():
    orders = pd.DataFrame({"order_id": ["a", "b"], "customer_state": ["SP", "SP"], "revenue_brl": [10.0, 30.0],
                           "late": [1.0, 0.0], "delivery_days": [5, 7], "freight_total": [2.0, 2.0],
                           "items_value": [10.0, 30.0], "review_score": [2, 4]})
    s = kpi_agent.state_breakdown(orders)[0]
    assert (s["late_delivery_rate_pct"], s["median_delivery_days"], s["freight_to_revenue_pct"],
            s["avg_review_score"]) == (50.0, 6.0, 10.0, 3.0)


# --------------------------------------------------------------------------- Analyste
SIGNALS = [{"id": "S1", "severity": "alert", "summary": "x", "facts": {"growth_pct": 118.4}},
           {"id": "S2", "severity": "info", "summary": "y", "facts": {"share": 48.0}}]


def test_analyst_flags_invented_numbers_and_unknown_signals():
    res = analyst_agent.validate({"insights": [
        {"finding": "Croissance de 118.4 %", "signal_ids": ["S1"]},
        {"finding": "Croissance de 999.9 %", "signal_ids": ["S1", "S99"]},
    ]}, SIGNALS)
    ok, bad = res["insights"]
    assert ok["verified"] is True
    assert bad["verified"] is False and bad["ungrounded_numbers"] == ["999.9"] and bad["unknown_signal_ids"] == ["S99"]


def test_rule_insights_cover_alerts():
    sig = [{"id": "S1", "kind": "category_decline", "severity": "alert", "title": "Baisse",
            "summary": "s", "recommendation_hint": "r", "facts": {}}]
    out = analyst_agent.rule_insights(sig)
    assert out["insights"][0]["signal_ids"] == ["S1"] and out["insights"][0]["priority"] == "high"


def test_kpi_fingerprint_ignores_generation_time():
    a = {"generated_at": "2026-01-01", "headline": {"x": 1}, "definitions": {"d": "texte"}}
    b = {"generated_at": "2026-12-31", "headline": {"x": 1}, "definitions": {"d": "texte reformulé"}}
    c = {"generated_at": "2026-12-31", "headline": {"x": 2}, "definitions": {"d": "texte"}}
    assert analyst_agent.kpi_fingerprint(a) == analyst_agent.kpi_fingerprint(b) != analyst_agent.kpi_fingerprint(c)


# --------------------------------------------------------------------------- Auditor
def _facts():
    sales = pd.DataFrame({"order_id": ["a", "b"], "seller_id": ["s1", "s2"], "category": ["x", "y"],
                          "is_revenue_order": [True, True], "revenue_brl": [100.0, 50.0], "margin_brl": [30.0, 15.0]})
    orders = pd.DataFrame({"order_id": ["a", "b", "c"], "customer_unique_id": ["u1", "u1", "u2"],
                           "order_status": ["delivered", "delivered", "canceled"],
                           "is_revenue_order": [True, True, False], "revenue_brl": [100.0, 50.0, 0.0],
                           "is_late": ["True", "False", None], "delivery_days": [10, 4, None],
                           "review_score": [1, 5, None], "freight_total": [10.0, 5.0, 0.0],
                           "items_value": [100.0, 50.0, 0.0]})
    return {"sales": sales, "orders": orders}


def test_auditor_lineage_cards_detect_falsified_kpi():
    good = {"headline": {"total_revenue_brl": 150.0, "delivered_orders": 2, "late_delivery_rate_pct": 50.0,
                         "lost_order_rate_pct": 33.33, "repeat_customer_rate_pct": 100.0}}
    cards = {c["id"]: c["status"] for c in auditor_agent.build_cards(good, _facts())}
    assert cards["headline.total_revenue_brl"] == "ok" and cards["headline.late_delivery_rate_pct"] == "ok"
    assert cards["headline.lost_order_rate_pct"] == "ok" and cards["headline.repeat_customer_rate_pct"] == "ok"
    assert cards["headline.median_delivery_days"] == "non_rapporte"      # absent : pas un écart

    bad = {"headline": {**good["headline"], "total_revenue_brl": 157.5}}   # +5 % falsifié
    card = next(c for c in auditor_agent.build_cards(bad, _facts()) if c["id"] == "headline.total_revenue_brl")
    assert card["status"] == "ecart" and card["delta"] == 7.5


def test_registry_matches_current_kpi_format():
    # chaque chemin audité doit exister dans le format produit par l'agent KPI
    import json
    from pathlib import Path
    kpi_file = Path(auditor_agent.KPI_RESULTS)
    if kpi_file.exists():
        k = json.loads(kpi_file.read_text(encoding="utf-8"))
        missing = [p for p in auditor_agent.REGISTRY if auditor_agent.get_path(k, p) is None]
        assert missing == []


# --------------------------------------------------------------------------- Dashboard
def test_dashboard_shows_all_insights_sorted_with_evidence():
    import dashboard_agent
    analyst = {"insights": [{"id": "A1", "priority": "low", "signal_ids": ["S1"], "verified": True},
                            {"id": "A2", "priority": "high", "signal_ids": ["S2", "S9"], "verified": False}],
               "signals": [{"id": "S1", "kind": "weekday", "severity": "info", "summary": "lundi", "facts": {}},
                           {"id": "S2", "kind": "region_delivery", "severity": "alert", "summary": "MA en retard",
                            "facts": {"states": [{"state": "MA"}]}}]}
    view = dashboard_agent.build_analyst_view(analyst, {"breakdowns": {}, "limitations": [],
                                                        "margin_simulated": {"warning": "simulée"}})
    assert [i["id"] for i in view["insights"]] == ["A2", "A1"]          # priorité haute d'abord
    assert view["insights"][0]["evidence"] == ["S2 : MA en retard"]     # signal inconnu S9 ignoré
    assert view["alert_states"] == ["MA"] and view["alerts"] == 1


# --------------------------------------------------------------------------- Orchestrateur
def test_deterministic_agents_are_not_retried(monkeypatch):
    monkeypatch.setattr(orchestrator, "USE_LLM", False)
    d = orchestrator.decide("etl", {"status": "failed"}, {}, {"etl": 1})
    assert d["next"] == "finish"


def test_audit_failure_follows_auditor_suggestion(monkeypatch):
    monkeypatch.setattr(orchestrator, "USE_LLM", False)
    step = {"status": "success", "result": {"passed": False, "suggested_rerun": "etl"}}
    assert orchestrator.decide("auditor", step, {"audit_loops": 0}, {})["next"] == "retry_etl"
    step["result"]["suggested_rerun"] = "kpi"
    assert orchestrator.decide("auditor", step, {"audit_loops": 0}, {})["next"] == "retry_kpi"
