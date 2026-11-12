"""Filtres du KPI Dashboard : olist_kpi_agent.compute / filter_frames."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agents"))
import olist_kpi_agent as K  # noqa: E402

KPI_JSON = ROOT / "data" / "kpi" / "kpi_results.json"
pytestmark = pytest.mark.skipif(not (K.CURATED / "fact_orders.csv").exists() or not KPI_JSON.exists(),
                                reason="données curated / KPI absentes")


@pytest.fixture(scope="module")
def tables():
    return K.load()


def test_compute_without_filter_equals_published_results(tables):
    published = json.loads(KPI_JSON.read_text(encoding="utf-8"))
    fresh = K.compute(*tables)
    for d in (published, fresh):
        d.pop("generated_at")
    assert fresh == published


def test_state_filters_partition_total_revenue(tables):
    o, s, d = tables
    total = K.compute(o, s, d)["headline"]["total_revenue_brl"]
    parts = 0.0
    for st in o["customer_state"].dropna().unique():
        oo, ss = K.filter_frames(o, s, states=(st,))
        parts += K.compute(oo, ss, d)["headline"]["total_revenue_brl"] or 0
    assert parts == pytest.approx(total, abs=1)


def test_category_and_period_filter(tables):
    o, s, d = tables
    oo, ss = K.filter_frames(o, s, categories=("bed_bath_table",))
    k = K.compute(oo, ss, d, period=("2018-01", "2018-08"))
    expected = ss[ss["is_revenue_order"] & ss["purchase_date"].between("2018-01-01", "2018-08-31")]["revenue_brl"].sum()
    assert k["headline"]["total_revenue_brl"] == pytest.approx(expected, abs=0.01)
    assert [m["month"] for m in k["breakdowns"]["monthly"]] == [f"2018-0{i}" for i in range(1, 9)]
    assert k["breakdowns"]["yoy_summary"]["growth_pct"] is not None   # N-1 lu dans l'historique
    assert all(e["date"].startswith("2018") for e in k["breakdowns"]["events"])
