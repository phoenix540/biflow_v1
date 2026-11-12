"""Pages Clients, Paiements, Produits et Carte d'Olist Pilot : cohérence avec les chiffres officiels."""
import numpy as np
import pytest

from manager import assistant as A
from manager import insights as I
from manager import metrics as M

pytestmark = pytest.mark.skipif(not (M.CUR / "fact_orders.csv").exists(), reason="données curated absentes")


@pytest.fixture(scope="module")
def store():
    return M.load_store()


@pytest.fixture(scope="module")
def extra():
    return I.load_extra()


@pytest.fixture(scope="module")
def sc(store):
    return M.Scope(store.default_start, store.last_full_month)


def test_customer_overview_matches_kpi_summary(store, sc):
    ov, s = I.customers_overview(store, sc), M.summary(store, sc)
    assert ov["clients"] == s["clients_uniques"]
    assert ov["taux_recurrents_pct"] == pytest.approx(s["clients_recurrents_pct"], abs=0.01)


def test_rfm_segments_cover_every_customer_once(store, sc):
    seg, per_customer = I.rfm_segments(store, sc)
    assert seg["clients"].sum() == I.customers_overview(store, sc)["clients"] == per_customer["customer_unique_id"].nunique()
    assert seg["part_clients_pct"].sum() == pytest.approx(100) and seg["part_ca_pct"].sum() == pytest.approx(100)
    assert set(seg["segment"]) <= set(I.RFM_ORDER)
    champions = per_customer[per_customer["segment"] == "Champions"]
    assert (champions["frequence"] >= 2).all() and (champions["recence"] <= 180).all()


def test_cohort_retention_is_cumulative_and_bounded(store, sc):
    co = I.cohort_retention(store, sc)
    cols = [c for c in co.columns if c.startswith("M+")]
    vals = co[cols].to_numpy(dtype=float)
    assert np.nanmin(vals) >= 0 and np.nanmax(vals) <= 100
    diffs = np.diff(vals, axis=1)
    assert np.all(diffs[~np.isnan(diffs)] >= 0)               # part cumulée : jamais décroissante
    assert np.isnan(vals[-1, -1])                              # dernière cohorte : pas encore 12 mois de recul


def test_payment_mix_revenue_matches_total(store, sc):
    mix = I.payments_mix(store, sc)
    assert mix["ca"].sum() == pytest.approx(M.summary(store, sc)["chiffre_affaires_brl"], rel=1e-3)
    assert mix["part_commandes_pct"].sum() == pytest.approx(100)


def test_pareto_and_best_sellers(store, extra, sc):
    pa = I.pareto(store, extra, sc)
    assert 20 <= pa["part_ca_top20_pct"] <= 100
    assert pa["curve"]["part_ca_pct"].iloc[-1] == pytest.approx(100)
    bs = I.best_sellers(store, extra, sc, 10)
    assert len(bs) == 10 and bs["unites"].is_monotonic_decreasing


def test_map_and_state_flows_are_consistent(store, extra, sc):
    total = M.summary(store, sc)["chiffre_affaires_brl"]
    pts = I.sales_points(store, extra, sc)
    assert 0.99 * total <= pts["ca"].sum() <= total + 1          # quelques codes postaux sans coordonnées
    sv = I.same_vs_other_state(store, sc).set_index("flux")
    assert sv["commandes"].sum() == pytest.approx(M.summary(store, sc)["commandes"], rel=1e-3)
    assert sv.loc["Autre État", "delai_median"] > sv.loc["Même État", "delai_median"]


def test_new_analyses_survive_an_almost_empty_scope(store, extra):
    tiny = M.Scope("2017-01", "2017-01", ("RR",), ("security_and_services",))
    for name in ("customers_overview", "cohort_retention", "rfm_segments", "payments_mix", "installments_vs_basket",
                 "same_vs_other_state", "state_flows"):
        getattr(I, name)(store, tiny)
    for name in ("pareto", "best_sellers", "listing_quality", "weight_vs_freight", "sales_points"):
        getattr(I, name)(store, extra, tiny)


@pytest.mark.parametrize("question, tool", [
    ("Quels segments de clients relancer en priorité ?", "segments_clients"),
    ("Le boleto augmente-t-il les annulations ?", "analyse_paiements"),
    ("Quels produits se vendent le plus ?", "concentration_produits"),
    ("Les vendeurs éloignés livrent-ils plus tard ?", "flux_livraison"),
    ("Que vaut une fidélisation à 6 % ?", "simuler"),
])
def test_rules_assistant_routes_new_questions(store, question, tool):
    ans = A.ask(store, question, [], use_llm=False)
    assert ans.mode == "regles" and tool in [t["outil"] for t in ans.trace], question
