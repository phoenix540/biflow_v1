"""Tests du profilage et des transformations de l'ETL (données synthétiques, rapides)."""
import json

import numpy as np
import pandas as pd

import etl_agent
import profiler


class Log:
    def __init__(self):
        self.entries = []

    def add(self, step, action, rows, reason):
        self.entries.append((step, action, int(rows), reason))


def test_profile_boolean_column_does_not_crash():
    # br_holidays avait une colonne booléenne qui faisait planter le profilage
    df = pd.DataFrame({"date": ["2017-01-01", "2017-02-27"], "flag": [True, False]})
    info = profiler.profile_table(df)["columns"]["flag"]
    assert info["top_values"] == {"True": 1, "False": 1}


def test_cost_above_price_rule():
    tables = {
        "olist_order_items_dataset": pd.DataFrame({"product_id": ["a", "b"], "price": [10.0, 10.0]}),
        "olist_products_dataset": pd.DataFrame({"product_id": ["a", "b"], "production_cost_brl": [5.0, 15.0]}),
    }
    rule = next(r for r in profiler.business_rules(tables) if "sous le coût" in r.get("rule", ""))
    assert (rule["violations"], rule["total"]) == (1, 2)


def _orders(delivered, estimated, status="delivered"):
    return pd.DataFrame({
        "order_id": ["o1"], "order_status": [status],
        "order_purchase_timestamp": ["2017-10-01 10:00:00"], "order_approved_at": [None],
        "order_delivered_carrier_date": [None], "order_delivered_customer_date": [delivered],
        "order_estimated_delivery_date": [estimated],
    })


def test_delivery_on_promised_day_is_not_late():
    # Bug corrigé : livré le jour promis à 15 h ne doit pas compter en retard
    o = etl_agent.tool_clean_orders(_orders("2017-10-05 15:00:00", "2017-10-05 00:00:00"),
                                    {"revenue_statuses": ["delivered"]}, Log())
    assert bool(o.loc[0, "is_late"]) is False and o.loc[0, "delay_days"] == 0


def test_delivery_one_day_after_is_late():
    o = etl_agent.tool_clean_orders(_orders("2017-10-06 08:00:00", "2017-10-05 00:00:00"),
                                    {"revenue_statuses": ["delivered"]}, Log())
    assert bool(o.loc[0, "is_late"]) is True and o.loc[0, "delay_days"] == 1


def test_human_translation_overrides_config(tmp_path, monkeypatch):
    cfg = tmp_path / "etl_config.json"
    cfg.write_text(json.dumps({"translation_overrides": {"pc_gamer": "from_config"}}), encoding="utf-8")
    human = tmp_path / "validated.json"
    human.write_text(json.dumps({"translations": {"pc_gamer": "from_human"}, "rejected": []}), encoding="utf-8")
    monkeypatch.setattr(etl_agent, "CONFIG_PATH", cfg)
    monkeypatch.setattr(etl_agent, "VALIDATED", human)
    assert etl_agent.load_config()["approved_translations"]["pc_gamer"] == "from_human"


def test_sales_use_production_cost_with_fallback():
    items = pd.DataFrame({"order_id": ["o1", "o1"], "order_item_id": [1, 2], "product_id": ["p1", "p2"],
                          "seller_id": ["s1", "s1"], "price": [100.0, 100.0], "freight_value": [10.0, 10.0]})
    orders = pd.DataFrame({"order_id": ["o1"], "customer_id": ["c1"], "customer_unique_id": ["u1"],
                           "customer_state": ["SP"], "order_status": ["delivered"], "is_revenue_order": [True],
                           "purchase_date": [pd.Timestamp("2017-10-01")]})
    products = pd.DataFrame({"product_id": ["p1", "p2"], "product_name": ["A", "B"], "category": ["x", "x"],
                             "production_cost_brl": [60.0, np.nan]})
    sellers = pd.DataFrame({"seller_id": ["s1"], "seller_name": ["Shop"], "seller_state": ["SP"]})
    cfg = {"cost_source": "production_cost", "default_cost_pct": 0.7, "cost_pct_by_category": {}, "brl_to_tnd": 1}
    f = etl_agent.tool_build_sales(items, orders, products, sellers, cfg, Log())
    assert list(f["cost_brl"]) == [60.0, 70.0]                       # coût réel, puis repli 70 %
    assert list(f["cost_source"]) == ["production_cost", "estimated_pct"]
    assert list(f["margin_brl"]) == [40.0, 30.0]
    assert list(f["seller_name"]) == ["Shop", "Shop"]
