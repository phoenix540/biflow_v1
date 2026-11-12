"""Socle générique : l'ETL et le KPI fonctionnent sur un jeu de CSV inconnu (ici une boutique fictive,
sans aucun nom de colonne Olist). Tout tourne dans un dossier temporaire, sans LLM."""
import json

import numpy as np
import pandas as pd
import pytest

import etl_agent
import kpi_agent


@pytest.fixture
def shop(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    raw = tmp_path / "raw"
    raw.mkdir()
    n_cust, n_orders, n_lines = 40, 120, 300
    customers = pd.DataFrame({"client_id": [f"C{i:03d}" for i in range(n_cust)],
                              "region": rng.choice(["Nord", "Sud", "Est", "Ouest"], n_cust)})
    orders = pd.DataFrame({"order_ref": [f"O{i:04d}" for i in range(n_orders)],
                           "client_id": rng.choice(customers["client_id"], n_orders),
                           "created_date": pd.date_range("2024-01-01", periods=n_orders, freq="3D")
                                             .strftime("%Y-%m-%d"),
                           "channel": rng.choice(["web", "store"], n_orders)})
    lines = pd.DataFrame({"line_id": [f"L{i:05d}" for i in range(n_lines)],
                          "order_ref": rng.choice(orders["order_ref"], n_lines),
                          "amount": rng.gamma(2.0, 30.0, n_lines).round(2),
                          "quantity": rng.integers(1, 6, n_lines)})
    lines.loc[:5, "amount"] = np.nan                       # quelques manquants à imputer
    lines = pd.concat([lines, lines.iloc[:3]])             # 3 doublons exacts à retirer
    customers.to_csv(raw / "customers.csv", index=False)
    orders.to_csv(raw / "orders.csv", sep=";", index=False)  # séparateur différent : détecté
    lines.to_csv(raw / "order_lines.csv", index=False)

    out = tmp_path / "data"
    monkeypatch.setenv("BIFLOW_RAW_DIR", str(raw))
    for mod, attrs in ((etl_agent, {"CLEAN": out / "clean", "CURATED": out / "curated", "OUT": out / "etl",
                                    "PROFILER_DECISION": out / "none.json", "CONFIG_PATH": out / "none.json"}),
                       (kpi_agent, {"CLEAN": out / "clean", "CURATED": out / "curated",
                                    "ETL_DIR": out / "etl", "OUT": out / "kpi"})):
        for k, v in attrs.items():
            monkeypatch.setattr(mod, k, v)
    return out


def test_generic_etl_cleans_and_infers_star_schema(shop):
    res = etl_agent.run()
    assert res["ok"] and res["fact_table"] == "order_lines"
    assert res["rows_analytical_base"] == 300               # doublons retirés, aucune ligne ajoutée
    model = json.loads((shop / "etl" / "model.json").read_text(encoding="utf-8"))
    assert [f["table"] for f in model["facts"]] == ["order_lines"]
    assert "amount" in model["facts"][0]["measures"]
    assert {d["table"] for d in model["dimensions"]} == {"orders", "customers"}
    assert model["schema_type"] == "flocon"                 # order_lines -> orders -> customers
    clean = pd.read_csv(shop / "clean" / "order_lines.csv")
    assert clean["amount"].notna().all() and "amount_was_missing" in clean.columns


def test_generic_kpi_plans_and_computes_without_llm(shop):
    etl_agent.run()
    res = kpi_agent.run()
    assert res["ok"] and res["planner"] == "deterministic_rule"
    out = json.loads((shop / "kpi" / "generic_kpi_results.json").read_text(encoding="utf-8"))
    by_id = {h["id"]: h for h in out["headline"]}
    base = pd.read_csv(shop / "curated" / "analytical_base.csv")
    assert by_id["sum_amount"]["value"] == pytest.approx(base["amount"].sum(), abs=0.01)
    assert by_id["total_records"]["value"] == 300
    assert out["time_trends"] and out["time_trends"][0]["date_column"].endswith("created_date")
    assert out["attempts_log"][-1]["validation_errors"] == []


def test_raw_dir_ignores_backup_folders(tmp_path, monkeypatch):
    for name, n in (("backup", 5), ("shop", 2)):
        d = tmp_path / "data" / "raw" / name
        d.mkdir(parents=True)
        for i in range(n):
            (d / f"t{i}.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    monkeypatch.delenv("BIFLOW_RAW_DIR", raising=False)
    monkeypatch.setattr(etl_agent, "ROOT", tmp_path)
    assert etl_agent.find_raw_dir().name == "shop"
