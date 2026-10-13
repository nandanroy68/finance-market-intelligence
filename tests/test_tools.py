"""
Unit tests for market data and portfolio risk calculator tools.
"""
from src.tools.market_data_tool import (
    get_stock_info,
    get_sector_stocks,
    calculate_portfolio_risk,
    suggest_allocation,
)


def test_get_stock_info():
    # TCS is in stock_data.csv
    res = get_stock_info.invoke({"symbol": "TCS"})
    assert "error" not in res
    assert res["symbol"] == "TCS"
    assert "IT" in res.get("sector", "")


def test_get_stock_info_invalid():
    res = get_stock_info.invoke({"symbol": "NONEXISTENT_XYZ"})
    assert "error" in res


def test_get_sector_stocks():
    res = get_sector_stocks.invoke({"sector": "IT"})
    assert isinstance(res, list)
    assert len(res) >= 1
    assert any(s.get("symbol") in ["TCS", "INFY", "WIPRO"] for s in res)


def test_calculate_portfolio_risk():
    holdings = [
        {"symbol": "TCS", "weight": 0.5},
        {"symbol": "INFY", "weight": 0.5},
    ]
    res = calculate_portfolio_risk.invoke({"holdings": holdings})
    assert "portfolio_beta" in res
    assert "risk_level" in res
    assert "sector_weights" in res
    assert "IT" in res["sector_weights"]
    assert res["sector_concentration_risk"] == "high"


def test_suggest_allocation():
    res_mod = suggest_allocation.invoke({"risk_appetite": "moderate", "investment_amount": 1000000})
    assert "allocation" in res_mod
    assert res_mod["risk_appetite"] == "moderate"

    res_agg = suggest_allocation.invoke({"risk_appetite": "aggressive", "investment_amount": 500000})
    assert "allocation" in res_agg
    assert res_agg["risk_appetite"] == "aggressive"
