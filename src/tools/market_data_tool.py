"""
Market data tool and risk calculator tools for the Finance Market Intelligence agents.
"""
import json
import os
from langchain_core.tools import tool

from src.config import RAW_DATA_DIR


def _load_stock_data() -> list[dict]:
    """Load stock data from CSV and return as list of dicts."""
    import csv
    stock_path = os.path.join(RAW_DATA_DIR, "stock_data.csv")
    if not os.path.exists(stock_path):
        return []
    with open(stock_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


@tool
def get_stock_info(symbol: str) -> dict:
    """
    Get financial information for a specific stock by its NSE symbol.
    Returns key metrics like P/E, ROE, beta, market cap, etc.
    """
    stocks = _load_stock_data()
    stock = next((s for s in stocks if s["symbol"].upper() == symbol.upper()), None)
    if stock is None:
        return {"error": f"Stock '{symbol}' not found in database."}
    return stock


@tool
def get_sector_stocks(sector: str) -> list[dict]:
    """
    Get all stocks belonging to a specific sector.
    Returns a list of stocks with their financial metrics.
    """
    stocks = _load_stock_data()
    matches = [s for s in stocks if s.get("sector", "").lower() == sector.lower()]
    if not matches:
        return [{"error": f"No stocks found for sector '{sector}'."}]
    return matches


@tool
def calculate_portfolio_risk(holdings: list[dict]) -> dict:
    """
    Calculate portfolio-level risk metrics given a list of holdings.
    Each holding should have: symbol, weight (0-1).
    Returns portfolio beta, sector concentration, and risk assessment.
    """
    stocks = _load_stock_data()
    stock_map = {s["symbol"].upper(): s for s in stocks}

    portfolio_beta = 0.0
    sector_weights = {}
    total_weight = 0.0
    missing = []

    for holding in holdings:
        symbol = holding.get("symbol", "").upper()
        weight = float(holding.get("weight", 0))
        total_weight += weight

        stock = stock_map.get(symbol)
        if stock is None:
            missing.append(symbol)
            continue

        beta = float(stock.get("beta", 1.0))
        portfolio_beta += beta * weight

        sector = stock.get("sector", "Unknown")
        sector_weights[sector] = sector_weights.get(sector, 0) + weight

    # Determine risk level
    if portfolio_beta < 0.8:
        risk_level = "low"
    elif portfolio_beta < 1.2:
        risk_level = "moderate"
    else:
        risk_level = "high"

    # Check sector concentration
    max_sector_weight = max(sector_weights.values()) if sector_weights else 0
    concentration_risk = "high" if max_sector_weight > 0.40 else ("moderate" if max_sector_weight > 0.25 else "low")

    return {
        "portfolio_beta": round(portfolio_beta, 3),
        "risk_level": risk_level,
        "sector_weights": {k: round(v, 3) for k, v in sector_weights.items()},
        "sector_concentration_risk": concentration_risk,
        "max_sector_weight": round(max_sector_weight, 3),
        "total_weight": round(total_weight, 3),
        "missing_symbols": missing,
        "recommendation": _generate_risk_recommendation(portfolio_beta, concentration_risk, sector_weights),
    }


def _generate_risk_recommendation(beta: float, concentration: str, sectors: dict) -> str:
    """Generate a textual risk recommendation based on metrics."""
    recs = []
    if beta > 1.2:
        recs.append("Portfolio beta is high (>1.2). Consider adding defensive stocks (FMCG, Pharma) to reduce volatility.")
    elif beta < 0.7:
        recs.append("Portfolio beta is very low (<0.7). You may be missing growth opportunities. Consider adding selective mid-cap exposure.")

    if concentration == "high":
        top_sector = max(sectors, key=sectors.get) if sectors else "Unknown"
        recs.append(f"High sector concentration in {top_sector} ({sectors.get(top_sector, 0)*100:.0f}%). Diversify across at least 5 sectors.")

    if not recs:
        recs.append("Portfolio risk profile appears well-balanced. Continue monitoring sector weights and beta quarterly.")

    return " ".join(recs)


@tool
def suggest_allocation(risk_appetite: str, investment_amount: float = 1000000) -> dict:
    """
    Suggest a portfolio allocation based on the investor's risk appetite.
    Risk appetite can be: conservative, moderate, or aggressive.
    Returns suggested allocation percentages across asset classes.
    """
    allocations = {
        "conservative": {
            "large_cap_equity": 0.25,
            "mid_cap_equity": 0.05,
            "debt_instruments": 0.40,
            "gold": 0.15,
            "fixed_deposits": 0.10,
            "international_equity": 0.05,
            "expected_return_range": "8-10%",
            "risk_level": "low",
        },
        "moderate": {
            "large_cap_equity": 0.35,
            "mid_cap_equity": 0.15,
            "debt_instruments": 0.25,
            "gold": 0.10,
            "fixed_deposits": 0.05,
            "international_equity": 0.10,
            "expected_return_range": "10-14%",
            "risk_level": "moderate",
        },
        "aggressive": {
            "large_cap_equity": 0.30,
            "mid_cap_equity": 0.25,
            "small_cap_equity": 0.15,
            "debt_instruments": 0.10,
            "gold": 0.05,
            "international_equity": 0.15,
            "expected_return_range": "14-18%",
            "risk_level": "high",
        },
    }

    appetite = risk_appetite.lower()
    if appetite not in allocations:
        return {"error": f"Invalid risk appetite '{risk_appetite}'. Use: conservative, moderate, or aggressive."}

    alloc = allocations[appetite]
    # Calculate amounts
    amounts = {}
    for key, pct in alloc.items():
        if isinstance(pct, (int, float)):
            amounts[key] = {"percentage": f"{pct*100:.0f}%", "amount": f"₹{pct * investment_amount:,.0f}"}

    return {
        "risk_appetite": appetite,
        "investment_amount": f"₹{investment_amount:,.0f}",
        "allocation": amounts,
        "expected_return_range": alloc.get("expected_return_range", ""),
        "risk_level": alloc.get("risk_level", ""),
    }
