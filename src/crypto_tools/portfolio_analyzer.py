import logging
import json
import math
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from datetime import datetime, timedelta

from .price_checker import get_token_price
from .token_security import analyze_token_security, security_score

logger = logging.getLogger("crypto_tools.portfolio_analyzer")


class PortfolioAnalysisError(Exception):
    """Exception raised for errors in the portfolio analysis process."""
    pass


async def analyze_portfolio(
        portfolio: List[Dict[str, Any]],
        okx_client=None,
        third_party_client=None,
        binance_tickers=None
) -> Dict[str, Any]:
    """
    Analyze an existing portfolio of tokens.

    Args:
        portfolio: List of tokens with amounts, e.g. [{"symbol": "ETH", "amount": 1.5, "chain_id": "1", "address": "0x..."}]
        okx_client: Optional OkxWeb3Client instance
        third_party_client: Optional ThirdPartyClient for security data
        binance_tickers: Optional function to get prices from Binance

    Returns:
        A dictionary containing portfolio analysis results
    """
    logger.debug(f"Analyzing portfolio with {len(portfolio)} assets")

    try:
        analysis_result = {
            "timestamp": datetime.now().isoformat(),
            "total_assets": len(portfolio),
            "total_value_usd": 0,
            "assets": [],
            "risk_metrics": {},
            "diversity_score": 0,
            "recommendations": []
        }

        # Process each asset
        for asset in portfolio:
            symbol = asset.get("symbol", "").upper()
            amount = float(asset.get("amount", 0))
            chain_id = asset.get("chain_id")
            address = asset.get("address")

            # Skip invalid assets
            if not symbol or amount <= 0:
                continue

            # Get token price
            price_data = await get_token_price(
                token_symbol=symbol,
                chain_id=chain_id,
                token_address=address,
                okx_client=okx_client,
                binance_tickers=binance_tickers
            )

            # Check token security if address provided
            security_data = None
            security_risk = 0

            if address and chain_id and third_party_client:
                security_data = await analyze_token_security(
                    original_token=symbol,
                    chain_id=chain_id,
                    target_token=address,
                    third_party_client=third_party_client
                )
                security_risk = security_data.get("risk_score", 5)

            # Calculate asset value
            asset_value = 0
            if price_data.get("found", False):
                price = price_data.get("price", 0)
                asset_value = price * amount
                analysis_result["total_value_usd"] += asset_value

            # Add to assets list
            asset_info = {
                "symbol": symbol,
                "amount": amount,
                "price_usd": price_data.get("price", 0) if price_data.get("found", False) else None,
                "value_usd": asset_value,
                "percentage": 0,  # Will be calculated after total is known
                "chain_id": chain_id,
                "address": address,
                "security_risk": security_risk,
                "security_data": security_data
            }

            analysis_result["assets"].append(asset_info)

        # Calculate percentages and finalize asset data
        if analysis_result["total_value_usd"] > 0:
            for asset in analysis_result["assets"]:
                if asset["value_usd"] is not None:
                    asset["percentage"] = (asset["value_usd"] / analysis_result["total_value_usd"]) * 100

        # Calculate risk metrics
        security_risks = [asset["security_risk"] for asset in analysis_result["assets"]
                          if "security_risk" in asset and asset["security_risk"] is not None]

        if security_risks:
            avg_security_risk = sum(security_risks) / len(security_risks)
            max_security_risk = max(security_risks)

            analysis_result["risk_metrics"] = {
                "average_security_risk": avg_security_risk,
                "max_security_risk": max_security_risk,
                "high_risk_assets": sum(1 for risk in security_risks if risk >= 7),
                "medium_risk_assets": sum(1 for risk in security_risks if 3 <= risk < 7),
                "low_risk_assets": sum(1 for risk in security_risks if risk < 3)
            }

        # Calculate diversity score (0-10)
        if len(analysis_result["assets"]) > 0:
            # Higher score for more diverse assets
            num_assets_factor = min(len(analysis_result["assets"]) / 10, 1) * 5

            # Higher score for more balanced distribution
            distribution_factor = 0
            if len(analysis_result["assets"]) > 1:
                percentages = [asset["percentage"] for asset in analysis_result["assets"] if "percentage" in asset]
                if percentages:
                    # Calculate normalized standard deviation as a measure of balance
                    mean = sum(percentages) / len(percentages)
                    variance = sum((p - mean) ** 2 for p in percentages) / len(percentages)
                    std_dev = math.sqrt(variance)

                    # Lower std dev means more balanced
                    # normalize between 0-5
                    normalized_std_dev = max(0, 5 - (std_dev / 10))
                    distribution_factor = normalized_std_dev

            analysis_result["diversity_score"] = round(num_assets_factor + distribution_factor, 1)

        # Generate recommendations
        recommendations = []

        # Check for high concentration
        high_concentration_assets = [asset for asset in analysis_result["assets"]
                                     if asset.get("percentage", 0) > 50]
        if high_concentration_assets:
            recommendations.append({
                "type": "diversification",
                "message": f"Consider diversifying away from {high_concentration_assets[0]['symbol']} which represents {high_concentration_assets[0]['percentage']:.1f}% of your portfolio.",
                "importance": "high"
            })

        # Check for high risk assets
        high_risk_assets = [asset for asset in analysis_result["assets"]
                            if asset.get("security_risk", 0) >= 7]
        if high_risk_assets:
            recommendations.append({
                "type": "security",
                "message": f"Consider reducing exposure to high-risk assets: {', '.join(asset['symbol'] for asset in high_risk_assets)}",
                "importance": "high"
            })

        # Add general recommendation for small portfolios
        if len(analysis_result["assets"]) < 3:
            recommendations.append({
                "type": "diversification",
                "message": "Consider adding more assets to improve diversification.",
                "importance": "medium"
            })

        analysis_result["recommendations"] = recommendations

        return analysis_result

    except Exception as e:
        logger.error(f"Error analyzing portfolio: {str(e)}")
        raise PortfolioAnalysisError(f"Failed to analyze portfolio: {str(e)}")


async def suggest_portfolio(
        total_investment: float,
        risk_profile: str = "medium",
        chain_id: str = "8453",  # Default to Base chain
        hot_tokens: Optional[List[Dict[str, Any]]] = None,
        okx_client=None,
        third_party_client=None
) -> Dict[str, Any]:
    """
    Suggest a portfolio based on investment amount and risk profile.

    Args:
        total_investment: Total amount to invest in USD
        risk_profile: Risk profile ("low", "medium", or "high")
        chain_id: Target blockchain chain ID
        hot_tokens: Optional list of trending tokens to consider
        okx_client: Optional OkxWeb3Client instance
        third_party_client: Optional ThirdPartyClient for security data

    Returns:
        A dictionary containing the suggested portfolio
    """
    logger.debug(f"Suggesting portfolio for ${total_investment} with {risk_profile} risk on chain {chain_id}")

    try:
        # Define base allocations based on risk profile
        allocations = {
            "low": {
                "major": 0.60,  # 60% to major assets (ETH, etc.)
                "medium": 0.30,  # 30% to medium-risk
                "speculative": 0.10  # 10% to speculative/high-risk
            },
            "medium": {
                "major": 0.40,  # 40% to major assets
                "medium": 0.40,  # 40% to medium-risk
                "speculative": 0.20  # 20% to speculative/high-risk
            },
            "high": {
                "major": 0.20,  # 20% to major assets
                "medium": 0.40,  # 40% to medium-risk
                "speculative": 0.40  # 40% to speculative/high-risk
            }
        }

        # Define assets by risk category
        # These would ideally be dynamically determined based on current market data
        base_assets = {
            "major": [
                {"symbol": "ETH", "address": None, "name": "Ethereum"},
                {"symbol": "BTC", "address": None, "name": "Bitcoin"}
            ],
            "medium": [
                {"symbol": "LINK", "address": "0x1Bc24f4AFBe1A395c10Ef382cF8c33a50E1C53B2", "name": "Chainlink"},
                {"symbol": "AAVE", "address": "0x8b09aebcaed17ac2a8e003c09b4c3db3e986bd5a", "name": "Aave"}
            ],
            "speculative": []  # This will be filled with hot tokens if available
        }

        # If hot tokens are provided, use them for speculative allocation
        if hot_tokens and len(hot_tokens) > 0:
            # Filter tokens with security risk under threshold
            safe_hot_tokens = []
            for token in hot_tokens:
                if token.get("symbol") and token.get("address"):
                    # Check security if possible
                    if third_party_client:
                        security_data = await analyze_token_security(
                            token_symbol=token["symbol"],
                            chain_id=chain_id,
                            token_address=token["address"],
                            third_party_client=third_party_client
                        )
                        # Only include tokens below certain risk threshold
                        risk_threshold = 8 if risk_profile == "high" else (6 if risk_profile == "medium" else 4)
                        if security_data.get("risk_score", 10) <= risk_threshold:
                            safe_hot_tokens.append(token)
                    else:
                        # If we can't check security, still include but limit to fewer tokens
                        safe_hot_tokens.append(token)

            # Limit to top 3 tokens
            base_assets["speculative"] = safe_hot_tokens[:3]

        # If no speculative tokens, adjust allocation
        if not base_assets["speculative"]:
            # Redistribute speculative allocation to medium and major
            spec_alloc = allocations[risk_profile]["speculative"]
            allocations[risk_profile]["major"] += spec_alloc * 0.5
            allocations[risk_profile]["medium"] += spec_alloc * 0.5
            allocations[risk_profile]["speculative"] = 0

        # Create the suggestion result
        result = {
            "total_investment": total_investment,
            "risk_profile": risk_profile,
            "chain_id": chain_id,
            "timestamp": datetime.now().isoformat(),
            "allocations": {},
            "tokens": []
        }

        # Calculate token allocations
        for category, allocation in allocations[risk_profile].items():
            category_amount = total_investment * allocation
            tokens_in_category = base_assets[category]

            if not tokens_in_category:
                continue

            # Distribute evenly within category
            token_amount = category_amount / len(tokens_in_category)

            for token in tokens_in_category:
                symbol = token["symbol"]

                # Get current price if possible
                price = 0
                if okx_client:
                    price_data = await get_token_price(
                        token_symbol=symbol,
                        chain_id=chain_id,
                        token_address=token.get("address"),
                        okx_client=okx_client
                    )
                    if price_data.get("found", False):
                        price = price_data.get("price", 0)

                # Calculate token quantity
                quantity = 0
                if price > 0:
                    quantity = token_amount / price

                token_allocation = {
                    "symbol": symbol,
                    "name": token.get("name", symbol),
                    "address": token.get("address"),
                    "chain_id": chain_id,
                    "category": category,
                    "allocation_percentage": (allocation / sum(allocations[risk_profile].values())) * 100,
                    "allocation_amount": token_amount,
                    "price": price,
                    "quantity": quantity
                }

                result["tokens"].append(token_allocation)

        # Calculate category allocations for the result
        category_totals = {}
        for category in allocations[risk_profile].keys():
            category_tokens = [t for t in result["tokens"] if t["category"] == category]
            category_total = sum(t["allocation_amount"] for t in category_tokens)
            category_totals[category] = {
                "amount": category_total,
                "percentage": (category_total / total_investment) * 100,
                "tokens": len(category_tokens)
            }

        result["allocations"] = category_totals

        return result

    except Exception as e:
        logger.error(f"Error suggesting portfolio: {str(e)}")
        raise PortfolioAnalysisError(f"Failed to suggest portfolio: {str(e)}")


async def calculate_portfolio_metrics(
        portfolio: List[Dict[str, Any]],
        historical_days: int = 30,
        okx_client=None
) -> Dict[str, Any]:
    """
    Calculate various metrics for a portfolio based on historical data.

    Args:
        portfolio: List of tokens with amounts
        historical_days: Number of days of historical data to analyze
        okx_client: Optional OkxWeb3Client instance

    Returns:
        A dictionary containing portfolio metrics
    """
    logger.debug(f"Calculating portfolio metrics over {historical_days} days")

    try:
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "period_days": historical_days,
            "volatility": None,
            "performance": {
                "7d": None,
                "30d": None
            },
            "max_drawdown": None,
            "sharpe_ratio": None,
            "token_correlations": {},
            "risk_adjusted_return": None
        }

        # This is a placeholder for a more complex implementation
        # A full implementation would:
        # 1. Get historical prices for all tokens in the portfolio
        # 2. Calculate portfolio value over time
        # 3. Calculate volatility, performance, drawdown, etc.

        return metrics

    except Exception as e:
        logger.error(f"Error calculating portfolio metrics: {str(e)}")
        raise PortfolioAnalysisError(f"Failed to calculate portfolio metrics: {str(e)}")