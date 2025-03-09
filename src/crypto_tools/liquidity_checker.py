import logging
import json
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from datetime import datetime, timedelta

logger = logging.getLogger("crypto_tools.liquidity_checker")


class LiquidityCheckError(Exception):
    """Exception raised for errors in the liquidity checking process."""
    pass


async def check_liquidity(
        token_symbol: str,
        chain_id: str,
        token_address: str,
        cave_client=None
) -> Dict[str, Any]:
    """
    Check liquidity for a token.

    Args:
        token_symbol: The token symbol
        chain_id: Chain ID where the token is deployed
        token_address: Token contract address
        cave_client: Optional CaveClient for on-chain analysis

    Returns:
        A dictionary containing liquidity information
    """
    logger.debug(f"Checking liquidity for {token_symbol} ({token_address}) on chain {chain_id}")

    try:
        # Initialize result
        liquidity_result = {
            "symbol": token_symbol,
            "address": token_address,
            "chain_id": chain_id,
            "timestamp": datetime.now().isoformat(),
            "total_liquidity_usd": 0,
            "liquidity_sources": [],
            "is_sufficient": False,
            "liquidity_score": 0,
            "recommended_max_swap": 0
        }

        # Use CARV client if available
        if cave_client:
            try:
                # Form query to get liquidity information
                query = f"""Analyze the liquidity for token {token_symbol} with address {token_address} on chain {chain_id}.
                Please provide:
                1. Total liquidity in USD
                2. Liquidity sources/pools
                3. Liquidity depth (how much can be swapped without significant slippage)
                4. Liquidity concentration (are there a few large LPs that could pull liquidity)
                """

                carv_response = await cave_client.fetch_on_chain_data_by_llm(query)

                if carv_response:
                    # Process the results
                    # This would need to be adapted based on actual CARV response format

                    # For demonstration, update with placeholder data
                    liquidity_result["total_liquidity_usd"] = 1000000  # Example value
                    liquidity_result["liquidity_sources"] = [
                        {
                            "dex": "Uniswap V3",
                            "pool_address": "0x123...",
                            "liquidity_usd": 800000,
                            "percentage": 80
                        },
                        {
                            "dex": "SushiSwap",
                            "pool_address": "0x456...",
                            "liquidity_usd": 200000,
                            "percentage": 20
                        }
                    ]

                    # Determine if liquidity is sufficient (> $100k is reasonable)
                    liquidity_result["is_sufficient"] = liquidity_result["total_liquidity_usd"] > 100000

                    # Calculate liquidity score (0-10)
                    # Factors:
                    # - Total amount (higher is better)
                    # - Number of sources (more is better)
                    # - Concentration (lower is better)

                    # Score based on total amount (0-5 points)
                    amount_score = min(5, liquidity_result["total_liquidity_usd"] / 200000)

                    # Score based on number of sources (0-3 points)
                    sources_score = min(3, len(liquidity_result["liquidity_sources"]))

                    # Score based on concentration (0-2 points)
                    concentration_score = 0
                    if liquidity_result["liquidity_sources"]:
                        max_percentage = max(source["percentage"] for source in liquidity_result["liquidity_sources"])
                        concentration_score = 2 * (1 - (max_percentage / 100))

                    liquidity_result["liquidity_score"] = round(amount_score + sources_score + concentration_score, 1)

                    # Calculate recommended max swap (0.5% of total liquidity is a safe amount)
                    liquidity_result["recommended_max_swap"] = liquidity_result["total_liquidity_usd"] * 0.005
            except Exception as e:
                logger.warning(f"Error getting liquidity data from CARV: {str(e)}")

        # If we couldn't get data, provide a basic assessment
        if liquidity_result["total_liquidity_usd"] == 0:
            liquidity_result["is_sufficient"] = False
            liquidity_result["liquidity_score"] = 0
            liquidity_result["recommended_max_swap"] = 0

        return liquidity_result

    except Exception as e:
        logger.error(f"Error checking liquidity: {str(e)}")
        raise LiquidityCheckError(f"Failed to check liquidity: {str(e)}")


async def analyze_liquidity_depth(
        token_symbol: str,
        chain_id: str,
        token_address: str,
        swap_amounts: List[float] = [1000, 5000, 10000, 50000, 100000],
        cave_client=None
) -> Dict[str, Any]:
    """
    Analyze liquidity depth by simulating swaps of different sizes.

    Args:
        token_symbol: The token symbol
        chain_id: Chain ID where the token is deployed
        token_address: Token contract address
        swap_amounts: List of USD amounts to simulate swaps for
        cave_client: Optional CaveClient for on-chain analysis

    Returns:
        A dictionary containing liquidity depth analysis
    """
    logger.debug(f"Analyzing liquidity depth for {token_symbol} on chain {chain_id}")

    try:
        depth_result = {
            "symbol": token_symbol,
            "address": token_address,
            "chain_id": chain_id,
            "timestamp": datetime.now().isoformat(),
            "swap_simulations": []
        }

        # Use CARV client if available
        if cave_client:
            try:
                # Form query for liquidity depth analysis
                amounts_str = ", ".join(f"${amount}" for amount in swap_amounts)
                query = f"""Analyze the liquidity depth for token {token_symbol} with address {token_address} on chain {chain_id}.
                Please simulate swaps of the following amounts: {amounts_str}.
                For each amount, calculate:
                1. The expected slippage
                2. The equivalent token amount received
                3. The impact on pool price
                """

                carv_response = await cave_client.fetch_on_chain_data_by_llm(query)

                if carv_response:
                    # Process the results
                    # This would need to be adapted based on actual CARV response format

                    # For demonstration, create placeholder simulations
                    for amount in swap_amounts:
                        # Scale slippage with amount (just an example formula)
                        slippage = min(20, 0.1 * (amount / 1000) ** 1.5)

                        depth_result["swap_simulations"].append({
                            "amount_usd": amount,
                            "slippage_percent": slippage,
                            "token_amount": amount / 10,  # Placeholder calculation
                            "price_impact_percent": slippage * 0.8,
                            "is_problematic": slippage > 3  # More than 3% slippage is problematic
                        })
            except Exception as e:
                logger.warning(f"Error analyzing liquidity depth: {str(e)}")

        # If we couldn't get data, return empty simulations
        if not depth_result["swap_simulations"]:
            for amount in swap_amounts:
                depth_result["swap_simulations"].append({
                    "amount_usd": amount,
                    "slippage_percent": None,
                    "token_amount": None,
                    "price_impact_percent": None,
                    "is_problematic": None
                })

        return depth_result

    except Exception as e:
        logger.error(f"Error analyzing liquidity depth: {str(e)}")
        raise LiquidityCheckError(f"Failed to analyze liquidity depth: {str(e)}")


async def get_liquidity_report(
        token_symbol: str,
        chain_id: str,
        token_address: str,
        cave_client=None
) -> Dict[str, Any]:
    """
    Get a comprehensive liquidity report for a token.

    Args:
        token_symbol: The token symbol
        chain_id: Chain ID where the token is deployed
        token_address: Token contract address
        cave_client: Optional CaveClient for on-chain analysis

    Returns:
        A dictionary containing a comprehensive liquidity report
    """
    logger.debug(f"Generating liquidity report for {token_symbol} on chain {chain_id}")

    try:
        # Get basic liquidity information
        liquidity_info = await check_liquidity(
            token_symbol=token_symbol,
            chain_id=chain_id,
            token_address=token_address,
            cave_client=cave_client
        )

        # Get liquidity depth analysis
        depth_analysis = await analyze_liquidity_depth(
            token_symbol=token_symbol,
            chain_id=chain_id,
            token_address=token_address,
            cave_client=cave_client
        )

        # Combine into a comprehensive report
        liquidity_report = {
            "symbol": token_symbol,
            "address": token_address,
            "chain_id": chain_id,
            "timestamp": datetime.now().isoformat(),
            "liquidity_info": liquidity_info,
            "liquidity_depth_analysis": depth_analysis
        }

        return liquidity_report

    except Exception as e:
        logger.error(f"Error generating liquidity report: {str(e)}")
        raise LiquidityCheckError(f"Failed to generate liquidity report: {str(e)}")