import logging
import json
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from datetime import datetime, timedelta

from .carv_dataframe_client import CarvDataframeClient, query_on_chain_data

logger = logging.getLogger("on_chain.blockchain_analyzer")


class BlockchainAnalysisError(Exception):
    """Exception raised for errors in blockchain analysis."""
    pass


async def analyze_blockchain_activity(
        chain: str = "base",
        days: int = 1,
        metrics: List[str] = ["transactions", "gas", "active_addresses"],
        cave_client=None
) -> Dict[str, Any]:
    """
    Analyze blockchain activity for a specified chain.

    Args:
        chain: Blockchain chain name
        days: Number of days to analyze
        metrics: List of metrics to include
        cave_client: Optional CaveClient instance

    Returns:
        Dictionary containing blockchain activity analysis
    """
    logger.debug(f"Analyzing {chain} blockchain activity for past {days} days")

    try:
        if not cave_client:
            raise BlockchainAnalysisError("CARV client is not available")

        # Build query based on requested metrics
        metrics_str = ", ".join(metrics)
        query = f"""
        Analyze the following metrics on {chain} blockchain for the past {days} days:
        {metrics_str}.

        For each metric, provide:
        1. Current value
        2. Change compared to previous period
        3. Any notable trends or anomalies
        """

        result = await query_on_chain_data(query, cave_client)

        # Process the result to extract activity data
        # This would need to be adapted based on actual CARV response format

        # For demonstration, return a structured placeholder
        activity = {
            "chain": chain,
            "analysis_period_days": days,
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "insights": []
        }

        # Add placeholder data for each requested metric
        for metric in metrics:
            activity["metrics"][metric] = {
                "current_value": 0,
                "previous_value": 0,
                "change_percentage": 0,
                "trend": "stable"
            }

        # Parse the response to extract insights
        # This is a placeholder as actual implementation would depend on CARV response format

        return activity

    except Exception as e:
        logger.error(f"Error analyzing blockchain activity: {str(e)}")
        raise BlockchainAnalysisError(f"Failed to analyze blockchain activity: {str(e)}")


async def get_top_addresses(
        chain: str = "base",
        category: str = "active",
        limit: int = 10,
        time_period: str = "24h",
        cave_client=None
) -> List[Dict[str, Any]]:
    """
    Get top addresses by various metrics.

    Args:
        chain: Blockchain chain name
        category: Category of addresses ("active", "new", "whales", etc.)
        limit: Number of addresses to return
        time_period: Time period for calculation
        cave_client: Optional CaveClient instance

    Returns:
        List of top addresses with metadata
    """
    logger.debug(f"Getting top {limit} {category} addresses on {chain} for {time_period}")

    try:
        if not cave_client:
            raise BlockchainAnalysisError("CARV client is not available")

        query = f"""
        What are the top {limit} {category} addresses on {chain} chain in the last {time_period}?

        For each address please provide:
        1. Address
        2. Transaction count
        3. Total volume
        4. Most interacted contracts
        """

        result = await query_on_chain_data(query, cave_client)

        # Process the result to extract address data
        # This would need to be adapted based on actual CARV response format

        # For demonstration, return a placeholder list
        addresses = []

        # Example response processing (would need to be adapted to actual format)
        if isinstance(result, dict) and "data" in result:
            for address_data in result.get("data", []):
                address = {
                    "address": address_data.get("address", ""),
                    "chain": chain,
                    "transaction_count": int(address_data.get("transaction_count", 0)),
                    "volume_usd": float(address_data.get("volume", 0)),
                    "top_contracts": address_data.get("contracts", [])
                }
                addresses.append(address)

        return addresses

    except Exception as e:
        logger.error(f"Error getting top addresses: {str(e)}")
        raise BlockchainAnalysisError(f"Failed to get top addresses: {str(e)}")


async def analyze_token_transfers(
        token_address: str,
        chain: str = "base",
        days: int = 7,
        min_amount: Optional[float] = None,
        cave_client=None
) -> Dict[str, Any]:
    """
    Analyze transfers for a specific token.

    Args:
        token_address: Token contract address
        chain: Blockchain chain name
        days: Number of days to analyze
        min_amount: Minimum transfer amount to include
        cave_client: Optional CaveClient instance

    Returns:
        Dictionary containing token transfer analysis
    """
    logger.debug(f"Analyzing transfers for token {token_address} on {chain} for {days} days")

    try:
        if not cave_client:
            raise BlockchainAnalysisError("CARV client is not available")

        # Build query with optional minimum amount filter
        amount_filter = f"larger than ${min_amount}" if min_amount else ""
        query = f"""
        Analyze transfers for token {token_address} on {chain} chain over the past {days} days {amount_filter}.

        Please provide:
        1. Total number of transfers
        2. Total volume transferred
        3. Largest transfers
        4. Transfer patterns (e.g., accumulation, distribution)
        5. Any notable wallet behaviors
        """

        result = await query_on_chain_data(query, cave_client)

        # Process the result to extract transfer data
        # This would need to be adapted based on actual CARV response format

        # For demonstration, return a structured placeholder
        transfers = {
            "token_address": token_address,
            "chain": chain,
            "analysis_period_days": days,
            "minimum_amount": min_amount,
            "timestamp": datetime.now().isoformat(),
            "total_transfers": 0,
            "total_volume": 0,
            "largest_transfers": [],
            "transfer_patterns": [],
            "notable_wallets": []
        }

        # Parse the response to extract transfer data
        # This is a placeholder as actual implementation would depend on CARV response format

        return transfers

    except Exception as e:
        logger.error(f"Error analyzing token transfers: {str(e)}")
        raise BlockchainAnalysisError(f"Failed to analyze token transfers: {str(e)}")