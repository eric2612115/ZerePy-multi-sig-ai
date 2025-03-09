import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Union
import asyncio
from datetime import datetime

logger = logging.getLogger("on_chain.carv_dataframe_client")


class CarvQueryError(Exception):
    """Exception raised for errors in CARV queries."""
    pass


class CarvDataframeClient:
    """Client for interacting with CARV Dataframe API."""

    def __init__(self, cave_client):
        """
        Initialize the CARV Dataframe client.

        Args:
            cave_client: CaveClient instance for making API calls
        """
        self.cave_client = cave_client
        self.supported_chains = ["ethereum", "base", "arbitrum", "optimism"]
        logger.debug("CarvDataframeClient initialized")

    async def execute_llm_query(self, query_text: str) -> Dict[str, Any]:
        """
        Execute a natural language query using CARV's LLM interface.

        Args:
            query_text: Natural language query text

        Returns:
            Query result as dictionary
        """
        logger.debug(f"Executing LLM query: {query_text[:50]}...")

        try:
            if not self.cave_client:
                raise CarvQueryError("CARV client is not available")

            result = await self.cave_client.fetch_on_chain_data_by_llm(query_text)

            if not result:
                raise CarvQueryError("Empty result returned from CARV")

            return result

        except Exception as e:
            logger.error(f"Error executing LLM query: {str(e)}")
            raise CarvQueryError(f"Failed to execute LLM query: {str(e)}")

    async def execute_sql_query(self, sql_content: str) -> Dict[str, Any]:
        """
        Execute a SQL query using CARV's SQL interface.

        Args:
            sql_content: SQL query string

        Returns:
            Query result as dictionary
        """
        logger.debug(f"Executing SQL query: {sql_content[:50]}...")

        try:
            if not self.cave_client:
                raise CarvQueryError("CARV client is not available")

            result = await self.cave_client.fetch_on_chain_data_by_sql_query(sql_content)

            if not result:
                raise CarvQueryError("Empty result returned from CARV")

            return result

        except Exception as e:
            logger.error(f"Error executing SQL query: {str(e)}")
            raise CarvQueryError(f"Failed to execute SQL query: {str(e)}")

    async def get_top_tokens_by_volume(
            self,
            chain: str = "base",
            limit: int = 5,
            time_period: str = "24h"
    ) -> List[Dict[str, Any]]:
        """
        Get top tokens by volume on a specified chain.

        Args:
            chain: Blockchain chain name
            limit: Number of tokens to return
            time_period: Time period for volume calculation ("24h", "7d", etc.)

        Returns:
            List of top tokens with metadata
        """
        logger.debug(f"Getting top {limit} tokens by volume on {chain} chain for {time_period}")

        query = f"""
        What are the top {limit} tokens on {chain} chain by trading volume in the last {time_period}?
        Please provide symbol, contract address, trading volume, and price change percentage.
        """

        try:
            result = await self.execute_llm_query(query)

            # Process the result to extract token information
            # This would need to be adapted based on actual CARV response format

            # For demonstration, return a placeholder structure
            tokens = []

            # Example response processing (would need to be adapted to actual format)
            if isinstance(result, dict) and "data" in result:
                for token_data in result.get("data", []):
                    token = {
                        "symbol": token_data.get("symbol", ""),
                        "name": token_data.get("name", ""),
                        "address": token_data.get("address", ""),
                        "chain": chain,
                        "volume_24h": float(token_data.get("volume", 0)),
                        "price_change_24h": float(token_data.get("price_change", 0))
                    }
                    tokens.append(token)

            return tokens

        except Exception as e:
            logger.error(f"Error getting top tokens by volume: {str(e)}")
            raise CarvQueryError(f"Failed to get top tokens by volume: {str(e)}")

    async def analyze_token_activity(
            self,
            token_address: str,
            chain: str = "base",
            days: int = 7
    ) -> Dict[str, Any]:
        """
        Analyze token activity over a time period.

        Args:
            token_address: Token contract address
            chain: Blockchain chain name
            days: Number of days to analyze

        Returns:
            Token activity analysis
        """
        logger.debug(f"Analyzing token activity for {token_address} on {chain} for {days} days")

        query = f"""
        Analyze the token with address {token_address} on {chain} chain over the past {days} days.
        Please provide:
        1. Daily transaction count
        2. Daily unique active addresses
        3. Whale movements (transactions > $100k)
        4. Distribution of token holders
        """

        try:
            result = await self.execute_llm_query(query)

            # Process the result to extract token activity data
            # This would need to be adapted based on actual CARV response format

            # For demonstration, return a structured placeholder
            activity = {
                "token_address": token_address,
                "chain": chain,
                "analysis_period_days": days,
                "timestamp": datetime.now().isoformat(),
                "transaction_data": {
                    "total_count": 0,
                    "daily_average": 0,
                    "trend": ""
                },
                "unique_addresses": {
                    "total_count": 0,
                    "daily_average": 0,
                    "trend": ""
                },
                "whale_movements": [],
                "holder_distribution": {},
                "insights": []
            }

            # Parse the response to extract insights
            # This is a placeholder as actual implementation would depend on CARV response format

            return activity

        except Exception as e:
            logger.error(f"Error analyzing token activity: {str(e)}")
            raise CarvQueryError(f"Failed to analyze token activity: {str(e)}")


# Standalone functions that use the client

async def query_on_chain_data(query_text: str, cave_client) -> Dict[str, Any]:
    """
    Execute a natural language query for on-chain data.

    Args:
        query_text: Natural language query
        cave_client: CaveClient instance

    Returns:
        Query result as dictionary
    """
    client = CarvDataframeClient(cave_client)
    return await client.execute_llm_query(query_text)


async def query_tokens_by_volume(
        chain: str = "base",
        limit: int = 5,
        time_period: str = "24h",
        cave_client=None
) -> List[Dict[str, Any]]:
    """
    Query for top tokens by volume.

    Args:
        chain: Blockchain chain name
        limit: Number of tokens to return
        time_period: Time period for volume calculation
        cave_client: CaveClient instance

    Returns:
        List of top tokens with metadata
    """
    if not cave_client:
        return []

    client = CarvDataframeClient(cave_client)
    return await client.get_top_tokens_by_volume(chain, limit, time_period)