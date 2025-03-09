import logging
import json
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from datetime import datetime, timedelta

from .token_security import analyze_token_security

logger = logging.getLogger("crypto_tools.market_scanner")


class MarketScanError(Exception):
    """Exception raised for errors in the market scanning process."""
    pass


async def get_trending_tokens(
        limit: int = 10,
        chain_id: Optional[str] = None,
        cave_client=None
) -> List[Dict[str, Any]]:
    """
    Get trending tokens across all chains or a specific chain.

    Args:
        limit: Maximum number of tokens to return
        chain_id: Optional chain ID to filter by
        cave_client: Optional CaveClient for data

    Returns:
        A list of trending tokens with metadata
    """
    logger.debug(f"Getting top {limit} trending tokens" + (f" on chain {chain_id}" if chain_id else ""))

    try:
        trending_tokens = []

        # Use CARV client for on-chain data if available
        if cave_client:
            try:
                # Form query based on parameters
                chain_filter = f"on {chain_id} chain" if chain_id else "across all chains"
                query = f"What are the top {limit} trending tokens by trading volume {chain_filter} in the last 24 hours? Please provide symbol, contract address, and volume."

                # Execute query
                carv_response = await cave_client.fetch_on_chain_data_by_llm(query)

                if carv_response:
                    # Process the results
                    # This is a placeholder as actual implementation would depend on CARV response format
                    trending_tokens = []

                    # Example output format that we might extract from response
                    # trending_tokens = [
                    #     {
                    #         "symbol": "TOKEN1",
                    #         "name": "Token One",
                    #         "address": "0x123...",
                    #         "chain_id": chain_id or "1",
                    #         "volume_24h": 1000000,
                    #         "price_change_24h": 5.2,
                    #         "market_cap": 10000000
                    #     },
                    #     ...
                    # ]
            except Exception as e:
                logger.warning(f"Error getting trending tokens from CARV: {str(e)}")

        # If we couldn't get data or it's empty, return a fallback or empty list
        if not trending_tokens:
            return []

        return trending_tokens[:limit]  # Ensure we don't exceed requested limit

    except Exception as e:
        logger.error(f"Error scanning for trending tokens: {str(e)}")
        raise MarketScanError(f"Failed to get trending tokens: {str(e)}")


async def scan_new_listings(
        hours_ago: int = 24,
        chain_id: Optional[str] = None,
        min_liquidity: float = 10000,
        cave_client=None,
        third_party_client=None
) -> List[Dict[str, Any]]:
    """
    Scan for new token listings within a specified time period.

    Args:
        hours_ago: How many hours back to scan
        chain_id: Optional chain ID to filter by
        min_liquidity: Minimum liquidity in USD
        cave_client: Optional CaveClient for data
        third_party_client: Optional ThirdPartyClient for security data

    Returns:
        A list of new token listings with metadata
    """
    logger.debug(f"Scanning for new listings in the past {hours_ago} hours" +
                 (f" on chain {chain_id}" if chain_id else ""))

    try:
        new_listings = []

        # Use CARV client for on-chain data if available
        if cave_client:
            try:
                # Form query based on parameters
                chain_filter = f"on {chain_id} chain" if chain_id else "across all chains"
                query = f"What are the new token listings {chain_filter} in the past {hours_ago} hours with at least ${min_liquidity} in liquidity? Please provide symbol, name, contract address, initial price, and current price."

                # Execute query
                carv_response = await cave_client.fetch_on_chain_data_by_llm(query)

                if carv_response:
                    # Process the results
                    # This is a placeholder as actual implementation would depend on CARV response format
                    new_listings = []

                    # If third_party_client is available, check security for each token
                    if third_party_client and new_listings:
                        for token in new_listings:
                            if token.get("address") and (token.get("chain_id") or chain_id):
                                try:
                                    security_data = await analyze_token_security(
                                        token_symbol=token.get("symbol", ""),
                                        chain_id=token.get("chain_id", chain_id),
                                        token_address=token["address"],
                                        third_party_client=third_party_client
                                    )
                                    token["security"] = security_data
                                except Exception as e:
                                    logger.warning(f"Error analyzing token security: {str(e)}")
            except Exception as e:
                logger.warning(f"Error scanning for new listings: {str(e)}")

        # If we couldn't get data or it's empty, return a fallback or empty list
        if not new_listings:
            return []

        return new_listings

    except Exception as e:
        logger.error(f"Error scanning for new listings: {str(e)}")
        raise MarketScanError(f"Failed to scan for new listings: {str(e)}")


async def get_hot_tokens_by_chain(
        chain_id: str,
        limit: int = 5,
        min_market_cap: float = 100000,
        min_liquidity: float = 50000,
        okx_client=None,
        cave_client=None,
        third_party_client=None
) -> List[Dict[str, Any]]:
    """
    Get hot tokens on a specific chain with filtering.

    Args:
        chain_id: Chain ID to scan
        limit: Maximum number of tokens to return
        min_market_cap: Minimum market cap in USD
        min_liquidity: Minimum liquidity in USD
        okx_client: Optional OkxWeb3Client instance
        cave_client: Optional CaveClient for data
        third_party_client: Optional ThirdPartyClient for security data

    Returns:
        A list of hot tokens on the specified chain
    """
    logger.debug(f"Getting {limit} hot tokens on chain {chain_id}")

    try:
        hot_tokens = []

        # Use CARV client for on-chain data if available
        if cave_client:
            try:
                # Form query based on parameters
                query = f"""What are the top {limit} tokens on chain {chain_id} by trading volume in the past 24 hours 
                with minimum market cap of ${min_market_cap} and minimum liquidity of ${min_liquidity}? 
                Please provide symbol, name, contract address, volume, and price change percentage."""

                # Execute query
                carv_response = await cave_client.fetch_on_chain_data_by_llm(query)

                if carv_response:
                    # Process the results
                    # This is a placeholder as actual implementation would depend on CARV response format
                    hot_tokens = []

                    # Example processing (would need to be adapted to actual response format)
                    if isinstance(carv_response, dict) and "data" in carv_response:
                        # This is just a placeholder structure
                        for token_data in carv_response.get("data", []):
                            token = {
                                "symbol": token_data.get("symbol", ""),
                                "name": token_data.get("name", ""),
                                "address": token_data.get("address", ""),
                                "chain_id": chain_id,
                                "volume_24h": float(token_data.get("volume", 0)),
                                "price_change_24h": float(token_data.get("price_change", 0)),
                                "liquidity": float(token_data.get("liquidity", 0)),
                                "market_cap": float(token_data.get("market_cap", 0))
                            }
                            hot_tokens.append(token)
            except Exception as e:
                logger.warning(f"Error getting hot tokens: {str(e)}")

        # Fallback to a default set if needed
        if not hot_tokens and chain_id == "8453":  # Base chain
            hot_tokens = [
                {
                    "symbol": "ETH",
                    "name": "Ethereum",
                    "address": None,
                    "chain_id": "8453",
                    "volume_24h": 1000000,
                    "price_change_24h": 2.5,
                    "liquidity": 1000000000,
                    "market_cap": 500000000000
                },
                {
                    "symbol": "USDC",
                    "name": "USD Coin",
                    "address": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
                    "chain_id": "8453",
                    "volume_24h": 500000,
                    "price_change_24h": 0.1,
                    "liquidity": 500000000,
                    "market_cap": 26000000000
                }
            ]

        # Check security for each token if third_party_client is available
        if third_party_client and hot_tokens:
            for token in hot_tokens:
                if token.get("address"):
                    try:
                        security_data = await analyze_token_security(
                            token_symbol=token.get("symbol", ""),
                            chain_id=chain_id,
                            token_address=token["address"],
                            third_party_client=third_party_client
                        )
                        token["security"] = security_data
                    except Exception as e:
                        logger.warning(f"Error analyzing token security: {str(e)}")

        # Sort by volume and limit results
        hot_tokens = sorted(hot_tokens, key=lambda x: x.get("volume_24h", 0), reverse=True)
        return hot_tokens[:limit]

    except Exception as e:
        logger.error(f"Error getting hot tokens: {str(e)}")
        raise MarketScanError(f"Failed to get hot tokens: {str(e)}")