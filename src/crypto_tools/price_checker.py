import logging
import json
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from datetime import datetime, timedelta

logger = logging.getLogger("crypto_tools.price_checker")


class PriceCheckError(Exception):
    """Exception raised for errors in the price checking process."""
    pass


async def get_token_price(
        token_symbol: str,
        chain_id: Optional[str] = None,
        token_address: Optional[str] = None,
        okx_client=None,
        binance_tickers=None
) -> Dict[str, Any]:
    """
    Get the current price of a token.

    Args:
        token_symbol: The token symbol (e.g., 'ETH', 'BTC')
        chain_id: Optional chain ID to specify which network to check
        token_address: Optional token contract address
        okx_client: Optional OkxWeb3Client instance
        binance_tickers: Optional function to get prices from Binance

    Returns:
        A dictionary containing price information
    """
    logger.debug(f"Getting price for {token_symbol} on chain {chain_id}")

    try:
        price_data = {}

        # Normalize symbol format
        token_symbol = token_symbol.upper()
        if token_symbol.endswith('USDT') or token_symbol.endswith('USD'):
            base_symbol = token_symbol
        else:
            base_symbol = f"{token_symbol}USDT"

        # Try OKX client if available and token address is provided
        if okx_client and chain_id and token_address:
            try:
                okx_result = await okx_client.get_token_price(chain_id, token_address)
                if okx_result and okx_result.get('code') == '0':
                    price_data['okx'] = {
                        'price': float(okx_result['data'][0]['price']),
                        'timestamp': datetime.now().isoformat(),
                        'source': 'OKX Web3'
                    }
                    logger.debug(f"Got OKX price for {token_symbol}: {price_data['okx']['price']}")
            except Exception as e:
                logger.warning(f"Failed to get OKX price for {token_symbol}: {str(e)}")

        # Try Binance API
        if binance_tickers:
            try:
                binance_data = await binance_tickers([base_symbol])
                if binance_data and base_symbol in binance_data:
                    price_data['binance'] = {
                        'price': float(binance_data[base_symbol]['price']),
                        'change_24h': float(binance_data[base_symbol]['change_24h']),
                        'timestamp': datetime.now().isoformat(),
                        'source': 'Binance'
                    }
                    logger.debug(f"Got Binance price for {token_symbol}: {price_data['binance']['price']}")
            except Exception as e:
                logger.warning(f"Failed to get Binance price for {token_symbol}: {str(e)}")

        # If no prices were found
        if not price_data:
            logger.warning(f"No price data found for {token_symbol}")
            return {
                'symbol': token_symbol,
                'found': False,
                'timestamp': datetime.now().isoformat(),
                'message': f"No price data found for {token_symbol}"
            }

        # Calculate average price if multiple sources
        prices = [data['price'] for source, data in price_data.items()]
        avg_price = sum(prices) / len(prices)

        # Determine best source based on recency and reliability
        best_source = list(price_data.keys())[0]  # Default to first source

        return {
            'symbol': token_symbol,
            'found': True,
            'price': avg_price,
            'sources': price_data,
            'best_source': best_source,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting price for {token_symbol}: {str(e)}")
        raise PriceCheckError(f"Failed to get price for {token_symbol}: {str(e)}")


async def get_historical_prices(
        token_symbol: str,
        days: int = 7,
        interval: str = '1d',
        chain_id: Optional[str] = None,
        token_address: Optional[str] = None,
        okx_client=None
) -> Dict[str, Any]:
    """
    Get historical price data for a token.

    Args:
        token_symbol: The token symbol
        days: Number of days of history to retrieve
        interval: Time interval between data points ('1h', '1d', etc.)
        chain_id: Optional chain ID
        token_address: Optional token contract address
        okx_client: Optional OkxWeb3Client instance

    Returns:
        A dictionary containing historical price data
    """
    logger.debug(f"Getting {days} days of historical prices for {token_symbol} with {interval} interval")

    try:
        if okx_client and chain_id and token_address:
            period_map = {
                '1h': '1H',
                '4h': '4H',
                '12h': '12H',
                '1d': '1D',
                '1w': '1W',
                '1M': '1M'
            }

            okx_period = period_map.get(interval, '1D')
            limit = min(days * 24 if interval == '1h' else days, 100)  # OKX API limit

            result = await okx_client.get_historical_price(
                chain_index=chain_id,
                token_address=token_address,
                limit=limit,
                period=okx_period
            )

            if result and result.get('code') == '0' and 'data' in result:
                price_data = []
                for price_point in result['data']:
                    try:
                        time_str = price_point.get('time')
                        price = float(price_point.get('price', 0))
                        if time_str and price > 0:
                            price_data.append({
                                'timestamp': time_str,
                                'price': price
                            })
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error parsing price point: {e}")

                return {
                    'symbol': token_symbol,
                    'interval': interval,
                    'days': days,
                    'data_points': len(price_data),
                    'prices': price_data,
                    'source': 'OKX'
                }

        # If we couldn't get data from OKX or necessary parameters weren't provided
        # Default to returning an empty result
        return {
            'symbol': token_symbol,
            'interval': interval,
            'days': days,
            'data_points': 0,
            'prices': [],
            'source': None,
            'message': 'No historical price data available'
        }

    except Exception as e:
        logger.error(f"Error getting historical prices for {token_symbol}: {str(e)}")
        raise PriceCheckError(f"Failed to get historical prices for {token_symbol}: {str(e)}")


async def compare_token_prices(
        tokens: List[Dict[str, str]],
        okx_client=None,
        binance_tickers=None
) -> Dict[str, Any]:
    """
    Compare prices for multiple tokens.

    Args:
        tokens: List of token dictionaries with 'symbol' and optional 'chain_id' and 'address'
        okx_client: Optional OkxWeb3Client instance
        binance_tickers: Optional function to get prices from Binance

    Returns:
        A dictionary containing comparative price information
    """
    logger.debug(f"Comparing prices for {len(tokens)} tokens")

    try:
        price_results = {}

        # Get prices for each token
        for token in tokens:
            symbol = token.get('symbol')
            if not symbol:
                continue

            price_data = await get_token_price(
                token_symbol=symbol,
                chain_id=token.get('chain_id'),
                token_address=token.get('address'),
                okx_client=okx_client,
                binance_tickers=binance_tickers
            )

            price_results[symbol] = price_data

        # Calculate some comparison metrics
        if price_results:
            found_tokens = {s: data for s, data in price_results.items() if data.get('found', False)}

            if found_tokens:
                avg_price = sum(data.get('price', 0) for data in found_tokens.values()) / len(found_tokens)

                return {
                    'tokens_compared': len(tokens),
                    'tokens_found': len(found_tokens),
                    'average_price': avg_price,
                    'prices': price_results,
                    'timestamp': datetime.now().isoformat()
                }

        return {
            'tokens_compared': len(tokens),
            'tokens_found': 0,
            'prices': price_results,
            'timestamp': datetime.now().isoformat(),
            'message': 'No comparable price data found'
        }

    except Exception as e:
        logger.error(f"Error comparing token prices: {str(e)}")
        raise PriceCheckError(f"Failed to compare token prices: {str(e)}")