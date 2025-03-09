import logging
import json
from datetime import datetime
import re
from typing import Dict, List, Any, Optional

from src.action_handler import register_action
from src.helpers import print_h_bar
from src.thinking_framework.thinking_depth_selector import analyze_thinking_depth
from src.structured_output.json_formatter import format_structured_response
from src.crypto_tools.price_checker import get_token_price, compare_token_prices
from src.crypto_tools.token_security import analyze_token_security
from src.crypto_tools.portfolio_analyzer import analyze_portfolio, suggest_portfolio
from src.crypto_tools.market_scanner import get_hot_tokens_by_chain
from src.crypto_tools.liquidity_checker import check_liquidity
from src.on_chain.carv_dataframe_client import query_on_chain_data

logger = logging.getLogger("actions.crypto_actions")


# Helper functions
def extract_token_info(text: str) -> Dict[str, Any]:
    """Extract token symbol, chain, and address information from text."""
    # Default values
    token_info = {
        "symbol": None,
        "chain_id": "8453",  # Default to Base chain
        "address": None
    }

    # Try to extract symbol
    symbol_match = re.search(r'[$]?([A-Z]{2,10})', text.upper())
    if symbol_match:
        token_info["symbol"] = symbol_match.group(1)

    # Try to extract chain ID
    chain_map = {
        "ethereum": "1",
        "eth": "1",
        "base": "8453",
        "arbitrum": "42161",
        "arb": "42161",
        "optimism": "10",
        "polygon": "137",
        "matic": "137"
    }

    for chain_name, chain_id in chain_map.items():
        if chain_name.lower() in text.lower():
            token_info["chain_id"] = chain_id
            break

    # Try to extract address (0x format)
    address_match = re.search(r'(0x[a-fA-F0-9]{40})', text)
    if address_match:
        token_info["address"] = address_match.group(1)

    return token_info


@register_action("check-token-price")
def check_token_price(agent, **kwargs):
    """Check the current price of a cryptocurrency token."""
    query = kwargs.get("query", "")
    if not query:
        agent.logger.error("No query provided for token price check")
        return "I need to know which token to check. Please provide a token symbol."

    agent.logger.info(f"\n💰 CHECKING TOKEN PRICE FOR: {query}")
    print_h_bar()

    # Extract token information
    token_info = extract_token_info(query)
    symbol = token_info["symbol"]
    chain_id = token_info["chain_id"]
    address = token_info["address"]

    if not symbol:
        return "I couldn't identify a token symbol in your request. Please specify which token you're interested in."

    try:
        # Check thinking depth for proper response formatting
        thinking_result = analyze_thinking_depth(query)
        thinking_depth = thinking_result["depth"]

        # Get clients from agent's connection manager
        okx_client = None
        binance_tickers = None

        if "okx" in agent.connection_manager.connections:
            okx_client = agent.connection_manager.connections["okx"]

        # Get token price
        price_data = agent.connection_manager.perform_action(
            connection_name="timetool",
            action_name="get-symbol-price",
            params=[f"{symbol}USDT"]  # Assuming standard USDT pair format
        )

        if price_data:
            # Format response based on thinking depth
            if thinking_depth == "light":
                response = f"The current price of {symbol} is ${price_data}."
            else:
                # For medium/deep, we'd normally add more context and analysis
                response = f"The current price of {symbol} is ${price_data}. This price was fetched from Binance's API and represents the latest trading price for the {symbol}/USDT pair."

            agent.logger.info(f"✅ Successfully retrieved {symbol} price: ${price_data}")
            return response
        else:
            return f"I'm unable to fetch the current price for {symbol}. The token may not be listed on major exchanges or there might be an issue with the price feed."

    except Exception as e:
        agent.logger.error(f"❌ Error checking token price: {str(e)}")
        return f"I encountered an error while checking the price for {symbol}: {str(e)}"


@register_action("check-token-security")
def check_token_security(agent, **kwargs):
    """Analyze a token's security profile and potential risks."""
    query = kwargs.get("query", "")
    if not query:
        agent.logger.error("No query provided for token security check")
        return "I need to know which token to analyze. Please provide a token symbol and preferably the contract address."

    agent.logger.info(f"\n🔒 CHECKING TOKEN SECURITY FOR: {query}")
    print_h_bar()

    # Extract token information
    token_info = extract_token_info(query)
    symbol = token_info["symbol"]
    chain_id = token_info["chain_id"]
    address = token_info["address"]

    if not symbol or not address:
        return "I need both a token symbol and contract address to perform a security analysis. Please provide both."

    try:
        # Get third-party client for security checks
        third_party_client = None
        if "third_party" in agent.connection_manager.connections:
            third_party_client = agent.connection_manager.connections["third_party"]

        if not third_party_client:
            return f"I'm unable to perform a security check for {symbol} because the security analysis service isn't available."

        # Analyze token security
        security_data = analyze_token_security(
            token_symbol=symbol,
            chain_id=chain_id,
            token_address=address,
            third_party_client=third_party_client
        )

        if security_data and security_data.get("analysis_complete", False):
            # Format response based on security findings
            risk_level = security_data.get("risk_level", "unknown")
            issues = security_data.get("issues_found", [])

            response = f"Security Analysis for {symbol} ({address[:8]}...{address[-6:]}):\n\n"
            response += f"Risk Level: {risk_level.upper()}\n"

            if issues:
                response += "\nPotential Issues Found:\n"
                for issue in issues:
                    response += f"- {issue.get('description', 'Unknown issue')} ({issue.get('severity', 'unknown')} severity)\n"
                    response += f"  Impact: {issue.get('impact', 'Unknown impact')}\n"
            else:
                response += "\nNo significant security issues were found. However, always conduct your own research before investing.\n"

            # Add recommendation based on risk level
            if risk_level in ["critical", "high"]:
                response += "\nRecommendation: Exercise extreme caution. This token shows significant security risks that could indicate a scam or vulnerable contract."
            elif risk_level == "medium":
                response += "\nRecommendation: Proceed with caution. While no critical issues were found, there are some security concerns to be aware of."
            else:
                response += "\nRecommendation: The token appears to have a reasonable security profile, but always do your own research and invest only what you can afford to lose."

            agent.logger.info(f"✅ Successfully analyzed security for {symbol}")
            return response
        else:
            return f"I was unable to complete a security analysis for {symbol}. Please check the contract address and try again."

    except Exception as e:
        agent.logger.error(f"❌ Error checking token security: {str(e)}")
        return f"I encountered an error while analyzing security for {symbol}: {str(e)}"


@register_action("analyze-portfolio")
def analyze_crypto_portfolio(agent, **kwargs):
    """Analyze a cryptocurrency portfolio for performance, risk, and recommendations."""
    query = kwargs.get("query", "")
    portfolio_data = kwargs.get("portfolio", [])

    agent.logger.info(f"\n📊 ANALYZING CRYPTO PORTFOLIO")
    print_h_bar()

    # If no portfolio data provided, try to extract from query
    if not portfolio_data:
        try:
            # Simple regex-based extraction (would be more sophisticated in production)
            matches = re.findall(r'(\d+(?:\.\d+)?)\s*([A-Z]{2,10})', query.upper())
            portfolio_data = [{"symbol": symbol, "amount": float(amount)} for amount, symbol in matches]
        except Exception as e:
            agent.logger.error(f"Error extracting portfolio data: {e}")

    if not portfolio_data:
        return "I need portfolio data to analyze. Please provide your holdings in a format like '2.5 ETH, 1000 USDC'."

    try:
        # Get necessary clients
        okx_client = None
        third_party_client = None

        if "okx" in agent.connection_manager.connections:
            okx_client = agent.connection_manager.connections["okx"]
        if "third_party" in agent.connection_manager.connections:
            third_party_client = agent.connection_manager.connections["third_party"]

        # Analyze portfolio
        analysis_result = analyze_portfolio(
            portfolio=portfolio_data,
            okx_client=okx_client,
            third_party_client=third_party_client
        )

        if analysis_result:
            # Format response
            response = "Portfolio Analysis Results:\n\n"

            # Add portfolio overview
            total_value = analysis_result.get("total_value_usd", 0)
            response += f"Total Portfolio Value: ${total_value:,.2f}\n"
            response += f"Total Assets: {analysis_result.get('total_assets', 0)}\n"
            response += f"Diversity Score: {analysis_result.get('diversity_score', 0)}/10\n\n"

            # Add asset breakdown
            assets = analysis_result.get("assets", [])
            if assets:
                response += "Asset Breakdown:\n"
                for asset in assets:
                    response += f"- {asset.get('symbol', 'Unknown')}: {asset.get('amount', 0)} tokens"
                    if asset.get("value_usd") is not None:
                        response += f" (${asset.get('value_usd', 0):,.2f}, {asset.get('percentage', 0):.1f}% of portfolio)"
                    if asset.get("security_risk") is not None and asset.get("security_risk") > 5:
                        response += f" ⚠️ Security Risk: {asset.get('security_risk')}/10"
                    response += "\n"
                response += "\n"

            # Add risk metrics
            risk_metrics = analysis_result.get("risk_metrics", {})
            if risk_metrics:
                response += "Risk Assessment:\n"
                response += f"- Average Security Risk: {risk_metrics.get('average_security_risk', 0)}/10\n"
                response += f"- High Risk Assets: {risk_metrics.get('high_risk_assets', 0)}\n"
                response += f"- Medium Risk Assets: {risk_metrics.get('medium_risk_assets', 0)}\n\n"

            # Add recommendations
            recommendations = analysis_result.get("recommendations", [])
            if recommendations:
                response += "Recommendations:\n"
                for rec in recommendations:
                    response += f"- {rec.get('message', '')}\n"

            agent.logger.info(f"✅ Successfully analyzed portfolio with {len(assets)} assets")
            return response
        else:
            return "I wasn't able to analyze your portfolio. Please check your portfolio data and try again."

    except Exception as e:
        agent.logger.error(f"❌ Error analyzing portfolio: {str(e)}")
        return f"I encountered an error while analyzing your portfolio: {str(e)}"


@register_action("get-hot-tokens")
def get_hot_tokens(agent, **kwargs):
    """Get trending/hot tokens on a specific blockchain."""
    query = kwargs.get("query", "")
    chain_id = kwargs.get("chain_id", "8453")  # Default to Base chain
    limit = kwargs.get("limit", 5)

    agent.logger.info(f"\n🔥 GETTING HOT TOKENS ON CHAIN: {chain_id}")
    print_h_bar()

    # Try to extract chain ID from query if not provided
    if not chain_id or chain_id == "8453":
        token_info = extract_token_info(query)
        if token_info.get("chain_id"):
            chain_id = token_info["chain_id"]

    # Map chain ID to name
    chain_map = {
        "1": "Ethereum",
        "56": "BNB Smart Chain",
        "137": "Polygon",
        "8453": "Base",
        "42161": "Arbitrum",
        "10": "Optimism"
    }
    chain_name = chain_map.get(chain_id, f"Chain {chain_id}")

    try:
        # Get necessary clients
        okx_client = None
        cave_client = None
        third_party_client = None

        if "okx" in agent.connection_manager.connections:
            okx_client = agent.connection_manager.connections["okx"]
        if "cave" in agent.connection_manager.connections:
            cave_client = agent.connection_manager.connections["cave"]
        if "third_party" in agent.connection_manager.connections:
            third_party_client = agent.connection_manager.connections["third_party"]

        # Get hot tokens
        hot_tokens = get_hot_tokens_by_chain(
            chain_id=chain_id,
            limit=limit,
            okx_client=okx_client,
            cave_client=cave_client,
            third_party_client=third_party_client
        )

        if hot_tokens and len(hot_tokens) > 0:
            # Format response
            response = f"Hot Tokens on {chain_name}:\n\n"

            for i, token in enumerate(hot_tokens):
                symbol = token.get("symbol", "Unknown")
                name = token.get("name", symbol)
                address = token.get("address", "")
                volume = token.get("volume_24h", 0)
                price_change = token.get("price_change_24h", 0)

                response += f"{i + 1}. {symbol} ({name})\n"
                if address:
                    response += f"   Address: {address[:8]}...{address[-6:]}\n"
                if volume > 0:
                    response += f"   24h Volume: ${volume:,.2f}\n"
                if price_change != 0:
                    change_sign = "+" if price_change > 0 else ""
                    response += f"   24h Change: {change_sign}{price_change:.2f}%\n"

                # Add security info if available
                if "security" in token:
                    security = token["security"]
                    risk_score = security.get("risk_score", 0)
                    if risk_score > 7:
                        response += f"   ⚠️ HIGH RISK: Security Score {10 - risk_score}/10\n"
                    elif risk_score > 4:
                        response += f"   ⚠️ MEDIUM RISK: Security Score {10 - risk_score}/10\n"

                response += "\n"

            agent.logger.info(f"✅ Successfully retrieved {len(hot_tokens)} hot tokens on {chain_name}")
            return response
        else:
            return f"I couldn't find any trending tokens on {chain_name} at the moment. Please try again later or check another chain."

    except Exception as e:
        agent.logger.error(f"❌ Error getting hot tokens: {str(e)}")
        return f"I encountered an error while fetching hot tokens on {chain_name}: {str(e)}"


@register_action("check-token-liquidity")
def check_token_liquidity(agent, **kwargs):
    """Check liquidity for a token on a specific chain."""
    query = kwargs.get("query", "")
    if not query:
        agent.logger.error("No query provided for liquidity check")
        return "I need to know which token to check liquidity for. Please provide a token symbol and preferably the contract address."

    agent.logger.info(f"\n💧 CHECKING TOKEN LIQUIDITY FOR: {query}")
    print_h_bar()

    # Extract token information
    token_info = extract_token_info(query)
    symbol = token_info["symbol"]
    chain_id = token_info["chain_id"]
    address = token_info["address"]

    if not symbol or not address:
        return "I need both a token symbol and contract address to check liquidity. Please provide both."

    try:
        # Get necessary clients
        cave_client = None
        if "cave" in agent.connection_manager.connections:
            cave_client = agent.connection_manager.connections["cave"]

        # Check liquidity
        liquidity_result = check_liquidity(
            token_symbol=symbol,
            chain_id=chain_id,
            token_address=address,
            cave_client=cave_client
        )

        if liquidity_result:
            # Format response
            response = f"Liquidity Analysis for {symbol} ({address[:8]}...{address[-6:]}):\n\n"

            total_liquidity = liquidity_result.get("total_liquidity_usd", 0)
            response += f"Total Liquidity: ${total_liquidity:,.2f}\n"

            is_sufficient = liquidity_result.get("is_sufficient", False)
            if is_sufficient:
                response += "Liquidity Status: SUFFICIENT ✅\n"
            else:
                response += "Liquidity Status: INSUFFICIENT ⚠️\n"

            liquidity_score = liquidity_result.get("liquidity_score", 0)
            response += f"Liquidity Score: {liquidity_score}/10\n\n"

            # Add sources breakdown
            sources = liquidity_result.get("liquidity_sources", [])
            if sources:
                response += "Liquidity Sources:\n"
                for source in sources:
                    dex = source.get("dex", "Unknown DEX")
                    amount = source.get("liquidity_usd", 0)
                    percentage = source.get("percentage", 0)
                    response += f"- {dex}: ${amount:,.2f} ({percentage:.1f}%)\n"
                response += "\n"

            # Add recommendation based on liquidity
            max_swap = liquidity_result.get("recommended_max_swap", 0)
            response += "Recommendations:\n"

            if total_liquidity < 10000:
                response += "- CAUTION: Very low liquidity. High risk of significant slippage.\n"

            if max_swap > 0:
                response += f"- Maximum recommended swap size: ${max_swap:,.2f}\n"

            if liquidity_score < 3:
                response += "- Not recommended for significant trading due to liquidity constraints.\n"

            agent.logger.info(f"✅ Successfully checked liquidity for {symbol}")
            return response
        else:
            return f"I couldn't analyze liquidity for {symbol}. The token may not have established liquidity pools."

    except Exception as e:
        agent.logger.error(f"❌ Error checking token liquidity: {str(e)}")
        return f"I encountered an error while checking liquidity for {symbol}: {str(e)}"


@register_action("carv-llm-query")
def carv_llm_query(agent, **kwargs):
    """Execute an on-chain data query using natural language."""
    query = kwargs.get("query", "")
    if not query:
        agent.logger.error("No query provided for on-chain data analysis")
        return "I need a question about on-chain data to analyze. Please provide a specific query."

    agent.logger.info(f"\n🔍 EXECUTING ON-CHAIN DATA QUERY: {query}")
    print_h_bar()

    try:
        # Get CARV client
        cave_client = None
        if "cave" in agent.connection_manager.connections:
            cave_client = agent.connection_manager.connections["cave"]

        if not cave_client:
            return "I'm unable to query on-chain data because the necessary service isn't available."

        # Execute query
        result = query_on_chain_data(query, cave_client)

        if result:
            # Format response
            response = "On-Chain Data Analysis Results:\n\n"

            # Process and format the result (implementation would depend on CARV response format)
            # This is a placeholder for the actual implementation
            response += "The on-chain data query returned the following insights:\n\n"
            response += str(result)

            agent.logger.info(f"✅ Successfully executed on-chain data query")
            return response
        else:
            return "I didn't receive any results from the on-chain data query. Please try a different query."

    except Exception as e:
        agent.logger.error(f"❌ Error executing on-chain data query: {str(e)}")
        return f"I encountered an error while querying on-chain data: {str(e)}"


@register_action("suggest-portfolio")
def suggest_crypto_portfolio(agent, **kwargs):
    """Suggest a diversified cryptocurrency portfolio based on user requirements."""
    query = kwargs.get("query", "")
    amount = kwargs.get("amount", 0)
    risk_profile = kwargs.get("risk", "medium")

    agent.logger.info(f"\n💼 GENERATING PORTFOLIO SUGGESTION")
    print_h_bar()

    # Try to extract investment amount from query if not provided
    if not amount:
        amount_match = re.search(r'(\$?[\d,]+(?:\.\d+)?)', query)
        if amount_match:
            try:
                # Convert string like "$1,000" to float 1000.0
                amount_str = amount_match.group(1).replace('$', '').replace(',', '')
                amount = float(amount_str)
            except:
                amount = 1000  # Default amount

    # Use default if still not set
    if not amount:
        amount = 1000

    # Try to extract risk profile from query if not provided
    if risk_profile == "medium":
        if re.search(r'\b(low\s*risk|conservative|safe)\b', query.lower()):
            risk_profile = "low"
        elif re.search(r'\b(high\s*risk|aggressive|risky)\b', query.lower()):
            risk_profile = "high"

    # Extract preferred chain from query
    chain_id = "8453"  # Default to Base
    token_info = extract_token_info(query)
    if token_info.get("chain_id"):
        chain_id = token_info["chain_id"]

    try:
        # Get necessary clients
        okx_client = None
        third_party_client = None

        if "okx" in agent.connection_manager.connections:
            okx_client = agent.connection_manager.connections["okx"]
        if "third_party" in agent.connection_manager.connections:
            third_party_client = agent.connection_manager.connections["third_party"]

        # Get hot tokens to consider for portfolio
        hot_tokens = get_hot_tokens_by_chain(
            chain_id=chain_id,
            limit=10,
            okx_client=okx_client,
            third_party_client=third_party_client
        )

        # Generate portfolio suggestion
        portfolio = suggest_portfolio(
            total_investment=amount,
            risk_profile=risk_profile,
            chain_id=chain_id,
            hot_tokens=hot_tokens,
            okx_client=okx_client,
            third_party_client=third_party_client
        )

        if portfolio:
            # Format response
            chain_map = {
                "1": "Ethereum",
                "56": "BNB Smart Chain",
                "137": "Polygon",
                "8453": "Base",
                "42161": "Arbitrum",
                "10": "Optimism"
            }
            chain_name = chain_map.get(chain_id, f"Chain {chain_id}")

            response = f"Suggested Portfolio for ${amount:,.2f} on {chain_name} ({risk_profile.upper()} risk profile):\n\n"

            # Add allocation breakdown
            allocations = portfolio.get("allocations", {})
            if allocations:
                response += "Allocation Strategy:\n"
                for category, details in allocations.items():
                    response += f"- {category.capitalize()}: ${details.get('amount', 0):,.2f} ({details.get('percentage', 0):.1f}%)\n"
                response += "\n"

            # Add token recommendations
            tokens = portfolio.get("tokens", [])
            if tokens:
                response += "Recommended Assets:\n"
                for token in tokens:
                    symbol = token.get("symbol", "Unknown")
                    name = token.get("name", symbol)
                    allocation = token.get("allocation_percentage", 0)
                    amount_usd = token.get("allocation_amount", 0)
                    quantity = token.get("quantity", 0)
                    category = token.get("category", "")

                    response += f"- {symbol} ({name})\n"
                    response += f"  Category: {category.capitalize()}\n"
                    response += f"  Allocation: ${amount_usd:,.2f} ({allocation:.1f}%)\n"
                    if quantity > 0:
                        response += f"  Quantity: ~{quantity:.6f} tokens\n"
                    response += "\n"

            # Add portfolio management advice based on risk profile
            response += "Portfolio Management Advice:\n"

            if risk_profile == "low":
                response += "- Monitor your portfolio weekly, focusing on preserving capital.\n"
                response += "- Consider DCA (Dollar-Cost Averaging) for entering positions gradually.\n"
                response += "- Set stop-loss orders at 10-15% below purchase price.\n"
            elif risk_profile == "medium":
                response += "- Review your portfolio bi-weekly and consider rebalancing monthly.\n"
                response += "- Set both stop-loss (15-20%) and take-profit targets.\n"
                response += "- Follow market trends and adjust allocations accordingly.\n"
            else:  # high risk
                response += "- Monitor the market daily and be prepared for high volatility.\n"
                response += "- Consider taking profits more aggressively on speculative assets.\n"
                response += "- Maintain disciplined position sizing despite higher risk tolerance.\n"

            response += "- Always do your own research before investing.\n"

            agent.logger.info(f"✅ Successfully generated portfolio suggestion with {len(tokens)} assets")
            return response
        else:
            return f"I wasn't able to generate a portfolio suggestion for ${amount:,.2f} on {chain_id}. Please try again with different parameters."

    except Exception as e:
        agent.logger.error(f"❌ Error generating portfolio suggestion: {str(e)}")
        return f"I encountered an error while creating your portfolio suggestion: {str(e)}"


@register_action("get-wallet-balance")
def get_wallet_balance(agent, **kwargs):
    """Check balance of a wallet address on specified chain."""
    wallet_address = kwargs.get("wallet_address", "")
    chain_id = kwargs.get("chain_id", "8453")  # Default to Base chain

    if not wallet_address:
        agent.logger.error("No wallet address provided for balance check")
        return "I need a wallet address to check the balance. Please provide an address."

    agent.logger.info(f"\n💼 CHECKING WALLET BALANCE FOR: {wallet_address} on chain {chain_id}")
    print_h_bar()

    try:
        # Get OKX client
        okx_client = None
        if "okx" in agent.connection_manager.connections:
            okx_client = agent.connection_manager.connections["okx"]

        if not okx_client:
            return f"I'm unable to check the wallet balance because the necessary service isn't available."

        # Get wallet balances
        balances = agent.connection_manager.perform_action(
            connection_name="okx",
            action_name="get-all-token-balances-by-address",
            params=[wallet_address, [chain_id]]
        )

        if balances:
            # Format response
            response = f"Wallet Balance for {wallet_address[:8]}...{wallet_address[-6:]} on chain {chain_id}:\n\n"

            total_value = 0
            for token in balances:
                symbol = token.get("symbol", "Unknown")
                balance = float(token.get("balance", 0))
                price = float(token.get("tokenPrice", 0))
                value = balance * price
                total_value += value

                response += f"- {symbol}: {balance:.6f} (${value:.2f})\n"

            response += f"\nTotal Value: ${total_value:.2f}"

            agent.logger.info(f"✅ Successfully retrieved wallet balance for {wallet_address}")
            return response
        else:
            return f"I couldn't retrieve the balance for wallet {wallet_address}. The wallet may not exist or have no tokens."

    except Exception as e:
        agent.logger.error(f"❌ Error checking wallet balance: {str(e)}")
        return f"I encountered an error while checking the wallet balance: {str(e)}"


@register_action("add-token-whitelist")
def add_token_whitelist(agent, **kwargs):
    """Add tokens to the multisig wallet whitelist."""
    wallet_address = kwargs.get("wallet_address", "")
    chain_id = kwargs.get("chain_id", 8453)  # Default to Base chain
    token_addresses = kwargs.get("token_addresses", [])

    if not wallet_address:
        agent.logger.error("No wallet address provided for whitelist addition")
        return "I need a wallet address to add tokens to whitelist. Please provide a multisig wallet address."

    if not token_addresses:
        agent.logger.error("No token addresses provided for whitelist addition")
        return "I need token addresses to add to the whitelist. Please provide at least one token address."

    agent.logger.info(f"\n➕ ADDING TOKENS TO WHITELIST FOR: {wallet_address} on chain {chain_id}")
    print_h_bar()

    try:
        # Get wallet service client
        wallet_service_client = None
        if "wallet_service" in agent.connection_manager.connections:
            wallet_service_client = agent.connection_manager.connections["wallet_service"]

        if not wallet_service_client:
            return f"I'm unable to add tokens to the whitelist because the wallet service isn't available."

        # Get AI agent address for signatures
        user = agent.connection_manager.perform_action(
            connection_name="mongodb",
            action_name="get-user",
            params=[wallet_address]
        )

        if not user or not user.get("agent_address"):
            return "I couldn't find the AI agent address associated with this wallet. Please make sure your account is properly set up."

        ai_address = user.get("agent_address")

        # Generate signatures (this would be handled by the wallet service)
        signatures = []  # In a real implementation, this would be proper signatures

        # Add tokens to whitelist
        result = agent.connection_manager.perform_action(
            connection_name="wallet_service",
            action_name="add-multi-sig-wallet-whitelist",
            params=[chain_id, wallet_address, signatures, token_addresses]
        )

        if result and result.get("success"):
            # Format response
            token_addresses_display = [f"{addr[:8]}...{addr[-6:]}" for addr in token_addresses]
            response = f"Successfully added {len(token_addresses)} tokens to the whitelist for wallet {wallet_address[:8]}...{wallet_address[-6:]}.\n\n"
            response += "Added tokens:\n"
            for addr in token_addresses_display:
                response += f"- {addr}\n"

            agent.logger.info(f"✅ Successfully added tokens to whitelist for {wallet_address}")
            return response
        else:
            return f"I couldn't add the tokens to the whitelist. There might be an issue with the multisig wallet or the provided addresses."

    except Exception as e:
        agent.logger.error(f"❌ Error adding tokens to whitelist: {str(e)}")
        return f"I encountered an error while adding tokens to the whitelist: {str(e)}"


@register_action("remove-token-whitelist")
def remove_token_whitelist(agent, **kwargs):
    """Remove tokens from the multisig wallet whitelist."""
    wallet_address = kwargs.get("wallet_address", "")
    chain_id = kwargs.get("chain_id", 8453)  # Default to Base chain
    token_addresses = kwargs.get("token_addresses", [])

    if not wallet_address:
        agent.logger.error("No wallet address provided for whitelist removal")
        return "I need a wallet address to remove tokens from whitelist. Please provide a multisig wallet address."

    if not token_addresses:
        agent.logger.error("No token addresses provided for whitelist removal")
        return "I need token addresses to remove from the whitelist. Please provide at least one token address."

    agent.logger.info(f"\n➖ REMOVING TOKENS FROM WHITELIST FOR: {wallet_address} on chain {chain_id}")
    print_h_bar()

    try:
        # Get wallet service client
        wallet_service_client = None
        if "wallet_service" in agent.connection_manager.connections:
            wallet_service_client = agent.connection_manager.connections["wallet_service"]

        if not wallet_service_client:
            return f"I'm unable to remove tokens from the whitelist because the wallet service isn't available."

        # Get AI agent address for signatures
        user = agent.connection_manager.perform_action(
            connection_name="mongodb",
            action_name="get-user",
            params=[wallet_address]
        )

        if not user or not user.get("agent_address"):
            return "I couldn't find the AI agent address associated with this wallet. Please make sure your account is properly set up."

        ai_address = user.get("agent_address")

        # Generate signatures (this would be handled by the wallet service)
        signatures = []  # In a real implementation, this would be proper signatures

        # Remove tokens from whitelist
        result = agent.connection_manager.perform_action(
            connection_name="wallet_service",
            action_name="remove-multi-sig-wallet-whitelist",
            params=[chain_id, wallet_address, signatures, token_addresses]
        )

        if result and result.get("success"):
            # Format response
            token_addresses_display = [f"{addr[:8]}...{addr[-6:]}" for addr in token_addresses]
            response = f"Successfully removed {len(token_addresses)} tokens from the whitelist for wallet {wallet_address[:8]}...{wallet_address[-6:]}.\n\n"
            response += "Removed tokens:\n"
            for addr in token_addresses_display:
                response += f"- {addr}\n"

            agent.logger.info(f"✅ Successfully removed tokens from whitelist for {wallet_address}")
            return response
        else:
            return f"I couldn't remove the tokens from the whitelist. There might be an issue with the multisig wallet or the provided addresses."

    except Exception as e:
        agent.logger.error(f"❌ Error removing tokens from whitelist: {str(e)}")
        return f"I encountered an error while removing tokens from the whitelist: {str(e)}"


@register_action("execute-token-swap")
def execute_token_swap(agent, **kwargs):
    """Execute a token swap through a multisig wallet."""
    wallet_address = kwargs.get("wallet_address", "")
    chain_id = kwargs.get("chain_id", 8453)  # Default to Base chain
    pair_address = kwargs.get("pair_address", "")
    input_token_address = kwargs.get("input_token_address", "")
    input_token_amount = kwargs.get("input_token_amount", "")
    output_token_address = kwargs.get("output_token_address", "")
    output_token_min_amount = kwargs.get("output_token_min_amount", "")

    # Validation
    if not wallet_address:
        agent.logger.error("No wallet address provided for token swap")
        return "I need a wallet address to execute a token swap. Please provide a multisig wallet address."

    if not pair_address:
        agent.logger.error("No pair address provided for token swap")
        return "I need a liquidity pair address to execute the swap."

    if not input_token_address or not input_token_amount:
        agent.logger.error("Input token details missing for token swap")
        return "I need both the input token address and amount to execute the swap."

    if not output_token_address:
        agent.logger.error("Output token address missing for token swap")
        return "I need the output token address to execute the swap."

    agent.logger.info(f"\n💱 EXECUTING TOKEN SWAP FOR: {wallet_address} on chain {chain_id}")
    print_h_bar()

    try:
        # Get wallet service client
        wallet_service_client = None
        if "wallet_service" in agent.connection_manager.connections:
            wallet_service_client = agent.connection_manager.connections["wallet_service"]

        if not wallet_service_client:
            return f"I'm unable to execute the token swap because the wallet service isn't available."

        # Get AI agent address
        user = agent.connection_manager.perform_action(
            connection_name="mongodb",
            action_name="get-user",
            params=[wallet_address]
        )

        if not user or not user.get("agent_address"):
            return "I couldn't find the AI agent address associated with this wallet. Please make sure your account is properly set up."

        ai_address = user.get("agent_address")

        # Set default slippage if min amount not provided
        if not output_token_min_amount:
            # Get the output token price
            output_token_price = agent.connection_manager.perform_action(
                connection_name="okx",
                action_name="get-token-price",
                params=[chain_id, output_token_address]
            )

            # Calculate expected output amount (simplified estimation)
            expected_output = "0"  # In a real implementation, this would be calculated based on prices

            # Apply 2% slippage
            output_token_min_amount = expected_output  # In a real implementation, this would be 98% of expected

        # Execute the swap
        swap_result = agent.connection_manager.perform_action(
            connection_name="wallet_service",
            action_name="multi-sig-wallet-swap",
            params=[
                chain_id,
                ai_address,
                wallet_address,
                pair_address,
                input_token_address,
                input_token_amount,
                output_token_address,
                output_token_min_amount
            ]
        )

        if swap_result and swap_result.get("success"):
            # Format response
            tx_hash = swap_result.get("txHash", "Unknown")

            response = f"✅ Token swap executed successfully!\n\n"
            response += f"Transaction Hash: {tx_hash}\n"
            response += f"Swapped {input_token_amount} of token {input_token_address[:8]}...{input_token_address[-6:]}\n"
            response += f"For token {output_token_address[:8]}...{output_token_address[-6:]}\n"
            response += f"On chain {chain_id}\n\n"
            response += f"You can track this transaction on the blockchain explorer."

            agent.logger.info(f"✅ Successfully executed token swap for {wallet_address}")
            return response
        else:
            return f"I couldn't execute the token swap. There might be an issue with the multisig wallet, insufficient funds, or the tokens may not be whitelisted."

    except Exception as e:
        agent.logger.error(f"❌ Error executing token swap: {str(e)}")
        return f"I encountered an error while executing the token swap: {str(e)}"