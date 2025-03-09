import logging
from typing import Dict, List, Any
from datetime import datetime

from src.connections.base_connection import BaseConnection, Action, ActionParameter
from src.actions.crypto_actions import (
    check_token_price,
    check_token_security,
    analyze_crypto_portfolio,
    get_hot_tokens,
    check_token_liquidity,
    carv_llm_query,
    suggest_crypto_portfolio,
    get_wallet_balance,
    add_token_whitelist,
    remove_token_whitelist,
    execute_token_swap
)

logger = logging.getLogger("connections.crypto_tools_connection")


class CryptoToolsConnectionError(Exception):
    """Base exception for CryptoTools connection errors"""
    pass


class CryptoToolsConnection(BaseConnection):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.agent = None

    @property
    def is_llm_provider(self) -> bool:
        return False  # This is not a language model provider

    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate CryptoTools configuration"""
        # Currently, there is no specific configuration to validate
        return config

    def set_agent(self, agent):
        """Set the agent instance for tool execution"""
        self.agent = agent

    def register_actions(self) -> None:
        """Register available CryptoTools operations"""
        self.actions = {
            # Price and analysis related operations
            "check-token-price": Action(
                name="check-token-price",
                parameters=[
                    ActionParameter("query", True, str, "Token query text")
                ],
                description="Check the current price of a cryptocurrency token"
            ),
            "check-token-security": Action(
                name="check-token-security",
                parameters=[
                    ActionParameter("query", True, str, "Token security query")
                ],
                description="Analyze token security data and potential risks"
            ),
            "analyze-portfolio": Action(
                name="analyze-portfolio",
                parameters=[
                    ActionParameter("query", True, str, "Portfolio query text"),
                    ActionParameter("portfolio", False, list, "Portfolio data (optional)")
                ],
                description="Analyze the performance and risk of a cryptocurrency portfolio"
            ),

            # Market and liquidity related operations
            "get-hot-tokens": Action(
                name="get-hot-tokens",
                parameters=[
                    ActionParameter("query", True, str, "Hot tokens query text"),
                    ActionParameter("chain_id", False, str, "Chain ID"),
                    ActionParameter("limit", False, int, "Number of tokens to return")
                ],
                description="Get hot/trending tokens on a specific blockchain"
            ),
            "check-token-liquidity": Action(
                name="check-token-liquidity",
                parameters=[
                    ActionParameter("query", True, str, "Token liquidity query")
                ],
                description="Check the liquidity of a token on a specific chain"
            ),

            # On-chain data query
            "carv-llm-query": Action(
                name="carv-llm-query",
                parameters=[
                    ActionParameter("query", True, str, "On-chain data query text")
                ],
                description="Perform on-chain data queries using natural language"
            ),

            # Portfolio suggestion
            "suggest-portfolio": Action(
                name="suggest-portfolio",
                parameters=[
                    ActionParameter("query", True, str, "Portfolio suggestion query"),
                    ActionParameter("amount", False, float, "Investment amount (USD)"),
                    ActionParameter("risk", False, str, "Risk preference")
                ],
                description="Suggest a diversified cryptocurrency portfolio"
            ),

            # Wallet and transaction operations
            "get-wallet-balance": Action(
                name="get-wallet-balance",
                parameters=[
                    ActionParameter("wallet_address", True, str, "Wallet address to check"),
                    ActionParameter("chain_id", False, str, "Chain ID")
                ],
                description="Check the balance of a wallet address"
            ),
            "add-token-whitelist": Action(
                name="add-token-whitelist",
                parameters=[
                    ActionParameter("wallet_address", True, str, "Multisig wallet address"),
                    ActionParameter("chain_id", False, int, "Chain ID"),
                    ActionParameter("token_addresses", True, list, "List of token addresses")
                ],
                description="Add tokens to the multisig wallet whitelist"
            ),
            "remove-token-whitelist": Action(
                name="remove-token-whitelist",
                parameters=[
                    ActionParameter("wallet_address", True, str, "Multisig wallet address"),
                    ActionParameter("chain_id", False, int, "Chain ID"),
                    ActionParameter("token_addresses", True, list, "List of token addresses")
                ],
                description="Remove tokens from the multisig wallet whitelist"
            ),
            "execute-token-swap": Action(
                name="execute-token-swap",
                parameters=[
                    ActionParameter("wallet_address", True, str, "Multisig wallet address"),
                    ActionParameter("chain_id", False, int, "Chain ID"),
                    ActionParameter("pair_address", True, str, "Trading pair address"),
                    ActionParameter("input_token_address", True, str, "Input token address"),
                    ActionParameter("input_token_amount", True, str, "Input token amount"),
                    ActionParameter("output_token_address", True, str, "Output token address"),
                    ActionParameter("output_token_min_amount", False, str, "Minimum output token amount")
                ],
                description="Execute a token swap through a multisig wallet"
            )
        }

    # Method implementations for various operations
    def check_token_price(self, query: str, **kwargs) -> str:
        """Check the current price of a cryptocurrency token"""
        return check_token_price(self.agent, query=query, **kwargs)

    def check_token_security(self, query: str, **kwargs) -> str:
        """Analyze token security data and potential risks"""
        return check_token_security(self.agent, query=query, **kwargs)

    def analyze_portfolio(self, query: str, portfolio: List = None, **kwargs) -> str:
        """Analyze the performance and risk of a cryptocurrency portfolio"""
        return analyze_crypto_portfolio(self.agent, query=query, portfolio=portfolio, **kwargs)

    def get_hot_tokens(self, query: str, chain_id: str = None, limit: int = None, **kwargs) -> str:
        """Get hot/trending tokens on a specific blockchain"""
        return get_hot_tokens(self.agent, query=query, chain_id=chain_id, limit=limit, **kwargs)

    def check_token_liquidity(self, query: str, **kwargs) -> str:
        """Check the liquidity of a token on a specific chain"""
        return check_token_liquidity(self.agent, query=query, **kwargs)

    def carv_llm_query(self, query: str, **kwargs) -> str:
        """Perform on-chain data queries using natural language"""
        return carv_llm_query(self.agent, query=query, **kwargs)

    def suggest_portfolio(self, query: str, amount: float = None, risk: str = None, **kwargs) -> str:
        """Suggest a diversified cryptocurrency portfolio"""
        return suggest_crypto_portfolio(self.agent, query=query, amount=amount, risk=risk, **kwargs)

    def get_wallet_balance(self, wallet_address: str, chain_id: str = None, **kwargs) -> str:
        """Check the balance of a wallet address"""
        return get_wallet_balance(self.agent, wallet_address=wallet_address, chain_id=chain_id, **kwargs)

    def add_token_whitelist(self, wallet_address: str, token_addresses: List[str], chain_id: int = 8453,
                            **kwargs) -> str:
        """Add tokens to the multisig wallet whitelist"""
        return add_token_whitelist(self.agent, wallet_address=wallet_address, chain_id=chain_id,
                                   token_addresses=token_addresses, **kwargs)

    def remove_token_whitelist(self, wallet_address: str, token_addresses: List[str], chain_id: int = 8453,
                               **kwargs) -> str:
        """Remove tokens from the multisig wallet whitelist"""
        return remove_token_whitelist(self.agent, wallet_address=wallet_address, chain_id=chain_id,
                                      token_addresses=token_addresses, **kwargs)

    def execute_token_swap(self, wallet_address: str, pair_address: str, input_token_address: str,
                           input_token_amount: str, output_token_address: str, output_token_min_amount: str = None,
                           chain_id: int = 8453, **kwargs) -> str:
        """Execute a token swap through a multisig wallet"""
        return execute_token_swap(self.agent, wallet_address=wallet_address, pair_address=pair_address,
                                  input_token_address=input_token_address, input_token_amount=input_token_amount,
                                  output_token_address=output_token_address,
                                  output_token_min_amount=output_token_min_amount, chain_id=chain_id, **kwargs)

    def configure(self) -> bool:
        """crypto tools connection that does not require special configuration"""
        logger.info("\n✅ CryptoTools connection is configured by default")
        return True

    def is_configured(self, verbose=False) -> bool:
        """The CryptoTools connection is always configured"""
        return True

    def perform_action(self, action_name: str, kwargs) -> Any:
        """Perform CryptoTools operations with validation"""
        if action_name not in self.actions:
            raise KeyError(f"Unknown action: {action_name}")

        action = self.actions[action_name]
        errors = action.validate_params(kwargs)
        if errors:
            raise ValueError(f"Invalid parameters: {', '.join(errors)}")

        # If an agent instance is provided, store it
        if "agent" in kwargs:
            self.agent = kwargs.pop("agent")

        # If not yet set, get the agent instance
        if self.agent is None:
            # Try to get it from the global context or create a new one
            try:
                # Try to get the agent in the current context
                from src.agent import ZerePyAgent
                from src.cli import ZerePyCLI
                cli = ZerePyCLI()
                self.agent = cli.agent
            except:
                logger.error("Agent instance not found")
                return "Error: No agent instance available to execute crypto tools"

        # Call the appropriate method based on the operation name
        method_name = action_name.replace('-', '_')
        method = getattr(self, method_name)
        return method(**kwargs)