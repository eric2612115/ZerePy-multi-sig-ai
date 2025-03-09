import logging
import os
import json
import requests
from typing import Dict, List, Any
from datetime import datetime
from dotenv import load_dotenv, set_key
from anthropic import Anthropic

from src.connections.base_connection import BaseConnection, Action, ActionParameter
from src.actions.crypto_actions import (
    check_token_price,
    check_token_security,
    analyze_crypto_portfolio,
    get_hot_tokens,
    check_token_liquidity,
    suggest_crypto_portfolio,
    get_wallet_balance
)

logger = logging.getLogger("connections.timetool_connection")


class TimeToolConnectionError(Exception):
    """Base exception for TimeTool connection errors"""
    pass


class TimeToolConfigurationError(TimeToolConnectionError):
    """Raised when there are configuration/credential issues"""
    pass


class TimeToolAPIError(TimeToolConnectionError):
    """Raised when TimeTool API requests fail"""
    pass


class TimeToolConnection(BaseConnection):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._client = None
        self.agent = None  # Will be set later when handling tool requests

        # Define time tools
        self.time_tools = [
            # Basic time tools
            {
                "name": "get_time",
                "description": "Get the current time",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "get_date",
                "description": "Get the current date",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "get_datetime",
                "description": "Get the current date and time",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "get_symbol_price",
                "description": "Get the current price of a symbol",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Symbol to get the price of"
                        }
                    },
                    "required": ["symbol"]
                }
            },

            # Crypto tools from crypto_actions.py
            {
                "name": "check_token_price",
                "description": "Check the current price of a cryptocurrency token",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The token query text (e.g., 'Check ETH price')"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "check_token_security",
                "description": "Analyze a token's security profile and potential risks",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The token query with symbol and address"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "analyze_portfolio",
                "description": "Analyze a cryptocurrency portfolio for performance, risk, and recommendations",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The portfolio description with token amounts"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_hot_tokens",
                "description": "Get trending/hot tokens on a specific blockchain",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Query describing which chain to look at"
                        },
                        "chain_id": {
                            "type": "string",
                            "description": "Optional specific chain ID (e.g., '8453' for Base)"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of tokens to return"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "check_token_liquidity",
                "description": "Check liquidity for a token on a specific chain",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Query with token symbol and address"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "suggest_portfolio",
                "description": "Suggest a diversified cryptocurrency portfolio based on user requirements",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Query with investment amount and risk preference"
                        },
                        "amount": {
                            "type": "number",
                            "description": "Investment amount in USD"
                        },
                        "risk": {
                            "type": "string",
                            "description": "Risk profile: 'low', 'medium', or 'high'"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_wallet_balance",
                "description": "Check balance of a wallet address on specified chain",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "wallet_address": {
                            "type": "string",
                            "description": "Wallet address to check"
                        },
                        "chain_id": {
                            "type": "string",
                            "description": "Chain ID to query"
                        }
                    },
                    "required": ["wallet_address"]
                }
            }
        ]

    @property
    def is_llm_provider(self) -> bool:
        return True  # This is a language model provider since it uses Claude

    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate TimeTool configuration from JSON"""
        required_fields = ["model"]
        missing_fields = [field for field in required_fields if field not in config]

        if missing_fields:
            raise ValueError(f"Missing required configuration fields: {', '.join(missing_fields)}")

        if not isinstance(config["model"], str):
            raise ValueError("model must be a string")

        return config

    def register_actions(self) -> None:
        """Register available TimeTool actions"""
        self.actions = {
            "generate-text": Action(
                name="generate-text",
                parameters=[
                    ActionParameter("prompt", True, str, "The input prompt for text generation"),
                    ActionParameter("system_prompt", True, str, "System prompt to guide the model"),
                    ActionParameter("model", False, str, "Model to use for generation")
                ],
                description="Generate text using Claude with TimeTool capabilities"
            ),
            "get-time": Action(
                name="get-time",
                parameters=[],
                description="Get the current time"
            ),
            "get-date": Action(
                name="get-date",
                parameters=[],
                description="Get the current date"
            ),
            "get-datetime": Action(
                name="get-datetime",
                parameters=[],
                description="Get the current date and time"
            ),
            "get-symbol-price": Action(
                name="get-symbol-price",
                parameters=[
                    ActionParameter("symbol", True, str, "Symbol to get the price of")
                ],
                description="Get the current price of a symbol"
            ),

            # Add crypto actions
            "check-token-price": Action(
                name="check-token-price",
                parameters=[
                    ActionParameter("query", True, str, "The token query text")
                ],
                description="Check the current price of a cryptocurrency token"
            ),
            "check-token-security": Action(
                name="check-token-security",
                parameters=[
                    ActionParameter("query", True, str, "The token security query")
                ],
                description="Analyze a token's security profile and potential risks"
            ),
            "analyze-portfolio": Action(
                name="analyze-portfolio",
                parameters=[
                    ActionParameter("query", True, str, "The portfolio query text"),
                    ActionParameter("portfolio", False, list, "Portfolio data (optional)")
                ],
                description="Analyze a cryptocurrency portfolio for performance and risk"
            ),
            "get-hot-tokens": Action(
                name="get-hot-tokens",
                parameters=[
                    ActionParameter("query", True, str, "The hot tokens query text"),
                    ActionParameter("chain_id", False, str, "Chain ID"),
                    ActionParameter("limit", False, int, "Number of tokens to return")
                ],
                description="Get trending/hot tokens on a specific blockchain"
            ),
            "check-token-liquidity": Action(
                name="check-token-liquidity",
                parameters=[
                    ActionParameter("query", True, str, "The token liquidity query")
                ],
                description="Check liquidity for a token on a specific chain"
            ),
            "suggest-portfolio": Action(
                name="suggest-portfolio",
                parameters=[
                    ActionParameter("query", True, str, "The portfolio suggestion query"),
                    ActionParameter("amount", False, float, "Investment amount in USD"),
                    ActionParameter("risk", False, str, "Risk profile")
                ],
                description="Suggest a diversified cryptocurrency portfolio"
            ),
            "get-wallet-balance": Action(
                name="get-wallet-balance",
                parameters=[
                    ActionParameter("wallet_address", True, str, "Wallet address to check"),
                    ActionParameter("chain_id", False, str, "Chain ID")
                ],
                description="Check balance of a wallet address"
            )
        }

    def _get_client(self) -> Anthropic:
        """Get or create Anthropic client"""
        if not self._client:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise TimeToolConfigurationError("Anthropic API key not found in environment")
            self._client = Anthropic(api_key=api_key)
        return self._client

    def set_agent(self, agent):
        """Set the agent instance for tool execution"""
        self.agent = agent

    def _handle_tool_use(self, message, original_query, system_prompt):
        """Handle tool use requests and return results"""
        logger.debug("Checking for tool use...")

        # Check if there are any tool uses in the message content
        has_tool_use = False
        tool_uses = []

        for content in message.content:
            if content.type == 'tool_use':
                has_tool_use = True
                tool_uses.append(content)
                logger.debug(f"Tool use found: {content.name} (ID: {content.id})")

        if not has_tool_use:
            logger.debug("No tool use requests found")

            # Extract and return text content
            text_content = ""
            for content in message.content:
                if content.type == 'text':
                    text_content += content.text
            return text_content

        # Process tool uses
        logger.debug(f"Found {len(tool_uses)} tool use requests")
        tool_result_contents = []

        for tool_use in tool_uses:
            tool_id = tool_use.id
            tool_name = tool_use.name

            logger.debug(f"Processing tool: {tool_name}, ID: {tool_id}")

            try:
                # Basic time and price tools
                if tool_name == "get_time":
                    result = self.get_time()
                elif tool_name == "get_date":
                    result = self.get_date()
                elif tool_name == "get_datetime":
                    result = self.get_datetime()
                elif tool_name == "get_symbol_price":
                    symbol = tool_use.input.get("symbol")
                    result = self.get_symbol_price(symbol)

                # Crypto tools - requires agent to be set
                elif tool_name == "check_token_price":
                    if not self.agent:
                        result = "Error: Agent not configured for crypto tools"
                    else:
                        query = tool_use.input.get("query", "")
                        result = check_token_price(self.agent, query=query)

                elif tool_name == "check_token_security":
                    if not self.agent:
                        result = "Error: Agent not configured for crypto tools"
                    else:
                        query = tool_use.input.get("query", "")
                        result = check_token_security(self.agent, query=query)

                elif tool_name == "analyze_portfolio":
                    if not self.agent:
                        result = "Error: Agent not configured for crypto tools"
                    else:
                        query = tool_use.input.get("query", "")
                        result = analyze_crypto_portfolio(self.agent, query=query)

                elif tool_name == "get_hot_tokens":
                    if not self.agent:
                        result = "Error: Agent not configured for crypto tools"
                    else:
                        query = tool_use.input.get("query", "")
                        chain_id = tool_use.input.get("chain_id", "8453")  # Default to Base
                        limit = tool_use.input.get("limit", 5)
                        result = get_hot_tokens(self.agent, query=query, chain_id=chain_id, limit=limit)

                elif tool_name == "check_token_liquidity":
                    if not self.agent:
                        result = "Error: Agent not configured for crypto tools"
                    else:
                        query = tool_use.input.get("query", "")
                        result = check_token_liquidity(self.agent, query=query)

                elif tool_name == "suggest_portfolio":
                    if not self.agent:
                        result = "Error: Agent not configured for crypto tools"
                    else:
                        query = tool_use.input.get("query", "")
                        amount = tool_use.input.get("amount", 0)
                        risk = tool_use.input.get("risk", "medium")
                        result = suggest_crypto_portfolio(self.agent, query=query, amount=amount, risk=risk)

                elif tool_name == "get_wallet_balance":
                    if not self.agent:
                        result = "Error: Agent not configured for crypto tools"
                    else:
                        wallet_address = tool_use.input.get("wallet_address", "")
                        chain_id = tool_use.input.get("chain_id", "8453")
                        result = get_wallet_balance(self.agent, wallet_address=wallet_address, chain_id=chain_id)

                else:
                    result = f"Unknown tool: {tool_name}"

                logger.debug(f"Tool {tool_name} result: {result}")

                # Create individual tool_result content for each result
                tool_result_contents.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": result
                })

            except Exception as e:
                logger.error(f"Tool execution error: {e}")
                tool_result_contents.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": f"Error: {str(e)}"
                })

        # Send tool results back to Claude
        try:
            logger.debug(f"Sending tool results back to Claude: {tool_result_contents}")

            final_response = self._get_client().messages.create(
                model=self.config["model"],
                system=system_prompt,
                messages=[
                    {"role": "user", "content": original_query},
                    {"role": "assistant", "content": message.content},
                    {"role": "user", "content": tool_result_contents}
                ],
                max_tokens=1000
            )

            # Extract response text
            response_text = ""
            for content in final_response.content:
                if content.type == 'text':
                    response_text += content.text

            return response_text

        except Exception as e:
            logger.error(f"Error getting final response: {e}")
            return f"I got the information but couldn't generate a proper response due to an error: {str(e)}"

    def generate_text(self, prompt: str, system_prompt: str, model: str = None, **kwargs) -> str:
        """Generate text using Claude with TimeTool capabilities"""
        try:
            client = self._get_client()

            # Use configured model if none provided
            if not model:
                model = self.config["model"]

            # Send initial request with tool definitions
            response = client.messages.create(
                model=model,
                system=system_prompt,
                tools=self.time_tools,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )

            # Handle potential tool use
            final_response = self._handle_tool_use(response, prompt, system_prompt)
            return final_response

        except Exception as e:
            raise TimeToolAPIError(f"Text generation failed: {e}")

    def get_time(self) -> str:
        """Get the current time"""
        try:
            return datetime.now().strftime("%H:%M:%S")
        except Exception as e:
            raise TimeToolAPIError(f"Failed to get time: {e}")

    def get_date(self) -> str:
        """Get the current date"""
        try:
            return datetime.now().strftime("%Y-%m-%d")
        except Exception as e:
            raise TimeToolAPIError(f"Failed to get date: {e}")

    def get_datetime(self) -> str:
        """Get the current date and time"""
        try:
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            raise TimeToolAPIError(f"Failed to get datetime: {e}")

    def get_symbol_price(self, symbol: str) -> str:
        """Get the current price of a symbol"""
        try:
            url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
            response = requests.get(url)
            data = response.json()
            return data["price"]
        except Exception as e:
            raise TimeToolAPIError(f"Failed to get symbol price: {e}")

    def configure(self) -> bool:
        """Sets up TimeTool API authentication"""
        logger.info("\n🔧 TIMETOOL API SETUP")

        if self.is_configured():
            logger.info("\nTimeTool API is already configured.")
            response = input("Do you want to reconfigure? (y/n): ")
            if response.lower() != 'y':
                return True

        logger.info("\n📝 To use TimeTool, you need an Anthropic API key:")
        logger.info("1. Go to https://console.anthropic.com/settings/keys")
        logger.info("2. Create a new API key.")

        api_key = input("\nEnter your Anthropic API key: ")

        try:
            if not os.path.exists('.env'):
                with open('.env', 'w') as f:
                    f.write('')

            set_key('.env', 'ANTHROPIC_API_KEY', api_key)

            # Validate the API key
            client = Anthropic(api_key=api_key)
            client.models.list()

            logger.info("\n✅ TimeTool API configuration successfully saved!")
            logger.info("Your API key has been stored in the .env file.")
            return True

        except Exception as e:
            logger.error(f"Configuration failed: {e}")
            return False

    def is_configured(self, verbose=False) -> bool:
        """Check if TimeTool API key is configured and valid"""
        try:
            load_dotenv()
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                return False

            client = Anthropic(api_key=api_key)
            client.models.list()
            return True

        except Exception as e:
            if verbose:
                logger.debug(f"Configuration check failed: {e}")
            return False

    def perform_action(self, action_name: str, kwargs) -> Any:
        """Execute a TimeTool action with validation"""
        if action_name not in self.actions:
            raise KeyError(f"Unknown action: {action_name}")

        action = self.actions[action_name]
        errors = action.validate_params(kwargs)
        if errors:
            raise ValueError(f"Invalid parameters: {', '.join(errors)}")

        # For crypto tools actions, we need to set the agent
        if action_name in ["check-token-price", "check-token-security", "analyze-portfolio",
                           "get-hot-tokens", "check-token-liquidity", "suggest-portfolio",
                           "get-wallet-balance"] and self.agent is None:
            # If agent is passed in kwargs, use it
            if "agent" in kwargs:
                self.agent = kwargs.pop("agent")
            else:
                from src.agent import ZerePyAgent
                # Try to get agent from global or create one
                try:
                    from src.cli import ZerePyCLI
                    cli = ZerePyCLI()
                    self.agent = cli.agent
                except:
                    pass

        # Call the appropriate method based on action name
        method_name = action_name.replace('-', '_')
        method = getattr(self, method_name)
        return method(**kwargs)