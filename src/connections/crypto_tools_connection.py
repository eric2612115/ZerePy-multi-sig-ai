import json
import logging
import os
from typing import Dict, List, Any
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from anthropic import Anthropic

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
        load_dotenv(find_dotenv())
        self._client = None
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

    def _handle_tool_use(self, message, original_query, system_prompt):
        """Handle tool use requests and return results"""
        logger.debug("Checking for tool use...")

        print(f"Message: {message}")
        print(f"Original Query: {original_query}")
        print(f"System Prompt: {system_prompt}")


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
                if tool_name == "think-and-pick-tool":
                    # Get tool name from the query and execute
                    tool_query = original_query
                    tool_name = tool_query.split(" ")[-1]
                    result = self.think_and_pick_tool(tool_query, tool_name)

                elif tool_name == "check-token-price":
                    result = self.check_token_price(original_query)

                elif tool_name == "check-token-security":
                    result = self.check_token_security(original_query)

                elif tool_name == "analyze-portfolio":
                    result = self.analyze_portfolio(original_query)


                elif tool_name == "get-hot-tokens":
                    result = self.get_hot_tokens(original_query)

                elif tool_name == "check-token-liquidity":
                    result = self.check_token_liquidity(original_query)

                elif tool_name == "carv-llm-query":
                    result = self.carv_llm_query(original_query)

                elif tool_name == "suggest-portfolio":
                    result = self.suggest_portfolio(original_query)

                elif tool_name == "get-wallet-balance":
                    result = self.get_wallet_balance(original_query)

                elif tool_name == "add-token-whitelist":

                    # Extract wallet address and token addresses from the query
                    wallet_address = original_query.split(" ")[-1]
                    token_addresses = original_query.split(" ")[1:-1]
                    result = self.add_token_whitelist(wallet_address, token_addresses)

                elif tool_name == "remove-token-whitelist":

                    # Extract wallet address and token addresses from the query
                    wallet_address = original_query.split(" ")[-1]
                    token_addresses = original_query.split(" ")[1:-1]
                    result = self.remove_token_whitelist(wallet_address, token_addresses)

                elif tool_name == "execute-token-swap":
                    result = self.execute_token_swap(original_query)


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

    def _get_tools_list(self):
        return {
            # Price and analysis related operations
            "check-token-price": {
                "name": "check-token-price",
                "parameters": [
                    {"name": "query", "required": True, "type": "string", "description": "Token query text"}
                ],
                "description": "Check the current price of a cryptocurrency token"
            },
            "check-token-security": {
                "name": "check-token-security",
                "parameters": [
                    {"name": "query", "required": True, "type": "string", "description": "Token security query"}
                ],
                "description": "Analyze token security data and potential risks"
            },
            "analyze-portfolio": {
                "name": "analyze-portfolio",
                "parameters": [
                    {"name": "query", "required": True, "type": "string", "description": "Portfolio query text"},
                    {"name": "portfolio", "required": False, "type": "list", "description": "Portfolio data (optional)"}
                ],
                "description": "Analyze the performance and risk of a cryptocurrency portfolio"
            },

            # Market and liquidity related operations
            "get-hot-tokens": {
                "name": "get-hot-tokens",
                "parameters": [
                    {"name": "query", "required": True, "type": "string", "description": "Hot tokens query text"},
                    {"name": "chain_id", "required": False, "type": "string", "description": "Chain ID"},
                    {"name": "limit", "required": False, "type": "integer", "description": "Number of tokens to return"}
                ],
                "description": "Get hot/trending tokens on a specific blockchain"
            },
            "check-token-liquidity": {
                "name": "check-token-liquidity",
                "parameters": [
                    {"name": "query", "required": True, "type": "string", "description": "Token liquidity query"}
                ],
                "description": "Check the liquidity of a token on a specific chain"
            },

            # On-chain data query
            "carv-llm-query": {
                "name": "carv-llm-query",
                "parameters": [
                    {"name": "query", "required": True, "type": "string", "description": "On-chain data query text"}
                ],
                "description": "Perform on-chain data queries using natural language"
            },

            # Portfolio suggestion
            "suggest-portfolio": {
                "name": "suggest-portfolio",
                "parameters": [
                    {"name": "query", "required": True, "type": "string", "description": "Portfolio suggestion query"},
                    {"name": "amount", "required": False, "type": "float", "description": "Investment amount (USD)"},
                    {"name": "risk", "required": False, "type": "string", "description": "Risk preference"}
                ],
                "description": "Suggest a diversified cryptocurrency portfolio"
            },

            # Wallet and transaction operations
            "get-wallet-balance": {
                "name": "get-wallet-balance",
                "parameters": [
                    {"name": "wallet_address", "required": True, "type": "string",
                     "description": "Wallet address to check"},
                    {"name": "chain_id", "required": False, "type": "string", "description": "Chain ID"}
                ],
                "description": "Check the balance of a wallet address"
            },
            "add-token-whitelist": {
                "name": "add-token-whitelist",
                "parameters": [
                    {"name": "wallet_address", "required": True, "type": "string",
                     "description": "Multisig wallet address"},
                    {"name": "chain_id", "required": False, "type": "integer", "description": "Chain ID"},
                    {"name": "token_addresses", "required": True, "type": "list",
                     "description": "List of token addresses"}
                ],
                "description": "Add tokens to the multisig wallet whitelist"
            },
            "remove-token-whitelist": {
                "name": "remove-token-whitelist",
                "parameters": [
                    {"name": "wallet_address", "required": True, "type": "string",
                     "description": "Multisig wallet address"},
                    {"name": "chain_id", "required": False, "type": "integer", "description": "Chain ID"},
                    {"name": "token_addresses", "required": True, "type": "list",
                     "description": "List of token addresses"}
                ],
                "description": "Remove tokens from the multisig wallet whitelist"
            },
            "execute-token-swap": {
                "name": "execute-token-swap",
                "parameters": [
                    {"name": "wallet_address", "required": True, "type": "string",
                     "description": "Multisig wallet address"},
                    {"name": "chain_id", "required": False, "type": "integer", "description": "Chain ID"},
                    {"name": "pair_address", "required": True, "type": "string", "description": "Trading pair address"},
                    {"name": "input_token_address", "required": True, "type": "string",
                     "description": "Input token address"},
                    {"name": "input_token_amount", "required": True, "type": "string",
                     "description": "Input token amount"},
                    {"name": "output_token_address", "required": True, "type": "string",
                     "description": "Output token address"},
                    {"name": "output_token_min_amount", "required": False, "type": "string",
                     "description": "Minimum output token amount"}
                ],
                "description": "Execute a token swap through a multisig wallet"
            },
            "think-and-pick-tool": {
                "name": "think_and_pick_tool",
                "parameters": [
                    {"name": "query", "required": True, "type": "string", "description": "Query text"},
                    {"name": "tool", "required": True, "type": "string", "description": "Tool name"}
                ],
                "description": "Select a tool based on the query and execute it"
            }
        }

    def _get_client(self) -> Anthropic:
        """Get or create Anthropic client"""
        if not self._client:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise CryptoToolsConnectionError("Anthropic API key not found in environment")
            self._client = Anthropic(api_key=api_key)
        return self._client

    def function_call(self, prompt: str, system_prompt: str, model: str = None, **kwargs) -> str:
        """
        Selects the appropriate tool and its parameters based on the user's query.

        Args:
            user_query: The user's natural language query.
            action_registry: A dictionary containing the available actions (tools).  This
                             should be the dictionary created from the prompt in the
                             previous responses (the JSON-formatted tool list).

        Returns:
            A dictionary representing the selected tool and its parameters, or
            {"tool_name": None, "tool_input": None} if no tool is needed.  The
            dictionary has the format:
            {
                "tool_name": "name_of_selected_tool",
                "tool_input": {
                    "param1_name": "param1_value",
                    "param2_name": "param2_value",
                    ...
                }
            }
        """
        action_registry = {"tools": self._get_tools_list()}
        tools_json = json.dumps(
            self.create_action_registry(action_registry["tools"]))  # Convert the action registry to JSON
        tools_prompt = f"""Based on the user's query, determine the most appropriate tool to use and the values for its parameters.

        Available tools (in JSON format):
        {tools_json}

        The user's query is: {prompt}

        If a tool is needed, respond with a JSON object in the following format:
        ```json
        {{
          "tool_name": "Selected Tool Name",
          "tool_input": {{
            "Parameter1Name": "Parameter1Value",
            "Parameter2Name": "Parameter2Value",
            ...
          }}
        }}
        ```

        If no tool is needed, respond with:
        ```json
        {{
          "tool_name": null,
          "tool_input": null
        }}
        ```
        Only respond with the JSON, do not add other texts.
        """

        system_prompt = "You are an AI assistant that analyzes user requests and determines which tools to use, responding only in JSON."

        try:
            client = self._get_client()

            # Use configured model if none provided
            if not model:
                model = self.config["model"]

            # Send initial request with tool definitions
            tools_prompt = self._get_tools_list()
            response = client.messages.create(
                model=model,
                system=system_prompt,
                tools=tools_prompt,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )

            # Handle potential tool use
            final_response = self._handle_tool_use(response, prompt, system_prompt)
            return final_response

        except Exception as e:
            raise CryptoToolsConnectionError(f"Text generation failed: {e}")

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
            ),
            "think-and-pick-tool": Action(
                name="think_and_pick_tool",
                parameters=[
                    ActionParameter("query", True, str, "Query text"),
                    ActionParameter("tool", True, str, "Tool name")
                ],
                description="Select a tool based on the query and execute it"
            )
        }

    # Method implementations for various operations
    def think_and_pick_tool(self, query: str, tool: str, **kwargs) -> str:
        """Select a tool based on the query and execute it"""
        if tool not in self.actions:
            raise ValueError(f"Unknown tool: {tool}")

        action = self.actions[tool]
        return self.perform_action(action.name, {"query": query, **kwargs})

    def check_token_price(self, query: str, **kwargs) -> str:
        print(f"Checking token price for query: {query}")
        """Check the current price of a cryptocurrency token"""
        return check_token_price(self.agent, query=query, **kwargs)

    def check_token_security(self, query: str, **kwargs) -> str:
        """Analyze token security data and potential risks"""
        return check_token_security(self.agent, query=query, **kwargs)

    def analyze_portfolio(self, query: str, portfolio: List = None, **kwargs) -> str:
        """Analyze the performance and risk of a cryptocurrency portfolio"""
        return analyze_crypto_portfolio(self.agent, query=query, portfolio=portfolio, **kwargs)

    def get_hot_tokens(self, query: str, **kwargs) -> str:
        """Get hot/trending tokens on a specific blockchain"""
        return get_hot_tokens(self.agent, query=query, **kwargs)

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
        print(f"Performing action: {action_name}")
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
