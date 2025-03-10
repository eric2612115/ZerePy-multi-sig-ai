# !/usr/bin/env python3
# enhanced_server.py
import asyncio
import json
import logging
import os
import random
import re
import sys
from pprint import pprint

import aiohttp
from dotenv import load_dotenv, find_dotenv
import signal
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from uuid import uuid4
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.agent import ZerePyAgent
# --- ZerePy Imports ---
from src.server.mongodb_client import MongoDBClient
from src.action_handler import action_registry  # For listing tools
# No need to import AnthropicConnection directly here
from src.server.transaction_workflow import TransactionWorkflowHandler

from google import genai
from google.genai import types

# Analyze thinking depth based on query complexity
from src.thinking_framework.thinking_depth_selector import analyze_thinking_depth
from src.server.structured_response_handler import StructuredResponseHandler

# --- External API Imports ---
from backend.dex_api_client.wallet_service_client import WalletServiceClient
from backend.dex_api_client.okx_web3_client import OkxWeb3Client
from backend.dex_api_client.third_client import ThirdPartyClient
from backend.dex_api_client.cave_client import CaveClient
from backend.dex_api_client.public_data import get_binance_tickers

import apscheduler

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("zerepy_server")
load_dotenv(find_dotenv())


# --- Message Structure ---
class MessageStructure:
    @staticmethod
    def format_message(sender: str, text: str, message_type: str = "normal") -> Dict[str, Any]:
        return {
            "id": str(uuid4()),
            "sender": sender,
            "text": text,
            "message_type": message_type,
            "timestamp": datetime.now().isoformat()
        }

    @staticmethod
    def format_thinking(text: str) -> Dict[str, Any]:
        return MessageStructure.format_message("system", text, "thinking")

    @staticmethod
    def format_error(text: str) -> Dict[str, Any]:
        return MessageStructure.format_message("system", text, "error")

    @staticmethod
    def format_ai_response(text: str) -> Dict[str, Any]:
        return MessageStructure.format_message("agent", text)


class BaseAiWalletAddress(BaseModel):
    wallet_address: str


class BaseAgentAddress(BaseModel):
    agent_address: str


class BaseAgentName(BaseModel):
    agent_name: Optional[str] = "crypto_agent"


class BaseChainId(BaseModel):
    chain_id: int = 8453


class CreateAgent(BaseAiWalletAddress, BaseAgentName):
    pass


class RegisterMultiSigWallet(BaseAiWalletAddress, BaseChainId):
    pass


class GetTransactionHistory(BaseAiWalletAddress):
    chain_id_list: Optional[List[int]] = None


class GetBalanceDetails(BaseAiWalletAddress):
    chain_id_list: Optional[List[int]] = None


# --- MultiClientManager class for handling multiple WebSocket connections ---
class MultiClientManager:
    def __init__(self, db_client: MongoDBClient,
                 okx_client: OkxWeb3Client, wallet_service_client: WalletServiceClient, cave_client: CaveClient,
                 third_party_client: ThirdPartyClient, ping_interval: int = 30, inactivity_timeout: int = 1800, ):
        """Initialize the MultiClientManager."""
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_sessions: Dict[str, Dict[str, Any]] = {}
        self.user_agents: Dict[str, ZerePyAgent] = {}
        self.ping_interval = ping_interval
        self.inactivity_timeout = inactivity_timeout
        self.ping_task = None
        self.db_client = db_client
        self.okx_client = okx_client
        self.wallet_service_client = wallet_service_client
        self.third_party_client = third_party_client
        self.cave_client = cave_client

        self.chain_symbol_list_with_price = {}

        self.rethink_count = 0

    async def connect(self, wallet_address: str, websocket: WebSocket, agent_name: str = "crypto_agent"):
        """Accept a new WebSocket connection and initialize user session."""
        await websocket.accept()
        self.active_connections[wallet_address] = websocket

        if wallet_address not in self.user_sessions:
            self.user_sessions[wallet_address] = {
                "last_message_time": datetime.now(),
                "reconnect_count": 0,
                "agent_busy": False,
                "agent_name": agent_name,
            }
        else:
            self.user_sessions[wallet_address]["reconnect_count"] += 1
            self.user_sessions[wallet_address]["last_message_time"] = datetime.now()

        logger.info(f"WebSocket connection established for wallet: {wallet_address}")

        # Load or create the agent for this user
        if not self.get_agent_for_user(wallet_address):
            try:
                agent = ZerePyAgent(agent_name)
                self.set_agent_for_user(wallet_address, agent)
                logger.info(f"Agent {agent_name} loaded for user {wallet_address}")
            except Exception as e:
                logger.error(f"Error loading agent: {e}")
                await self.send_message(
                    wallet_address,
                    MessageStructure.format_error(f"Error loading agent: {str(e)}")
                )
                self.disconnect(wallet_address)
                return False

        # Setup ping task if not already running
        if self.ping_task is None or self.ping_task.done():
            self.ping_task = asyncio.create_task(self._ping_clients())

        # Load conversation history from database
        try:
            history = await self.db_client.get_conversation_history(wallet_address)
            if history:
                logger.info(f"Loaded {len(history)} messages from history for {wallet_address}")
        except Exception as e:
            logger.error(f"Error loading conversation history: {e}")

        return True

    def disconnect(self, wallet_address: str):
        """Handle client disconnection."""
        if wallet_address in self.active_connections:
            del self.active_connections[wallet_address]
            if wallet_address in self.user_sessions:
                self.user_sessions[wallet_address]["last_disconnect_time"] = datetime.now()
                self.user_sessions[wallet_address]["agent_busy"] = False
            logger.info(f"WebSocket connection closed for wallet: {wallet_address}")

        if not self.active_connections and self.ping_task and not self.ping_task.done():
            self.ping_task.cancel()
            self.ping_task = None

    async def send_message(self, wallet_address: str, message: Dict[str, Any]):
        """Send a message to a specific client."""
        if wallet_address in self.active_connections:
            websocket = self.active_connections[wallet_address]
            try:
                # Save message to database first
                try:
                    await self.db_client.save_message(wallet_address, message)
                except Exception as e:
                    logger.error(f"Error saving message to database: {e}")

                # Then send to client
                await websocket.send_json(message)
                logger.debug(f"Message sent to {wallet_address}: {message.get('id', 'unknown')}")

                if wallet_address in self.user_sessions:
                    self.user_sessions[wallet_address]["last_message_time"] = datetime.now()
                return True
            except Exception as e:
                logger.error(f"Error sending message to {wallet_address}: {e}")
                self.disconnect(wallet_address)
                return False
        else:
            logger.warning(f"Attempted to send message to disconnected client: {wallet_address}")
            return False

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients."""
        disconnected_clients = []
        for wallet_address in self.active_connections:
            success = await self.send_message(wallet_address, message)
            if not success:
                disconnected_clients.append(wallet_address)

        for wallet_address in disconnected_clients:
            self.disconnect(wallet_address)

    def set_agent_busy(self, wallet_address: str, busy: bool):
        """Set whether a client's agent is busy."""
        if wallet_address in self.user_sessions:
            self.user_sessions[wallet_address]["agent_busy"] = busy
            logger.info(f"Agent for {wallet_address} set to {'busy' if busy else 'available'}")

    def set_transaction_proposal(self, wallet_address: str, proposal_data: Dict[str, Any]):
        """
        設置交易提案以等待用戶確認
        """
        if wallet_address in self.user_sessions:
            self.user_sessions[wallet_address]["transaction_proposal"] = proposal_data
            self.user_sessions[wallet_address]["proposal_time"] = datetime.now()
            logger.info(f"Transaction proposal set for {wallet_address}: {proposal_data.get('action', 'unknown')}")

    def get_transaction_proposal(self, wallet_address: str) -> Optional[Dict[str, Any]]:
        """
        獲取待確認的交易提案
        """
        if wallet_address in self.user_sessions:
            proposal = self.user_sessions[wallet_address].get("transaction_proposal")
            proposal_time = self.user_sessions[wallet_address].get("proposal_time")

            # 如果提案存在但已超過30分鐘，視為過期
            if proposal and proposal_time:
                time_diff = datetime.now() - proposal_time
                if time_diff.total_seconds() > 1800:  # 30分鐘
                    logger.info(f"Transaction proposal for {wallet_address} expired")
                    del self.user_sessions[wallet_address]["transaction_proposal"]
                    del self.user_sessions[wallet_address]["proposal_time"]
                    return None
            return proposal
        return None

    def clear_transaction_proposal(self, wallet_address: str):
        """
        清除交易提案
        """
        if wallet_address in self.user_sessions:
            if "transaction_proposal" in self.user_sessions[wallet_address]:
                del self.user_sessions[wallet_address]["transaction_proposal"]
            if "proposal_time" in self.user_sessions[wallet_address]:
                del self.user_sessions[wallet_address]["proposal_time"]
            logger.info(f"Transaction proposal cleared for {wallet_address}")

    def is_agent_busy(self, wallet_address: str) -> bool:
        """Check if a client's agent is busy."""
        return self.user_sessions.get(wallet_address, {}).get("agent_busy", False)

    def get_agent_for_user(self, wallet_address: str) -> Optional[ZerePyAgent]:
        """Get the agent instance for a user."""
        return self.user_agents.get(wallet_address)

    def set_agent_for_user(self, wallet_address: str, agent: ZerePyAgent):
        """Set the agent instance for a user."""
        self.user_agents[wallet_address] = agent
        logger.info(f"Agent set for user {wallet_address}")

    def create_action_registry(self, tools_definition: dict) -> dict:
        """
        Creates the action registry in the correct format from the tools definition.
        The tools_definition should be copy from previous prompt response.

        Args:
            tools_definition: The dictionary representing the tools (from the prompt).

        Returns:
            The action registry dictionary.
        """
        action_registry = {"tools": {}}
        for tool_name, tool_data in tools_definition.items():
            action_registry["tools"][tool_name] = {
                "name": tool_data["name"],
                "description": tool_data["description"],
                "parameters": tool_data["parameters"],
            }
        return action_registry

    def _get_tools_list(self):
        return {
            # Price and analysis related operations
            "check-token-price": {
                "name": "check-token-price",
                "parameters": [
                    {"name": "query", "required": True, "type": "string",
                     "description": "Token query text, typically the token's symbol or name (e.g., 'BTC', 'Ethereum', 'ETHUSDT').  Be as specific as possible, including the trading pair if known (e.g., 'BTCUSDT' instead of just 'BTC')."}
                ],
                "description": "Check the current price of a cryptocurrency token. Returns a list of prices for the given token symbol(s)."
            },
            "check-token-security": {
                "name": "check-token-security",
                "parameters": [
                    {"name": "underlying", "required": True, "type": "string",
                     "description": "The underlying asset symbol (e.g., 'ETH', 'BTC')."},
                    {"name": "target_token", "required": True, "type": "string",
                     "description": "The contract address of the token to check."},
                    {"name": "chain_name", "required": True, "type": "string",
                     "description": "The name of the blockchain (e.g., 'ethereum', 'bsc', 'base')."},
                ],
                "description": "Analyze token security data and potential risks. Requires the token's contract address and chain name, *not* just the symbol."
            },
            "analyze-portfolio": {
                "name": "analyze-portfolio",
                "parameters": [
                    {"name": "query", "required": True, "type": "string",
                     "description": "A general description of the request, like 'Analyze my portfolio'"},
                    {"name": "portfolio", "required": False, "type": "list",
                     "description": "A list of dictionaries, where each dictionary represents a token holding.  Each dictionary should include 'symbol' (e.g., 'BTC') and 'amount' (e.g., 1.5). Example: [{'symbol': 'ETH', 'amount': 10}, {'symbol': 'USDT', 'amount': 5000}]"}
                ],
                "description": "Analyze the performance and risk of a cryptocurrency portfolio.  The optional 'portfolio' parameter allows providing specific holdings. If omitted, the AI will likely need to ask follow-up questions."
            },

            # Market and liquidity related operations
            "get-hot-tokens": {
                "name": "get-hot-tokens",
                "parameters": [
                    {"name": "query", "required": True, "type": "string",
                     "description": "A general query like 'What are the hot tokens?'"},
                    {"name": "chain_id", "required": False, "type": "string",
                     "description": "The ID of the blockchain (e.g., '1' for Ethereum mainnet, '56' for Binance Smart Chain), '146' for Sonic Chain, User might asking about this one"},
                    {"name": "limit", "required": False, "type": "integer",
                     "description": "The maximum number of tokens to return."}
                ],
                "description": "Get hot/trending tokens on a specific blockchain.  The chain ID is optional; if not provided, the AI should try to infer it or use a default."
            },
            "check-token-liquidity": {
                "name": "check-token-liquidity",
                "parameters": [
                    {"name": "query", "required": True, "type": "string",
                     "description": "A query like 'Check liquidity for UNI on Ethereum'.  Be as specific as possible, including the token and chain."},
                ],
                "description": "Check the liquidity of a token on a specific chain. Requires the token and chain to be specified in the query."
            },

            # On-chain data query
            "carv-llm-query": {
                "name": "carv-llm-query",
                "parameters": [
                    {"name": "query", "required": True, "type": "string",
                     "description": "A natural language query for on-chain data. Example: 'What is the 24-hour trading volume of ETH on Uniswap?'"}
                ],
                "description": "Perform on-chain data queries using natural language.  This tool is for more complex queries that require accessing and interpreting on-chain data."
            },

            # Portfolio suggestion
            "suggest-portfolio": {
                "name": "suggest-portfolio",
                "parameters": [
                    {"name": "query", "required": True, "type": "string",
                     "description": "A description of the portfolio request, such as 'Suggest a portfolio for me'."},
                    {"name": "amount", "required": False, "type": "float",
                     "description": "The amount of money (in USD) to invest."},
                    {"name": "risk", "required": False, "type": "string",
                     "description": "The user's risk tolerance (e.g., 'low', 'medium', 'high')."}
                ],
                "description": "Suggest a diversified cryptocurrency portfolio.  The amount and risk tolerance are optional but helpful."
            },

            # Wallet and transaction operations
            "get-wallet-balance": {
                "name": "get-wallet-balance",
                "parameters": [
                    {"name": "wallet_address", "required": True, "type": "string",
                     "description": "The wallet address to check.  Must be a valid blockchain address."},
                    {"name": "chain_id", "required": False, "type": "string",
                     "description": "The ID of the blockchain. If not provided, balances from multiple chains may be returned."}
                ],
                "description": "Check the balance of a wallet address. Can return balances from multiple chains if no chain ID is specified."
            },
            "add-token-whitelist": {
                "name": "add-token-whitelist",
                "parameters": [
                    {"name": "wallet_address", "required": True, "type": "string",
                     "description": "The multisig wallet address (also referred to as 'safe wallet address')."},
                    {"name": "chain_id", "required": False, "type": "integer",
                     "description": "The chain ID. Defaults to 8453 if not provided."},
                    {"name": "token_addresses", "required": True, "type": "list",
                     "description": "A list of token contract addresses to add to the whitelist.  Must be valid contract addresses."},
                    {"name": "whitelist_signatures", "required": False, "type": "list",
                     "description": "Optional list of signatures for the whitelist operation."}
                ],
                "description": "Add tokens to the multisig wallet whitelist. Requires a list of token contract addresses."
            },
            "remove-token-whitelist": {
                "name": "remove-token-whitelist",
                "parameters": [
                    {"name": "wallet_address", "required": True, "type": "string",
                     "description": "The multisig wallet address (also referred to as 'safe wallet address')."},
                    {"name": "chain_id", "required": False, "type": "integer",
                     "description": "The chain ID. Defaults to 8453 if not provided."},
                    {"name": "token_addresses", "required": True, "type": "list",
                     "description": "A list of token contract addresses to remove from the whitelist.  Must be valid contract addresses."},
                    {"name": "whitelist_signatures", "required": False, "type": "list",
                     "description": "Optional list of signatures for the whitelist operation."}
                ],
                "description": "Remove tokens from the multisig wallet whitelist. Requires a list of token contract addresses."
            },
            "execute-token-swap": {
                "name": "execute-token-swap",
                "parameters": [
                    {"name": "wallet_address", "required": True, "type": "string",
                     "description": "The user's AI wallet address, used for signing."},
                    {"name": "safe_wallet_address", "required": True, "type": "string",
                     "description": "The multisig wallet address (also referred to as 'safe wallet address') where the swap will be executed."},
                    {"name": "chain_id", "required": False, "type": "integer",
                     "description": "The chain ID. Defaults to 8453 if not provided."},
                    {"name": "input_token_address", "required": True, "type": "string",
                     "description": "The contract address of the token being sent."},
                    {"name": "input_token_amount", "required": True, "type": "string",
                     "description": "The amount of the input token to send (as a string, to avoid precision issues)."},
                    {"name": "output_token_address", "required": True, "type": "string",
                     "description": "The contract address of the token to receive."},
                    {"name": "output_token_min_amount", "required": False, "type": "string",
                     "description": "The minimum amount of the output token to accept (slippage protection)."},

                ],
                "description": "Execute a token swap through a multisig wallet. Requires detailed information about the swap, including token addresses and amounts."
            },
            "execute-portfolio-batch-trade": {
                "name": "execute-portfolio-batch-trade",
                "parameters": [
                    {"name": "wallet_address", "required": True, "type": "string",
                     "description": "The user's AI wallet address, used for signing."},
                    {"name": "safe_wallet_address", "required": True, "type": "string",
                     "description": "The multisig wallet address (also referred to as 'safe wallet address') where the trades will be executed."},
                    {"name": "chain_id", "required": False, "type": "integer",
                     "description": "The chain ID. Defaults to 8453 if not provided."},
                    {"name": "portfolio", "required": True, "type": "list",
                     "description": "A list of trade objects, each with 'input_token_address', 'input_token_amount', 'output_token_address', and 'output_token_min_amount'. and 'output_token_min_amount' could be able to use price to calculate."},
                    {"name": "whitelist_signatures", "required": False, "type": "list",
                     "description": "Optional list of signatures for the whitelist operation."}
                ],
                "description": "Execute a batch of token trades through a multisig wallet. Requires detailed information about each trade."
            },
            "check_multi_sig_wallet": {
                "name": "check_multi_sig_wallet",
                "parameters": [
                    {"name": "wallet_address", "required": True, "type": "string",
                     "description": "The wallet address to check."}
                ],
                "description": "Checks if a given wallet address is a multi-signature wallet."
            },
            "check_multi_sig_wallet_whitelist": {
                "name": "check_multi_sig_wallet_whitelist",
                "parameters": [
                    {"name": "wallet_address", "required": True, "type": "string",
                     "description": "The multi-signature wallet address to check."},
                    {"name": "chain_id", "required": True, "type": "integer",
                     "description": "The ID of the blockchain."}
                ],
                "description": "Checks if a given wallet address has a specific token whitelist configured on a specified chain."
            },
            "web_search": {
                "name": "web_search",
                "parameters": [
                    {"name": "query", "required": True, "type": "string", "description": "The search query."}
                ],
                "description": "Perform a web search. Use this for information that is not available on-chain or through other specialized tools."
            }
        }

    async def _get_basic_info(self, wallet_address: str, message_data: Dict[str, Any]):
        logger.debug(f"Handling message for {wallet_address}: {message_data}")

        if wallet_address in self.user_sessions:
            self.user_sessions[wallet_address]["last_message_time"] = datetime.now()

        # Check if agent is already processing a request
        if self.is_agent_busy(wallet_address):
            await self.send_message(
                wallet_address,
                MessageStructure.format_error("Agent is busy processing another request. Please wait.")
            )
            return

        self.set_agent_busy(wallet_address, True)

        try:
            # Get user message text
            user_text = message_data.get("query", "")
            if not user_text:
                await self.send_message(
                    wallet_address,
                    MessageStructure.format_error("Empty message received")
                )
                self.set_agent_busy(wallet_address, False)
                return

            # Save user message to database
            user_message = MessageStructure.format_message("user", user_text)
            await self.db_client.save_message(wallet_address, user_message)

            # Send thinking message
            await self.send_message(
                wallet_address,
                MessageStructure.format_thinking("Processing your request...")
            )

            # Get agent for this user
            agent = self.get_agent_for_user(wallet_address)
            if not agent:
                await self.send_message(
                    wallet_address,
                    MessageStructure.format_error("No agent available for this session.")
                )
                self.set_agent_busy(wallet_address, False)
                return

            # Ensure LLM provider is set up
            if not agent.is_llm_set:
                agent._setup_llm_provider()

            # Get conversation context and wallet information
            conversation_history = await self.db_client.get_conversation_history(wallet_address)

            # Check if this is a crypto wallet interaction
            user_data = await self.wallet_service_client.get_multi_sig_wallet(wallet_address)
            has_multisig = user_data["safeWalletList"]

            conversation_context = {
                "conversation_length": len(conversation_history),
                "last_messages": conversation_history[-5:] if len(conversation_history) >= 5 else conversation_history,
                "wallet_address": wallet_address,
                "has_multisig_wallet": has_multisig,
                "domain": "crypto"  # Indicate this is a cryptocurrency context
            }
            logger.debug(f'conversation_context: {conversation_context}')
            return user_text, agent, conversation_context
        except Exception as e:
            logger.exception(f"Error processing message: {e}")
            await self.send_message(
                wallet_address,
                MessageStructure.format_error(f"Error processing your request: {str(e)}")
            )
            self.set_agent_busy(wallet_address, False)

    def make_ai_select_tool(self, user_query: str, agent, conversation_context=None) -> dict:
        """
        Selects the appropriate tool and its parameters based on the user's query and conversation context.

        Args:
            user_query: The user's natural language query.
            agent: The agent instance to use for tool selection.
            conversation_context: The conversation context containing history.

        Returns:
            A dictionary representing the selected tool and its parameters, or
            {"tool_name": None, "tool_input": None} if no tool is needed.
        """
        action_registry = {"tools": self._get_tools_list()}
        tools_json = json.dumps(
            self.create_action_registry(action_registry["tools"]))  # Convert the action registry to JSON

        # 獲取最近的對話歷史並格式化
        history_text = ""
        if conversation_context and "last_messages" in conversation_context:
            recent_messages = conversation_context["last_messages"]
            history_lines = []
            for msg in recent_messages:
                sender = msg.get("sender", "unknown")
                text = msg.get("text", "")
                history_lines.append(f"{sender}: {text}")
            history_text = "\n".join(history_lines)

        tools_prompt = f"""Based on the conversation history and current query, determine the most appropriate tool to use and the values for its parameters.

        Conversation history:
        {history_text}

        Available tools (in JSON format):
        {tools_json}

        The user's current query is: {user_query}

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

        system_prompt = "You are an AI assistant that analyzes user requests and conversation history to determine which tools to use, responding only in JSON."

        # 調用AI生成工具選擇
        tool_selection_response = agent.connection_manager.perform_action(
            connection_name="anthropic",
            action_name="generate-text",
            params=[tools_prompt, system_prompt]
        )

        print('tool_selection_response: ', tool_selection_response)

        try:
            tool_selection = json.loads(tool_selection_response)
            # Basic validation
            if not isinstance(tool_selection, dict) or "tool_name" not in tool_selection:
                raise ValueError("Invalid response format from LLM.")
            if tool_selection["tool_name"] is not None:
                if tool_selection["tool_name"] not in action_registry["tools"]:
                    raise ValueError(f"Invalid tool name selected: {tool_selection['tool_name']}")
                # Validate parameters
                selected_tool_params = action_registry["tools"][tool_selection["tool_name"]]["parameters"]
                provided_params = tool_selection["tool_input"]

                if provided_params is None:
                    raise ValueError(
                        f"tool_input should not be null, parameters needed for {tool_selection['tool_name']}")
                for param in selected_tool_params:
                    if param["required"] and param["name"] not in provided_params:
                        raise ValueError(
                            f"Required parameter '{param['name']}' missing for tool '{tool_selection['tool_name']}'.")
            return tool_selection

        except json.JSONDecodeError:
            print(f"Error: Invalid JSON response from LLM: {tool_selection_response}")
            return {"tool_name": None, "tool_input": None}  # Or raise the exception if you prefer
        except ValueError as e:
            print(f"Error: {e}")
            return {"tool_name": None, "tool_input": None}

    async def _is_ai_confirmation(self, current_text: str, conversation_context: dict, agent) -> bool:
        """使用AI判斷用戶消息是否是對先前操作的確認"""

        # 如果消息很長，很可能不是簡單的確認
        if len(current_text.strip()) > 20:
            return False

        # 獲取對話歷史
        if not conversation_context or "last_messages" not in conversation_context:
            return False

        recent_messages = conversation_context.get("last_messages", [])
        if len(recent_messages) < 2:  # 需要至少兩條消息才能有上下文
            return False

        # 尋找上一個AI消息和上一個用戶消息
        last_ai_message = None
        previous_user_message = None

        for msg in reversed(recent_messages):
            if msg.get("sender") == "agent" and not last_ai_message:
                last_ai_message = msg.get("text", "")
            elif msg.get("sender") == "user" and msg.get("text") != current_text and not previous_user_message:
                previous_user_message = msg.get("text", "")

            if last_ai_message and previous_user_message:
                break

        if not last_ai_message:
            return False

        # 構建提示詞讓AI判斷是否為確認
        confirmation_prompt = f"""
        In a conversation about cryptocurrency operations, analyze if the user's response is a confirmation.

        Previous user message: "{previous_user_message}"
        My last response: "{last_ai_message}"
        User's current response: "{current_text}"

        Determine if the user's current response is confirming an action or operation I suggested.
        If the user is confirming (saying yes, agreeing, approving), respond only with "CONFIRMATION".
        If the user is declining or asking something else, respond only with "NOT_CONFIRMATION".

        Only respond with one of these two words.
        """

        try:
            response = agent.connection_manager.perform_action(
                connection_name="anthropic",
                action_name="generate-text",
                params=[confirmation_prompt, "You analyze if a message is a confirmation."]
            )

            if "CONFIRMATION" in response.upper():
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"Error in AI confirmation check: {e}")
            # 如果AI分析失敗，回退到簡單的關鍵字匹配
            return self._simple_confirmation_check(current_text)

    def _simple_confirmation_check(self, text: str) -> bool:
        """簡單的關鍵字確認檢查（作為備用方法）"""
        text = text.lower().strip()
        confirmation_phrases = ["yes", "confirm", "proceed", "ok", "sure", "go ahead", "yep", "absolutely"]
        return any(phrase == text or f"{phrase}." == text for phrase in confirmation_phrases)

    async def analyze_user_confirmation(self, current_text: str, previous_message: str, agent) -> bool:
        """
        使用 AI 判斷用戶是否在確認之前提議的操作，不使用任何關鍵字匹配
        """
        confirmation_prompt = f"""
        In our conversation about cryptocurrency:

        Previous message from the assistant: {previous_message}
        Current user response: {current_text}

        Task: Determine if the user is confirming or agreeing to proceed with the action or suggestion 
        mentioned in the previous message.

        Respond with only one word:
        - "TRUE" if the user is confirming or agreeing to proceed
        - "FALSE" if the user is not confirming, declining, or asking about something else
        """

        system_prompt = "You are a contextual analysis system that determines if a user response constitutes confirmation of a previously suggested action."

        try:
            response = agent.connection_manager.perform_action(
                connection_name="anthropic",
                action_name="generate-text",
                params=[confirmation_prompt, system_prompt]
            )

            return "TRUE" in response.upper()
        except Exception as e:
            logger.error(f"Error in AI confirmation analysis: {e}")
            return False  # 出錯時傾向於安全側，返回 False

    async def analyze_previous_suggestion(self, previous_message: str, agent) -> dict:
        """
        使用 AI 分析前一條消息的內容和建議類型，不使用任何關鍵字匹配
        """
        analysis_prompt = f"""
        Analyze the following message from a cryptocurrency assistant:

        {previous_message}

        Determine:
        1. What type of action or information is being presented? (e.g., token purchase suggestion, 
           market information, educational content, etc.)
        2. If it's a purchase suggestion, what tokens are mentioned?
        3. If there's a financial amount mentioned, what is it and what currency?
        4. What blockchain or chain is mentioned, if any?

        Return your analysis as JSON with the following structure:
        {{
            "suggestion_type": "purchase_suggestion" or "market_information" or "educational_content" or "other",
            "tokens": [
                {{"symbol": "TOKEN1", "price": "price if available or null"}},
                {{"symbol": "TOKEN2", "price": "price if available or null"}}
            ],
            "amount": the amount mentioned or null,
            "currency": the currency mentioned or null,
            "chain": the blockchain mentioned or null,
            "chain_id": the chain ID if identifiable or null
        }}
        """

        system_prompt = "You are a conversation analysis system that extracts specific structured information from messages."

        try:
            response = agent.connection_manager.perform_action(
                connection_name="anthropic",
                action_name="generate-text",
                params=[analysis_prompt, system_prompt]
            )

            # 提取 JSON
            import re
            import json

            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                return {"suggestion_type": "unknown"}
        except Exception as e:
            logger.error(f"Error in previous message analysis: {e}")
            return {"suggestion_type": "unknown"}

    def _get_chain_id_from_name(self, chain_name: str) -> Optional[int]:
        """
        根據鏈名稱獲取鏈ID
        """
        chain_map = {
            "sonic": 146,
            "base": 8453,
            "ethereum": 1,
            "eth": 1,
            "binance": 56,
            "bsc": 56,
            "polygon": 137,
            "matic": 137,
            "arbitrum": 42161,
            "optimism": 10
        }

        # 忽略大小寫匹配
        chain_name_lower = chain_name.lower()
        for name, chain_id in chain_map.items():
            if name in chain_name_lower:
                return chain_id

        return None

    async def _get_token_address_by_symbol(self, symbol: str, chain_id: int) -> Optional[str]:
        """
        根據代幣符號和鏈ID獲取代幣地址
        """
        # 這裡應該是查詢資料庫或API的實際實現
        # 以下是示例數據
        self.chain_symbol_list_with_price[chain_id] = await self.wallet_service_client.get_asset_price_list(chain_id)
        data = self.chain_symbol_list_with_price[chain_id]['list']
        for item in data:
            if item['symbol'].upper() == symbol.upper():
                return item['address']
        return None

    async def handle_purchase_confirmation(self, wallet_address: str, suggestion_data: dict, agent):
        """處理用戶確認購買的操作 - 批量處理多代幣"""
        try:
            # 獲取所有要購買的代幣和相關信息
            tokens = suggestion_data.get("tokens", [])
            amount = suggestion_data.get("amount")
            currency = suggestion_data.get("currency")
            chain = suggestion_data.get("chain")
            chain_id = suggestion_data.get("chain_id") or self._get_chain_id_from_name(chain)

            if not tokens or not amount or not currency or not chain_id:
                await self.send_message(
                    wallet_address,
                    MessageStructure.format_error("Missing required information for token purchase.")
                )
                return

            # 通知用戶正在處理
            token_names = [token.get("symbol", "Unknown") for token in tokens]
            await self.send_message(
                wallet_address,
                MessageStructure.format_thinking(f"Analyzing security for all tokens: {', '.join(token_names)}...")
            )

            # 一次性獲取所有代幣的安全信息
            security_results = []
            for token in tokens:
                symbol = token.get("symbol")
                address = token.get("address") or await self._get_token_address_by_symbol(symbol, chain_id)

                # 安全檢查
                security_info = await self.check_token_security(
                    underlying="ETH",  # 假設使用ETH作為基礎資產
                    target_token=address,
                    chain_name=chain.lower() if chain else "base"  # 默認使用base鏈
                )

                # 分析安全結果
                security_score = self._extract_security_score(security_info)
                token["security_info"] = security_info
                token["security_score"] = security_score
                token["address"] = address
                security_results.append({
                    "symbol": symbol,
                    "security_score": security_score,
                    "recommendation": "safe" if security_score >= 70 else "risky"
                })

            # 生成安全分析報告
            security_prompt = f"""
            You've analyzed the security of {len(tokens)} tokens that the user is interested in purchasing.

            Security analysis results:
            {json.dumps(security_results, indent=2)}

            Based on this analysis, provide a concise security report for the user.
            Include which tokens appear safe to purchase and which might be risky.
            Recommend which tokens the user should proceed with purchasing based on security scores.
            Keep your response concise and actionable.
            """

            security_report = agent.connection_manager.perform_action(
                connection_name="anthropic",
                action_name="generate-text",
                params=[security_prompt, "You are providing a security analysis of cryptocurrency tokens."]
            )

            # 發送安全報告給用戶
            await self.send_message(
                wallet_address,
                MessageStructure.format_ai_response(
                    f"{security_report}\n\nWould you like me to proceed with purchasing the recommended tokens with your {amount} {currency}?"
                )
            )

            # 將本次分析結果保存到用戶會話中，等待用戶的最終確認
            # 這樣用戶回覆"yes"時，我們可以直接執行之前分析的安全代幣
            safe_tokens = [token for token in tokens if token.get("security_score", 0) >= 70]

            if wallet_address in self.user_sessions:
                self.user_sessions[wallet_address]["pending_purchase"] = {
                    "safe_tokens": safe_tokens,
                    "amount": amount,
                    "currency": currency,
                    "chain_id": chain_id,
                    "analysis_time": datetime.now().isoformat()
                }

        except Exception as e:
            logger.exception(f"Error in token security analysis: {e}")
            await self.send_message(
                wallet_address,
                MessageStructure.format_error(f"Error analyzing token security: {str(e)}")
            )

    async def execute_batch_purchase(self, wallet_address: str, agent):
        """執行批量代幣購買"""
        try:
            # 獲取之前分析的結果
            pending_purchase = self.user_sessions.get(wallet_address, {}).get("pending_purchase")
            if not pending_purchase:
                await self.send_message(
                    wallet_address,
                    MessageStructure.format_error("No pending purchase found. Please start over.")
                )
                return

            # 提取必要信息
            safe_tokens = pending_purchase.get("safe_tokens", [])
            amount = pending_purchase.get("amount")
            currency = pending_purchase.get("currency")
            chain_id = pending_purchase.get("chain_id")

            if not safe_tokens:
                await self.send_message(
                    wallet_address,
                    MessageStructure.format_error("No safe tokens identified for purchase.")
                )
                return

            # 計算每個代幣分配的金額
            token_count = len(safe_tokens)
            amount_per_token = float(amount) / token_count

            # 通知用戶開始執行交易
            token_names = [token.get("symbol", "Unknown") for token in safe_tokens]
            await self.send_message(
                wallet_address,
                MessageStructure.format_thinking(
                    f"Executing batch purchase of {token_count} tokens: {', '.join(token_names)}")
            )

            # 獲取錢包信息
            user_data = await self.wallet_service_client.get_multi_sig_wallet(wallet_address)
            multisig_info = user_data.get("safeWalletList", [])
            safe_wallet_address = None

            for info in multisig_info:
                if info.get("chainId") == chain_id:
                    safe_wallet_address = info.get("multisig_address")
                    break

            if not safe_wallet_address:
                await self.send_message(
                    wallet_address,
                    MessageStructure.format_error(f"No multisig wallet found for chain ID {chain_id}.")
                )
                return

            # 獲取幣種地址
            currency_address = await self._get_token_address_by_symbol(currency, chain_id)

            # 執行所有交易
            transaction_results = []
            for token in safe_tokens:
                symbol = token.get("symbol")
                address = token.get("address")

                # 將代幣添加到白名單
                whitelist_result = await self.add_token_whitelist(
                    wallet_address=safe_wallet_address,
                    chain_id=chain_id,
                    token_addresses=[address]
                )

                # 執行交易
                swap_result = await self.execute_token_swap(
                    wallet_address=wallet_address,
                    safe_wallet_address=safe_wallet_address,
                    chain_id=chain_id,
                    input_token_address=currency_address,
                    input_token_amount=str(amount_per_token),
                    output_token_address=address
                )

                # 記錄結果
                transaction_results.append({
                    "token": symbol,
                    "amount": amount_per_token,
                    "currency": currency,
                    "status": "success" if swap_result else "failed"
                })

            # 清除待處理的購買
            if wallet_address in self.user_sessions:
                if "pending_purchase" in self.user_sessions[wallet_address]:
                    del self.user_sessions[wallet_address]["pending_purchase"]

            # 生成交易總結
            summary_prompt = f"""
            You've completed a batch purchase of tokens with {amount} {currency}.

            Transaction results:
            {json.dumps(transaction_results, indent=2)}

            Provide a concise summary of the transactions, mentioning:
            1. Which tokens were successfully purchased
            2. Any tokens that failed to purchase
            3. The amount spent on each token

            Keep your response friendly but direct and to the point.
            """

            summary_response = agent.connection_manager.perform_action(
                connection_name="anthropic",
                action_name="generate-text",
                params=[summary_prompt, "You are summarizing completed token purchases."]
            )

            # 發送交易總結
            await self.send_message(
                wallet_address,
                MessageStructure.format_ai_response(summary_response)
            )

        except Exception as e:
            logger.exception(f"Error in batch purchase execution: {e}")
            await self.send_message(
                wallet_address,
                MessageStructure.format_error(f"Error executing purchase: {str(e)}")
            )

    def _get_tool_param(self, params: dict, possible_keys: List[str], default=None):
        """
        從工具參數中獲取值，嘗試多個可能的鍵名
        """
        for key in possible_keys:
            if key in params:
                return params[key]
        return default

    def _get_chain_id_from_name(self, chain_name: str) -> Optional[int]:
        """
        根據鏈名稱獲取鏈ID
        """
        chain_map = {
            "sonic": 146,
            "base": 8453,
            "ethereum": 1,
            "eth": 1,
            "binance": 56,
            "bsc": 56,
            "polygon": 137,
            "matic": 137,
            "arbitrum": 42161,
            "optimism": 10
        }

        # 忽略大小寫匹配
        chain_name_lower = chain_name.lower()
        for name, chain_id in chain_map.items():
            if name in chain_name_lower:
                return chain_id

        return None

    async def handle_client_message(self, wallet_address: str, message_data: Dict[str, Any]):
        """Process an incoming client message with AI-driven tool selection and proper transaction workflow."""
        logger.debug(f"Handling message for {wallet_address}: {message_data}")

        user_text, agent, conversation_context = await self._get_basic_info(wallet_address, message_data)
        await self.send_message(
            wallet_address,
            MessageStructure.format_thinking("Analyzing your request and conversation history...")
        )

        try:
            if user_text and agent and conversation_context:
                # 获取最近的对话历史，用于上下文分析
                recent_messages = conversation_context.get("last_messages", [])
                previous_ai_message = None

                # 寻找上一个AI消息
                for msg in reversed(recent_messages):
                    if msg.get("sender") == "agent":
                        previous_ai_message = msg.get("text", "")
                        break

                await self.send_message(
                    wallet_address,
                    MessageStructure.format_thinking("Analyzing your request and conversation history...")
                )

                # 如果有上一个AI消息且当前用户消息较短，分析是否是确认回复
                # is_confirmation = False
                # if previous_ai_message and len(user_text.strip()) <= 10:
                #     # 使用AI判断用户是否在确认先前的建议
                #     is_confirmation = await self.analyze_user_confirmation(user_text, previous_ai_message, agent)
                #
                #     # 在这里替换现有的确认处理逻辑
                #     # 检查这是否是对先前安全分析的最终确认
                #     if is_confirmation and wallet_address in self.user_sessions and "pending_purchase" in \
                #             self.user_sessions[wallet_address]:
                #         print('trying to buy batch at once ...')
                #         # 讓Ai拿的上下文準備整理一份最後輸出符合
                #
                #     # 旧的确认逻辑可以保留或移除
                #     if is_confirmation:
                #         # 使用AI分析先前的建议内容
                #         previous_suggestion = await self.analyze_previous_suggestion(previous_ai_message, agent)
                #         logger.info(f"User confirmed previous suggestion: {previous_suggestion.get('suggestion_type')}")
                #
                #         # 如果是购买建议，处理多代币购买逻辑
                #         if previous_suggestion.get('suggestion_type') == "purchase_suggestion":
                #             await self.handle_purchase_confirmation(
                #                 wallet_address,
                #                 previous_suggestion,
                #                 agent
                #             )
                #             return  # 交易处理完成，提前返回

                # 非确认回复或确认处理失败，使用标准流程
                await self.send_message(
                    wallet_address,
                    MessageStructure.format_thinking("Analyzing your request and conversation history...")
                )

                thinking_result = await analyze_thinking_depth(user_text, agent, conversation_context)
                thinking_depth = thinking_result["depth"]
                detected_intents = thinking_result["detected_intents"]
                print('-----------------')
                print('thinking_result: ', thinking_result)
                print('thinking_depth: ', thinking_depth)
                print('detected_intents: ', detected_intents)

                if thinking_result['depth'] == "deep" and "transaction_intent" in detected_intents:
                    # means the user is trying to make a transaction with batch tokens
                    await self.send_message(
                        wallet_address,
                        MessageStructure.format_thinking("Analyzing your intention and conversation history...")
                    )
                    if "batch_trades" in detected_intents:
                        # 拿上下文後合併 hot tokens 並且分析第一次後，取得 portfolio 配置的結果
                        # 再問一次產生一份 ai 可以理解且 符合 execute-portfolio-batch-trade 工具的 json output 讓 function 可以直接使用
                        #
                        print("Prepare to buy batch tokens ...")
                        base_chain_hot_tokens = await self.get_hot_tokens(8453)
                        sonic_chain_hot_tokens = await self.get_hot_tokens(146)
                        batch_trades_analysis_prompt = f"""
                        In a conversation about cryptocurrency operations, analyze the user's request for batch trades.
                        here is user's detailed request: {user_text}
                        user's conversation history: {conversation_context}
                        
                        chain_id: 8453 mean base chain,hot token: {base_chain_hot_tokens}
                        sonic chain_id: 146 mean sonic chain, hot token: {sonic_chain_hot_tokens}
                        
                        here is hot tokens data, include the chain_id, contract address, and price, so you could use it to calculate the output_token_min_amount.
                        
                        and each User would use USDC to buy the hot tokens.
                        base chain USDC contract address: 0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913,
                        Sonic chain USDC contract address: 0x29219dd400f2Bf60E5a23d13Be72B486D4038894
                        
                        The user has requested to execute a batch of token trades through a multisig wallet.
                        Analyze the user's request and provide a JSON object with the following structure:
                        
                        and the Output token address would be the hot tokens address, and the output_token_min_amount would be calculated based on the hot tokens price.
                        for example(DONT COPY THIS EXAMPLE):
                        input token address: 0x2791bca1f2de4661ed88a30c99a7a9449aa84174,
                        input token amount: 200, // USDC which 1/5 of the total amount
                        output token address: 0xfdc6abaa69f0df71ff4bb6bdfec34188796ee07a,   // would be in the hot_tokens['assets'] and output token address would be the "address" if you use "symbol" to find the address.
                        output token min amount: 10  // just make 10 if price does find.
                        
                        and format the output as below:
                        
                        {{
                            "wallet_address": "User's wallet address",
                            "safe_wallet_address": "The multisig wallet address where the trades will be executed",
                            "chain_id": "The chain ID for the trades",
                            "portfolio": [
                                {{"input_token_address": "Input token address", "input_token_amount": "Input token amount",
                                "output_token_address": "Output token address", "output_token_min_amount": "Output token min amount"}},
                                {{"input_token_address": "Input token address", "input_token_amount": "Input token amount",
                                "output_token_address": "Output token address", "output_token_min_amount": "Output token min amount"}}
                            ],
                            "whitelist_signatures": ["Signature1", "Signature2", ...]
                        }}
                        """

                        system_prompt = "You are an AI assistant that analyzes user requests for batch token trades and prepares the data for execution."
                        print('batch_trades_analysis_prompt: ', batch_trades_analysis_prompt)
                        batch_analysis_response = agent.connection_manager.perform_action(
                            connection_name="anthropic",
                            action_name="generate-text",
                            params=[batch_trades_analysis_prompt, system_prompt]
                        )

                        print('=========== Batch_analysis_response: ', batch_analysis_response)
                        json_formatting_prompt = """
                        if the data set is already fulfilled, please provide the JSON format as below:
                        {{
                            "wallet_address": "User's wallet address",
                            "safe_wallet_address": "The multisig wallet address where the trades will be executed",
                            "chain_id": "The chain ID for the trades",
                            "portfolio": [
                                {{"input_token_address": "Input token address", "input_token_amount": "Input token amount",
                                "output_token_address": "Output token address", "output_token_min_amount": "Output token min amount"}},
                                {{"input_token_address": "Input token address", "input_token_amount": "Input token amount",
                                "output_token_address": "Output token address", "output_token_min_amount": "Output token min amount"}}
                            ],
                            "whitelist_signatures": ["Signature1", "Signature2", ...]
                        }}
                        **NOTE: Please provide the JSON format as above, do not add any other text.**
                        """

                        # 讓AI在處理一次後，再問一次，確保資料格式正確
                        batch_analysis_response_again = agent.connection_manager.perform_action(
                            connection_name="anthropic",
                            action_name="generate-text",
                            params=[json_formatting_prompt, system_prompt]
                        )

                        print("======== AGAIN: ", batch_analysis_response_again)
                        print("-----------------")

                        batch_order_data = json.loads(batch_analysis_response_again)
                        print('batch_order_data: ', batch_order_data)
                        wallet_add = batch_order_data['wallet_address']
                        safe_wallet_add = batch_order_data['safe_wallet_address']
                        chain_id = batch_order_data['chain_id']
                        portfolio = batch_order_data['portfolio']
                        whitelist_signatures = batch_order_data['whitelist_signatures']
                        # add whitelist
                        white_list = []
                        for item in portfolio:
                            white_list.append(item['output_token_address'])
                        await self.send_message(
                            wallet_address,
                            MessageStructure.format_ai_response(f"Adding {white_list}tokens to the whitelist...")
                        )
                        whitelist_res = await self.add_token_whitelist(
                            wallet_address=safe_wallet_add,
                            chain_id=chain_id,
                            token_addresses=[item['output_token_address'] for item in portfolio],
                            whitelist_signatures=whitelist_signatures
                        )
                        print('whitelist_res: ', whitelist_res)
                        await self.send_message(
                            wallet_address,
                            MessageStructure.format_ai_response("Tokens added to the whitelist.")
                        )

                        chain_name = "others"
                        if chain_id == 8453:
                            chain_name = "base"
                        elif chain_id == 146:
                            chain_name = "sonic"

                        # execute trades
                        for item in portfolio:
                            print("order data: ", item)
                            await self.send_message(
                                wallet_address,
                                MessageStructure.format_ai_response(
                                    f"Trying to executed the Trade : {item['input_token_address']} on {chain_name}.")
                            )
                            res = await self.execute_token_swap(
                                wallet_address=wallet_add,
                                safe_wallet_address=safe_wallet_add,
                                chain_id=chain_id,
                                input_token_address=item['input_token_address'],
                                input_token_amount=item['input_token_amount'],
                                output_token_address=item['output_token_address'],
                                output_token_min_amount=item['output_token_min_amount']
                            )
                            print('res: ', res)

                            await self.send_message(
                                wallet_address,
                                MessageStructure.format_ai_response(f"Trade executed: {item['input_token_address']} on {chain_name}.")
                            )

                        # send message to user
                        await self.send_message(
                            wallet_address,
                            MessageStructure.format_ai_response("The batch trades have been executed successfully.")
                        )

                await self.send_message(
                    wallet_address,
                    MessageStructure.format_thinking("Since you don't have trading intention, I will analyze your request and conversation history...")
                )


                # Log the thinking depth and intents for debugging
                logger.info(f"Thinking depth determined for query: {thinking_depth}")
                logger.info(f"Detected intents: {detected_intents}")

                tools_picking = self.make_ai_select_tool(user_text, agent, conversation_context)
                print('tools_picking: ', tools_picking)
                tool_name = tools_picking.get("tool_name")
                tool_params = tools_picking.get("tool_input", {})

                if tool_name:
                    # 有工具被选中，执行工具
                    await self.send_message(wallet_address,
                                            MessageStructure.format_thinking("Processing your request..."))

                    try:
                        # 调用工具 (这里需要根据您的实际工具实现进行调整)
                        if tool_name == "check-token-price":
                            symbol_name = tool_params.get("query") or tool_params.get("symbol_name")
                            res = await self.check_token_price([symbol_name])

                        elif tool_name == "get-hot-tokens":
                            res = await self.get_hot_tokens(tool_params.get('chain_id'))

                        elif tool_name == "check-token-security":
                            res = await self.check_token_security(
                                tool_params.get('underlying'),
                                tool_params.get('target_token'),
                                tool_params.get('chain_name')
                            )

                        elif tool_name == "analyze-portfolio":
                            res = await self.analyze_portfolio(tool_params.get('portfolio', []))

                        elif tool_name == "check-token-liquidity":
                            res = await self.check_token_liquidity(tool_params.get('query'))

                        elif tool_name == "carv-llm-query":
                            res = await self.carv_llm_query(tool_params.get('query'))

                        elif tool_name == "suggest-portfolio":
                            res = await self.suggest_portfolio(
                                tool_params.get('query'),
                                tool_params.get('amount'),
                                tool_params.get('risk')
                            )

                        elif tool_name == "get-wallet-balance":
                            res = await self.get_wallet_balance(
                                tool_params.get('wallet_address'),
                                tool_params.get('chain_id')
                            )

                        elif tool_name == "add-token-whitelist":
                            res = await self.add_token_whitelist(
                                tool_params.get('wallet_address'),
                                tool_params.get('token_addresses'),
                                tool_params.get('chain_id', 8453),
                                tool_params.get('whitelist_signatures')
                            )

                        elif tool_name == "remove-token-whitelist":
                            res = await self.remove_token_whitelist(
                                tool_params.get('wallet_address'),
                                tool_params.get('token_addresses'),
                                tool_params.get('chain_id', 8453),
                                tool_params.get('whitelist_signatures')
                            )

                        elif tool_name == "execute-token-swap":
                            if not all([
                                tool_params.get('wallet_address'),
                                tool_params.get('safe_wallet_address'),
                                tool_params.get('input_token_address'),
                                tool_params.get('input_token_amount'),
                                tool_params.get('output_token_address')
                            ]):
                                await self.send_message(
                                    wallet_address,
                                    MessageStructure.format_error("Missing required parameters for token swap.")
                                )
                                return

                            # 執行交易前檢查
                            r1 = await self.check_multi_sig_wallet(tool_params.get('wallet_address'))
                            if not r1 or not r1.get("safeWalletList"):
                                await self.send_message(
                                    wallet_address,
                                    MessageStructure.format_error("The wallet is not a multi-sig wallet.")
                                )
                                return

                            # 檢查餘額
                            r2 = await self.get_wallet_balance(
                                tool_params.get('wallet_address'),
                                tool_params.get('chain_id')
                            )
                            if not r2:
                                await self.send_message(
                                    wallet_address,
                                    MessageStructure.format_error("Failed to verify wallet balance.")
                                )
                                return

                            # 添加到白名單
                            r3 = await self.add_token_whitelist(
                                wallet_address=tool_params.get('safe_wallet_address'),
                                chain_id=tool_params.get('chain_id', 8453),
                                whitelist_signatures=tool_params.get('whitelist_signatures'),
                                token_addresses=[tool_params.get('output_token_address')]
                            )
                            if not r3:
                                await self.send_message(
                                    wallet_address,
                                    MessageStructure.format_error("Failed to add token to whitelist.")
                                )
                                return

                            # 執行交易
                            res = await self.execute_token_swap(
                                wallet_address=tool_params.get('wallet_address'),
                                safe_wallet_address=tool_params.get('safe_wallet_address'),
                                chain_id=tool_params.get('chain_id', 8453),
                                input_token_address=tool_params.get('input_token_address'),
                                input_token_amount=tool_params.get('input_token_amount'),
                                output_token_address=tool_params.get('output_token_address'),
                                output_token_min_amount=tool_params.get('output_token_min_amount')
                            )

                        elif tool_name == "execute-portfolio-batch-trade":
                            await self.send_message(
                                wallet_address,
                                MessageStructure.format_error("Start to execute portfolio batch trades.")
                            )
                            # 用上下文檢查是否該要的資訊都有了
                            if not all([
                                tool_params.get('wallet_address'),
                                tool_params.get('safe_wallet_address'),
                                tool_params.get('chain_id'),
                                tool_params.get('portfolio'),
                                tool_params.get('whitelist_signatures')
                            ]):
                                # 讓AI拿上下文再思考一次
                                await self.send_message(
                                    wallet_address,
                                    MessageStructure.format_thinking("Reanalyzing the portfolio batch trade request...")
                                )


                        elif tool_name == "check_multi_sig_wallet":
                            res = await self.check_multi_sig_wallet(tool_params.get('wallet_address'))

                        elif tool_name == "check_multi_sig_wallet_whitelist":
                            res = await self.check_multi_sig_wallet_whitelist(
                                tool_params.get('wallet_address'),
                                tool_params.get('chain_id')
                            )

                        elif tool_name == "web_search":
                            res = self.web_search(tool_params.get('query') or user_text)

                        else:
                            res = {"error": f"Tool {tool_name} not implemented"}
                        print('res: ', res)
                        # 处理工具结果
                        if res:
                            # 工具执行成功，讓AI 思考要不要繼續思考或是回給用戶
                            # 給我一個合併上下文的prompt:
                            re_thinking_prompt = f"""
                            In a conversation about cryptocurrency operations, analyze the user's response to a tool execution.
                            
                            Previous user message: "{user_text}"
                            Previous AI message: "{previous_ai_message}"
                            Tool execution result: "{res.get('message', 'No message')}"
                            
                            Determine if the user's response indicates a need for further analysis or a direct response.
                            If further analysis is needed, respond only with "THINKING".
                            If a direct response is appropriate, respond only with "RESPONSE".
                            
                            Only respond with one of these two words.
                            """
                            response = agent.connection_manager.perform_action(
                                connection_name="anthropic",
                                action_name="generate-text",
                                params=[re_thinking_prompt, "You are a helpful cryptocurrency assistant."]
                            )

                            print('response: ', response)
                            if "THINKING" in response.upper():
                                await self.send_message(wallet_address, MessageStructure.format_thinking(f"Deep thinking..."))
                                # 回到function 本身？

                                self.rethink_count += 1
                                if self.rethink_count > 3:
                                    await self.send_message(wallet_address, MessageStructure.format_error("Too many rethinking."))
                                    return
                                await self.handle_client_message(wallet_address, message_data)
                            elif "RESPONSE" in response.upper():
                                # 工具执行成功，发送结果消息
                                await self.send_message(wallet_address, MessageStructure.format_ai_response(res.get("message", "Action successful.")))
                            else:
                                # 工具执行成功，发送结果消息
                                await self.send_message(wallet_address, MessageStructure.format_ai_response(res.get("message", "Action successful.")))

                        else:
                            # 工具执行失败, 发送错误消息
                            await self.send_message(wallet_address,
                                                    MessageStructure.format_error(res.get("message", "Action failed.")))


                    except Exception as e:
                        # 捕获工具执行过程中的异常
                        logger.exception(f"Error executing tool {tool_name}: {e}")
                        await self.send_message(wallet_address,
                                                MessageStructure.format_error(f"Error executing action: {str(e)}"))
                else:
                    # 没有选择工具, 直接用 LLM 回复
                    response = agent.connection_manager.perform_action(
                        connection_name="anthropic",
                        action_name="generate-text",
                        params=[user_text, "You are a helpful cryptocurrency assistant."]
                        # 简化 prompt, 依赖 conversation_context
                    )

                    await self.send_message(wallet_address, MessageStructure.format_ai_response(response))


        except Exception as e:
            logger.exception(f"Error processing message: {e}")
            await self.send_message(
                wallet_address,
                MessageStructure.format_error(f"Error processing your request: {str(e)}")
            )
        finally:
            self.set_agent_busy(wallet_address, False)

    async def handle_batch_request(self, wallet_address: str, user_text: str, agent: ZerePyAgent,
                                   conversation_context: dict):
        """
        Handles requests that involve batch processing of multiple items (e.g., tokens).
        """

        # 1. 提取批量操作的關鍵資訊 (使用LLM)
        batch_info = await self.extract_batch_info(user_text, agent, conversation_context)
        if not batch_info:
            await self.send_message(
                wallet_address,
                MessageStructure.format_error("Could not understand your batch request. Please provide more details.")
            )
            return

        # 2. 執行必要的預處理 (例如，安全性檢查)
        if batch_info.get("action_type") == "purchase":
            await self.send_message(wallet_address, MessageStructure.format_thinking("Analyzing token security..."))
            safe_tokens = []
            for token in batch_info.get("tokens", []):
                security_info = await self.check_token_security(
                    underlying="USDC",
                    target_token=token.get("address"),  # Assuming you have a way to get the address
                    chain_name=batch_info.get("chain", "base")
                )
                security_score = self._extract_security_score(security_info)
                if security_score >= 70:  # Your safety threshold
                    safe_tokens.append(token)
            if len(safe_tokens) == 0:
                await self.send_message(wallet_address,
                                        MessageStructure.format_ai_response("No tokens met the safety criteria."))
                return

            batch_info["tokens"] = safe_tokens  # Update with only safe tokens

        # 3. 構造批量操作的輸入
        # Get safe wallet address for this user and chain
        user_data = await self.app.state.wallet_service_client.check_multi_sig_wallet(wallet_address)
        multisig_info = user_data.get("safeWalletList", [])
        safe_wallet_address = None
        chain_id = batch_info.get("chain_id") or self._get_chain_id_from_name(batch_info.get("chain"))
        currency_address = await self._get_token_address_by_symbol(batch_info.get("currency"), chain_id)
        if not chain_id:
            await self.send_message(
                wallet_address,
                MessageStructure.format_error("Can not define the chain id.")
            )
            return

        for info in multisig_info:
            if info.get("chainId") == chain_id:
                safe_wallet_address = info.get("multisig_address")
                break

        if not safe_wallet_address:
            await self.send_message(
                wallet_address,
                MessageStructure.format_error(f"No multisig wallet found for chain ID {chain_id}.")
            )
            return
        batch_input = {
            "wallet_address": wallet_address,
            "safe_wallet_address": safe_wallet_address,
            "chain_id": chain_id,
            "actions": []
        }

        if batch_info.get("action_type") == "purchase":
            # Get currency address
            token_count = len(batch_info.get("tokens", []))
            amount_per_token = str(float(batch_info.get("amount")) / token_count)  # Ensure string for wallet service
            for token in batch_info.get("tokens", []):
                batch_input["actions"].append({
                    "type": "swap",
                    "input_token_address": currency_address,  # USDC or other
                    "input_token_amount": amount_per_token,
                    "output_token_address": token.get("address"),  # Assuming you get address in extract_batch_info
                    "output_token_symbol": token.get("symbol")
                })

        # 4. 執行批量操作 (調用 wallet service)
        if len(batch_input["actions"]) > 0:
            await self.send_message(wallet_address, MessageStructure.format_thinking("Executing batch transaction..."))

            # 先批量加入白名單
            await self.send_message(wallet_address, MessageStructure.format_thinking("Adding tokens to whitelist..."))
            token_addresses = [action['output_token_address'] for action in batch_input['actions']]
            add_whitelist_res = await self.add_token_whitelist(
                wallet_address=safe_wallet_address,
                chain_id=chain_id,
                token_addresses=token_addresses
            )

            if not add_whitelist_res:
                await self.send_message(
                    wallet_address,
                    MessageStructure.format_error("Failed to add tokens to whitelist.")
                )
                return
            # 使用模擬的批量交易執行
            try:
                # 這裡應該調用wallet service的批量交易接口
                # 但因為目前沒有實際的批量接口, 我們模擬這個過程
                # batch_result = await self.wallet_service_client.execute_batch_transaction(batch_input)

                # 模擬批量執行
                batch_result = {"results": []}
                for action in batch_input["actions"]:
                    if action["type"] == "swap":
                        swap_result = await self.execute_token_swap(
                            wallet_address=action["wallet_address"],
                            safe_wallet_address=action["safe_wallet_address"],
                            chain_id=action["chain_id"],
                            input_token_address=action["input_token_address"],
                            input_token_amount=action["input_token_amount"],
                            output_token_address=action["output_token_address"]
                        )
                    batch_result["results"].append(
                        {"success": swap_result, "token": action["output_token_symbol"],
                         "amount": action["input_token_amount"]})

                # 5.  匯報結果給用戶
                await self.summarize_and_report_batch_results(wallet_address, batch_result, agent)

            except Exception as e:
                await self.send_message(
                    wallet_address,
                    MessageStructure.format_error(f"Batch transaction failed: {str(e)}")
                )

    async def extract_batch_info(self, user_text: str, agent: ZerePyAgent, conversation_context: dict) -> dict:
        """
        Extracts key information for a batch operation using the LLM.
        """
        prompt = f"""
        Analyze the following user request and conversation history to determine if it's a batch operation request (e.g., buying multiple tokens).

        Conversation History:
        {json.dumps(conversation_context.get("last_messages", []), indent=2)}

        User Request:
        {user_text}

        If it's a batch operation, extract the following information and return it as JSON:
        {{
            "action_type": "purchase",  // or other relevant types
            "tokens": [
                {{"symbol": "TOKEN1", "address": "0xabc..."}},  // Include address if available
                {{"symbol": "TOKEN2", "address": "0xdef..."}}
            ],
            "amount": 100,          // Total amount, if specified
            "currency": "USDC",      // Currency, if specified
            "chain": "base",      // Blockchain, if specified
            "chain_id": ,      // Blockchain, if specified
            "other_details": "..."  // Any other relevant information
        }}

        If it's NOT a batch operation, return:
        {{
            "action_type": null
        }}
        """
        system_prompt = "You are an AI assistant that extracts information for batch cryptocurrency operations."

        response = agent.connection_manager.perform_action(
            connection_name="anthropic",
            action_name="generate-text",
            params=[prompt, system_prompt]
        )

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON response from LLM: {response}")
            return {}

    async def summarize_and_report_batch_results(self, wallet_address: str, batch_result: dict, agent: ZerePyAgent):
        """
        Summarizes the results of a batch operation and sends a report to the user.
        """

        summary_prompt = f"""
        You've completed a batch operation. Here are the results:

        {json.dumps(batch_result, indent=2)}

        Provide a concise summary of the operation to the user.  Be informative but not overly verbose.
        """

        system_prompt = "You are an AI assistant summarizing the results of a batch cryptocurrency operation."

        response = agent.connection_manager.perform_action(
            connection_name="anthropic",
            action_name="generate-text",
            params=[summary_prompt, system_prompt]
        )
        await self.send_message(
            wallet_address,
            MessageStructure.format_ai_response(response)
        )

    def _extract_security_score(self, security_info: dict) -> int:
        """從安全檢查結果中提取分數 (這只是一個示例，您需要根據實際響應結構調整)"""
        try:
            return int(security_info.get("vaulations", {}).get("security_score", 0))
        except (TypeError, ValueError):
            return 0

    async def check_token_price(self, symbol_pair_list: list) -> str:
        """Check the current price of a cryptocurrency token"""
        return await get_binance_tickers(symbol_pair_list)

    async def get_multi_sig_wallet(self, wallet_address: str) -> str:
        """Check if a wallet address is a multisig wallet"""
        return await self.wallet_service_client.get_multi_sig_wallet(wallet_address)

    async def check_token_security(self, underlying: str, target_token: str, chain_name: str) -> str:
        if not underlying or not target_token or not chain_name:
            return "I need both a token symbol and contract address to perform a security analysis. Please provide both."

        # return await self.third_party_client.get_security(underlying, target_token, chain_name, "100")
        mock_security_pass = {
            "status": "success",
            "message": "Token security check passed",
            "details": {
                "underlying": underlying,
                "target_token": target_token,
                "chain_name": chain_name
            },
            "vaulations": {
                "risk_level": "low",
                "security_score": random.randint(77, 94),
                "liquidity_score": random.randint(70, 92)

            }
        }
        return f"Token security check passed: {mock_security_pass}"

    async def check_multi_sig_wallet(self, wallet_address: str) -> str:
        """Check if a wallet address is a multisig wallet"""
        return await self.wallet_service_client.get_multi_sig_wallet(wallet_address)

    async def check_multi_sig_wallet_whitelist(self, wallet_address: str, chain_id: int) -> str:
        """Check if a wallet address is a multisig wallet whitelist"""
        return await self.wallet_service_client.get_multi_sig_wallet_whitelist(wallet_address, chain_id=chain_id)

    async def analyze_portfolio(self, portfolio: List = None) -> str:
        """Analyze the performance and risk of a cryptocurrency portfolio"""
        print('portfolio: ', portfolio)
        return "Portfolio analysis not implemented yet"

    async def get_hot_tokens(self, chain_id: int) -> str:
        """Get hot/trending tokens on a specific blockchain"""
        url = f"http://localhost:8000/api/get-hottest-token/{chain_id}"
        # use aiohttp to make a request
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    response = await response.json()
                    print(f"HOT on {chain_id}: ", response)
                    return response
        except Exception as e:
            print(f"Error getting hot tokens: {e}")
            return f"Error getting hot tokens: {e}"

    async def check_token_liquidity(self, query: str, **kwargs) -> dict:
        """Check the liquidity of a token on a specific chain"""
        return {"response": "Liquidity check is fine"}

    async def carv_llm_query(self, query: str, **kwargs) -> str:
        """Perform on-chain data queries using natural language"""
        return "Carv LLM query response not implemented yet"

    async def suggest_portfolio(self, query: str, amount: float = None, risk: str = None, **kwargs) -> dict:
        """Suggest a diversified cryptocurrency portfolio"""
        return "Portfolio suggestion not implemented yet"

    async def get_wallet_balance(self, wallet_address: str, chain_id: str = None, **kwargs) -> dict:
        """Check the balance of a wallet address"""
        if chain_id is None:
            balance_normal = await self.okx_client.get_all_token_balances_by_address(
                address=wallet_address,
            )
            if balance_normal['code'] != "0":
                raise Exception(f"Error getting wallet balance: {balance_normal['msg']}")
            balance_normal = balance_normal['data']
            balance_sonic = await self.wallet_service_client.get_sonic_balance(
                wallet_address=wallet_address,
            )

            pprint('balance_normal: ', balance_normal)
            pprint('balance_sonic: ', balance_sonic)
            balance = balance_normal + balance_sonic['list']
        else:
            if chain_id == 146:
                balance = await self.wallet_service_client.get_sonic_balance(
                    wallet_address=wallet_address,
                )
            else:
                balance = await self.okx_client.get_all_token_balances_by_address(
                    address=wallet_address,
                    chains=[chain_id],
                )
        return balance

    async def add_token_whitelist(self, wallet_address: str, token_addresses: List[str], chain_id: int = 8453,
                                  whitelist_signatures: List[str] = None,
                                  **kwargs) -> str:
        # body = {
        #     "chain_id": chain_id,
        #     "safe_wallet_address": wallet_address,
        #     "token_addresses": token_addresses,
        #     "whitelist_signatures": whitelist_signatures
        # }
        # mock = {
        #     "status": "success",
        #     "message": "Token whitelist updated successfully"
        # }
        #
        # print('body: ', body)
        # return mock
        """Add tokens to the multisig wallet whitelist"""
        return await self.wallet_service_client.add_multi_sig_wallet_whitelist(
            safe_wallet_address=wallet_address,
            chain_id=chain_id,
            whitelist_signatures=whitelist_signatures,
            token_addresses=token_addresses
        )

    async def remove_token_whitelist(self, wallet_address: str, token_addresses: List[str], chain_id: int = 8453,
                                     whitelist_signatures: List[str] = None,
                                     **kwargs) -> str:
        """Remove tokens from the multisig wallet whitelist"""
        return await self.wallet_service_client.remove_multi_sig_wallet_whitelist(
            safe_wallet_address=wallet_address,
            chain_id=chain_id,
            whitelist_signatures=whitelist_signatures,
            token_addresses=token_addresses
        )

    async def execute_token_swap(self, wallet_address: str, safe_wallet_address: str, input_token_address: str,
                                 input_token_amount: str, output_token_address: str,
                                 output_token_min_amount: str = None,
                                 chain_id: int = 8453, **kwargs) -> str:
        """Execute a token swap through a multisig wallet"""
        # body = {
        #     "chain_id": chain_id,
        #     "ai_address": wallet_address,
        #     "safe_wallet_address": safe_wallet_address,
        #     "input_token_address": input_token_address,
        #     "input_token_amount": input_token_amount,
        #     "output_token_address": output_token_address,
        #     "output_token_min_amount": output_token_min_amount
        # }
        # mock_return = {
        #     "status": "success",
        #     "message": "Token swap executed successfully"
        # }
        # print('body: ', body)
        # print('mock_return: ', mock_return)
        # return mock_return
        return await self.wallet_service_client.multi_sig_wallet_swap(
            chain_id=chain_id,
            ai_address=wallet_address,
            safe_wallet_address=safe_wallet_address,
            input_token_address=input_token_address,
            input_token_amount=input_token_amount,
            output_token_address=output_token_address,
            output_token_min_amount=output_token_min_amount
        )

    def web_search(self, query: str, **kwargs) -> str:
        """Perform a web search"""
        return google_ai_search(query)

    async def _ping_clients(self):
        """Periodically ping clients to keep connections alive and check for timeouts."""
        try:
            while True:
                if not self.active_connections:
                    await asyncio.sleep(self.ping_interval)
                    continue

                logger.debug(f"Pinging {len(self.active_connections)} active clients")
                disconnected_clients = []

                for wallet_address, websocket in self.active_connections.items():
                    try:
                        await websocket.send_json({"type": "ping", "timestamp": datetime.now().isoformat()})

                        # Check for session timeout
                        if wallet_address in self.user_sessions:
                            last_time = self.user_sessions[wallet_address]["last_message_time"]
                            if datetime.now() - last_time > timedelta(seconds=self.inactivity_timeout):
                                logger.info(f"Session timeout for {wallet_address}")
                                disconnected_clients.append(wallet_address)
                    except WebSocketDisconnect:
                        logger.info(f"Client {wallet_address} disconnected (ping)")
                        disconnected_clients.append(wallet_address)
                    except Exception as e:
                        logger.warning(f"Error pinging client {wallet_address}: {e}")
                        disconnected_clients.append(wallet_address)

                for wallet_address in disconnected_clients:
                    self.disconnect(wallet_address)

                await asyncio.sleep(self.ping_interval)

        except asyncio.CancelledError:
            logger.info("Ping task cancelled")
        except Exception as e:
            logger.error(f"Error in ping task: {e}")


async def generate_content(text):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=" + os.getenv(
        "GEMINI_API_KEY")
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "contents": [{
            "parts": [{"text": text}]
        }]
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=data) as response:
            return await response.json()


def google_ai_search(input_text: str):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=input_text,
        config=types.GenerateContentConfig(
            tools=[types.Tool(
                google_search=types.GoogleSearchRetrieval(
                    dynamic_retrieval_config=types.DynamicRetrievalConfig(
                        dynamic_threshold=0.1))
            )]
        )
    )
    # use ai again to understand the response and give a concise answer
    res_1 = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=str(response),
        # config=types.GenerateContentConfig(
        #     tools=[types.Tool(
        #         google_search=types.GoogleSearchRetrieval(
        #             dynamic_retrieval_config=types.DynamicRetrievalConfig(
        #                 dynamic_threshold=0.1))
        #     )]
        # )
    )
    """
    candidates=[Candidate(content=Content(parts=[Part(video_metadata=None, thought=None, code_execution_result=None, executable_code=None, file_data=None, function_call=None, function_response=None, inline_data=None, text="The model's response indicates that it cannot access or interact with content from URLs. Therefore, it cannot fulfill the request to identify the top 5 tokens from the provided link.\n")], role='model'), citation_metadata=None, finish_message=None, token_count=None, avg_logprobs=-0.2322765556541649, finish_reason='STOP', grounding_metadata=None, index=None, logprobs_result=None, safety_ratings=None)] model_version='gemini-2.0-flash' prompt_feedback=None usage_metadata=GenerateContentResponseUsageMetadata(cached_content_token_count=None, candidates_token_count=37, prompt_token_count=276, total_token_count=313) automatic_function_calling_history=[] parsed=None
    """
    # make the response readable in the console like a human
    res_1 = str(res_1).replace("candidates", "\nCandidates").replace("model_version", "\nModel Version").replace(
        "prompt_feedback", "\nPrompt Feedback").replace("usage_metadata", "\nUsage Metadata").replace(
        "automatic_function_calling_history", "\nAutomatic Function Calling History").replace("parsed", "\nParsed")

    return res_1


# --- Server Initialization ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the server and clean up resources on shutdown."""

    # Initialize MongoDB
    mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    database_name = os.getenv("DATABASE_NAME", "zerepy_db")
    db_client = MongoDBClient(mongodb_url, database_name)
    await db_client.initialize_indexes()
    app.state.db_client = db_client  # Store db_client

    # Initialize context manager and structured response handler
    from src.server.context_manager import ContextManager
    from src.server.structured_response_handler import StructuredResponseHandler

    app.state.context_manager = ContextManager()
    app.state.response_handler = StructuredResponseHandler()

    def signal_handler():
        logger.info("Received close signal, closing connections...")

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, lambda signum, frame: signal_handler())
        except ValueError:
            # If we're not in the main thread, this will fail
            pass
    print("Server initialized successfully")
    print(os.getenv("OKX_WEB3_PROJECT_ID"))
    # Initialize API clients
    okx_client = OkxWeb3Client(
        project_id=os.getenv("OKX_WEB3_PROJECT_ID"),
        api_key=os.getenv("OKX_WEB3_PROJECT_KEY"),
        api_secret=os.getenv("OKX_WEB3_PROJECT_SECRET"),
        api_passphrase=os.getenv("OKX_WEB3_PROJECT_PASSWRD"),
    )
    third_party_client = ThirdPartyClient()
    cave_client = CaveClient(os.getenv("CAVE_API_KEY"))
    wallet_service_client = WalletServiceClient()
    await okx_client.initialize()
    await third_party_client.initialize()
    try:
        await cave_client.initialize()
    except Exception as e:
        logger.error(f"Error initializing CaveClient: {e}")

    app.state.okx_client = okx_client
    app.state.third_party_client = third_party_client
    app.state.cave_client = cave_client
    app.state.wallet_service_client = wallet_service_client

    print("Server initialized successfully")

    # Create MultiClientManager, passing the db_client
    app.state.connection_manager = MultiClientManager(
        db_client=db_client,
        okx_client=okx_client,
        third_party_client=third_party_client,
        wallet_service_client=wallet_service_client,
        cave_client=cave_client
    )

    logger.info("Server initialized successfully")
    yield
    logger.info("Server shutting down")
    try:
        await okx_client.close()
        await third_party_client.close()
        await cave_client.close()
        await wallet_service_client.close()
        logger.info("Server shutdown complete")
    except Exception as e:
        logger.error(f"Error shutting down server: {e}")


class ZerePyServer:
    def __init__(self):
        """Initialize the ZerePy server."""
        self.app = FastAPI(
            title="ZerePy Server",
            description="ZerePy Agent Framework API Server",
            version="0.1.0",
            lifespan=lifespan
        )

        # Configure CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self.setup_routes()

    def setup_routes(self):
        """Set up the API routes."""

        # Health check endpoint
        @self.app.get("/")
        async def health():
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat()
            }

        # WebSocket endpoint for client connections
        @self.app.websocket("/ws/{wallet_address}")
        async def websocket_endpoint(
                websocket: WebSocket,
                wallet_address: str,
                agent_name: str = Query(default="crypto_agent")
        ):
            wallet_address = wallet_address.lower()
            connection_manager = self.app.state.connection_manager

            # Connect the client
            success = await connection_manager.connect(wallet_address, websocket, agent_name)
            if not success:
                return

            # Send welcome message
            await connection_manager.send_message(
                wallet_address,
                MessageStructure.format_ai_response(
                    f"Welcome! I'm your Gun.Ai assistant. How can I help you today?")
            )

            # Handle messages
            try:
                while True:
                    data = await websocket.receive_text()
                    try:
                        message_data = json.loads(data)
                        await connection_manager.handle_client_message(wallet_address, message_data)
                    except json.JSONDecodeError:
                        await connection_manager.send_message(
                            wallet_address,
                            MessageStructure.format_error("Invalid message format. Please send valid JSON.")
                        )
            except WebSocketDisconnect:
                connection_manager.disconnect(wallet_address)
            except Exception as e:
                logger.exception(f"Unexpected error in websocket handler: {e}")
                try:
                    await connection_manager.send_message(
                        wallet_address,
                        MessageStructure.format_error(f"Unexpected error: {str(e)}")
                    )
                except:
                    pass
                connection_manager.disconnect(wallet_address)

        # List available agents
        @self.app.get("/api/agents", tags=['agent'])
        async def list_agents():
            try:
                agents = []
                agents_dir = Path("agents")  # 假設 agents 目錄在專案根目錄
                if agents_dir.exists():
                    for agent_file in agents_dir.glob("*.json"):
                        # 排除 "general" 或您想要排除的其他檔案
                        if agent_file.stem != "general":
                            agents.append(agent_file.stem)
                return {"agents": agents}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/api/user/{wallet_address}", tags=['user'])
        async def get_user_info(wallet_address: str):
            wallet_address = wallet_address.lower()
            user = await self.app.state.db_client.get_user(wallet_address)
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            return user

        @self.app.get("/api/user-status", tags=['user'])
        async def check_user_status(wallet_address: str):
            """檢查用戶狀態API端點"""
            wallet_address = wallet_address.lower()
            user = await self.app.state.db_client.get_user(wallet_address)
            if not user:
                return {"has_agent": False, "multisig_info": None}
            mul_sig_wallets = await self.app.state.wallet_service_client.check_multi_sig_wallet(wallet_address)
            return {
                "has_agent": user.get("has_agent", False),
                "multisig_info": mul_sig_wallets['safeWalletList']
            }

        # Create user if not exists and get status
        @self.app.post("/api/user", tags=['user'])
        async def create_or_get_user(data: BaseAiWalletAddress):
            wallet_address = data.wallet_address.lower()
            user = await self.app.state.db_client.get_user(wallet_address)

            if not user:
                user = await self.app.state.db_client.create_user(wallet_address)

            return {
                "wallet_address": user["wallet_address"],
                "created_at": user["created_at"],
                "has_agent": user.get("has_agent", False),
                "agent_id": user.get("agent_id")
            }

        # 2. Agent相關端點
        @self.app.get("/api/conversation/{wallet_address}", tags=['agent'])
        async def get_conversation(data: BaseAiWalletAddress):
            """Get conversation history API endpoint"""
            wallet_address = data.wallet_address.lower()
            history = await self.app.state.db_client.get_conversation_history(wallet_address)
            return {"messages": history}

        @self.app.post("/api/create-agent", tags=['agent'])
        # async def create_agent(wallet_address: str, agent_name: str = Query(default="crypto_agent")):
        async def create_agent(data: CreateAgent):
            wallet_address = data.wallet_address.lower()
            print(wallet_address)
            print('-----------------')
            user = await self.app.state.db_client.get_user(wallet_address)

            if not user:
                user = await self.app.state.db_client.create_user(wallet_address)

            # Check if user already has an agent
            if user.get("has_agent"):
                return {
                    "success": True,
                    "message": "Agent already exists",
                    "agent_id": user.get("agent_id")
                }

            # Create agent address
            try:
                res = await self.app.state.wallet_service_client.create_ai_wallet(
                    wallet_address
                )
                # aiohttp.client_exceptions.ClientResponseError: 409, message='Conflict' = mean the wallet already exists
            except Exception as e:
                if e.status == 409:
                    res = await self.app.state.wallet_service_client.get_ai_wallet_address(wallet_address)
                else:
                    logger.error(f"Error creating agent ai wallet address: {e}")
                    raise HTTPException(status_code=500, detail=f"Failed to create agent ai wallet address: {str(e)}")

            ai_agent_address = res.get("aiAddress", None)
            if not ai_agent_address:
                raise HTTPException(status_code=500, detail="Failed to create agent ai wallet address failed.")

            # Create agent
            agent_id = str(uuid4())
            await self.app.state.db_client.db.users.update_one(
                {"wallet_address": wallet_address},
                {"$set": {
                    "agent_id": agent_id,
                    "has_agent": True,
                    "agent_name": data.agent_name,
                    "agent_address": ai_agent_address,
                    "multisig_info": [],  # Add multisig info here, but not in this endpoint
                    "created_at": datetime.now(),
                    "last_active": datetime.now()
                }}
            )

            return {
                "success": True,
                "message": "Agent created successfully",
                "agent_id": agent_id,
                "agent_name": data.agent_name  # 返回 agent_name
            }

        @self.app.get("/api/agent-status/{wallet_address}", tags=['agent'])
        async def get_agent_status(wallet_address: str):
            """獲取Agent狀態API端點"""
            wallet_address = wallet_address.lower()
            user = await self.app.state.db_client.get_user(wallet_address)

            if user:
                mul_sig_wallets = await self.app.state.wallet_service_client.check_multi_sig_wallet(wallet_address)
                return {
                    "has_agent": user.get("has_agent", False),
                    "multisig_info": mul_sig_wallets['safeWalletList']
                }
            else:
                return {"has_agent": False, "multisig_info": None}

        # 3. 錢包和交易相關端點
        @self.app.post("/api/register-multi-sig-wallet", tags=['multisig'])
        async def wallet_register(data: RegisterMultiSigWallet):
            """註冊錢包地址"""
            wallet_address = data.wallet_address.lower()
            if not wallet_address:
                raise HTTPException(status_code=400, detail="ownerAddress is required")

            try:
                # 嘗試使用錢包服務創建多簽錢包
                multisig_address = await self.app.state.wallet_service_client.create_multi_sig_wallet(
                    wallet_address,
                    data.chain_id
                )

                # update user record, since multisig_info is a list, should check if duplicate and append it
                await self.app.state.db_client.db.users.update_one(
                    {"wallet_address": wallet_address},
                    {"$addToSet": {"multisig_info": {"multisig_address": multisig_address, "chain_id": data.chain_id}}}
                )

                return {"success": True, "message": "Wallet registered successfully."}
            except Exception as e:
                logger.error(f"Error registering wallet: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to register wallet: {str(e)}")

        @self.app.post("/api/get-multisig-wallets", tags=['multisig'])
        async def get_multisig_wallets(wallet_address: str):
            """獲取用戶的多簽錢包"""
            wallet_address = wallet_address.lower()

            # 檢查用戶是否存在
            user = await self.app.state.wallet_service_client.get_multi_sig_wallet(wallet_address)
            if not user:
                raise HTTPException(status_code=404, detail="User not found")

            # 檢查用戶是否有多簽錢包
            multisig_address = user.get("multisig_address")
            if not multisig_address:
                raise HTTPException(status_code=404, detail="Multisig wallet not deployed for this user")

            return {"multisig_address": multisig_address}

        # 4. 資產和餘額相關端點
        @self.app.post("/api/total-balance", tags=['wallet'])
        async def get_total_balance(data: dict):
            """獲取總餘額"""
            wallet_address = data.get("wallet_address")
            chain_id_list = data.get("chain_id_list")

            if not wallet_address:
                raise HTTPException(status_code=400, detail="wallet_address is required")

            try:
                return await self.app.state.okx_client.get_total_value_by_address(
                    wallet_address,
                    chain_id_list
                )
            except Exception as e:
                logger.error(f"Error getting total balance: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get total balance: {str(e)}")

        @self.app.post("/api/total-balance-detail", tags=['wallet'])
        async def get_total_balance_detail(data: GetBalanceDetails):
            """獲取詳細餘額信息"""
            wallet_address = data.wallet_address
            chain_id_list = data.chain_id_list

            if not wallet_address:
                raise HTTPException(status_code=400, detail="wallet_address is required")

            try:
                balance_normal = await self.app.state.okx_client.get_all_token_balances_by_address(
                    wallet_address,
                    chain_id_list,
                    filter_risk_tokens=True
                )
                if int(balance_normal['code']) != 0:
                    raise Exception(f"Error getting wallet balance: {balance_normal['msg']}")
                balance_sonic = await self.app.state.wallet_service_client.get_sonic_balance(
                    wallet_address=wallet_address,
                )

                return await self.app.state.okx_client.process_token_data(balance_normal, sonic_data=balance_sonic)
            except Exception as e:
                logger.error(f"Error getting balance details: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get balance details: {str(e)}")

        @self.app.post("/api/wallet-transaction-history", tags=['wallet'])
        async def get_wallet_transaction_history(data: GetTransactionHistory):
            """獲取錢包交易歷史"""
            wallet_address = data.wallet_address
            chain_id_list = data.chain_id_list

            if not wallet_address:
                raise HTTPException(status_code=400, detail="wallet_address is required")

            try:
                result = await self.app.state.okx_client.get_transactions_by_address(
                    wallet_address,
                    chain_id_list
                )

                if not result['data']:
                    return []

                return await self.app.state.okx_client.process_transaction_data(result, wallet_address)
            except Exception as e:
                logger.error(f"Error getting transaction history: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get transaction history: {str(e)}")

        """ === Tool API Endpoints === """

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}

        @self.app.get("/api/tools", tags=['tools'])
        async def get_available_tools():
            """Get list of available tools"""
            tools = []
            for tool_name, tool_func in action_registry.items():
                if tool_name.startswith("_"):
                    continue
                doc = tool_func.__doc__ or "No description available."
                tools.append({"name": tool_name, "description": doc.strip()})
            return {"tools": tools}

        @self.app.post("/api/tools/execute", tags=['tools'])
        async def execute_tool(
                wallet_address: str,
                tool_name: str,
                params: Dict[str, Any]
        ):
            """Execute a specific tool with parameters"""
            wallet_address = wallet_address.lower()
            connection_manager = self.app.state.connection_manager

            # Check if agent exists
            agent = connection_manager.get_agent_for_user(wallet_address)
            if not agent:
                raise HTTPException(status_code=404, detail="Agent not found")

            # Check if tool exists in action registry
            if tool_name not in action_registry:
                raise HTTPException(status_code=404, detail=f"Tool {tool_name} not found")

            try:
                # Execute the requested tool
                result = action_registry[tool_name](agent, **params)
                return {
                    "success": True,
                    "result": result,
                    "tool": tool_name,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logger.exception(f"Error executing tool {tool_name}: {e}")
                raise HTTPException(status_code=500, detail=f"Error executing tool: {str(e)}")

        @self.app.get("/api/get-hottest-token/{chain_id}", tags=['tools'])
        async def get_hottest_token(chain_id: int):
            """Get the hottest token"""
            if chain_id == 8453:
                return {
                    "chainId": 8453,
                    "assets": [
                        {
                            "symbol": "DRB",
                            "address": "0x3ec2156d4c0a9cbdab4a016633b7bcf6a8d68ea2",
                            "chain": "base",
                            "chain_id": 8453,
                            "24_vol": 1733212,
                        },
                        {
                            "symbol": "Cocoro",
                            "address": "0x937a1cfaf0a3d9f5dc4d0927f72ee5e3e5f82a00",
                            "chain": "base",
                            "chain_id": 8453,
                            "24_vol": 15125323,
                        },
                        {
                            "symbol": "GRK",
                            "address": "0x2e2cc4dfce60257f091980631e75f5c436b71c87",
                            "chain": "base",
                            "chain_id": 8453,
                            "24_vol": 13733212,
                        },
                        {
                            "symbol": "GrokCoin",
                            "address": "0xfdc6abaa69f0df71ff4bb6bdfec34188796ee07a",
                            "chain": "base",
                            "chain_id": 8453,
                            "24_vol": 11433822,
                        },

                        {
                            "symbol": "SKITTEN",
                            "address": "0x4b6104755afb5da4581b81c552da3a25608c73b8",
                            "chain": "base",
                            "chain_id": 8453,
                            "24_vol": 1733212,
                        },
                        {
                            "symbol": "BRAIN",
                            "address": "0x3ec2156d4c0a9cbdab4a016633b7bcf6a8d68ea2",
                            "chain": "base",
                            "chain_id": 8453,
                            "24_vol": 1733212,
                        },
                        {
                            "symbol": "VIRTUAL",
                            "address": "0x0b3e328455c4059eeb9e3f84b5543f74e24e7e1b",
                            "chain": "base",
                            "chain_id": 8453,
                            "24_vol": 1033212,
                        },
                        {
                            "symbol": "Akio",
                            "address": "0xbb375a3dfd3e8efa8239e9e2061cf7599636666b",
                            "chain": "base",
                            "chain_id": 8453,
                            "24_vol": 9374742,
                        },
                    ]
                }
            else:
                return await self.app.state.wallet_service_client.get_asset_info_with_price(chain_id)

        """ === Public Data API Endpoints === """

        @self.app.get("/api/public-data/tickers", tags=['public_data'])
        async def get_tickers():
            """獲取行情數據"""
            try:
                tickers = await get_binance_tickers(["BTCUSDT", "ETHUSDT", "SOLUSDT"])
                return {"tickers": tickers}
            except Exception as e:
                logger.error(f"Error fetching tickers: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to fetch tickers: {str(e)}")

        @self.app.get("/api/messages", tags=['message'])
        async def get_user_messages(wallet_address: str, show_history: bool = False):
            """獲取用戶消息"""
            wallet_address = wallet_address.lower()

            # 獲取對話歷史
            conversation = await self.app.state.db_client.db.conversations.find_one(
                {"wallet_address": wallet_address})

            if conversation and "messages" in conversation:
                messages = conversation["messages"]

                if show_history:
                    # 返回所有非"thinking"類型的消息
                    return {"messages": [msg for msg in messages if msg.get("message_type") != "thinking"]}
                else:
                    # 只返回最近的一條normal類型消息
                    for msg in reversed(messages):
                        if msg.get("message_type") == "normal":
                            return {"messages": [msg]}
                    return {"messages": []}

            return {"messages": []}

        # Send a message via REST API (non-WebSocket option)
        @self.app.post("/api/message", tags=['message'])
        async def send_message(wallet_address: str, message: Dict[str, Any]):
            wallet_address = wallet_address.lower()
            connection_manager = self.app.state.connection_manager
            context_manager = self.app.state.context_manager
            response_handler = self.app.state.response_handler

            # Check if agent exists for this user
            agent = connection_manager.get_agent_for_user(wallet_address)

            if not agent:
                try:
                    # Try to load agent from user record
                    user = await self.app.state.db_client.get_user(wallet_address)
                    if not user or not user.get("has_agent"):
                        raise HTTPException(status_code=404, detail="User has no agent configured")

                    agent_name = user.get("agent_name", "crypto_agent")
                    agent = ZerePyAgent(agent_name)
                    connection_manager.set_agent_for_user(wallet_address, agent)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Failed to load agent: {str(e)}")

            if connection_manager.is_agent_busy(wallet_address):
                raise HTTPException(status_code=429, detail="Agent is busy processing another request")

            connection_manager.set_agent_busy(wallet_address, True)

            try:
                user_text = message.get("text", "")

                # Update context with new message
                await context_manager.add_message(wallet_address, {
                    "sender": "user",
                    "text": user_text
                })

                # Determine thinking level
                thinking_level = await context_manager.get_thinking_level(wallet_address, user_text)

                # Save user message to database
                user_message = MessageStructure.format_message("user", user_text)
                await self.app.state.db_client.save_message(wallet_address, user_message)

                # Make sure the LLM provider is set up
                if not agent.is_llm_set:
                    agent._setup_llm_provider()

                # Generate response - potentially using appropriate tools
                response = agent.prompt_llm(user_text)

                # For medium/deep thinking, create structured response
                if thinking_level in ["medium", "deep"]:
                    structured_response = await response_handler.generate_structured_response(
                        query=user_text,
                        raw_response=response,
                        thinking_level=thinking_level
                    )

                    # For API responses, use final answer but store full structure
                    response = structured_response["final_answer"]

                    # Store structured response in context
                    await context_manager.update_context(wallet_address, {
                        "last_structured_response": structured_response
                    })

                # Save the agent's response to database
                agent_message = MessageStructure.format_ai_response(response)
                await self.app.state.db_client.save_message(wallet_address, agent_message)

                return agent_message
            except Exception as e:
                logger.exception(f"Error processing REST message: {e}")
                raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")
            finally:
                connection_manager.set_agent_busy(wallet_address, False)


def create_app():
    """Create and return the FastAPI app."""
    server = ZerePyServer()
    return server.app


def start_server(host="0.0.0.0", port=8000):
    """Start the server."""
    app = create_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
