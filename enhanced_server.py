# !/usr/bin/env python3
# enhanced_server.py
import asyncio
import json
import logging
import os
import sys
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

# Analyze thinking depth based on query complexity
from src.thinking_framework.thinking_depth_selector import analyze_thinking_depth
from src.server.structured_response_handler import StructuredResponseHandler

# --- External API Imports ---
from backend.dex_api_client.wallet_service_client import WalletServiceClient
from backend.dex_api_client.okx_web3_client import OkxWeb3Client
from backend.dex_api_client.third_client import ThirdPartyClient
from backend.dex_api_client.cave_client import CaveClient
from backend.dex_api_client.public_data import get_binance_tickers

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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


# --- MultiClientManager class for handling multiple WebSocket connections ---
class MultiClientManager:
    def __init__(self, db_client: MongoDBClient, ping_interval: int = 30, inactivity_timeout: int = 1800):
        """Initialize the MultiClientManager."""
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_sessions: Dict[str, Dict[str, Any]] = {}
        self.user_agents: Dict[str, ZerePyAgent] = {}
        self.ping_interval = ping_interval
        self.inactivity_timeout = inactivity_timeout
        self.ping_task = None
        self.db_client = db_client

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

    async def handle_client_message(self, wallet_address: str, message_data: Dict[str, Any]):
        """Process an incoming client message with AI-driven tool selection and thinking depth evaluation."""
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
                return

            # Ensure LLM provider is set up
            if not agent.is_llm_set:
                agent._setup_llm_provider()

            # Get conversation context and wallet information
            conversation_history = await self.db_client.get_conversation_history(wallet_address)

            # Check if this is a crypto wallet interaction
            user_data = await self.db_client.get_user(wallet_address)
            has_multisig = user_data and user_data.get("multisig_info") and len(user_data.get("multisig_info", [])) > 0

            conversation_context = {
                "conversation_length": len(conversation_history),
                "last_messages": conversation_history[-5:] if len(conversation_history) >= 5 else conversation_history,
                "wallet_address": wallet_address,
                "has_multisig_wallet": has_multisig,
                "domain": "crypto"  # Indicate this is a cryptocurrency context
            }
            print('conversation_context: ', conversation_context)

            # Use LLM to determine thinking depth
            thinking_result = await analyze_thinking_depth(user_text, agent, conversation_context)
            thinking_depth = thinking_result["depth"]
            detected_intents = thinking_result["detected_intents"]

            print('thinking_result: ', thinking_result)
            print('thinking_depth: ', thinking_depth)
            print('detected_intents: ', detected_intents)

            # Transaction-related queries require additional preparation steps
            transaction_intents = ["transaction_intent", "defi_operation", "whitelist_operation",
                                   "chain_interaction", "multisig_operation"]
            has_transaction_intent = any(intent in transaction_intents for intent in detected_intents)

            print('has_transaction_intent: ', has_transaction_intent)
            # For transaction intents, force deep thinking and prepare necessary services
            if has_transaction_intent and thinking_depth != "deep":
                logger.warning(f"Transaction intent detected but thinking depth was {thinking_depth}. Forcing to DEEP.")
                thinking_depth = "deep"

            # Log the thinking depth and intents for debugging
            logger.info(f"Thinking depth determined for query: {thinking_depth}")
            logger.info(f"Detected intents: {detected_intents}")

            # For deep thinking transaction operations, do additional setup
            if thinking_depth == "deep" and has_transaction_intent:
                # Add a preliminary thinking step to show user we're preparing a thorough response
                await self.send_message(
                    wallet_address,
                    MessageStructure.format_thinking("This requires careful analysis. Preparing a detailed response...")
                )

            # Create prompt for tool selection
            tools_prompt = f"""Based on the user's query, determine if you need to use a specific tool to respond.
    The query requires {thinking_depth} thinking depth and has these intents: {', '.join(detected_intents) if detected_intents else 'no specific intent'}.

    Available tools:
    """

            # List available tools with descriptions
            available_tools = []
            for tool_name, tool_func in action_registry.items():
                if not tool_name.startswith("_") and hasattr(tool_func, "__doc__") and tool_func.__doc__:
                    tool_desc = tool_func.__doc__.strip().split("\n")[0]  # Get first line of docstring
                    tools_prompt += f"- {tool_name}: {tool_desc}\n"
                    available_tools.append(tool_name)

            tools_prompt += "\nIf a tool is needed, respond with JSON in this format: {\"tool\": \"tool_name\", \"params\": {\"param1\": \"value1\"}}"
            tools_prompt += "\nIf no tool is needed, respond with {\"tool\": null}"
            tools_prompt += f"\nUser query: {user_text}"

            # Ask the LLM to decide which tool to use
            system_prompt = "You are an AI assistant that analyzes user requests and determines which tools to use."
            tool_selection_response = agent.connection_manager.perform_action(
                connection_name="anthropic",
                action_name="generate-text",
                params=[tools_prompt, system_prompt]
            )

            print('tool_selection_response77777: ', tool_selection_response)
            # Try to parse JSON response to get tool selection
            tool_name = None
            tool_params = {}
            try:
                import re
                import json

                # Extract JSON from the response (it might be wrapped in text)
                json_match = re.search(r'\{[\s\S]*\}', tool_selection_response)
                if json_match:
                    tool_data = json.loads(json_match.group(0))
                    tool_name = tool_data.get("tool")
                    tool_params = tool_data.get("params", {})

                    # Add the user text as a parameter if not already present
                    if tool_name and "query" not in tool_params:
                        tool_params["query"] = user_text
            except Exception as e:
                logger.warning(f"Failed to parse tool selection: {e}")

            # Generate the response - either using the selected tool or standard generation
            if tool_name and tool_name in available_tools:
                logger.info(f"Using tool {tool_name} with params {tool_params}")

                # Execute the selected tool
                response = agent.connection_manager.perform_action(
                    connection_name="crypto_tools",
                    action_name=tool_name,
                    params=tool_params
                )

                print('tool_selection_response8888: ', tool_selection_response)
            else:
                # Standard text generation for queries not requiring specific tools
                logger.info(f"No specific tool selected, using standard text generation with {thinking_depth} depth")

                # Craft system prompt that includes thinking depth guidance and crypto-specific instructions
                base_system_prompt = agent._construct_system_prompt()

                # Add crypto-specific instruction based on thinking depth and intents
                crypto_instructions = ""
                if "transaction_intent" in detected_intents:
                    crypto_instructions = """
    As a crypto assistant, prioritize security and thorough analysis for transaction-related queries:
    1. Check if the token exists on the specified chain
    2. Evaluate token security and legitimacy 
    3. Check for sufficient liquidity
    4. Consider gas costs and slippage
    5. Always warn about potential risks
    6. For direct transactions, explain all the necessary steps
    """
                elif "security_assessment" in detected_intents:
                    crypto_instructions = """
    When evaluating token or contract security:
    1. Check contract code patterns if available
    2. Look for red flags like unlimited minting, hidden owners, or transfer restrictions
    3. Consider liquidity depth and token distribution
    4. Check for audit status
    5. Be appropriately cautious in assessment
    """

                # Add thinking depth guidance
                enhanced_system_prompt = f"{base_system_prompt}\n{crypto_instructions}\n\nThis query requires {thinking_depth} thinking depth. "

                if thinking_depth == "light":
                    enhanced_system_prompt += "Provide a concise, factual answer focused on the specific information requested."
                elif thinking_depth == "medium":
                    enhanced_system_prompt += "Provide a balanced response with relevant market context and moderate reasoning."
                else:  # deep
                    enhanced_system_prompt += """Provide a comprehensive response with:
    1. Thorough analysis broken into clear sections
    2. Security considerations and risk assessment
    3. Step-by-step guidance where appropriate
    4. Explicit warnings about potential risks
    5. Clear conclusions and recommendations"""

                # For transaction intents, use a specialized system prompt
                if has_transaction_intent and thinking_depth == "deep":
                    # Show the user we're doing a thorough analysis
                    await self.send_message(
                        wallet_address,
                        MessageStructure.format_thinking("Analyzing token security and transaction parameters...")
                    )

                response = agent.connection_manager.perform_action(
                    connection_name="anthropic",
                    action_name="generate-text",
                    params=[user_text, enhanced_system_prompt]
                )
                print('response0000: ', response)

            # For medium/deep thinking, structure the response
            if thinking_depth in ["medium", "deep"]:
                response_handler = StructuredResponseHandler()
                structured_response = await response_handler.generate_structured_response(
                    query=user_text,
                    raw_response=response,
                    thinking_level=thinking_depth
                )
                print('response1111: ', response)

                # Use the final answer from the structured response
                response = structured_response["final_answer"]

                print('response22222 : ', response)
            # Send response
            await self.send_message(
                wallet_address,
                MessageStructure.format_ai_response(response)
            )

        except Exception as e:
            logger.exception(f"Error processing message: {e}")
            await self.send_message(
                wallet_address,
                MessageStructure.format_error(f"Error processing your request: {str(e)}")
            )
        finally:
            self.set_agent_busy(wallet_address, False)

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

    # Create the connection manager
    app.state.connection_manager = MultiClientManager(db_client)

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
    app.state.connection_manager = MultiClientManager(db_client=db_client)

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
                    f"Welcome! I'm your {agent_name} assistant. How can I help you today?")
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
            return {
                "has_agent": user.get("has_agent", False),
                "multisig_info": user.get("multisig_info")
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
                return {
                    "has_agent": user.get("has_agent", False),
                    "multisig_info": user.get("multisig_info"),
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
        async def get_total_balance_detail(data: dict):
            """獲取詳細餘額信息"""
            wallet_address = data.get("wallet_address")
            chain_id_list = data.get("chain_id_list")

            if not wallet_address:
                raise HTTPException(status_code=400, detail="wallet_address is required")

            try:
                result = await self.app.state.okx_client.get_all_token_balances_by_address(
                    wallet_address,
                    chain_id_list,
                    filter_risk_tokens=True
                )

                if not result['data']:
                    return []

                return await self.app.state.okx_client.process_token_data(result)
            except Exception as e:
                logger.error(f"Error getting balance details: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get balance details: {str(e)}")

        @self.app.post("/api/wallet-transaction-history", tags=['wallet'])
        async def get_wallet_transaction_history(data: dict):
            """獲取錢包交易歷史"""
            wallet_address = data.get("wallet_address")
            chain_id_list = data.get("chain_id_list")

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
