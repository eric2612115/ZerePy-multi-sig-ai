# !/usr/bin/env python3
# enhanced_server.py
import asyncio
import json
import logging
import os
import sys
import signal
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from uuid import uuid4
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import ZerePy modules
from src.agent import ZerePyAgent
from src.connection_manager import ConnectionManager as ZerePyConnectionManager
from src.helpers import print_h_bar

# Additional imports for server functionality
from src.server.mongodb_client import MongoDBClient

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("zerepy_server")


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

    async def connect(self, wallet_address: str, websocket: WebSocket, agent_name: str = "timetool_agent"):
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
        """Process an incoming client message."""
        logger.debug(f"Handling message for {wallet_address}: {message_data}")
        print('-----------------')
        print("message_data: ", message_data, "wallet_address: ", wallet_address)
        print(self.user_sessions)

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

        print('-------- Set agent busy --------')

        try:
            # 獲取用戶訊息文本
            user_text = message_data.get("query", "")
            print('user_text: ', message_data)
            if not user_text:
                await self.send_message(
                    wallet_address,
                    MessageStructure.format_error("Empty message received")
                )
                return

            # 首先保存用戶訊息到資料庫
            user_message = MessageStructure.format_message("user", user_text)
            await self.db_client.save_message(wallet_address, user_message)

            # 發送思考中訊息
            await self.send_message(
                wallet_address,
                MessageStructure.format_thinking("Processing your request...")
            )

            # 獲取此用戶的agent
            agent = self.get_agent_for_user(wallet_address)
            if not agent:
                await self.send_message(
                    wallet_address,
                    MessageStructure.format_error("No agent available for this session.")
                )
                return

            # 確保LLM提供者已設置
            if not agent.is_llm_set:
                agent._setup_llm_provider()

            # --- 關鍵修改：完全移除關鍵字檢查 ---
            try:
                system_prompt = agent._construct_system_prompt()
                response = agent.connection_manager.perform_action(
                    "timetool",  # 始終使用 timetool 連接
                    "generate-text",
                    [user_text, system_prompt]
                )

                # 發送響應
                await self.send_message(
                    wallet_address,
                    MessageStructure.format_ai_response(response)
                )
            except Exception as e:
                logger.error(f"TimeTool處理錯誤: {e}")
                await self.send_message(
                    wallet_address,
                    MessageStructure.format_error(f"處理TimeTool請求時出錯: {str(e)}")
                )
            # --- 關鍵修改結束 ---


        except Exception as e:
            logger.exception(f"處理消息時出錯: {e}")
            await self.send_message(
                wallet_address,
                MessageStructure.format_error(f"處理消息時出錯: {str(e)}")
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
    # Initialize MongoDB client
    mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    database_name = os.getenv("DATABASE_NAME", "zerepy_db")
    db_client = MongoDBClient(mongodb_url, database_name)

    # 嘗試初始化索引，但不讓失敗阻止應用程序啟動
    try:
        await db_client.initialize_indexes()
    except Exception as e:
        logger.error(f"索引初始化失敗，但應用程序將繼續啟動: {e}")

    # Store the db_client in the app state
    app.state.db_client = db_client

    # Create and store the connection manager
    app.state.connection_manager = MultiClientManager(db_client)

    # Handle graceful shutdown
    def signal_handler():
        logger.info("接收到關閉信號，正在關閉連接...")

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, lambda signum, frame: signal_handler())
        except ValueError:
            # If we're not in the main thread, this will fail
            pass

    logger.info("伺服器初始化完成")
    yield
    logger.info("伺服器正在關閉")


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
                agent_name: str = Query(default="timetool_agent")
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
                    print('-----------------')
                    print(data)
                    try:
                        message_data = json.loads(data)
                        print("message_data: ", message_data, "wallet_address: ", wallet_address)
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
        @self.app.get("/api/agents")
        async def list_agents():
            try:
                agents = []
                agents_dir = Path("agents")
                if agents_dir.exists():
                    for agent_file in agents_dir.glob("*.json"):
                        if agent_file.stem != "general":
                            agents.append(agent_file.stem)
                return {"agents": agents}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # Get user information
        @self.app.get("/api/user/{wallet_address}")
        async def get_user_info(wallet_address: str):
            wallet_address = wallet_address.lower()
            user = await self.app.state.db_client.get_user(wallet_address)
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            return user

        # Create user if not exists and get status
        @self.app.post("/api/user/{wallet_address}")
        async def create_or_get_user(wallet_address: str):
            wallet_address = wallet_address.lower()
            user = await self.app.state.db_client.get_user(wallet_address)

            if not user:
                user = await self.app.state.db_client.create_user(wallet_address)

            return {
                "wallet_address": user["wallet_address"],
                "created_at": user["created_at"],
                "has_agent": user.get("has_agent", False),
                "agent_id": user.get("agent_id")
            }

        # Get conversation history
        @self.app.get("/api/conversation/{wallet_address}")
        async def get_conversation(wallet_address: str, limit: int = Query(50, gt=0, lt=1000)):
            wallet_address = wallet_address.lower()
            history = await self.app.state.db_client.get_conversation_history(wallet_address)

            # Only return the last 'limit' messages
            if len(history) > limit:
                history = history[-limit:]

            return {"messages": history}

        # Create a user agent
        @self.app.post("/api/create-agent/{wallet_address}")
        async def create_agent(wallet_address: str, agent_name: str = Query(default="timetool_agent")):
            wallet_address = wallet_address.lower()
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

            # Create agent
            agent_id = str(uuid4())
            await self.app.state.db_client.db.users.update_one(
                {"wallet_address": wallet_address},
                {"$set": {
                    "agent_id": agent_id,
                    "has_agent": True,
                    "agent_name": agent_name,
                    "created_at": datetime.now(),
                    "last_active": datetime.now()
                }}
            )

            return {
                "success": True,
                "message": "Agent created successfully",
                "agent_id": agent_id,
                "agent_name": agent_name
            }

        # Send a message via REST API (non-WebSocket option)
        @self.app.post("/api/message/{wallet_address}")
        async def send_message(wallet_address: str, message: Dict[str, Any]):
            wallet_address = wallet_address.lower()
            connection_manager = self.app.state.connection_manager

            # Check if agent exists for this user
            agent = connection_manager.get_agent_for_user(wallet_address)

            if not agent:
                try:
                    # Try to load agent from user record
                    user = await self.app.state.db_client.get_user(wallet_address)
                    if not user or not user.get("has_agent"):
                        raise HTTPException(status_code=404, detail="User has no agent configured")

                    agent_name = user.get("agent_name", "timetool_agent")
                    agent = ZerePyAgent(agent_name)
                    connection_manager.set_agent_for_user(wallet_address, agent)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Failed to load agent: {str(e)}")

            if connection_manager.is_agent_busy(wallet_address):
                raise HTTPException(status_code=429, detail="Agent is busy processing another request")

            connection_manager.set_agent_busy(wallet_address, True)

            try:
                # First, save the user message to database
                print(f"user_message = MessageStructure.format_message message: {message.get('text', '')}")
                print(f"user_message = MessageStructure.format_message message: {message}")
                user_message = MessageStructure.format_message("user", message.get("text", ""))
                await self.app.state.db_client.save_message(wallet_address, user_message)

                # Make sure the LLM provider is set up
                if not agent.is_llm_set:
                    agent._setup_llm_provider()

                # Generate the response using the agent
                print(f"agent.prompt_llm message: {message.get('text', '')}")
                print(f"agent.prompt_llm message: {message}")
                response = agent.prompt_llm(message.get("text", ""))

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