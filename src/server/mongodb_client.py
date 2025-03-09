#!/usr/bin/env python3
# src/server/mongodb_client.py

import motor.motor_asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from uuid import uuid4
from pymongo.errors import OperationFailure

logger = logging.getLogger("mongodb_client")


class MongoDBClient:
    """MongoDB client for persisting user conversations and data"""

    def __init__(self, mongodb_url: str, database_name: str):
        """Initialize the MongoDB client.

        Args:
            mongodb_url: MongoDB connection URL
            database_name: Name of the database to use
        """
        self.client = motor.motor_asyncio.AsyncIOMotorClient(mongodb_url)
        self.db = self.client[database_name]
        logger.info(f"MongoDB client initialized for database: {database_name}")

    async def initialize_indexes(self):
        """Create necessary database indexes, handling existing indexes gracefully"""
        try:
            # 使用 try-except 分別處理每個索引的創建，避免一個失敗導致全部失敗

            # Users collection indexes
            try:
                await self.db.users.create_index("wallet_address", unique=True, background=True)
                logger.info("Created unique index on users.wallet_address")
            except OperationFailure as e:
                # 檢查是否是因為索引已存在但配置不同
                if "IndexKeySpecsConflict" in str(e):
                    logger.warning("Index on users.wallet_address already exists with different settings")
                    # 嘗試刪除現有索引並重新創建
                    try:
                        await self.db.users.drop_index("wallet_address_1")
                        await self.db.users.create_index("wallet_address", unique=True, background=True)
                        logger.info("Recreated index on users.wallet_address")
                    except Exception as drop_err:
                        logger.warning(f"Failed to recreate index: {drop_err}")
                else:
                    logger.error(f"Error creating index on users.wallet_address: {e}")

            # Conversations collection indexes - 這裡是問題的根源
            try:
                # 檢查索引是否已存在
                indexes = await self.db.conversations.list_indexes().to_list(length=100)
                wallet_index_exists = any(idx.get('name') == 'wallet_address_1' for idx in indexes)

                if wallet_index_exists:
                    # 如果已存在，先刪除再創建
                    await self.db.conversations.drop_index("wallet_address_1")
                    logger.info("Dropped existing index on conversations.wallet_address")

                # 創建新索引
                await self.db.conversations.create_index("wallet_address", unique=True, background=True)
                logger.info("Created unique index on conversations.wallet_address")
            except Exception as e:
                logger.warning(f"Error handling index on conversations.wallet_address: {e}")
                # 如果出錯，嘗試不使用唯一約束
                try:
                    await self.db.conversations.create_index("wallet_address", background=True)
                    logger.info("Created non-unique index on conversations.wallet_address")
                except Exception as fallback_err:
                    logger.error(f"Failed to create fallback index: {fallback_err}")

            # Agent data collection indexes
            try:
                await self.db.agent_data.create_index("wallet_address", background=True)
                logger.info("Created index on agent_data.wallet_address")
            except Exception as e:
                logger.warning(f"Error creating index on agent_data.wallet_address: {e}")

            logger.info("Database indexes setup completed")

        except Exception as e:
            logger.error(f"Error in index initialization process: {e}")
            # 不引發異常，讓應用程序可以繼續啟動
            # 在最壞的情況下，我們只是沒有最佳的索引性能

    async def get_user(self, wallet_address: str) -> Optional[Dict[str, Any]]:
        """Get a user by wallet address"""
        return await self.db.users.find_one({"wallet_address": wallet_address})

    async def create_user(self, wallet_address: str) -> Dict[str, Any]:
        """Create a new user"""
        user = {
            "wallet_address": wallet_address,
            "created_at": datetime.now(),
            "has_agent": False,
            "agent_id": None,
            "agent_name": None,
            "last_active": datetime.now()
        }

        try:
            await self.db.users.insert_one(user)
            logger.info(f"Created new user: {wallet_address}")
            return user
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            raise

    async def update_user(self, wallet_address: str, update_data: Dict[str, Any]) -> bool:
        """Update user data"""
        try:
            result = await self.db.users.update_one(
                {"wallet_address": wallet_address},
                {"$set": update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating user: {e}")
            return False

    async def save_message(self, wallet_address: str, message: Dict[str, Any]) -> bool:
        """Save a message to the conversation history"""
        # Ensure message has required fields
        if "id" not in message:
            message["id"] = str(uuid4())

        if "timestamp" not in message:
            message["timestamp"] = datetime.now().isoformat()

        try:
            # Update user's last_active timestamp
            await self.db.users.update_one(
                {"wallet_address": wallet_address},
                {"$set": {"last_active": datetime.now()}},
                upsert=True
            )

            # Save message to conversation history
            result = await self.db.conversations.update_one(
                {"wallet_address": wallet_address},
                {"$push": {"messages": message}},
                upsert=True
            )

            return result.modified_count > 0 or result.upserted_id is not None
        except Exception as e:
            logger.error(f"Error saving message: {e}")
            return False

    async def get_conversation_history(self, wallet_address: str) -> List[Dict[str, Any]]:
        """Get conversation history for a user"""
        try:
            conversation = await self.db.conversations.find_one({"wallet_address": wallet_address})
            return conversation.get("messages", []) if conversation else []
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []

    async def delete_conversation_history(self, wallet_address: str) -> bool:
        """Delete conversation history for a user"""
        try:
            result = await self.db.conversations.delete_one({"wallet_address": wallet_address})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting conversation history: {e}")
            return False

    async def save_agent_data(self, wallet_address: str, key: str, data: Any) -> bool:
        """Save agent-specific data for a user"""
        try:
            result = await self.db.agent_data.update_one(
                {"wallet_address": wallet_address},
                {"$set": {key: data, "updated_at": datetime.now()}},
                upsert=True
            )
            return result.modified_count > 0 or result.upserted_id is not None
        except Exception as e:
            logger.error(f"Error saving agent data: {e}")
            return False

    async def get_agent_data(self, wallet_address: str, key: str = None) -> Any:
        """Get agent-specific data for a user"""
        try:
            agent_data = await self.db.agent_data.find_one({"wallet_address": wallet_address})
            if not agent_data:
                return None

            if key:
                return agent_data.get(key)
            else:
                return agent_data
        except Exception as e:
            logger.error(f"Error getting agent data: {e}")
            return None

    async def delete_agent_data(self, wallet_address: str, key: str = None) -> bool:
        """Delete agent-specific data for a user"""
        try:
            if key:
                result = await self.db.agent_data.update_one(
                    {"wallet_address": wallet_address},
                    {"$unset": {key: ""}}
                )
                return result.modified_count > 0
            else:
                result = await self.db.agent_data.delete_one({"wallet_address": wallet_address})
                return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting agent data: {e}")
            return False