import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import uuid
import re

logger = logging.getLogger("server.context_manager")


class ContextManager:
    """
    Manages conversation context and state for user sessions
    """

    def __init__(self):
        self.user_contexts = {}
        self.context_ttl = timedelta(hours=24)  # Time-to-live for context

    async def get_context(self, user_id: str) -> Dict[str, Any]:
        """
        Get the current context for a user

        Args:
            user_id: The unique user identifier

        Returns:
            User context dictionary
        """
        # Create new context if not exists or expired
        if user_id not in self.user_contexts:
            return await self.create_context(user_id)

        context = self.user_contexts[user_id]

        # Check if context is expired
        last_updated = datetime.fromisoformat(context.get("last_updated"))
        if datetime.now() - last_updated > self.context_ttl:
            return await self.create_context(user_id)

        return context

    async def create_context(self, user_id: str) -> Dict[str, Any]:
        """
        Create a new context for a user

        Args:
            user_id: The unique user identifier

        Returns:
            Newly created context dictionary
        """
        context = {
            "context_id": str(uuid.uuid4()),
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "message_history": [],
            "entities": {},
            "thinking_level": "medium",  # Default thinking level
            "preferences": {},
            "detected_intents": []
        }

        self.user_contexts[user_id] = context
        return context

    async def update_context(self, user_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update specific fields in a user's context

        Args:
            user_id: The unique user identifier
            updates: Dictionary of fields to update

        Returns:
            Updated context dictionary
        """
        if user_id not in self.user_contexts:
            context = await self.create_context(user_id)
        else:
            context = self.user_contexts[user_id]

        # Update provided fields
        for key, value in updates.items():
            if key != "context_id" and key != "user_id" and key != "created_at":
                context[key] = value

        # Always update last_updated timestamp
        context["last_updated"] = datetime.now().isoformat()

        return context

    async def add_message(self, user_id: str, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a message to the user's context history

        Args:
            user_id: The unique user identifier
            message: The message to add

        Returns:
            Updated context dictionary
        """
        context = await self.get_context(user_id)

        # Ensure the message has required fields
        if "id" not in message:
            message["id"] = str(uuid.uuid4())
        if "timestamp" not in message:
            message["timestamp"] = datetime.now().isoformat()

        # Add message to history
        if "message_history" not in context:
            context["message_history"] = []

        context["message_history"].append(message)

        # Limit history size (keep last 20 messages)
        if len(context["message_history"]) > 20:
            context["message_history"] = context["message_history"][-20:]

        # Update context entities based on message content
        if "text" in message and message.get("sender") == "user":
            await self._extract_entities(context, message["text"])

        # Update last_updated timestamp
        context["last_updated"] = datetime.now().isoformat()

        return context

    async def _extract_entities(self, context: Dict[str, Any], text: str) -> None:
        """
        Extract entities from message text and update context

        Args:
            context: User context dictionary
            text: Message text to analyze
        """
        # Ensure entities dictionary exists
        if "entities" not in context:
            context["entities"] = {}

        # Extract cryptocurrency tokens
        token_pattern = r'\b(?:[$]?[A-Z]{2,10}|0x[a-fA-F0-9]{40})\b'
        tokens = re.findall(token_pattern, text.upper())

        # Clean token symbols (remove $ prefix)
        clean_tokens = [token.replace('$', '') if token.startswith('$') else token for token in tokens]

        # Add to entities
        for token in clean_tokens:
            context["entities"][token] = {
                "type": "crypto_token",
                "mentioned_at": datetime.now().isoformat(),
                "mention_count": context["entities"].get(token, {}).get("mention_count", 0) + 1
            }

        # Extract blockchain names
        chain_names = ["ethereum", "eth", "base", "arbitrum", "arb", "optimism", "polygon", "matic"]
        for chain in chain_names:
            if chain.lower() in text.lower():
                context["entities"][chain] = {
                    "type": "blockchain",
                    "mentioned_at": datetime.now().isoformat(),
                    "mention_count": context["entities"].get(chain, {}).get("mention_count", 0) + 1
                }

        # Extract numeric values (potentially amounts)
        amount_pattern = r'\$?(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)'
        amounts = re.findall(amount_pattern, text)

        if amounts:
            # Store the last mentioned amount
            try:
                # Convert string like "1,000.50" to float 1000.50
                amount_value = float(amounts[-1].replace(',', ''))
                context["entities"]["last_amount"] = {
                    "type": "amount",
                    "value": amount_value,
                    "mentioned_at": datetime.now().isoformat()
                }
            except ValueError:
                pass

    async def get_thinking_level(self, user_id: str, message_text: str) -> str:
        """
        Determine the appropriate thinking level for a message

        Args:
            user_id: The unique user identifier
            message_text: The user's message text

        Returns:
            Thinking level ("light", "medium", or "deep")
        """
        context = await self.get_context(user_id)

        # Analyze message complexity
        complexity = self._analyze_complexity(message_text)

        # Determine thinking level based on complexity
        thinking_level = "medium"  # Default

        if complexity < 3:
            thinking_level = "light"
        elif complexity > 7:
            thinking_level = "deep"

        # Store in context
        context["thinking_level"] = thinking_level
        context["last_message_complexity"] = complexity

        return thinking_level

    def _analyze_complexity(self, text: str) -> int:
        """
        Analyze text complexity on a scale of 1-10

        Args:
            text: Text to analyze

        Returns:
            Complexity score (1-10)
        """
        # Simple complexity analysis based on text features
        # Length factor (0-4 points)
        words = text.split()
        length_factor = min(len(words) / 10, 4)

        # Question factor (0-2 points)
        question_factor = min(text.count('?'), 2)

        # Conjunction factor (0-2 points)
        conjunctions = ["and", "or", "but", "because", "if", "when", "while"]
        conjunction_count = sum(1 for word in words if word.lower() in conjunctions)
        conjunction_factor = min(conjunction_count, 2)

        # Structure factor (0-2 points)
        structure_words = ["compare", "analyze", "evaluate", "recommend", "suggest"]
        structure_count = sum(1 for word in words if word.lower() in structure_words)
        structure_factor = min(structure_count, 2)

        # Calculate total complexity (1-10)
        complexity = length_factor + question_factor + conjunction_factor + structure_factor
        return max(1, min(10, round(complexity)))