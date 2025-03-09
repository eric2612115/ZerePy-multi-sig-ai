import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import re

logger = logging.getLogger("weight_management.context_analyzer")


def analyze_context(query: str, session_history: List[Dict], user_data: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Analyze the current context based on query, session history, and user data.

    Args:
        query: The current user query
        session_history: List of previous exchanges in the session
        user_data: Optional user-specific data like preferences and history

    Returns:
        Dictionary containing context factors
    """
    context = {
        "timestamp": datetime.now().isoformat(),
        "context_type": "general",
        "is_weekend": datetime.now().weekday() >= 5,  # 5,6 = Saturday, Sunday
        "time_of_day": get_time_of_day(),
        "query_complexity": calculate_query_complexity(query),
        "chat_duration_minutes": calculate_chat_duration(session_history),
        "mentioned_tokens": extract_token_mentions(query, session_history),
        "crypto_intent": detect_crypto_intent(query),
        "user_preferences": extract_user_preferences(user_data),
        "thinking_level_indicators": detect_thinking_level(query)
    }

    # Determine overall context type
    context["context_type"] = determine_context_type(context)

    return context


def determine_context_type(context: Dict[str, Any]) -> str:
    """
    Determine the overall context type based on analyzed factors.

    Args:
        context: Context data from analyze_context

    Returns:
        Context type as string
    """
    crypto_intent = context.get("crypto_intent", "")
    query_complexity = context.get("query_complexity", 0)
    thinking_indicators = context.get("thinking_level_indicators", {})

    if crypto_intent in ["buy", "sell", "swap"] and query_complexity >= 5:
        return "crypto_transaction"

    if crypto_intent in ["analysis", "portfolio"] or thinking_indicators.get("deep", 0) > 0.7:
        return "crypto_analysis"

    if crypto_intent in ["price_check", "market_info"] or thinking_indicators.get("light", 0) > 0.7:
        return "crypto_info"

    if crypto_intent in ["security", "risk"] or "scam" in context.get("mentioned_tokens", []):
        return "security_analysis"

    # Default to general crypto context
    return "general_crypto"


def get_time_of_day() -> str:
    """Get the current time of day category."""
    hour = datetime.now().hour

    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    else:
        return "night"


def calculate_query_complexity(query: str) -> int:
    """
    Calculate the complexity of a query on a scale of 1-10.

    Args:
        query: User query text

    Returns:
        Complexity score (1-10)
    """
    # Basic complexity factors
    length_factor = min(len(query.split()), 30) / 6  # 0-5 points for length
    question_factor = min(query.count('?'), 3)  # 0-3 points for questions
    conjunction_factor = min(len(re.findall(r'\b(and|or|but|because|if|when|while)\b', query)), 5) / 2  # 0-2.5 points

    # Check for complexity indicators
    has_multiple_requests = len(re.findall(r'\b(and|also|additionally|moreover)\b', query)) > 1
    has_comparison = bool(re.search(r'\b(compare|versus|vs|against|better|best|worse|worst)\b', query))
    has_constraints = bool(re.search(r'\b(must|should|only|exactly|specifically|precisely)\b', query))

    # Calculate total score
    total = length_factor + question_factor + conjunction_factor

    if has_multiple_requests:
        total += 2
    if has_comparison:
        total += 1.5
    if has_constraints:
        total += 1

    # Ensure score is between 1-10
    return max(1, min(10, round(total)))


def calculate_chat_duration(session_history: List[Dict]) -> float:
    """
    Calculate how long the current session has been active in minutes.

    Args:
        session_history: List of previous exchanges in the session

    Returns:
        Duration in minutes
    """
    if not session_history:
        return 0

    try:
        # Get first and last message timestamps
        first_msg_time = datetime.fromisoformat(session_history[0].get("timestamp", datetime.now().isoformat()))
        last_msg_time = datetime.fromisoformat(session_history[-1].get("timestamp", datetime.now().isoformat()))

        # Calculate duration
        duration = (last_msg_time - first_msg_time).total_seconds() / 60
        return duration
    except (ValueError, KeyError, IndexError):
        return 0


def extract_token_mentions(query: str, session_history: List[Dict]) -> List[str]:
    """
    Extract cryptocurrency token mentions from query and recent history.

    Args:
        query: Current user query
        session_history: List of previous exchanges

    Returns:
        List of token symbols mentioned
    """
    # Common token regex pattern
    token_pattern = r'\b(?:[$]?[A-Z]{2,10}|0x[a-fA-F0-9]{40})\b'

    # Extract from current query
    current_tokens = re.findall(token_pattern, query.upper())

    # Extract from recent messages (last 3)
    recent_tokens = []
    for msg in session_history[-3:]:
        if "text" in msg:
            recent_tokens.extend(re.findall(token_pattern, msg["text"].upper()))

    # Combine and deduplicate
    all_tokens = list(set(current_tokens + recent_tokens))

    # Clean up (remove $ prefix if present)
    cleaned_tokens = [token.replace('$', '') for token in all_tokens]

    return cleaned_tokens


def detect_crypto_intent(query: str) -> str:
    """
    Detect the primary cryptocurrency-related intent in the query.

    Args:
        query: User query text

    Returns:
        Intent category as string
    """
    query = query.lower()

    # Check for different intents
    if re.search(r'price|worth|value|cost|rate', query):
        return "price_check"

    if re.search(r'buy|purchase|get|acquire', query):
        return "buy"

    if re.search(r'sell|trade|swap|exchange|convert', query):
        return "sell"

    if re.search(r'portfolio|invest|allocation|distribute|split', query):
        return "portfolio"

    if re.search(r'analy|review|evaluate|assess|study', query):
        return "analysis"

    if re.search(r'secure|safe|scam|risk|danger', query):
        return "security"

    if re.search(r'market|trend|move|news|development', query):
        return "market_info"

    # Default intent
    return "general"


def extract_user_preferences(user_data: Optional[Dict]) -> Dict[str, Any]:
    """
    Extract user preferences from user data.

    Args:
        user_data: User-specific data including preferences and history

    Returns:
        Dictionary of user preferences
    """
    if not user_data:
        return {}

    preferences = {
        "preferred_chains": user_data.get("preferred_chains", ["base"]),
        "risk_tolerance": user_data.get("risk_tolerance", "medium"),
        "portfolio_focus": user_data.get("portfolio_focus", "balanced"),
        "interaction_style": user_data.get("interaction_style", "normal")
    }

    return preferences


def detect_thinking_level(query: str) -> Dict[str, float]:
    """
    Detect indicators of thinking level requirements.

    Args:
        query: User query text

    Returns:
        Dictionary with confidence scores for each thinking level
    """
    query = query.lower()

    # Keywords associated with different thinking levels
    light_indicators = [
        "price", "current", "quick", "simple", "what is", "how much",
        "check", "now", "today"
    ]

    medium_indicators = [
        "explain", "help me", "guide", "steps", "process", "how to",
        "details", "buy", "sell", "compare"
    ]

    deep_indicators = [
        "analyze", "portfolio", "strategy", "risk", "comprehensive",
        "detailed", "thorough", "evaluation", "long-term", "complex"
    ]

    # Calculate matches for each level
    light_matches = sum(1 for word in light_indicators if word in query)
    medium_matches = sum(1 for word in medium_indicators if word in query)
    deep_matches = sum(1 for word in deep_indicators if word in query)

    # Calculate total matches
    total = light_matches + medium_matches + deep_matches

    # Calculate normalized confidence scores
    if total > 0:
        light_score = light_matches / total
        medium_score = medium_matches / total
        deep_score = deep_matches / total
    else:
        # Default to medium if no clear indicators
        light_score = 0.2
        medium_score = 0.6
        deep_score = 0.2

    return {
        "light": light_score,
        "medium": medium_score,
        "deep": deep_score
    }


def get_context_factors(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract key factors from context for weight adjustment.

    Args:
        context: The full context dictionary

    Returns:
        Dictionary with key factors for weight adjustment
    """
    return {
        "is_weekend": context.get("is_weekend", False),
        "is_night": context.get("time_of_day") == "night",
        "is_low_volume": context.get("is_night", False) or context.get("is_weekend", False),
        "user_active": context.get("chat_duration_minutes", 0) < 15,
        "has_crypto_intent": context.get("crypto_intent") != "general",
        "context_type": context.get("context_type", "general"),
        "thinking_level": get_thinking_level_from_context(context)
    }


def get_thinking_level_from_context(context: Dict[str, Any]) -> str:
    """
    Determine the appropriate thinking level based on context.

    Args:
        context: The context dictionary

    Returns:
        Thinking level as string ("light", "medium", or "deep")
    """
    # Get thinking level indicators
    indicators = context.get("thinking_level_indicators", {})

    # Get the level with highest confidence
    if indicators:
        levels = ["light", "medium", "deep"]
        scores = [indicators.get(level, 0) for level in levels]
        max_index = scores.index(max(scores))
        return levels[max_index]

    # Fallback based on query complexity
    complexity = context.get("query_complexity", 5)

    if complexity <= 3:
        return "light"
    elif complexity <= 7:
        return "medium"
    else:
        return "deep"