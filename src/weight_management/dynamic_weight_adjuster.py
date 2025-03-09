import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime

logger = logging.getLogger("weight_management.dynamic_weight_adjuster")


def adjust_weights_for_context(tasks: List[Dict[str, Any]], context: Dict[str, Any],
                               base_weights: List[float], multipliers: Dict[str, float]) -> List[float]:
    """
    Adjust task weights based on the current context.

    Args:
        tasks: List of tasks with their properties
        context: Context information including time, user history, etc.
        base_weights: Original weights for tasks
        multipliers: Dictionary of multipliers for different contexts

    Returns:
        Modified list of weights
    """
    if not tasks or not base_weights or len(tasks) != len(base_weights):
        logger.warning("Invalid tasks or weights input")
        return base_weights

    adjusted_weights = base_weights.copy()

    # Time-based adjustments
    current_hour = datetime.now().hour

    # Night hours (1 AM - 5 AM): Reduce frequency of certain tasks
    if 1 <= current_hour <= 5:
        for i, task in enumerate(tasks):
            if task["name"] in ["post-tweet", "reply-to-tweet", "send-discord-message"]:
                adjusted_weights[i] *= multipliers.get("tweet_night_multiplier", 0.4)

    # Day hours (8 AM - 8 PM): Increase engagement and trading
    if 8 <= current_hour <= 20:
        for i, task in enumerate(tasks):
            if task["name"] in ["reply-to-tweet", "like-tweet", "execute-token-swap"]:
                adjusted_weights[i] *= multipliers.get("engagement_day_multiplier", 1.5)
            if task["name"] in ["get-crypto-price", "check-token-details", "analyze-portfolio"]:
                adjusted_weights[i] *= multipliers.get("market_active_hours_multiplier", 1.2)

    # Weekend adjustments (if present in context)
    is_weekend = context.get("is_weekend", False)
    if is_weekend:
        for i, task in enumerate(tasks):
            if task["name"] in ["analyze-portfolio", "market-sentiment-analysis"]:
                adjusted_weights[i] *= multipliers.get("analysis_weekend_multiplier", 1.3)

    # Low market volume times
    is_low_volume = context.get("is_low_volume", False)
    if is_low_volume:
        for i, task in enumerate(tasks):
            if task["name"] in ["execute-token-swap", "add-token-whitelist"]:
                adjusted_weights[i] *= multipliers.get("low_volume_hours_multiplier", 0.8)

    # User activity-based adjustments
    user_active = context.get("user_active", False)
    if user_active:
        for i, task in enumerate(tasks):
            if "user-interaction" in task.get("tags", []):
                adjusted_weights[i] *= 1.5

    return adjusted_weights


def adjust_weights_for_thinking_level(tasks: List[Dict[str, Any]], thinking_level: str,
                                      base_weights: List[float]) -> List[float]:
    """
    Adjust task weights based on thinking level.

    Args:
        tasks: List of tasks with their properties
        thinking_level: Level of thinking ("light", "medium", or "deep")
        base_weights: Original weights for tasks

    Returns:
        Modified list of weights
    """
    if not tasks or not base_weights or len(tasks) != len(base_weights):
        logger.warning("Invalid tasks or weights input")
        return base_weights

    adjusted_weights = base_weights.copy()

    # Task category-to-thinking level mapping
    task_categories = {
        "light": [
            "get-crypto-price", "get-current-time", "get-current-date",
            "get-current-datetime", "like-tweet"
        ],
        "medium": [
            "check-token-details", "get-wallet-balance", "add-token-whitelist",
            "post-tweet", "reply-to-tweet", "get-market-news", "check-token-liquidity"
        ],
        "deep": [
            "analyze-portfolio", "check-token-security", "get-hot-tokens",
            "on-chain-data-analysis", "market-sentiment-analysis", "scam-detection",
            "technical-indicator-calculation"
        ]
    }

    # Multipliers for each thinking level
    multipliers = {
        "light": {"light": 2.0, "medium": 0.7, "deep": 0.3},
        "medium": {"light": 0.8, "medium": 1.8, "deep": 0.9},
        "deep": {"light": 0.5, "medium": 1.0, "deep": 2.0}
    }

    # Apply multipliers based on thinking level
    level_multipliers = multipliers.get(thinking_level, {"light": 1.0, "medium": 1.0, "deep": 1.0})

    for i, task in enumerate(tasks):
        task_name = task["name"]

        # Determine which category this task belongs to
        task_level = None
        for level, level_tasks in task_categories.items():
            if task_name in level_tasks:
                task_level = level
                break

        # Apply multiplier if we found a category
        if task_level:
            adjusted_weights[i] *= level_multipliers.get(task_level, 1.0)

    return adjusted_weights