import re
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from .thinking_templates import get_thinking_template
from .reasoning_chains import generate_reasoning_chain
from .self_questioning import generate_self_questions

logger = logging.getLogger("thinking_framework.depth_selector")


class ThinkingDepthSelector:
    """
    Analyzes queries and determines the appropriate thinking depth and structure.
    """

    def __init__(self):
        self.complexity_thresholds = {
            "light": 3,
            "medium": 7,
            "deep": 10
        }

        self.intent_categories = {
            "light": ["price", "time", "date", "current", "check", "what is", "how much"],
            "medium": ["buy", "sell", "trade", "swap", "purchase", "exchange", "convert"],
            "deep": ["portfolio", "strategy", "analysis", "compare", "recommend", "risk", "best", "optimize"]
        }

    def detect_intent(self, query: str) -> List[str]:
        """
        Detects intents from the user's query.

        Args:
            query: The user's query text

        Returns:
            A list of detected intents
        """
        query = query.lower()
        intents = []

        # Check for price query
        if re.search(r'price|worth|value|cost|rate', query):
            intents.append("price_check")

        # Check for buy intention
        if re.search(r'buy|purchase|get|acquire', query):
            intents.append("buy")

        # Check for sell intention
        if re.search(r'sell|trade|swap|exchange|convert', query):
            intents.append("sell")

        # Check for portfolio management
        if re.search(r'portfolio|invest|allocation|distribute|split', query):
            intents.append("portfolio")

        # Check for analysis request
        if re.search(r'analy|review|evaluate|assess|study', query):
            intents.append("analysis")

        # Check for recommendation request
        if re.search(r'recommend|suggest|advise|best|good|better', query):
            intents.append("recommendation")

        # Check for risk assessment
        if re.search(r'risk|safe|secure|danger|scam|security', query):
            intents.append("risk_assessment")

        # Check for time-related queries
        if re.search(r'time|date|when|now', query):
            intents.append("time_check")

        return intents

    def analyze_query_complexity(self, query: str) -> int:
        """
        Analyzes the complexity of a query and returns a score.

        Args:
            query: The user's query text

        Returns:
            A complexity score (1-10)
        """
        # Basic factors affecting complexity
        factors = {
            "length": min(len(query.split()), 25) / 5,  # 0-5 points for length
            "question_marks": min(query.count('?'), 3),  # 0-3 points for questions
            "conjunctions": min(len(re.findall(r'\b(and|or|but|because|if|when|while)\b', query)), 5) / 2
            # 0-2.5 points for logical structures
        }

        # Check for multiple requests in one query
        if len(re.findall(r'\b(and|also|additionally|moreover)\b', query)) > 1:
            factors["multiple_requests"] = 2

        # Check for comparative analysis
        if re.search(r'\b(compare|versus|vs|against|better|best|worse|worst)\b', query):
            factors["comparative"] = 1.5

        # Check for specific constraints or parameters
        if re.search(r'\b(must|should|only|exactly|specifically|precisely)\b', query):
            factors["constraints"] = 1

        # Calculate final score (1-10 scale)
        score = min(sum(factors.values()), 10)
        return round(score)

    def determine_thinking_depth(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Determines the appropriate thinking depth for a given query.

        Args:
            query: The user's query text
            context: Optional context information about the conversation

        Returns:
            A dictionary with thinking depth configuration
        """
        # Detect intents
        intents = self.detect_intent(query)

        # Calculate complexity
        complexity_score = self.analyze_query_complexity(query)

        # Determine depth based on complexity and intents
        depth = "medium"  # Default

        # Check for explicit intents matching depth categories
        for category, keywords in self.intent_categories.items():
            if any(keyword in query.lower() for keyword in keywords):
                depth = category
                break

        # Adjust based on complexity score
        if complexity_score < self.complexity_thresholds["light"]:
            depth = "light"
        elif complexity_score < self.complexity_thresholds["medium"]:
            depth = "medium"
        elif complexity_score >= self.complexity_thresholds["deep"]:
            depth = "deep"

        # Override with context if available
        if context and "thinking_depth" in context:
            depth = context["thinking_depth"]

        # Build the thinking framework
        template = get_thinking_template(depth)
        reasoning_chain = generate_reasoning_chain(depth, intents)
        self_questions = generate_self_questions(depth, intents)

        logger.debug(f"Query '{query[:50]}...' analyzed as {depth} thinking with complexity {complexity_score}")

        return {
            "depth": depth,
            "complexity_score": complexity_score,
            "detected_intents": intents,
            "template": template,
            "reasoning_chain": reasoning_chain,
            "self_questions": self_questions,
            "timestamp": datetime.now().isoformat()
        }

    def get_step_count(self, depth: str) -> int:
        """
        Returns the recommended number of steps for the given thinking depth.

        Args:
            depth: The thinking depth ("light", "medium", or "deep")

        Returns:
            The recommended number of steps
        """
        step_counts = {
            "light": 2,
            "medium": 4,
            "deep": 7
        }
        return step_counts.get(depth, 3)


# Create a singleton instance for easy import
selector = ThinkingDepthSelector()


def analyze_thinking_depth(query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function to analyze thinking depth using the singleton.

    Args:
        query: The user's query text
        context: Optional context information

    Returns:
        A dictionary with thinking depth configuration
    """
    return selector.determine_thinking_depth(query, context)