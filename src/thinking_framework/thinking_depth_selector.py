import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from .thinking_templates import get_thinking_template
from .reasoning_chains import generate_reasoning_chain
from .self_questioning import generate_self_questions

logger = logging.getLogger("thinking_framework.depth_selector")


class ThinkingDepthSelector:
    """
    Analyzes queries and determines the appropriate thinking depth and structure
    using LLM-based evaluation rather than keyword matching.
    """

    def __init__(self):
        self.depth_descriptions = {
            "light": "Simple, straightforward queries requiring minimal analysis. Usually factual, current information, or simple checks that can be answered directly.",
            "medium": "Moderately complex queries requiring some analysis, explanation, or multi-step reasoning. May involve explanations, comparisons, or decision-making.",
            "deep": "Complex queries requiring thorough analysis, deep reasoning, or comprehensive evaluation. Often involves strategy, optimization, risk assessment, or multi-faceted analysis."
        }

    async def evaluate_thinking_depth(self, query: str, agent) -> str:
        """
        Use the LLM itself to determine the appropriate thinking depth with crypto-specific understanding.

        Args:
            query: The user's query text
            agent: The agent instance with connection to LLM

        Returns:
            The thinking depth as string ("light", "medium", or "deep")
        """
        # First detect intents to inform depth evaluation
        intents = self.detect_intent(query, agent)

        # Predefine depth weights for known intent types
        intent_depth_mapping = {
            # Light intents
            "price_check": "light",
            "time_check": "light",
            "basic_info": "light",
            "status_check": "light",

            # Medium intents
            "market_analysis": "medium",
            "token_comparison": "medium",
            "educational": "medium",
            "portfolio_view": "medium",
            "liquidity_check": "medium",

            # Deep intents - most transaction and security operations
            "transaction_intent": "deep",
            "defi_operation": "deep",
            "whitelist_operation": "deep",
            "security_assessment": "deep",
            "investment_strategy": "deep",
            "portfolio_modification": "deep",
            "chain_interaction": "deep",
            "multisig_operation": "deep"
        }

        # Check for automatic deep thinking triggers (all transaction-related intents)
        deep_transaction_intents = [
            "transaction_intent", "defi_operation", "whitelist_operation",
            "chain_interaction", "multisig_operation"
        ]

        if any(intent in deep_transaction_intents for intent in intents):
            logger.info(f"Transaction intent detected in {intents} - elevating to DEEP thinking")
            return "deep"

        # Check for mapped intents
        detected_depths = [intent_depth_mapping.get(intent, None) for intent in intents]
        if "deep" in detected_depths:
            return "deep"
        elif "medium" in detected_depths:
            return "medium"
        elif "light" in detected_depths:
            return "light"

        # If no clear mapping, use LLM evaluation with crypto context
        eval_prompt = f"""
As a cryptocurrency and DeFi expert, evaluate the complexity of the following user query to determine the appropriate thinking depth needed.

User query: "{query}"

Thinking depth options:
1. LIGHT - Simple factual information, current prices, basic status checks.
2. MEDIUM - Market analysis, token comparisons, educational explanations, viewing portfolio status.
3. DEEP - Any transaction intentions (buy/sell/swap), security assessments, DeFi operations, investment strategy, 
   blockchain interactions, or anything involving user funds or security.

CRITICAL CRYPTO-SPECIFIC CRITERIA:
- Any query involving potential transactions (buy, sell, swap) MUST be treated as DEEP
- Any query about token security, contract risks, or scam assessment MUST be DEEP
- Any query requiring on-chain data analysis should be at least MEDIUM
- Any query involving portfolio changes or investment recommendations MUST be DEEP
- Simple price checks or factual information can be LIGHT

First, analyze the query considering these crypto-specific criteria. Then respond with only one word: "LIGHT", "MEDIUM", or "DEEP".
"""

        # Get LLM to evaluate the query
        try:
            response = agent.connection_manager.perform_action(
                connection_name="timetool",  # Use timetool or appropriate LLM provider
                action_name="generate-text",
                params=[eval_prompt, "You are a crypto expert evaluating query complexity."]
            )

            # Extract the depth from the response
            if "LIGHT" in response.upper():
                return "light"
            elif "DEEP" in response.upper():
                return "deep"
            elif "MEDIUM" in response.upper():
                return "medium"
            else:
                # Default to medium if unclear
                logger.warning(f"Unclear thinking depth evaluation: {response}")
                return "medium"
        except Exception as e:
            logger.error(f"Error evaluating thinking depth: {e}")
            return "medium"  # Default to medium on error

    def detect_intent(self, query: str, agent) -> List[str]:
        """
        Use the LLM to detect user intents from the query with crypto-specific understanding.

        Args:
            query: The user's query text
            agent: The agent instance with connection to LLM

        Returns:
            A list of detected intents
        """
        intent_prompt = f"""
Analyze the following user query in the context of cryptocurrency and DeFi operations.
Identify all relevant intents, focusing especially on transaction-related intentions that require careful analysis.

INTENT CATEGORIES:
1. Information Queries (Generally Light):
   - price_check: Checking price or value of a token
   - time_check: Asking about time or dates
   - basic_info: Basic facts about tokens, chains, or crypto concepts
   - status_check: Checking status of services or transactions

2. Analysis Queries (Generally Medium):
   - market_analysis: Requesting market trend or price movement analysis
   - token_comparison: Comparing different tokens or projects
   - educational: Seeking to learn or understand a crypto concept
   - portfolio_view: Viewing or checking portfolio status (without changes)
   - liquidity_check: Checking liquidity of a token or pool

3. Transaction & Risk Queries (Generally Deep):
   - transaction_intent: Any intention to perform an actual transaction (buy, sell, swap)
   - defi_operation: Operations involving DeFi protocols (farming, staking, lending)
   - whitelist_operation: Adding or removing tokens from whitelist
   - security_assessment: Evaluating security risks of tokens or contracts
   - investment_strategy: Seeking advice on investment approach
   - portfolio_modification: Changing portfolio allocation or structure
   - chain_interaction: Operations requiring interaction with the blockchain
   - multisig_operation: Operations involving multisig wallets

User query: "{query}"

Return only a JSON array of the most relevant intents, like: ["transaction_intent", "security_assessment"]
"""
        try:
            response = agent.connection_manager.perform_action(
                connection_name="timetool",
                action_name="generate-text",
                params=[intent_prompt, "You are a crypto-specialized intent analyzer."]
            )

            # Extract JSON array from response
            import re
            import json

            # Find JSON array pattern
            array_match = re.search(r'\[.*\]', response)
            if array_match:
                intents = json.loads(array_match.group(0))
                return intents

            # Fallback intent extraction if JSON parsing fails
            transaction_intents = [
                "transaction_intent", "defi_operation", "whitelist_operation",
                "multisig_operation", "chain_interaction"
            ]

            # Check if any transaction-related terms appear in the response
            for intent in transaction_intents:
                if intent.lower() in response.lower():
                    return [intent, "security_assessment"]  # Security is always important with transactions

            # Check for basic information types
            if any(term in query.lower() for term in ["price", "worth", "value", "cost", "how much"]):
                return ["price_check"]

            if any(term in query.lower() for term in ["analysis", "evaluate", "review", "study", "assess"]):
                return ["market_analysis"]

            return ["basic_info"]  # Default fallback
        except Exception as e:
            logger.error(f"Error detecting intents: {e}")
            return ["basic_info"]  # Safe default

    async def determine_thinking_depth(self, query: str, agent, context: Optional[Dict[str, Any]] = None) -> Dict[
        str, Any]:
        """
        Determines the appropriate thinking depth for a given query using LLM evaluation.

        Args:
            query: The user's query text
            agent: The agent instance with connection to LLM
            context: Optional context information about the conversation

        Returns:
            A dictionary with thinking depth configuration
        """
        # Override with context if available
        if context and "thinking_depth" in context:
            depth = context["thinking_depth"]
        else:
            # Use LLM to evaluate thinking depth
            depth = await self.evaluate_thinking_depth(query, agent)

        # Detect intents using LLM
        intents = self.detect_intent(query, agent)

        # Build the thinking framework
        template = get_thinking_template(depth, intents)
        reasoning_chain = generate_reasoning_chain(depth, intents)
        self_questions = generate_self_questions(depth, intents)

        logger.debug(f"Query '{query[:50]}...' analyzed as {depth} thinking with intents {intents}")

        return {
            "depth": depth,
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


async def analyze_thinking_depth(query: str, agent, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function to analyze thinking depth using the singleton.

    Args:
        query: The user's query text
        agent: The agent instance with connection to LLM
        context: Optional context information

    Returns:
        A dictionary with thinking depth configuration
    """
    return await selector.determine_thinking_depth(query, agent, context)