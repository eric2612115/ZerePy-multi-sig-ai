from typing import Dict, List, Any, Optional

# Reasoning chain templates for different thinking depths
REASONING_CHAINS = {
    "light": [
        "Identify the main request in the query",
        "Determine the specific information needed",
        "Provide the information directly"
    ],

    "medium": [
        "Identify the main request and any sub-requests in the query",
        "Determine what information is needed to fulfill these requests",
        "Identify any constraints or preferences mentioned by the user",
        "Gather the necessary information",
        "Process and analyze the information",
        "Formulate a clear and direct response",
        "Verify the response against the original query"
    ],

    "deep": [
        "Identify the explicit and implicit aspects of the query",
        "Break down the query into component parts",
        "Determine what information is needed for each component",
        "Identify any constraints, preferences, or special requirements",
        "Establish evaluation criteria for potential solutions",
        "Gather comprehensive information from multiple sources",
        "Process and synthesize the information",
        "Generate multiple potential approaches or solutions",
        "Evaluate each option against the established criteria",
        "Consider potential risks, downsides, and edge cases",
        "Formulate a recommendation with clear justification",
        "Verify the recommendation satisfies all aspects of the original query"
    ]
}

# Intent-specific reasoning chain elements
INTENT_REASONING_ELEMENTS = {
    "price_check": [
        "Identify the specific token or cryptocurrency",
        "Determine which blockchain network to check",
        "Gather current price data",
        "Check for any recent significant price movements",
        "Provide the current price with relevant context"
    ],

    "buy": [
        "Identify the specific token the user wants to buy",
        "Determine which blockchain network is relevant",
        "Check if the user specified an amount to purchase",
        "Verify the user has the necessary funds",
        "Evaluate the token's security and legitimacy",
        "Assess current market conditions for the purchase",
        "Determine the best execution method",
        "Outline steps for securely completing the transaction"
    ],

    "sell": [
        "Identify the specific token the user wants to sell",
        "Determine which blockchain network is relevant",
        "Check if the user specified an amount to sell",
        "Verify the user owns the token they want to sell",
        "Assess current market conditions for the sale",
        "Determine the best execution method",
        "Outline steps for securely completing the transaction"
    ],

    "portfolio": [
        "Understand the user's investment goals",
        "Assess the user's risk tolerance",
        "Analyze current market conditions",
        "Evaluate potential assets for inclusion",
        "Consider diversification across different cryptocurrencies",
        "Determine optimal allocation percentages",
        "Assess the security of each component",
        "Create a balanced portfolio strategy",
        "Plan for monitoring and rebalancing"
    ],

    "analysis": [
        "Identify what specific aspects need to be analyzed",
        "Determine what data points are relevant",
        "Gather comprehensive data from reliable sources",
        "Apply appropriate analytical frameworks",
        "Look for patterns, trends, and anomalies",
        "Consider multiple interpretations of the data",
        "Draw evidence-based conclusions",
        "Present findings in a structured manner"
    ],

    "risk_assessment": [
        "Identify specific security risks to evaluate",
        "Check contract code for vulnerabilities",
        "Analyze token distribution and concentration",
        "Evaluate team transparency and history",
        "Assess liquidity depth and stability",
        "Check for audit reports and security measures",
        "Rate overall security level with justification",
        "Provide specific risk mitigation recommendations"
    ]
}


def generate_reasoning_chain(depth: str, intents: List[str] = None) -> List[str]:
    """
    Generate a reasoning chain based on thinking depth and detected intents.

    Args:
        depth: The thinking depth ("light", "medium", or "deep")
        intents: Optional list of detected intents

    Returns:
        A list of reasoning steps
    """
    if depth not in REASONING_CHAINS:
        depth = "medium"  # Default to medium if unknown depth

    # Start with the base reasoning chain for the given depth
    base_chain = REASONING_CHAINS[depth].copy()

    # If no intents provided, return the base chain
    if not intents:
        return base_chain

    # Process intents to modify the reasoning chain
    intent_steps = []
    for intent in intents:
        if intent in INTENT_REASONING_ELEMENTS:
            intent_steps.extend(INTENT_REASONING_ELEMENTS[intent])

    # Merge and deduplicate steps
    combined_steps = []
    seen_steps = set()

    # First add base steps that aren't in intent steps
    for step in base_chain:
        step_lower = step.lower()
        if step_lower not in seen_steps:
            combined_steps.append(step)
            seen_steps.add(step_lower)

    # Then add intent steps that aren't duplicates of base steps
    for step in intent_steps:
        step_lower = step.lower()
        if step_lower not in seen_steps:
            # Find best position to insert this step
            if "verification" in step_lower or "verify" in step_lower:
                # Verification steps go near the end
                insert_pos = len(combined_steps) - 1
            elif "gather" in step_lower or "identify" in step_lower or "determine" in step_lower:
                # Information gathering steps go early
                insert_pos = min(2, len(combined_steps))
            elif "evaluate" in step_lower or "assess" in step_lower or "analyze" in step_lower:
                # Analysis steps go in the middle
                insert_pos = len(combined_steps) // 2
            else:
                # Default to appending
                insert_pos = len(combined_steps)

            combined_steps.insert(insert_pos, step)
            seen_steps.add(step_lower)

    # Limit the number of steps based on depth
    max_steps = {
        "light": 5,
        "medium": 10,
        "deep": 15
    }.get(depth, 10)

    if len(combined_steps) > max_steps:
        combined_steps = combined_steps[:max_steps]

    return combined_steps


def get_reasoning_chain_for_step(step_type: str, depth: str) -> List[str]:
    """
    Get a specific reasoning chain for a particular step type.

    Args:
        step_type: The type of step (e.g., "price_check", "risk_assessment")
        depth: The thinking depth ("light", "medium", or "deep")

    Returns:
        A list of reasoning steps specific to the step type
    """
    if step_type in INTENT_REASONING_ELEMENTS:
        return INTENT_REASONING_ELEMENTS[step_type]

    # Return a subset of the base reasoning chain based on step type
    base_chain = REASONING_CHAINS.get(depth, REASONING_CHAINS["medium"])

    if step_type == "query_understanding":
        return base_chain[:2]
    elif step_type == "information_gathering":
        return [s for s in base_chain if "gather" in s.lower() or "information" in s.lower()]
    elif step_type == "analysis":
        return [s for s in base_chain if "analyze" in s.lower() or "process" in s.lower() or "evaluate" in s.lower()]
    elif step_type == "recommendation":
        return [s for s in base_chain if "recommend" in s.lower() or "formulate" in s.lower()]
    else:
        # Default to returning a generic subset
        return base_chain[len(base_chain) // 2:]