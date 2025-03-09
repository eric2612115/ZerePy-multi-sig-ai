from typing import Dict, List, Any, Optional

# Base self-questioning templates for different thinking depths
SELF_QUESTIONING_TEMPLATES = {
    "light": [
        "What specific information is the user asking for?",
        "Is there any ambiguity in the request that I should clarify?"
    ],

    "medium": [
        "What specific information is the user asking for?",
        "Are there any implicit aspects to this request?",
        "What specific data points do I need to gather?",
        "Am I making any assumptions that should be verified?",
        "Is my response directly answering the user's query?",
        "Have I provided sufficient context for the information?"
    ],

    "deep": [
        "What is the central question or problem in this query?",
        "What are the explicit and implicit aspects of this request?",
        "What assumptions am I making that should be verified?",
        "What additional context would be helpful to know?",
        "What are the key factors that should influence my analysis?",
        "What alternative perspectives should I consider?",
        "What are the potential risks or downsides I should address?",
        "What evidence supports my conclusions?",
        "How confident am I in my analysis and recommendation?",
        "Have I addressed all aspects of the original query?"
    ]
}

# Intent-specific self-questioning elements
INTENT_SELF_QUESTIONING = {
    "price_check": [
        "Have I identified the correct token/coin?",
        "Am I checking the price on the correct blockchain?",
        "Is this the most up-to-date price information?",
        "Should I provide additional price context (e.g., 24h change)?"
    ],

    "buy": [
        "Have I identified the correct token the user wants to buy?",
        "Do I have enough information about the token's security?",
        "Have I considered the current market conditions?",
        "Is there sufficient liquidity for this purchase?",
        "Have I checked if the user has enough funds for this transaction?",
        "Have I clearly outlined the steps for executing the purchase?",
        "Have I addressed potential risks associated with this purchase?"
    ],

    "sell": [
        "Have I identified the correct token the user wants to sell?",
        "Have I checked if the user actually owns this token?",
        "Have I considered the current market conditions?",
        "Is there sufficient liquidity for this sale?",
        "Have I clearly outlined the steps for executing the sale?",
        "Have I calculated any fees or slippage that might affect the sale?"
    ],

    "portfolio": [
        "Have I considered the user's risk tolerance?",
        "Am I diversifying across different types of assets?",
        "Have I evaluated the security of each component?",
        "Is the allocation balanced appropriately?",
        "Have I considered how this portfolio will perform in different market conditions?",
        "Have I provided a clear rationale for each component of the portfolio?",
        "Have I addressed how the portfolio should be monitored and rebalanced?"
    ],

    "analysis": [
        "Am I considering all relevant data points?",
        "Have I looked at this from multiple perspectives?",
        "Am I being objective in my analysis?",
        "Have I considered contrary evidence?",
        "Is my reasoning clear and logical?",
        "Are there any biases affecting my analysis?"
    ],

    "risk_assessment": [
        "Have I identified all major security risks?",
        "Have I checked for contract vulnerabilities?",
        "Have I analyzed the token distribution?",
        "Have I evaluated the team's transparency and history?",
        "Have I assessed the liquidity situation?",
        "Have I checked for audit reports?",
        "Am I being appropriately cautious in my assessment?"
    ],

    "time_check": [
        "Am I providing the correct time or date information?",
        "Should I include the timezone?",
        "Is there any additional time-related context I should provide?"
    ]
}


def generate_self_questions(depth: str, intents: List[str] = None) -> List[str]:
    """
    Generate self-questions based on thinking depth and detected intents.

    Args:
        depth: The thinking depth ("light", "medium", or "deep")
        intents: Optional list of detected intents

    Returns:
        A list of self-questions
    """
    if depth not in SELF_QUESTIONING_TEMPLATES:
        depth = "medium"  # Default to medium if unknown depth

    # Start with the base self-questioning template for the given depth
    base_questions = SELF_QUESTIONING_TEMPLATES[depth].copy()

    # If no intents provided, return the base questions
    if not intents:
        return base_questions

    # Process intents to add relevant self-questions
    intent_questions = []
    for intent in intents:
        if intent in INTENT_SELF_QUESTIONING:
            intent_questions.extend(INTENT_SELF_QUESTIONING[intent])

    # Merge and deduplicate questions
    combined_questions = []
    seen_questions = set()

    for question in base_questions + intent_questions:
        question_lower = question.lower()
        if question_lower not in seen_questions:
            combined_questions.append(question)
            seen_questions.add(question_lower)

    # Limit number of questions based on depth
    max_questions = {
        "light": 3,
        "medium": 6,
        "deep": 12
    }.get(depth, 6)

    if len(combined_questions) > max_questions:
        combined_questions = combined_questions[:max_questions]

    return combined_questions


def get_self_questions_for_step(step_type: str) -> List[str]:
    """
    Get self-questions specific to a particular step type.

    Args:
        step_type: The type of step (e.g., "price_check", "risk_assessment")

    Returns:
        A list of self-questions relevant to the step type
    """
    if step_type in INTENT_SELF_QUESTIONING:
        return INTENT_SELF_QUESTIONING[step_type]

    # Map step types to intent types
    step_to_intent = {
        "price_verification": "price_check",
        "security_check": "risk_assessment",
        "purchase_plan": "buy",
        "risk_assessment": "risk_assessment",
        "allocation_planning": "portfolio",
        "query_understanding": ["price_check", "buy", "sell", "portfolio", "analysis", "risk_assessment"]
    }

    if step_type in step_to_intent:
        intent_mapping = step_to_intent[step_type]
        if isinstance(intent_mapping, list):
            # Combine questions from multiple intents
            questions = []
            for intent in intent_mapping:
                if intent in INTENT_SELF_QUESTIONING:
                    questions.extend(INTENT_SELF_QUESTIONING[intent][:2])  # Take top 2 from each
            return questions[:5]  # Limit to 5 questions
        elif intent_mapping in INTENT_SELF_QUESTIONING:
            return INTENT_SELF_QUESTIONING[intent_mapping]

    # Default self-questions for generic steps
    generic_questions = [
        "Have I understood this part of the query correctly?",
        "Am I addressing the user's actual needs?",
        "Is there additional information I should consider?",
        "Am I being thorough enough in my approach?",
        "Is my reasoning sound and logical?"
    ]

    return generic_questions