from typing import Dict, List, Any

# Templates for different thinking depths
THINKING_TEMPLATES = {
    "light": {
        "steps": [
            {
                "step_type": "query_understanding",
                "title": "Understand the Query",
                "description": "Identify what information the user is seeking"
            },
            {
                "step_type": "direct_response",
                "title": "Provide Direct Response",
                "description": "Give a straightforward answer to the query"
            }
        ],
        "max_steps": 2,
        "min_steps": 2,
        "requires_reasoning": False,
        "requires_verification": False
    },

    "medium": {
        "steps": [
            {
                "step_type": "query_understanding",
                "title": "Understand the Query",
                "description": "Identify the user's intentions and requirements"
            },
            {
                "step_type": "information_gathering",
                "title": "Gather Information",
                "description": "Collect relevant data needed to answer the query"
            },
            {
                "step_type": "analysis",
                "title": "Analyze Information",
                "description": "Process the gathered information to form insights"
            },
            {
                "step_type": "response_formulation",
                "title": "Formulate Response",
                "description": "Create a comprehensive response based on analysis"
            }
        ],
        "max_steps": 5,
        "min_steps": 3,
        "requires_reasoning": True,
        "requires_verification": True
    },

    "deep": {
        "steps": [
            {
                "step_type": "query_understanding",
                "title": "Understand the Query",
                "description": "Thoroughly analyze the user's intentions, requirements, and constraints"
            },
            {
                "step_type": "goal_clarification",
                "title": "Clarify Goals",
                "description": "Establish clear objectives based on the user's query"
            },
            {
                "step_type": "information_gathering",
                "title": "Gather Information",
                "description": "Collect comprehensive data from multiple sources"
            },
            {
                "step_type": "analysis",
                "title": "Analyze Information",
                "description": "Process and synthesize the gathered information"
            },
            {
                "step_type": "option_generation",
                "title": "Generate Options",
                "description": "Develop multiple potential approaches or solutions"
            },
            {
                "step_type": "evaluation",
                "title": "Evaluate Options",
                "description": "Assess the pros and cons of each option"
            },
            {
                "step_type": "recommendation",
                "title": "Make Recommendation",
                "description": "Provide a well-reasoned recommendation with justification"
            }
        ],
        "max_steps": 10,
        "min_steps": 5,
        "requires_reasoning": True,
        "requires_verification": True
    }
}

# Intent-specific template modifications
INTENT_TEMPLATE_MODIFIERS = {
    "price_check": {
        "additional_steps": [
            {
                "step_type": "price_verification",
                "title": "Verify Price Information",
                "description": "Check multiple sources to ensure price accuracy"
            }
        ],
        "priority": 1
    },

    "buy": {
        "additional_steps": [
            {
                "step_type": "security_check",
                "title": "Security Assessment",
                "description": "Evaluate the security aspects of the token"
            },
            {
                "step_type": "purchase_plan",
                "title": "Purchase Plan",
                "description": "Outline steps for executing the purchase"
            }
        ],
        "priority": 2
    },

    "portfolio": {
        "additional_steps": [
            {
                "step_type": "risk_assessment",
                "title": "Risk Assessment",
                "description": "Evaluate risks associated with the portfolio"
            },
            {
                "step_type": "diversification_analysis",
                "title": "Diversification Analysis",
                "description": "Assess how well the portfolio is diversified"
            },
            {
                "step_type": "allocation_planning",
                "title": "Allocation Planning",
                "description": "Plan optimal allocation of funds across assets"
            }
        ],
        "priority": 3
    },

    "risk_assessment": {
        "additional_steps": [
            {
                "step_type": "security_audit",
                "title": "Security Audit",
                "description": "Perform a detailed security analysis"
            },
            {
                "step_type": "risk_quantification",
                "title": "Risk Quantification",
                "description": "Measure and quantify the identified risks"
            }
        ],
        "priority": 2
    }
}


def get_thinking_template(depth: str, intents: List[str] = None) -> Dict[str, Any]:
    """
    Get a thinking template based on depth and intents.

    Args:
        depth: The thinking depth ("light", "medium", or "deep")
        intents: Optional list of detected intents

    Returns:
        A dictionary containing the thinking template
    """
    if depth not in THINKING_TEMPLATES:
        depth = "medium"  # Default to medium if unknown depth

    # Start with the base template
    template = THINKING_TEMPLATES[depth].copy()
    base_steps = template["steps"].copy()

    # If no intents provided, return the base template
    if not intents:
        return template

    # Process intents to modify template
    additional_steps = []

    # Sort intents by priority (if they exist in modifiers)
    sorted_intents = sorted(
        [i for i in intents if i in INTENT_TEMPLATE_MODIFIERS],
        key=lambda x: INTENT_TEMPLATE_MODIFIERS[x].get("priority", 0),
        reverse=True
    )

    # Add steps from highest priority intents first
    for intent in sorted_intents:
        if intent in INTENT_TEMPLATE_MODIFIERS:
            modifier = INTENT_TEMPLATE_MODIFIERS[intent]
            additional_steps.extend(modifier.get("additional_steps", []))

    # Insert additional steps at appropriate positions
    result_steps = []
    response_step_index = next((i for i, step in enumerate(base_steps)
                                if step["step_type"] in ["direct_response", "recommendation", "response_formulation"]),
                               len(base_steps) - 1)

    # Insert steps before the response step
    result_steps = base_steps[:response_step_index] + additional_steps + base_steps[response_step_index:]

    # Update the template with the modified steps
    template["steps"] = result_steps

    # Ensure we don't exceed max_steps
    if len(template["steps"]) > template["max_steps"]:
        template["steps"] = template["steps"][:template["max_steps"]]

    return template


def get_step_descriptions(depth: str, step_types: List[str] = None) -> Dict[str, str]:
    """
    Get descriptions for specific step types.

    Args:
        depth: The thinking depth ("light", "medium", or "deep")
        step_types: Optional list of step types to get descriptions for

    Returns:
        A dictionary mapping step types to descriptions
    """
    if depth not in THINKING_TEMPLATES:
        depth = "medium"

    template = THINKING_TEMPLATES[depth]
    step_descriptions = {step["step_type"]: step["description"] for step in template["steps"]}

    # Add descriptions from intent modifiers
    for intent, modifier in INTENT_TEMPLATE_MODIFIERS.items():
        for step in modifier.get("additional_steps", []):
            step_descriptions[step["step_type"]] = step["description"]

    if step_types:
        return {s: step_descriptions.get(s, "No description available") for s in step_types}

    return step_descriptions