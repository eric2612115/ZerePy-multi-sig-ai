import logging
from typing import Dict, List, Any, Optional, Tuple
from ..thinking_framework.thinking_templates import get_thinking_template
from ..thinking_framework.reasoning_chains import get_reasoning_chain_for_step
from ..thinking_framework.self_questioning import get_self_questions_for_step

logger = logging.getLogger("structured_output.step_processor")


def create_step_structure(
        step_number: int,
        step_title: str,
        step_target: str,
        step_content: str,
        sub_steps: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Create a structured representation of a thinking step.

    Args:
        step_number: The step number
        step_title: The title of the step
        step_target: The goal or target of the step
        step_content: The content or result of the step
        sub_steps: Optional list of sub-steps

    Returns:
        A dictionary representing the structured step
    """
    step = {
        "main_step": step_number,
        "main_step_title": step_title,
        "main_step_target": step_target,
        "main_step_content": step_content
    }

    if sub_steps:
        step["sub_steps"] = sub_steps
    else:
        step["sub_steps"] = []

    return step


def create_sub_step_structure(
        step_number: int,
        step_title: str,
        step_target: str,
        step_content: str
) -> Dict[str, Any]:
    """
    Create a structured representation of a thinking sub-step.

    Args:
        step_number: The sub-step number
        step_title: The title of the sub-step
        step_target: The goal or target of the sub-step
        step_content: The content or result of the sub-step

    Returns:
        A dictionary representing the structured sub-step
    """
    return {
        "sub_step": step_number,
        "sub_step_title": step_title,
        "sub_step_target": step_target,
        "sub_step_content": step_content
    }


def process_thinking_steps(
        query: str,
        thinking_depth: str,
        intents: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Process a query and generate structured thinking steps.

    Args:
        query: The user's query
        thinking_depth: The thinking depth to use
        intents: Optional list of detected intents

    Returns:
        A list of structured thinking steps
    """
    # Get the thinking template based on depth and intents
    template = get_thinking_template(thinking_depth, intents)
    steps = template["steps"]

    # Initialize structured steps
    structured_steps = []

    # Process each step
    for i, step in enumerate(steps):
        step_type = step["step_type"]
        step_title = step["title"]
        step_description = step["description"]

        # Get reasoning chain for this step
        reasoning_chain = get_reasoning_chain_for_step(step_type, thinking_depth)

        # Get self-questions for this step
        self_questions = get_self_questions_for_step(step_type)

        # Convert reasoning chain and self-questions to sub-steps
        sub_steps = []

        # Add reasoning steps as sub-steps
        for j, reasoning_step in enumerate(reasoning_chain[:3]):  # Limit to 3 reasoning steps
            sub_steps.append(create_sub_step_structure(
                j + 1,
                f"Reasoning {j + 1}",
                reasoning_step,
                f"I need to {reasoning_step.lower()}"
            ))

        # Add self-questioning as sub-steps (if this is medium or deep thinking)
        if thinking_depth in ["medium", "deep"]:
            for j, question in enumerate(self_questions[:2]):  # Limit to 2 questions
                sub_steps.append(create_sub_step_structure(
                    len(sub_steps) + 1,
                    f"Self-Question {j + 1}",
                    question,
                    f"I should consider: {question}"
                ))

        # Create the main step structure
        structured_steps.append(create_step_structure(
            i + 1,
            step_title,
            step_description,
            f"Will {step_description.lower()} to address the query",
            sub_steps
        ))

    return structured_steps


def extract_step_titles(steps: List[Dict[str, Any]]) -> List[str]:
    """
    Extract just the titles from a list of structured steps.

    Args:
        steps: A list of structured thinking steps

    Returns:
        A list of step titles
    """
    return [step["main_step_title"] for step in steps]


def update_step_content(
        steps: List[Dict[str, Any]],
        step_index: int,
        content: str
) -> List[Dict[str, Any]]:
    """
    Update the content of a specific step.

    Args:
        steps: The list of structured steps
        step_index: The index of the step to update
        content: The new content for the step

    Returns:
        The updated list of steps
    """
    if 0 <= step_index < len(steps):
        steps[step_index]["main_step_content"] = content
    return steps


def update_sub_step_content(
        steps: List[Dict[str, Any]],
        step_index: int,
        sub_step_index: int,
        content: str
) -> List[Dict[str, Any]]:
    """
    Update the content of a specific sub-step.

    Args:
        steps: The list of structured steps
        step_index: The index of the parent step
        sub_step_index: The index of the sub-step to update
        content: The new content for the sub-step

    Returns:
        The updated list of steps
    """
    if 0 <= step_index < len(steps):
        sub_steps = steps[step_index].get("sub_steps", [])
        if 0 <= sub_step_index < len(sub_steps):
            sub_steps[sub_step_index]["sub_step_content"] = content
    return steps