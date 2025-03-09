import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from .json_formatter import get_response_schema

logger = logging.getLogger("structured_output.schema_validator")


def validate_structured_output(output: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a structured output against the schema.

    Args:
        output: The structured output to validate

    Returns:
        A tuple containing (is_valid, error_messages)
    """
    schema = get_response_schema()
    errors = []

    # Check required fields
    for field in schema["required"]:
        if field not in output:
            errors.append(f"Missing required field: {field}")

    # Check thinking depth value
    if "thinking_depth" in output:
        if output["thinking_depth"] not in ["light", "medium", "deep"]:
            errors.append(f"Invalid thinking_depth: {output['thinking_depth']}")

    # Check timestamp format
    if "timestamp" in output:
        timestamp = output["timestamp"]
        if not isinstance(timestamp, str):
            errors.append(f"timestamp must be a string, got {type(timestamp).__name__}")
        elif not re.match(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$', timestamp):
            errors.append(f"Invalid timestamp format: {timestamp}")

    # Check confidence score
    if "confidence_score" in output:
        score = output["confidence_score"]
        if not isinstance(score, (int, float)):
            errors.append(f"confidence_score must be a number, got {type(score).__name__}")
        elif score < 0 or score > 1:
            errors.append(f"confidence_score must be between 0 and 1, got {score}")

    # Check main_steps structure
    if "main_steps" in output:
        if not isinstance(output["main_steps"], list):
            errors.append("main_steps must be an array")
        else:
            for i, step in enumerate(output["main_steps"]):
                step_errors = validate_step(step, i)
                errors.extend(step_errors)

    # Check for empty final answer
    if "final_answer" in output:
        if not output["final_answer"].strip():
            errors.append("final_answer cannot be empty")

    return len(errors) == 0, errors


def validate_step(step: Dict[str, Any], step_index: int) -> List[str]:
    """
    Validate a single main step.

    Args:
        step: The step to validate
        step_index: The index of the step

    Returns:
        A list of error messages
    """
    errors = []

    # Check required fields
    for field in ["main_step", "main_step_title", "main_step_content"]:
        if field not in step:
            errors.append(f"Step {step_index + 1}: Missing required field: {field}")

    # Check main_step number
    if "main_step" in step:
        if not isinstance(step["main_step"], int):
            errors.append(f"Step {step_index + 1}: main_step must be an integer")
        elif step["main_step"] != step_index + 1:
            errors.append(
                f"Step {step_index + 1}: main_step number should be {step_index + 1}, got {step['main_step']}")

    # Check main_step_title
    if "main_step_title" in step:
        if not isinstance(step["main_step_title"], str):
            errors.append(f"Step {step_index + 1}: main_step_title must be a string")
        elif not step["main_step_title"].strip():
            errors.append(f"Step {step_index + 1}: main_step_title cannot be empty")

    # Check main_step_content
    if "main_step_content" in step:
        if not isinstance(step["main_step_content"], str):
            errors.append(f"Step {step_index + 1}: main_step_content must be a string")
        elif not step["main_step_content"].strip():
            errors.append(f"Step {step_index + 1}: main_step_content cannot be empty")

    # Check sub_steps if present
    if "sub_steps" in step:
        if not isinstance(step["sub_steps"], list):
            errors.append(f"Step {step_index + 1}: sub_steps must be an array")
        else:
            for j, sub_step in enumerate(step["sub_steps"]):
                sub_step_errors = validate_sub_step(sub_step, step_index, j)
                errors.extend(sub_step_errors)

    return errors


def validate_sub_step(sub_step: Dict[str, Any], step_index: int, sub_step_index: int) -> List[str]:
    """
    Validate a single sub-step.

    Args:
        sub_step: The sub-step to validate
        step_index: The index of the parent step
        sub_step_index: The index of the sub-step

    Returns:
        A list of error messages
    """
    errors = []

    # Check required fields
    for field in ["sub_step", "sub_step_title", "sub_step_content"]:
        if field not in sub_step:
            errors.append(f"Step {step_index + 1}, Sub-step {sub_step_index + 1}: Missing required field: {field}")

    # Check sub_step number
    if "sub_step" in sub_step:
        if not isinstance(sub_step["sub_step"], int):
            errors.append(f"Step {step_index + 1}, Sub-step {sub_step_index + 1}: sub_step must be an integer")
        elif sub_step["sub_step"] != sub_step_index + 1:
            errors.append(
                f"Step {step_index + 1}, Sub-step {sub_step_index + 1}: sub_step number should be {sub_step_index + 1}, got {sub_step['sub_step']}")

    # Check sub_step_title
    if "sub_step_title" in sub_step:
        if not isinstance(sub_step["sub_step_title"], str):
            errors.append(f"Step {step_index + 1}, Sub-step {sub_step_index + 1}: sub_step_title must be a string")
        elif not sub_step["sub_step_title"].strip():
            errors.append(f"Step {step_index + 1}, Sub-step {sub_step_index + 1}: sub_step_title cannot be empty")

    # Check sub_step_content
    if "sub_step_content" in sub_step:
        if not isinstance(sub_step["sub_step_content"], str):
            errors.append(f"Step {step_index + 1}, Sub-step {sub_step_index + 1}: sub_step_content must be a string")
        elif not sub_step["sub_step_content"].strip():
            errors.append(f"Step {step_index + 1}, Sub-step {sub_step_index + 1}: sub_step_content cannot be empty")

    return errors


def get_schema_errors(output: Dict[str, Any]) -> List[str]:
    """
    Get validation errors for a structured output.

    Args:
        output: The structured output to validate

    Returns:
        A list of error messages
    """
    is_valid, errors = validate_structured_output(output)
    return errors