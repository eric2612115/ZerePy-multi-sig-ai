import json
import logging
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger("structured_output.json_formatter")


def get_response_schema() -> Dict[str, Any]:
    """
    Returns the schema for structured responses.

    Returns:
        A dictionary containing the response schema
    """
    return {
        "type": "object",
        "properties": {
            "response_id": {"type": "string"},
            "timestamp": {"type": "string", "format": "date-time"},
            "query": {"type": "string"},
            "rephrased_query": {"type": ["string", "null"]},
            "thinking_depth": {"type": "string", "enum": ["light", "medium", "deep"]},
            "main_steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "main_step": {"type": "integer"},
                        "main_step_title": {"type": "string"},
                        "main_step_target": {"type": "string"},
                        "main_step_content": {"type": "string"},
                        "sub_steps": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "sub_step": {"type": "integer"},
                                    "sub_step_title": {"type": "string"},
                                    "sub_step_target": {"type": "string"},
                                    "sub_step_content": {"type": "string"}
                                },
                                "required": ["sub_step", "sub_step_title", "sub_step_content"]
                            }
                        }
                    },
                    "required": ["main_step", "main_step_title", "main_step_content"]
                }
            },
            "final_answer": {"type": "string"},
            "confidence_score": {"type": "number", "minimum": 0, "maximum": 1},
            "additional_notes": {"type": ["string", "null"]}
        },
        "required": ["response_id", "timestamp", "query", "thinking_depth", "main_steps", "final_answer"]
    }


def format_structured_response(
        query: str,
        rephrased_query: Optional[str],
        thinking_depth: str,
        main_steps: List[Dict[str, Any]],
        final_answer: str,
        confidence_score: float,
        additional_notes: Optional[str] = None
) -> Dict[str, Any]:
    """
    Creates a structured JSON response based on provided parameters.

    Args:
        query: Original user query
        rephrased_query: Rephrased query (if any)
        thinking_depth: Thinking depth used ("light", "medium", or "deep")
        main_steps: List of main thinking steps with their sub-steps
        final_answer: The final answer or recommendation
        confidence_score: Confidence score (0-1)
        additional_notes: Additional notes or context (optional)

    Returns:
        A dictionary containing the structured response
    """
    response = {
        "response_id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "query": query,
        "rephrased_query": rephrased_query,
        "thinking_depth": thinking_depth,
        "main_steps": main_steps,
        "final_answer": final_answer,
        "confidence_score": max(0, min(1, confidence_score)),  # Ensure between 0-1
        "additional_notes": additional_notes
    }

    # Log response creation
    logger.debug(f"Created structured response {response['response_id']} with {len(main_steps)} main steps")

    return response


def serialize_structured_response(response: Dict[str, Any], indent: int = 2) -> str:
    """
    Serialize a structured response to a JSON string.

    Args:
        response: The structured response dictionary
        indent: Indentation level for JSON formatting

    Returns:
        A formatted JSON string
    """
    try:
        return json.dumps(response, indent=indent, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error serializing response: {str(e)}")
        # Return a simplified version in case of error
        return json.dumps({
            "response_id": response.get("response_id", "error"),
            "error": "Failed to serialize full response",
            "final_answer": response.get("final_answer", "")
        }, indent=indent, ensure_ascii=False)


def deserialize_structured_response(json_str: str) -> Dict[str, Any]:
    """
    Deserialize a JSON string into a structured response dictionary.

    Args:
        json_str: The JSON string to deserialize

    Returns:
        The structured response as a dictionary
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Error deserializing JSON: {str(e)}")
        return {
            "response_id": f"error-{uuid.uuid4()}",
            "error": f"Failed to deserialize JSON: {str(e)}",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }


def extract_final_answer(response: Dict[str, Any]) -> str:
    """
    Extract just the final answer from a structured response.

    Args:
        response: The structured response dictionary

    Returns:
        The final answer as a string
    """
    return response.get("final_answer", "No answer available")