import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import uuid

from src.thinking_framework.thinking_depth_selector import analyze_thinking_depth
from src.structured_output.json_formatter import format_structured_response
from src.structured_output.step_processor import process_thinking_steps
from src.structured_output.schema_validator import validate_structured_output

logger = logging.getLogger("server.structured_response_handler")


class StructuredResponseHandler:
    """
    Handles the generation and processing of structured responses
    """

    def __init__(self):
        self.response_cache = {}

    async def generate_structured_response(
            self,
            query: str,
            raw_response: str,
            thinking_level: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a structured response from a raw text response

        Args:
            query: The original user query
            raw_response: The raw text response from the AI
            thinking_level: Optional thinking level, automatically detected if not provided

        Returns:
            A structured response dictionary
        """
        logger.debug(f"Generating structured response for query: {query[:50]}...")

        # Detect thinking level if not provided
        thinking_result = {}
        print(f"Thinking level: {thinking_level}")
        if not thinking_level:
            thinking_result = analyze_thinking_depth(query)
            thinking_level = thinking_result["depth"]
            print(f"Thinking level: {thinking_level}")
            print(f"Thinking result: {thinking_result}")
        # Create thinking steps based on depth
        main_steps = process_thinking_steps(
            query=query,
            thinking_depth=thinking_level,
            intents=thinking_result.get("detected_intents", [])
        )

        # Generate rephrased query (simplified example)
        rephrased_query = f"Understood as: {query.strip()}"

        # Calculate confidence score (placeholder logic)
        confidence_score = 0.85
        if "not sure" in raw_response.lower() or "unclear" in raw_response.lower():
            confidence_score = 0.6
        elif "confident" in raw_response.lower() or "certain" in raw_response.lower():
            confidence_score = 0.95

        # Format structured response
        response = format_structured_response(
            query=query,
            rephrased_query=rephrased_query,
            thinking_depth=thinking_level,
            main_steps=main_steps,
            final_answer=raw_response,
            confidence_score=confidence_score
        )

        # Validate response
        is_valid, errors = validate_structured_output(response)
        if not is_valid:
            logger.warning(f"Generated structured response is invalid: {errors}")
            # Fix basic issues if possible
            response = self._fix_validation_issues(response, errors)

            # Cache the response
            self.response_cache[response["response_id"]] = response

        return response

    def _fix_validation_issues(self, response: Dict[str, Any], errors: List[str]) -> Dict[str, Any]:
        """
        Fix common validation issues in structured responses

        Args:
            response: The structured response with validation issues
            errors: List of validation error messages

        Returns:
            Fixed structured response dictionary
        """
        fixed_response = response.copy()

        # Check for missing required fields
        for error in errors:
            if "Missing required field" in error:
                field = error.split(": ")[1]
                if field == "response_id":
                    fixed_response["response_id"] = str(uuid.uuid4())
                elif field == "timestamp":
                    fixed_response["timestamp"] = datetime.now().isoformat() + "Z"
                elif field == "query" and "query" not in fixed_response:
                    fixed_response["query"] = "Unknown query"
                elif field == "thinking_depth" and "thinking_depth" not in fixed_response:
                    fixed_response["thinking_depth"] = "medium"
                elif field == "main_steps" and "main_steps" not in fixed_response:
                    fixed_response["main_steps"] = []
                elif field == "final_answer" and "final_answer" not in fixed_response:
                    fixed_response["final_answer"] = "No answer available."

        # Fix empty final answer
        if "final_answer cannot be empty" in str(errors):
            fixed_response["final_answer"] = "No specific answer could be generated."

        # Fix confidence score range
        if "confidence_score must be between 0 and 1" in str(errors):
            score = fixed_response.get("confidence_score", 0)
            fixed_response["confidence_score"] = max(0, min(score, 1))

        return fixed_response

    async def get_response_by_id(self, response_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a previously generated structured response by ID

        Args:
            response_id: The unique ID of the response

        Returns:
            The structured response dictionary or None if not found
        """
        return self.response_cache.get(response_id)

    async def serialize_response(self, response: Dict[str, Any], format_type: str = "json") -> str:
        """
        Serialize a structured response for transmission

        Args:
            response: The structured response dictionary
            format_type: Output format ("json" or "text")

        Returns:
            Serialized response as string
        """
        if format_type == "json":
            return json.dumps(response, ensure_ascii=False)
        elif format_type == "text":
            # Convert structured response to plain text
            text = f"Answer: {response.get('final_answer', '')}\n\n"

            # Add thinking steps if present
            main_steps = response.get("main_steps", [])
            if main_steps:
                text += "Reasoning process:\n"
                for step in main_steps:
                    step_num = step.get("main_step", "")
                    step_title = step.get("main_step_title", "")
                    step_content = step.get("main_step_content", "")
                    text += f"{step_num}. {step_title}: {step_content}\n"

            return text
        else:
            raise ValueError(f"Unsupported format type: {format_type}")