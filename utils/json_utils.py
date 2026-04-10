"""JSON parsing utilities for Multi-Stage Assist LLM outputs."""

import json
import logging
import re
from typing import Any, Dict

_LOGGER = logging.getLogger(__name__)


def extract_json_from_llm_string(text: str) -> Dict[str, Any]:
    """Robustly extract and parse JSON from a raw LLM output string.
    
    Handles cases where the LLM wraps the JSON in Markdown code blocks (e.g., ```json ... ```),
    has preamble/postamble text, or formatting issues.
    
    Args:
        text: The raw string response from the LLM.
        
    Returns:
        Dict: The parsed Python dictionary.
        
    Raises:
        json.JSONDecodeError: If valid JSON cannot be extracted.
    """
    cleaned = text.strip()
    
    # 1. Try to extract from Markdown JSON blocks
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.IGNORECASE | re.DOTALL)
    if match:
        cleaned = match.group(1).strip()
    
    # 2. Heuristically strip leading/trailing non-bracket text
    if ("{" in cleaned and "}" in cleaned) or ("[" in cleaned and "]" in cleaned):
        # Find first and last bracket of ANY kind
        start_idx_obj = cleaned.find("{")
        start_idx_arr = cleaned.find("[")
        
        # Determine start index (first of either)
        if start_idx_obj != -1 and start_idx_arr != -1:
            start_idx = min(start_idx_obj, start_idx_arr)
        else:
            start_idx = start_idx_obj if start_idx_obj != -1 else start_idx_arr
            
        # Determine end index (last of corresponding bracket)
        if start_idx == start_idx_obj:
            end_idx = cleaned.rfind("}")
        else:
            end_idx = cleaned.rfind("]")
            
        # Only slice if it looks like a JSON block
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            cleaned = cleaned[start_idx : end_idx + 1]
            
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        _LOGGER.warning("[JSON Utils] Failed to decode structured LLM output: %s", str(e))
        _LOGGER.debug("[JSON Utils] Attempted to parse: %s", cleaned)
        raise
