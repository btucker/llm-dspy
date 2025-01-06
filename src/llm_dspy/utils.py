from typing import Dict, List
import litellm
from .adapter import LLMAdapter
import logging
import sys

# Configure DSPy to use our adapter by default
adapter = LLMAdapter()

def setup_logging(verbose=False):
    """Configure logging for the llm_dspy package.
    
    Args:
        verbose (bool): If True, set logging level to DEBUG, otherwise INFO
    """
    logger = logging.getLogger('llm_dspy')
    if not logger.handlers:  # Only add handler if none exists
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    return logger

def completion_with_adapter(model: str, messages: List[Dict[str, str]], **kwargs):
    """Adapter function for LiteLLM to use our LLM adapter."""
    prompt = "\n".join(msg["content"] for msg in messages)
    response = adapter.llm.prompt(prompt)
    for chunk in response:
        pass
    response_text = response.text_or_raise()
    return litellm.ModelResponse(
        id="llm-" + str(hash(response_text))[:8],
        choices=[{
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": response_text,
                "role": "assistant"
            }
        }],
        model=model,
        usage={"total_tokens": 0}
    )
