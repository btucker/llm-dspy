from typing import Dict, List
import litellm
from .adapter import LLMAdapter

# Configure DSPy to use our adapter by default
adapter = LLMAdapter()

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
