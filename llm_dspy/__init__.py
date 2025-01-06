"""
LLM-DSPy: A bridge between LLM CLI and DSPy framework
"""

from .core.module import run_dspy_module
from .cli.commands import register_commands
from .adapter import LLMAdapter
from .utils import completion_with_adapter
from .rag import LLMRetriever, QueryTransformer, EnhancedRAGModule, ContextRewriter

__all__ = [
    'run_dspy_module',
    'register_commands',
    'LLMAdapter',
    'completion_with_adapter',
    'LLMRetriever',
    'QueryTransformer',
    'EnhancedRAGModule',
    'ContextRewriter'
]

__version__ = '0.1.0'

# Configure DSPy to use our adapter by default
import dspy
import litellm
from .utils import adapter, completion_with_adapter

# Register our provider with LiteLLM
litellm.provider_list.append("llm")
litellm.completion = completion_with_adapter

# Configure DSPy
dspy.configure(lm=dspy.LM(model="llm"))

# Register our modules with DSPy
setattr(dspy, 'EnhancedRAGModule', EnhancedRAGModule)
setattr(dspy, 'LLMRetriever', LLMRetriever)
setattr(dspy, 'QueryTransformer', QueryTransformer)
setattr(dspy, 'ContextRewriter', ContextRewriter)
