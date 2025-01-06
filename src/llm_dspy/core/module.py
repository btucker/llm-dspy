import dspy
import llm
import logging
from typing import Dict, Any, Optional
from dspy.signatures import ensure_signature
from ..retrieval import LLMRetriever

logger = logging.getLogger('llm_dspy.core')

def run_dspy_module(module_name: str, signature: str, **kwargs):
    """Run a DSPy module with the given signature and keyword arguments."""
    try:
        # Parse signature using DSPy's ensure_signature
        logger.debug(f"Parsing signature: {signature}")
        sig = ensure_signature(signature)
        logger.debug(f"Extracted input fields: {sig.input_fields}")
        
        # Process input fields
        logger.debug(f"Processing input fields: {sig.input_fields}")
        logger.debug(f"Available kwargs: {kwargs}")
        
        # Get the module class
        try:
            module_class = getattr(dspy, module_name)
        except AttributeError:
            error_msg = f"DSPy module {module_name} not found"
            logger.error(error_msg)
            raise AttributeError(error_msg)
        
        # Configure retrieval if any input value matches a collection name
        for field, value in kwargs.items():
            if isinstance(value, str) and value in llm.collections:
                logger.debug(f"Found collection '{value}' for field '{field}'")
                # Configure DSPy with our retriever
                retriever = LLMRetriever(k=3, collection_name=value)
                dspy.settings.configure(retrieve=retriever)
                break
        
        # Create and call the module
        module = module_class(signature)
        result = module(**kwargs)
        
        return result
        
    except Exception as e:
        logger.error(f"Error running DSPy module: {e}")
        raise
