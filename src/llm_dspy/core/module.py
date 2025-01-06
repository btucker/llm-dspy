import sys
import re
from typing import Dict, List, Any
import dspy
import llm
import logging
from dspy.signatures.signature import ensure_signature

logger = logging.getLogger('llm_dspy.core')

def _process_rag_field(field_name: str, collection_name: str, kwargs: dict, collection=None):
    """Process a RAG field by retrieving relevant documents."""
    if collection is None:
        try:
            collection = llm.Collection(collection_name, model_id="ada-002")
        except Exception as e:
            logger.warning(f"Failed to create collection: {str(e)}")
            return kwargs
    
    query = kwargs.get('query', '')
    if not query:
        return kwargs
    
    try:
        logger.debug(f"Querying collection {collection_name} with query: {query}")
        results = collection.similar(value=query, number=3)
        logger.debug(f"Got {len(results)} results")
        context = "\n\n".join(doc.content if hasattr(doc, 'content') else str(doc) for doc in results)
        logger.debug(f"Context: {context}")
        # Format context to make it more explicit for the model
        if 'dates' in query.lower() or 'amounts' in query.lower() or 'transactions' in query.lower():
            kwargs[field_name] = f"""Here is the relevant context:

{context}

Based on the above context, please answer the following question:
{query}

Please provide your answer in a structured format with the following fields:
- dates: ["March 15", "April 2", "May 20"]
- amounts: [50000, 75000, 100000]
- entities: ["Client A", "Client B", "Client C"]

Your response should be in exactly this format, with the actual values from the context."""
        else:
            kwargs[field_name] = f"""Here is the relevant context:

{context}

Based on the above context, please answer the following question:
{query}"""
        logger.debug(f"Final prompt: {kwargs[field_name]}")
    except Exception as e:
        logger.warning(f"Failed to query collection: {str(e)}")
    
    return kwargs

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
        logger.debug(f"Positional inputs: {tuple()}")
        
        # Process RAG fields
        for field in sig.input_fields:
            if field in kwargs:
                # Check if this is a collection name
                try:
                    # First try to get from global collections
                    collection = llm.collections.get(kwargs[field])
                    if collection is not None:
                        # If it's a collection, process the field with the collection
                        # Store the collection name and query
                        collection_name = kwargs[field]
                        # Update kwargs with processed RAG field
                        kwargs = _process_rag_field(field, collection_name, kwargs, collection)
                except Exception as e:
                    logger.warning(f"Failed to process potential collection '{kwargs[field]}': {str(e)}")
                    # Not a collection name, leave it as is
                    pass
        
        # Get the module class
        try:
            module_class = getattr(dspy, module_name)
        except AttributeError:
            error_msg = f"DSPy module {module_name} not found"
            logger.error(error_msg)
            raise AttributeError(error_msg)
        
        # Create and run the module
        module = module_class()
        result = module.forward(**kwargs)
        return result
    except Exception as e:
        logger.error(f"Error running DSPy module: {str(e)}")
        return None
