import sys
import re
from typing import Dict, List, Any
import dspy
import llm
from dspy.signatures.signature import ensure_signature

def _process_rag_field(field_name: str, collection_name: str, kwargs: dict, collection=None):
    """Process a RAG field by retrieving relevant documents."""
    if collection is None:
        try:
            collection = llm.Collection(collection_name, model_id="ada-002")
        except Exception as e:
            print(f"Warning: Failed to create collection: {str(e)}", file=sys.stderr)
            return kwargs
    
    query = kwargs.get('query', '')
    if not query:
        return kwargs
    
    try:
        print(f"Querying collection {collection_name} with query: {query}", file=sys.stderr)
        results = collection.similar(value=query, number=3)
        print(f"Got {len(results)} results", file=sys.stderr)
        context = "\n\n".join(doc.content if hasattr(doc, 'content') else str(doc) for doc in results)
        print(f"Context: {context}", file=sys.stderr)
        # Format context to make it more explicit for the model
        if 'dates' in query.lower() or 'amounts' in query.lower() or 'transactions' in query.lower():
            kwargs[field_name] = f"""Here is the relevant context:

{context}

Based on the above context, please answer the following question:
{query}

Please provide your answer in a structured format with the following fields:
- dates: List of dates mentioned in the transactions (e.g., ["March 15", "April 2", "May 20"])
- amounts: List of monetary amounts involved (e.g., [50000, 75000, 100000])
- entities: List of clients or parties involved (e.g., ["Client A", "Client B", "Client C"])"""
        else:
            kwargs[field_name] = f"""Here is the relevant context:

{context}

Based on the above context, please answer the following question:
{query}"""
        print(f"Final prompt: {kwargs[field_name]}", file=sys.stderr)
    except Exception as e:
        print(f"Warning: Failed to query collection: {str(e)}", file=sys.stderr)
    
    return kwargs

def run_dspy_module(module_name: str, signature: str, **kwargs):
    """Run a DSPy module with the given signature and keyword arguments."""
    try:
        # Parse signature using DSPy's ensure_signature
        print(f"Parsing signature: {signature}")
        sig = ensure_signature(signature)
        print(f"Extracted input fields: {sig.input_fields}")
        
        # Process input fields
        print(f"Processing input fields: {sig.input_fields}")
        print(f"Available kwargs: {kwargs}")
        print(f"Positional inputs: {tuple()}")
        
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
                        query = kwargs.get('query', '')
                        # Update kwargs with processed RAG field
                        kwargs = _process_rag_field(field, collection_name, {field: query}, collection)
                except Exception as e:
                    print(f"Warning: Failed to process potential collection '{kwargs[field]}': {str(e)}", file=sys.stderr)
                    # Not a collection name, leave it as is
                    pass
        
        # Get the module class
        try:
            module_class = getattr(dspy, module_name)
        except AttributeError:
            print(f"DSPy module {module_name} not found", file=sys.stderr)
            return None
        
        # Create and run the module
        module = module_class()
        result = module.forward(**kwargs)
        return result
    except Exception as e:
        print(f"Error running DSPy module: {str(e)}", file=sys.stderr)
        return None
