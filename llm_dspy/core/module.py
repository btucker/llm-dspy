import sys
import dspy
import llm

def _parse_signature(signature: str):
    """Parse a DSPy module signature into input and output fields."""
    parts = signature.split('->')
    if len(parts) != 2:
        raise ValueError("Invalid signature format. Expected: inputs -> outputs")
    
    input_str, output_str = parts
    input_fields = [f.strip() for f in input_str.split(',')]
    output_fields = [f.strip() for f in output_str.split(',')]
    
    return input_fields, output_fields

def _process_rag_field(field_name: str, collection_name: str, kwargs: dict, collection=None):
    """Process a RAG field by retrieving relevant documents."""
    if collection is None:
        collection = llm.Collection(collection_name, model_id="ada-002")
    
    query = kwargs.get(field_name)
    if not query:
        return kwargs
    
    try:
        results = collection.similar(text=query, n=3)
        context = "\n\n".join(doc.text if hasattr(doc, 'text') else str(doc) for doc in results)
        kwargs[field_name] = f"Context:\n{context}\n\nQuestion: {query}"
    except Exception as e:
        print(f"Warning: Failed to query collection: {str(e)}", file=sys.stderr)
    
    return kwargs

def run_dspy_module(module_name: str, signature: str, **kwargs):
    """Run a DSPy module with the given signature and keyword arguments."""
    try:
        # Parse signature
        print(f"Parsing signature: {signature}")
        input_fields, output_fields = _parse_signature(signature)
        print(f"Extracted input fields: {input_fields}")
        
        # Process input fields
        print(f"Processing input fields: {input_fields}")
        print(f"Available kwargs: {kwargs}")
        print(f"Positional inputs: {tuple()}")
        
        # Process RAG fields
        for field in input_fields:
            if field in kwargs and isinstance(kwargs[field], str):
                # Check if this is a collection name
                try:
                    collection = llm.Collection(kwargs[field], model_id="ada-002")
                    # If we get here, it's a collection name
                    kwargs = _process_rag_field(field, kwargs[field], kwargs, collection)
                except Exception as e:
                    print(f"Warning: Failed to process potential collection '{kwargs[field]}': {str(e)}", file=sys.stderr)
                    # Not a collection name, leave it as is
                    pass
        
        # Get the module class
        try:
            # Try to get the module from sys.modules first
            if 'dspy' in sys.modules:
                module_class = getattr(sys.modules['dspy'], module_name)
            else:
                module_class = getattr(dspy, module_name)
        except AttributeError:
            # Try to get it from our own modules
            try:
                from ..rag import EnhancedRAGModule
                if module_name == 'EnhancedRAGModule':
                    module_class = EnhancedRAGModule
                else:
                    raise AttributeError(f"DSPy module {module_name} not found")
            except ImportError:
                raise AttributeError(f"DSPy module {module_name} not found")
        
        # Create module instance
        module = module_class()
        
        # Run the module
        result = module.forward(**kwargs)
        
        # Return the result
        if hasattr(result, 'text'):
            return result.text
        elif hasattr(result, 'answer'):
            return result.answer
        else:
            return str(result)
            
    except Exception as e:
        print(f"Error running DSPy module: {str(e)}", file=sys.stderr)
        raise
