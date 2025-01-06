import sys
import re
import ast
from typing import Dict, List, Tuple, Any, Union, get_type_hints
import dspy
import llm

def _parse_type_annotation(annotation_str: str) -> Any:
    """Parse a type annotation string into a Python type."""
    try:
        # Handle Literal types
        if annotation_str.startswith('Literal['):
            values = re.findall(r'"([^"]*)"', annotation_str)
            if not values:
                values = re.findall(r'\'([^\']*)\'', annotation_str)
            from typing import Literal
            return Literal[tuple(values)]
        
        # Handle List types
        if annotation_str.startswith('List['):
            inner_type = _parse_type_annotation(annotation_str[5:-1])
            return List[inner_type]
        
        # Handle Dict types
        if annotation_str.startswith('Dict['):
            key_type, value_type = annotation_str[5:-1].split(',')
            return Dict[_parse_type_annotation(key_type.strip()), _parse_type_annotation(value_type.strip())]
        
        # Handle constrained types
        if annotation_str.startswith('conint('):
            args = re.findall(r'(\w+)=(\d+)', annotation_str)
            return int
        if annotation_str.startswith('confloat('):
            args = re.findall(r'(\w+)=(\d+)', annotation_str)
            return float
        
        # Handle basic types
        if annotation_str == 'str':
            return str
        if annotation_str == 'int':
            return int
        if annotation_str == 'float':
            return float
        if annotation_str == 'bool':
            return bool
        
        return str  # Default to str for unknown types
    except:
        return str

def _parse_field(field_str: str) -> Tuple[str, Any]:
    """Parse a field string into name and type."""
    parts = field_str.strip().split(':')
    if len(parts) == 1:
        return parts[0], str
    name, type_str = parts
    return name.strip(), _parse_type_annotation(type_str.strip())

def _parse_signature(signature: str):
    """Parse a DSPy module signature into input and output fields."""
    parts = signature.split('->')
    if len(parts) != 2:
        raise ValueError("Invalid signature format. Expected: inputs -> outputs")
    
    input_str, output_str = parts
    input_fields = [_parse_field(f) for f in input_str.split(',') if f.strip()]
    output_fields = [_parse_field(f) for f in output_str.split(',') if f.strip()]
    
    # Return both field names and types
    return dict(input_fields), dict(output_fields)

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
        
        # Create module instance with type information
        module = module_class(signature=signature, input_fields=input_fields, output_fields=output_fields)
        
        # Run the module
        result = module.forward(**kwargs)
        
        # Return the result
        if hasattr(result, 'text'):
            return result.text
        elif hasattr(result, 'answer'):
            return result.answer
        elif hasattr(result, '__dict__'):
            # Convert structured output to string
            output_str = "Prediction(\n"
            for key, value in result.__dict__.items():
                output_str += f"    {key}={repr(value)},\n"
            output_str += ")"
            return output_str
        else:
            return str(result)
            
    except Exception as e:
        print(f"Error running DSPy module: {str(e)}", file=sys.stderr)
        raise
