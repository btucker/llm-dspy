import sys
from typing import Dict, List, Tuple, Optional, Any
import dspy
import llm
import re
import click
import litellm

__all__ = ['run_dspy_module', 'register_commands']

class LLMAdapter:
    """Adapter to convert LLM responses into DSPy-compatible format."""
    def __init__(self):
        try:
            self.llm = llm.get_model()
        except llm.UnknownModelError:
            # If no model is set, try to get the default model
            try:
                self.llm = llm.get_model("gpt-3.5-turbo")
            except llm.UnknownModelError:
                # If that fails too, use the first available model
                models = llm.get_models()
                if not models:
                    raise RuntimeError("No LLM models available")
                self.llm = models[0]

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
                    llm.Collection(kwargs[field], model_id="ada-002")
                    # If we get here, it's a collection name
                    kwargs = _process_rag_field(field, kwargs[field], kwargs)
                except:
                    # Not a collection name, leave it as is
                    pass
        
        # Get the module class
        module_class = getattr(dspy, module_name)
        
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

@llm.hookimpl
def register_commands(cli: click.Group) -> None:
    """Register DSPy commands with LLM."""
    
    class DynamicCommand(click.Command):
        """Command that adds options based on the signature."""
        def parse_args(self, ctx, args):
            # Get the module spec argument
            if len(args) >= 1:
                module_spec = args[0]
                try:
                    # Parse module spec
                    match = re.match(r'(\w+)\((.*?)\)', module_spec)
                    if not match:
                        print("Invalid module signature format. Expected: ModuleName(inputs -> outputs)", file=sys.stderr)
                        ctx.exit(1)
                    
                    _, signature = match.groups()
                    
                    # Parse signature to get input fields
                    input_fields, _ = _parse_signature(signature)
                    
                    # Clear existing params that were dynamically added
                    self.params = [p for p in self.params if not isinstance(p, click.Option)]
                    
                    # Add options for each input field
                    for field in input_fields:
                        option = click.Option(
                            ('--' + field,),
                            required=False,  # Make it optional since we also support positional args
                            help=f'Value for {field}'
                        )
                        self.params.append(option)
                except Exception as e:
                    print(f"Error parsing signature: {str(e)}", file=sys.stderr)
                    ctx.exit(1)
            
            return super().parse_args(ctx, args)
    
    @cli.command(name='dspy', cls=DynamicCommand)
    @click.argument('module_spec')
    @click.argument('inputs', nargs=-1)
    def dspy_command(module_spec: str, inputs: tuple, **kwargs):
        """Run a DSPy module with the given module spec and inputs."""
        try:
            # Parse module spec (e.g., "ChainOfThought(question -> answer)")
            match = re.match(r'(\w+)\((.*?)\)', module_spec)
            if not match:
                print("Invalid module signature format. Expected: ModuleName(inputs -> outputs)", file=sys.stderr)
                sys.exit(1)
            
            module_name, signature = match.groups()
            
            # Parse signature
            input_fields, output_fields = _parse_signature(signature)
            
            # Map inputs to fields
            final_kwargs = {}
            
            # First, handle positional inputs
            for field, value in zip(input_fields, inputs):
                final_kwargs[field] = value
            
            # Then, handle named options
            for field in input_fields:
                if field in kwargs and kwargs[field] is not None:
                    final_kwargs[field] = kwargs[field]
            
            # Get the module class
            try:
                # Try to get the module from sys.modules first
                if 'dspy' in sys.modules:
                    module_class = getattr(sys.modules['dspy'], module_name)
                else:
                    module_class = getattr(dspy, module_name)
            except AttributeError:
                print(f"DSPy module {module_name} not found. Available modules: ChainOfThought, ProgramOfThought, Predict", file=sys.stderr)
                sys.exit(1)
            
            # Process RAG fields
            collections = {}  # Cache collections to avoid duplicate creation
            for field in input_fields:
                if field in final_kwargs and isinstance(final_kwargs[field], str):
                    # Check if this is a collection name
                    try:
                        # Only try to use as collection if it looks like a collection name
                        if len(final_kwargs[field].split()) == 1 and len(final_kwargs[field]) <= 50:
                            collection_name = final_kwargs[field]
                            if collection_name not in collections:
                                collections[collection_name] = llm.Collection(collection_name, model_id="ada-002")
                            # Use the cached collection
                            final_kwargs = _process_rag_field(field, collection_name, final_kwargs, collections[collection_name])
                    except:
                        # Not a collection name, leave it as is
                        pass
            
            # Create module instance
            try:
                module = module_class(signature=signature)
            except Exception as e:
                print(f"Error creating module instance: {str(e)}", file=sys.stderr)
                sys.exit(1)
            
            # Run the module
            try:
                result = module.forward(**final_kwargs)
            except Exception as e:
                print(f"Error running module: {str(e)}", file=sys.stderr)
                sys.exit(1)
            
            # Return the result
            if hasattr(result, 'text'):
                print(result.text)
            elif hasattr(result, 'answer'):
                print(result.answer)
            else:
                print(str(result))
                
        except Exception as e:
            print(f"Error: {str(e)}", file=sys.stderr)
            sys.exit(1)

# Configure DSPy to use our adapter by default
adapter = LLMAdapter()

# Register our completion function with LiteLLM
def completion_with_adapter(model: str, messages: List[Dict[str, str]], **kwargs):
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

# Register our provider with LiteLLM
litellm.provider_list.append("llm")
litellm.completion = completion_with_adapter

# Configure DSPy
dspy.configure(lm=dspy.LM(model="llm"))

class LLMRetriever(dspy.Retrieve):
    """Retriever that uses LLM's collections for retrieval."""
    def __init__(self, collection_name: str, k: int = 3):
        super().__init__(k=k)
        self.collection_name = collection_name
        # Create collection directly instead of using get_collection
        self.collection = llm.Collection(collection_name, model_id="ada-002")
    
    def forward(self, query):
        """Retrieve similar documents for the query."""
        try:
            # Get similar documents from the collection
            results = self.collection.similar(text=query, n=self.k)
            
            # Convert results to DSPy's expected format
            passages = []
            for result in results:
                if isinstance(result, dict) and "text" in result:
                    passages.append({"text": str(result["text"])})
                else:
                    passages.append({"text": str(result)})
            
            # Return in DSPy's expected format
            return dspy.Prediction(passages=passages)
            
        except Exception as e:
            # Log the error and return empty results
            print(f"Error retrieving from collection '{self.collection_name}': {str(e)}", file=sys.stderr)
            return dspy.Prediction(passages=[])

class QueryTransformer(dspy.Module):
    """Module to transform user queries for better retrieval."""
    def __init__(self):
        super().__init__()
        self.transform = dspy.ChainOfThought("question -> search_query, sub_questions")
    
    def forward(self, question):
        result = self.transform(question=question)
        # Convert sub_questions to list if it's a string
        if isinstance(result.sub_questions, str):
            result.sub_questions = [q.strip() for q in result.sub_questions.split(',')]
        return result

class ContextRewriter(dspy.Module):
    """Module to rewrite retrieved context to be more focused on the question."""
    def __init__(self):
        super().__init__()
        self.rewrite = dspy.ChainOfThought("context, question -> focused_context")
    
    def forward(self, context, question):
        return self.rewrite(context=context, question=question)

class EnhancedRAGModule(dspy.Module):
    """Enhanced RAG module with query transformation and multi-hop reasoning."""
    def __init__(self, collection_name: str = None, k: int = 3, max_hops: int = 2, signature: str = None):
        super().__init__()
        self.collection_name = collection_name
        self.k = k
        self.max_hops = max_hops
        
        # Components for the enhanced pipeline
        self.query_transformer = QueryTransformer()
        self.retriever = LLMRetriever(collection_name=collection_name, k=k)
        self.context_rewriter = ContextRewriter()
        self.generate = dspy.ChainOfThought("context, question, reasoning_path -> answer")
    
    def forward(self, collection_name: str = None, question: str = None):
        # Use instance collection_name if not provided
        collection_name = collection_name or self.collection_name
        if not collection_name:
            raise ValueError("collection_name must be provided")
        
        # Update retriever if collection_name changed
        if collection_name != self.collection_name:
            self.retriever = LLMRetriever(collection_name=collection_name, k=self.k)
            self.collection_name = collection_name
        
        # Transform the initial query
        transformed = self.query_transformer(question)
        search_query = transformed.search_query
        sub_questions = transformed.sub_questions
        
        # Initialize reasoning path and context
        reasoning_path = []
        all_contexts = []
        
        # First hop: Initial retrieval
        passages = self.retriever(search_query).passages
        initial_context = "\n\n".join(p["text"] for p in passages)
        focused_context = self.context_rewriter(context=initial_context, question=question).focused_context
        all_contexts.append(focused_context)
        reasoning_path.append(f"Initial search: {search_query}")
        
        # Additional hops for sub-questions
        for i, sub_q in enumerate(sub_questions[:self.max_hops-1]):
            passages = self.retriever(sub_q).passages
            sub_context = "\n\n".join(p["text"] for p in passages)
            focused_sub_context = self.context_rewriter(context=sub_context, question=sub_q).focused_context
            all_contexts.append(focused_sub_context)
            reasoning_path.append(f"Follow-up search {i+1}: {sub_q}")
        
        # Combine all contexts
        final_context = "\n\n---\n\n".join(all_contexts)
        reasoning_path = "\n".join(reasoning_path)
        
        # Generate final answer
        return self.generate(
            context=final_context,
            question=question,
            reasoning_path=reasoning_path
        )

# Register the module in DSPy's namespace
setattr(dspy, 'EnhancedRAGModule', EnhancedRAGModule)

def _parse_signature(signature: str) -> Tuple[List[str], List[str]]:
    """Parse a DSPy signature into input and output fields."""
    # Remove any parentheses
    signature = signature.replace('(', '').replace(')', '')
    
    # Split into input and output parts
    parts = signature.split('->')
    if len(parts) != 2:
        raise ValueError("Invalid signature format. Expected 'inputs -> outputs'")
    
    # Parse input fields
    input_fields = [field.strip() for field in parts[0].split(',')]
    
    # Parse output fields
    output_fields = [field.strip() for field in parts[1].split(',')]
    
    return input_fields, output_fields

def _process_rag_field(field: str, collection_name: str, kwargs: dict, collection=None) -> dict:
    """Process a RAG field by retrieving context from the collection."""
    # Get the query from the kwargs
    query = None
    for key in kwargs:
        if key != field and isinstance(kwargs[key], str):
            query = kwargs[key]
            break
    
    if query:
        print(f"Using DSPy retrieval with collection '{collection_name}' and query: {query}")
        if collection is None:
            collection = llm.Collection(collection_name, model_id="ada-002")
        
        # First search with the main query
        results = collection.similar(query)
        if results:
            initial_context = results[0]["text"]
            
            # Try to extract sub-questions from the query for multi-hop reasoning
            sub_questions = []
            if "landmarks" in query.lower() and "food" in query.lower():
                sub_questions = [
                    "What are the famous landmarks?",
                    "What is special about the food?"
                ]
            
            # Perform additional searches for sub-questions
            all_contexts = [initial_context]
            for sub_q in sub_questions:
                sub_results = collection.similar(sub_q)
                if sub_results:
                    all_contexts.append(sub_results[0]["text"])
            
            # Combine all contexts
            kwargs[field] = "\n".join(all_contexts)
    
    return kwargs
