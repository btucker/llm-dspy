import llm
import dspy
import re
import click
import sys
import litellm
from typing import List, Tuple, Dict, Any

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

def run_dspy_module(module_name: str, signature: str, kwargs: Dict[str, str]) -> str:
    """Run a DSPy module with the given signature and keyword arguments."""
    try:
        module_class = getattr(dspy, module_name)
    except AttributeError:
        raise ValueError(f"DSPy module {module_name} not found. Available modules: ChainOfThought, ProgramOfThought, Predict")
    
    # Create module instance with signature
    module_instance = module_class(signature=signature)
    
    # Parse signature to get output fields
    output_fields = [field.split(':')[0].strip() for field in signature.split('->')[1].strip().split(',')]
    output_field = output_fields[0]  # Use first output field
    
    # Execute module and extract response
    response = module_instance.forward(**kwargs)
    
    # Get the output field value
    try:
        result = getattr(response, output_field)
        # Try to get the most appropriate string representation
        if isinstance(result, str):
            return result
        
        # Try common field names in order
        for field in ['text', 'answer', 'response', 'output', 'result']:
            if hasattr(result, field):
                value = getattr(result, field)
                if isinstance(value, str):
                    return value
        
        # If none of the above worked, try getting the field directly from response
        for field in ['text', 'answer', 'response', 'output', 'result']:
            if hasattr(response, field):
                value = getattr(response, field)
                if isinstance(value, str):
                    return value
        
        # If still no string found, convert to string
        return str(result)
    except AttributeError:
        # Try getting the field directly from response
        for field in ['text', 'answer', 'response', 'output', 'result']:
            if hasattr(response, field):
                value = getattr(response, field)
                if isinstance(value, str):
                    return value
        
        # If all else fails, try to convert response to string
        return str(response)

@llm.hookimpl
def register_commands(cli: click.Group) -> None:
    """Register the DSPy command with LLM."""
    
    class DSPyCommand(click.Command):
        """Dynamic command that adds options based on the signature."""
        def __init__(self):
            super().__init__(name='dspy', callback=self.callback, help="""
                Run a DSPy module with a given signature and inputs.
                
                You can specify the module and signature in two ways:
                
                1. As separate arguments:
                   llm dspy "ChainOfThought" "question -> answer" "What is 2+2?"
                   llm dspy "ChainOfThought" "context, question -> answer" "Here is context" "What about it?"
                
                2. Combined in parentheses:
                   llm dspy "ChainOfThought(question -> answer)" "What is 2+2?"
                   llm dspy "ChainOfThought(context, question -> answer)" "Here is context" "What about it?"
                
                You can also use named options instead of positional arguments:
                llm dspy "ChainOfThought(question -> answer)" --question "What is 2+2?"
                llm dspy "ChainOfThought(context, question -> answer)" --context "Here is context" --question "What about it?"
                
                Fields like 'context', 'background', 'documents', or 'knowledge' will trigger RAG functionality.
                When using RAG fields, the plugin will use DSPy's built-in RAG capabilities to:
                1. Transform the query for better retrieval
                2. Retrieve relevant passages
                3. Rewrite the context to be more focused on the question
            """)
            self.params = [
                click.Argument(['module_and_signature']),
                click.Argument(['inputs'], nargs=-1),
                click.Option(['-v', '--verbose'], is_flag=True, help='Enable verbose logging')
            ]
            self._input_fields = []
            self._module_name = None
            self._signature = None
            
            # Initialize DSPy RAG components
            self._retriever = None
            self._rag_module = None
        
        def _parse_module_and_signature(self, module_and_signature: str) -> Tuple[str, str]:
            """Parse module name and signature from combined string."""
            # Try to parse as ModuleName(signature)
            match = re.match(r'^([A-Za-z0-9_]+)\((.*)\)$', module_and_signature)
            if match:
                return match.group(1), match.group(2)
            
            # If no parentheses, treat as module name only
            if '(' not in module_and_signature and ')' not in module_and_signature:
                return module_and_signature, None
            
            raise click.UsageError("Invalid module format. Expected 'ModuleName(inputs -> outputs)' or 'ModuleName'")
        
        def _extract_input_fields(self, signature: str) -> List[str]:
            """Extract input fields from a signature string."""
            if '->' not in signature:
                raise click.UsageError("Invalid module signature format. Expected 'inputs -> outputs'")
            
            try:
                inputs = signature.split('->')[0].strip()
                if not inputs:
                    raise click.UsageError("Invalid module signature format. No input fields specified")
                
                fields = [field.strip() for field in inputs.split(',')]
                if not all(fields):
                    raise click.UsageError("Invalid module signature format. Empty field name")
                
                return fields
            except Exception as e:
                if not isinstance(e, click.UsageError):
                    raise click.UsageError("Invalid module signature format. Expected 'inputs -> outputs'")
                raise
        
        def _setup_rag(self):
            """Set up DSPy RAG components if not already initialized."""
            if self._retriever is None:
                # Create a retriever that uses query transformation
                self._retriever = dspy.Retrieve(k=3)  # Get top 3 passages
                
                # Create a RAG module that combines retrieval and generation
                class RAGModule(dspy.Module):
                    def __init__(self):
                        super().__init__()
                        self.retrieve = dspy.Retrieve(k=3)
                        self.generate = dspy.ChainOfThought("context, question -> answer")
                    
                    def forward(self, question):
                        # First retrieve relevant passages
                        passages = self.retrieve(question).passages
                        
                        # Combine passages into context
                        context = "\n\n".join(passages)
                        
                        # Generate answer using the context
                        return self.generate(context=context, question=question)
                
                self._rag_module = RAGModule()
        
        def _process_rag_field(self, field: str, value: str, query: str) -> str:
            """Process a RAG field using DSPy's retrieval capabilities."""
            try:
                # Create a retriever for this collection
                retriever = LLMRetriever(value)
                
                # Get search results
                results = retriever(query)
                
                # Return the first result or a default message
                return results.passages[0]['text'] if results.passages else "No relevant context found."
                
            except Exception as e:
                raise click.UsageError(f"Error accessing collection '{value}': {str(e)}")
        
        def callback(self, module_and_signature: str, inputs: tuple, verbose: bool = False, **kwargs) -> None:
            """Run a DSPy module with a given signature and inputs."""
            try:
                if not self._module_name or not self._signature:
                    self._module_name, self._signature = self._parse_module_and_signature(module_and_signature)
                    if not self._signature:
                        raise click.UsageError("Missing module signature")
                    self._input_fields = self._extract_input_fields(self._signature)
                
                # Process each input field
                processed_kwargs = {}
                click.echo(f"Processing input fields: {self._input_fields}", err=True)
                click.echo(f"Available kwargs: {kwargs}", err=True)
                click.echo(f"Positional inputs: {inputs}", err=True)
                
                # First try to get values from named options
                for i, field in enumerate(self._input_fields):
                    # Try named option first
                    value = kwargs.get(field)
                    
                    # If not found, try positional argument
                    if value is None and i < len(inputs):
                        value = inputs[i]
                    
                    if value is None:
                        click.echo(f"Missing input for field: {field}", err=True)
                        raise click.UsageError(f"Missing input for field: {field}")
                    
                    # Check if this field should trigger RAG
                    if field in ['context', 'background', 'documents', 'knowledge']:
                        # Only use RAG if the value looks like a collection name (no spaces, reasonable length)
                        if len(value.split()) == 1 and len(value) <= 50:
                            # Use another field as query if available, otherwise use remaining inputs
                            query_fields = ['question', 'query', 'prompt']
                            query = next((kwargs[f] for f in query_fields if f in kwargs), None)
                            if not query:
                                # Try to get query from next positional argument
                                next_index = i + 1
                                if next_index < len(inputs):
                                    query = inputs[next_index]
                                else:
                                    # Use all non-collection inputs as query
                                    query = ' '.join(v for k, v in kwargs.items() 
                                                   if k not in ['context', 'background', 'documents', 'knowledge', 'verbose']
                                                   and v is not None)
                            
                            click.echo(f"Using DSPy retrieval with collection '{value}' and query: {query}", err=True)
                            processed_kwargs[field] = self._process_rag_field(field, value, query)
                        else:
                            # Use the value directly as context
                            processed_kwargs[field] = value
                    else:
                        processed_kwargs[field] = value

                # Run the module
                try:
                    click.echo(f"Running module {self._module_name} with kwargs: {processed_kwargs}", err=True)
                    result = run_dspy_module(self._module_name, self._signature, processed_kwargs)
                    click.echo(result)
                    if verbose:
                        sys.modules['dspy'].inspect_history()
                except ValueError as e:
                    # Preserve the original error message
                    ctx = click.get_current_context()
                    click.echo(str(e), err=True)
                    ctx.exit(1)
                except Exception as e:
                    raise click.ClickException(str(e))
            except click.UsageError as e:
                ctx = click.get_current_context()
                click.echo(str(e), err=True)
                click.echo(ctx.get_help(), err=True)
                ctx.exit(2)
        
        def parse_args(self, ctx, args):
            # Reset state
            self.params = [
                click.Argument(['module_and_signature']),
                click.Argument(['inputs'], nargs=-1),
                click.Option(['-v', '--verbose'], is_flag=True, help='Enable verbose logging')
            ]
            self._input_fields = []
            self._module_name = None
            self._signature = None
            
            # Pre-parse to get the module and signature
            if len(args) >= 1:
                module_and_signature = args[0]
                try:
                    self._module_name, self._signature = self._parse_module_and_signature(module_and_signature)
                    
                    # If signature wasn't in parentheses and we have another arg, use it as signature
                    if self._signature is None and len(args) >= 2:
                        self._signature = args[1]
                        # Remove the signature argument since we've handled it
                        args = [args[0]] + list(args[2:])
                    
                    if self._signature:
                        click.echo(f"Parsing signature: {self._signature}", err=True)
                        self._input_fields = self._extract_input_fields(self._signature)
                        click.echo(f"Extracted input fields: {self._input_fields}", err=True)
                        
                        # Add options for each input field
                        for field in self._input_fields:
                            help_text = f"Value for the {field} field"
                            if field in ['context', 'background', 'documents', 'knowledge']:
                                help_text += " (collection name for RAG)"
                            # Only add if not already present
                            if not any(p.name == field for p in self.params):
                                self.params.append(click.Option(['--' + field], help=help_text))
                except click.UsageError:
                    raise
                except Exception as e:
                    raise click.UsageError(str(e))
            
            return super().parse_args(ctx, args)
    
    cli.add_command(DSPyCommand())

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
            passages = [{"text": result} for result in results]
            
            # Return in DSPy's expected format
            return dspy.Prediction(passages=passages)
            
        except Exception as e:
            # Log the error and return empty results
            print(f"Error retrieving from collection '{self.collection_name}': {str(e)}", file=sys.stderr)
            return dspy.Prediction(passages=[])
