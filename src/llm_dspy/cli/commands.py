import sys
import re
import click
import llm
import dspy
from ..core.module import _parse_signature, _process_rag_field
from dspy.signatures.signature import ensure_signature

@llm.hookimpl
def register_commands(cli: click.Group) -> None:
    """Register DSPy commands with LLM."""
    
    class DynamicCommand(click.Command):
        """Command that adds options based on the signature."""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Add verbose flag
            self.params.append(
                click.Option(
                    ('-v', '--verbose'),
                    is_flag=True,
                    help='Enable verbose output'
                )
            )

        def parse_args(self, ctx, args):
            # Get the module spec argument
            if len(args) >= 1:
                module_spec = args[0]
                try:
                    # Parse module spec
                    match = re.match(r'(\w+)\(((?:[^()[\]]*|\[[^\[\]]*\]|\([^()]*\))*)\)', module_spec)
                    if not match:
                        print("Invalid module signature format. Expected: ModuleName(inputs -> outputs)", file=sys.stderr)
                        ctx.exit(1)
                    
                    _, signature_str = match.groups()
                    
                    # Parse signature using DSPy's ensure_signature
                    signature = ensure_signature(signature_str)
                    input_fields = signature.input_fields
                    
                    # Clear existing params that were dynamically added
                    self.params = [p for p in self.params if not isinstance(p, click.Option) or p.name == 'verbose']
                    
                    # Add options for each input field
                    for field_name in input_fields:
                        option = click.Option(
                            ('--' + field_name,),
                            required=False,  # Make it optional since we also support positional args
                            help=f'Value for {field_name}'
                        )
                        self.params.append(option)
                except Exception as e:
                    print(f"Error parsing signature: {str(e)}", file=sys.stderr)
                    ctx.exit(1)
            
            return super().parse_args(ctx, args)
    
    @cli.command(name='dspy', cls=DynamicCommand)
    @click.argument('module_spec')
    @click.argument('inputs', nargs=-1)
    def dspy_command(module_spec: str, inputs: tuple, verbose: bool = False, **kwargs):
        """Run a DSPy module with the given module spec and inputs."""
        try:
            # Parse module spec (e.g., "ChainOfThought(question -> answer)")
            match = re.match(r'(\w+)\(((?:[^()[\]]*|\[[^\[\]]*\]|\([^()]*\))*)\)', module_spec)
            if not match:
                print("Invalid module signature format. Expected: ModuleName(inputs -> outputs)", file=sys.stderr)
                sys.exit(1)
            
            module_name, signature = match.groups()
            
            # Parse signature using DSPy's ensure_signature
            signature = ensure_signature(signature)
            input_fields = signature.input_fields
            output_fields = signature.output_fields
            
            if verbose:
                print(f"Module: {module_name}", file=sys.stderr)
                print(f"Input fields: {input_fields}", file=sys.stderr)
                print(f"Output fields: {output_fields}", file=sys.stderr)
            
            # Initialize final kwargs
            final_kwargs = {}
            
            # Handle stdin input
            stdin_data = None
            if not sys.stdin.isatty():
                stdin_data = sys.stdin.read().strip()
            
            # Process inputs based on number of input fields
            if len(input_fields) == 1:
                # Single input field case
                field = next(iter(input_fields))
                if field in kwargs and kwargs[field] == "stdin":
                    # Explicit stdin for this field
                    if stdin_data is None:
                        print(f"Error: --{field} set to 'stdin' but no data provided via stdin", file=sys.stderr)
                        sys.exit(1)
                    final_kwargs[field] = stdin_data
                elif field in kwargs and kwargs[field]:
                    # Named option provided
                    final_kwargs[field] = kwargs[field]
                elif inputs:
                    # Positional argument provided
                    final_kwargs[field] = inputs[0]
                elif stdin_data is not None:
                    # Implicit stdin
                    final_kwargs[field] = stdin_data
                else:
                    print(f"Error: No input provided for field '{field}'", file=sys.stderr)
                    sys.exit(1)
            else:
                # Multiple input fields case - must use named options
                for field in input_fields:
                    if field not in kwargs or not kwargs[field]:
                        print(f"Error: Missing required option --{field}", file=sys.stderr)
                        sys.exit(1)
                    
                    if kwargs[field] == "stdin":
                        if stdin_data is None:
                            print(f"Error: --{field} set to 'stdin' but no data provided via stdin", file=sys.stderr)
                            sys.exit(1)
                        final_kwargs[field] = stdin_data
                    else:
                        final_kwargs[field] = kwargs[field]
            
            if verbose:
                print(f"Final kwargs: {final_kwargs}", file=sys.stderr)
            
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
            
            # Create module instance with type information
            try:
                module = module_class(signature=signature)
            except Exception as e:
                print(f"Error creating module instance: {str(e)}", file=sys.stderr)
                sys.exit(1)
            
            # Run the module
            try:
                result = module.forward(**final_kwargs)
                if verbose:
                    print(f"Module result: {result}", file=sys.stderr)
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
