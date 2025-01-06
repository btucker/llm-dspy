import sys
import re
import click
import llm
import dspy
import logging
from dspy.signatures.signature import ensure_signature
from ..utils import setup_logging
from ..rag.retriever import LLMRetriever

logger = logging.getLogger('llm_dspy.cli')

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
                        logger.error("Invalid module signature format. Expected: ModuleName(inputs -> outputs)")
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
                    logger.error(f"Error parsing signature: {str(e)}")
                    ctx.exit(1)
            
            return super().parse_args(ctx, args)
    
    @cli.command(name='dspy', cls=DynamicCommand)
    @click.argument('module_spec')
    @click.argument('inputs', nargs=-1)
    def dspy_command(module_spec: str, inputs: tuple, verbose: bool = False, **kwargs):
        """Run a DSPy module with the given module spec and inputs."""
        try:
            # Set up logging based on verbose flag
            setup_logging(verbose)
            
            # Parse module spec (e.g., "ChainOfThought(question -> answer)")
            match = re.match(r'(\w+)\(((?:[^()[\]]*|\[[^\[\]]*\]|\([^()]*\))*)\)', module_spec)
            if not match:
                logger.error("Invalid module signature format. Expected: ModuleName(inputs -> outputs)")
                sys.exit(1)
            
            module_name, signature = match.groups()
            
            # Parse signature using DSPy's ensure_signature
            signature = ensure_signature(signature)
            input_fields = signature.input_fields
            output_fields = signature.output_fields
            
            if verbose:
                logger.debug(f"Module: {module_name}")
                logger.debug(f"Input fields: {input_fields}")
                logger.debug(f"Output fields: {output_fields}")
            
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
                        logger.error(f"Error: --{field} set to 'stdin' but no data provided via stdin")
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
                    logger.error(f"Error: No input provided for field '{field}'")
                    sys.exit(1)
            else:
                # Multiple input fields case - must use named options
                for field in input_fields:
                    if field not in kwargs or not kwargs[field]:
                        logger.error(f"Error: Missing required option --{field}")
                        sys.exit(1)
                    
                    if kwargs[field] == "stdin":
                        if stdin_data is None:
                            logger.error(f"Error: --{field} set to 'stdin' but no data provided via stdin")
                            sys.exit(1)
                        final_kwargs[field] = stdin_data
                    else:
                        final_kwargs[field] = kwargs[field]
            
            # Check for collection names and retrieve context
            query = kwargs.get('query')
            if query:  # Only try RAG if we have a query
                for field, value in list(final_kwargs.items()):
                    try:
                        if hasattr(llm, 'collections') and value in llm.collections:
                            # This field contains a collection name, use RAG
                            collection = llm.collections[value]
                            retriever = LLMRetriever(collection_name=value, collection=collection)
                            result = retriever(query)
                            if result and result.passages:
                                context = "\n\n".join(p["text"] for p in result.passages)
                                logger.debug(f"Retrieved context for {field}: {context}")
                                final_kwargs[field] = context
                            else:
                                logger.warning(f"No passages retrieved for {field}")
                    except Exception as e:
                        logger.debug(f"Error checking collection for field {field}: {str(e)}")
                        continue  # Skip this field if there's an error
            
            if verbose:
                logger.debug(f"Final kwargs: {final_kwargs}")
            
            # Get the module class
            try:
                # Try to get the module from sys.modules first
                if 'dspy' in sys.modules:
                    module_class = getattr(sys.modules['dspy'], module_name)
                else:
                    module_class = getattr(dspy, module_name)
            except AttributeError:
                error_msg = f"DSPy module {module_name} not found. Available modules: ChainOfThought, ProgramOfThought, Predict"
                logger.error(error_msg)
                click.echo(error_msg, err=True)  # Ensure error is echoed to stderr
                sys.exit(1)
            
            # Create module instance with type information
            try:
                module = module_class(signature=signature)
            except Exception as e:
                logger.error(f"Error creating module instance: {str(e)}")
                sys.exit(1)
            
            # Run the module
            try:
                result = module.forward(**final_kwargs)
                if verbose:
                    logger.debug(f"Module result: {result}")
            except Exception as e:
                logger.error(f"Error running module: {str(e)}")
                sys.exit(1)
            
            # Return the result
            if hasattr(result, 'text'):
                click.echo(result.text)
            elif hasattr(result, 'answer'):
                click.echo(result.answer)
            else:
                click.echo(str(result))
                
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            sys.exit(1)
