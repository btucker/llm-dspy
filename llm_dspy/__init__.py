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

def run_dspy_module(module_name: str, signature: str, prompt: Tuple[str, ...]) -> str:
    """Run a DSPy module with the given signature and prompt."""
    try:
        module_class = getattr(dspy, module_name)
    except AttributeError:
        raise ValueError(f"DSPy module {module_name} not found")
    
    # Create module instance with signature
    module_instance = module_class(signature=signature)
    
    # Parse signature and inputs
    input_fields = [field.strip() for field in signature.split('->')[0].strip().split(',')]
    output_fields = [field.split(':')[0].strip() for field in signature.split('->')[1].strip().split(',')]
    output_field = output_fields[0]  # Use first output field
    
    # Handle input parsing
    try:
        # Convert string prompt to tuple if needed
        if isinstance(prompt, str):
            # If it's a string, split it on spaces while respecting quotes
            import shlex
            prompt = tuple(shlex.split(prompt))
            
        if len(prompt) < len(input_fields):
            raise ValueError(f"Expected {len(input_fields)} inputs but got {len(prompt)}")
        
        # If we have more prompts than input fields, join the excess with spaces
        if len(prompt) > len(input_fields):
            # Take the first n-1 prompts as-is
            processed_prompts = list(prompt[:len(input_fields)-1])
            # Join remaining prompts for the last input field
            processed_prompts.append(' '.join(prompt[len(input_fields)-1:]))
            kwargs = dict(zip(input_fields, processed_prompts))
        else:
            kwargs = dict(zip(input_fields, prompt))
    except Exception as e:
        raise ValueError(f"Failed to parse inputs: {str(e)}")
    
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
        
        # If none of the above worked, convert to string
        return str(result)
    except AttributeError:
        raise ValueError(f"Response does not have field '{output_field}'")

@llm.hookimpl
def register_commands(cli: click.Group) -> None:
    """Register the DSPy command with LLM."""
    
    class ModuleSignature(click.ParamType):
        """Custom parameter type for module signature parsing."""
        name = "module_signature"
        
        def convert(self, value, param, ctx):
            try:
                # Remove any escape characters as Click handles those
                value = value.replace("\\", "")
                match = re.match(r"(\w+)\((.*)\)", value)
                if not match:
                    self.fail("Invalid module signature format. Expected: ModuleName(inputs -> outputs)", param, ctx)
                return match.groups()
            except ValueError as e:
                self.fail(str(e), param, ctx)
    
    @cli.command()
    @click.argument("module_signature", type=ModuleSignature())
    @click.argument("inputs", nargs=-1, required=True)
    @click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
    def dspy(module_signature: Tuple[str, str], inputs: Tuple[str, ...], verbose: bool) -> None:
        """Run a DSPy module with a given signature and inputs.
        
        MODULE_SIGNATURE should be in the format: 'ModuleName(inputs -> outputs)'
        For example: 'ChainOfThought(question -> answer)'
        
        INPUTS are the input values to process. For single input signatures, all inputs
        are joined together. For multiple input signatures, each input should be properly
        quoted if it contains spaces.
        """
        try:
            module_name, signature = module_signature
            # Join inputs with spaces to preserve the original string
            result = run_dspy_module(module_name, signature, inputs)
            click.echo(result)
            if verbose:
                sys.modules['dspy'].inspect_history()
        except Exception as e:
            raise click.ClickException(str(e))

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
