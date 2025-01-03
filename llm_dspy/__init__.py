import llm
import dspy
import re
import click
import sys
import litellm
from typing import List, Tuple, Dict, Any, Optional
from dspy.primitives.prediction import Prediction

__all__ = ['run_dspy_module', 'register_commands']

# Core LLM Adapter
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
        
        self.kwargs = {
            "temperature": 0.7,  # Default temperature
            "max_tokens": 1000,  # Default max tokens
        }

    def __call__(self, prompt: str, **kwargs) -> str:
        response = self.llm.prompt(prompt)
        # Force the response to complete and get the text
        for chunk in response:
            pass
        return response.text_or_raise()

    def basic_create(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Basic completion creation that follows OpenAI's format."""
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt.get("messages", [])
        
        # Extract the actual prompt from the messages
        prompt_text = "\n".join(msg["content"] for msg in messages)
        response = self.__call__(prompt_text, **kwargs)
        
        return {
            "choices": [{
                "text": response,
                "message": {
                    "content": response,
                    "role": "assistant"
                }
            }],
            "model": "local",  # Use a generic model name
            "usage": {"total_tokens": 0}  # We don't track token usage
        }

# DSPy Module Runner
def run_dspy_module(module_name: str, signature: str, prompt: Tuple[str, ...]) -> str:
    """Run a DSPy module with the given signature and prompt."""
    try:
        module_class = getattr(dspy, module_name)
    except AttributeError:
        raise ValueError(f"DSPy module {module_name} not found")
    
    # Configure DSPy to use our LLM adapter
    adapter = LLMAdapter()
    
    # Configure DSPy
    dspy.configure(lm=dspy.LM(model="local"))
    
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

# CLI Integration
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

# Configure LiteLLM to use our adapter
def completion_with_adapter(**kwargs):
    messages = kwargs.get("messages", [])
    prompt = "\n".join(msg["content"] for msg in messages)
    response = adapter(prompt)
    return {
        "choices": [{
            "text": response,
            "message": {
                "content": response,
                "role": "assistant"
            },
            "index": 0,
            "finish_reason": "stop"
        }],
        "model": "local",
        "object": "chat.completion",
        "usage": {"total_tokens": 0}
    }

# Register our completion function with LiteLLM
litellm.completion = completion_with_adapter

# Configure DSPy
dspy.configure(lm=dspy.LM(model="local"))
