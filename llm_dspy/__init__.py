import llm
import dspy
import re
import click
import shlex
from typing import List, Tuple, Dict, Any, Optional
from dspy.clients.provider import Provider
from dspy.primitives.prediction import Prediction

__all__ = ['run_dspy_module', 'register_commands']

# Core LLM Adapter
class LLMAdapter:
    """Adapter to convert LLM responses into DSPy-compatible format."""
    def __init__(self):
        self.llm = llm.get_model()

    def __call__(self, prompt: str, **kwargs) -> str:
        response = self.llm.prompt(prompt)
        # Force the response to complete and get the text
        for chunk in response:
            pass
        return response.text_or_raise()

# DSPy Provider Implementation
class LLMProvider(Provider):
    """DSPy provider that uses LLM for completions."""
    def __init__(self):
        super().__init__()
        self.finetunable = False
        self._adapter = None

    @staticmethod
    def is_provider_model(model: str) -> bool:
        return model.startswith("llm/")

    def get_adapter(self):
        if self._adapter is None:
            self._adapter = LLMAdapter()
        return self._adapter

    def __call__(self, model: str, prompt=None, messages=None, **kwargs):
        adapter = self.get_adapter()

        if prompt is not None:
            response = adapter(prompt, **kwargs)
        elif messages is not None:
            prompt = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)
            response = adapter(prompt, **kwargs)
        else:
            raise ValueError("Either prompt or messages must be provided")

        # Create a dictionary with all possible output fields
        output_dict = {
            "answer": response,
            "text": response,
            "response": response,
            "output": response,
            "result": response,
            "confidence": 1.0  # Default confidence
        }
        return Prediction(**output_dict)

# LiteLLM Integration
def litellm_completion(request: Dict[str, Any], num_retries: int = 8, cache={"no-cache": True, "no-store": True}, **kwargs) -> Dict[str, Any]:
    """Handle litellm completion requests by using our LLM adapter."""
    adapter = LLMAdapter()
    messages = request.get("messages", [])
    prompt = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages) if messages else request.get("prompt", "")
    
    response = adapter(prompt)
    
    return {
        "choices": [{
            "text": response,  # Add text field for non-chat models
            "message": {
                "content": response,
                "role": "assistant"
            }
        }],
        "model": request.get("model", "llm/default"),
        "usage": {"total_tokens": 0},  # We don't track token usage
    }

def cached_litellm_completion(request: Dict[str, Any], num_retries: int = 8) -> Dict[str, Any]:
    """Cached version of litellm completion."""
    return litellm_completion(request, num_retries=num_retries, cache={"no-cache": False, "no-store": False})

# DSPy Module Runner
def run_dspy_module(module_name: str, signature: str, prompt: str) -> str:
    """Run a DSPy module with the given signature and prompt."""
    try:
        module_class = getattr(dspy, module_name)
    except AttributeError:
        raise ValueError(f"DSPy module {module_name} not found")
    
    # Configure DSPy to use our LLM adapter
    provider = LLMProvider()
    dspy.configure(lm=dspy.LM(model="llm/default", provider=provider))
    
    # Create module instance with signature
    module_instance = module_class(signature=signature)
    
    # Parse signature and inputs
    input_fields = [field.strip() for field in signature.split('->')[0].strip().split(',')]
    output_fields = [field.strip() for field in signature.split('->')[1].strip().split(',')]
    output_field = output_fields[0]  # Use first output field
    
    # Handle input parsing
    try:
        if len(input_fields) == 1:
            # For single input, use the entire prompt
            kwargs = {input_fields[0]: prompt}
        else:
            # For multiple inputs, split by quotes and spaces
            # Click has already handled escaping and quoting for us
            parts = shlex.split(prompt)
            
            if len(parts) != len(input_fields):
                raise ValueError(f"Expected {len(input_fields)} inputs but got {len(parts)}")
            
            kwargs = dict(zip(input_fields, parts))
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
    def dspy(module_signature: Tuple[str, str], inputs: Tuple[str, ...]) -> None:
        """Run a DSPy module with a given signature and inputs.
        
        MODULE_SIGNATURE should be in the format: 'ModuleName(inputs -> outputs)'
        For example: 'ChainOfThought(question -> answer)'
        
        INPUTS are the input values to process. For single input signatures, all inputs
        are joined together. For multiple input signatures, the number of inputs must
        match the number of input fields in the signature.
        """
        try:
            module_name, signature = module_signature
            # Let run_dspy_module handle the input parsing based on signature
            result = run_dspy_module(module_name, signature, " ".join(inputs))
            click.echo(result)
        except Exception as e:
            raise click.ClickException(str(e))

# Monkey patch litellm and dspy.clients.lm
import litellm
litellm.completion = litellm_completion
litellm.cached_completion = cached_litellm_completion

import dspy.clients.lm
dspy.clients.lm.litellm_completion = litellm_completion
dspy.clients.lm.cached_litellm_completion = cached_litellm_completion

# Configure DSPy to use our provider by default
dspy.configure(lm=dspy.LM(model="llm/default", provider=LLMProvider()))
