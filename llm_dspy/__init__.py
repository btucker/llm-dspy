import llm
import dspy
import re
import click
from typing import List, Tuple, Optional, Dict, Any
from dspy.clients.base_lm import BaseLM

class LLMLanguageModel(BaseLM):
    """A DSPy language model that delegates to LLM."""
    def __init__(self, temperature: float = 0.0, max_tokens: int = 1000, cache: bool = True, **kwargs):
        # Use the default model from LLM
        model = llm.get_model()
        super().__init__(model=model, model_type='chat', temperature=temperature, max_tokens=max_tokens, cache=cache, **kwargs)

    def __call__(self, prompt: Optional[str] = None, messages: Optional[List[Dict[str, str]]] = None, **kwargs) -> str:
        """Make a request to the language model."""
        if messages:
            # Convert DSPy message format to LLM format
            prompt = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)
        
        if not prompt:
            raise ValueError("Either prompt or messages must be provided")

        response = self.model.prompt(prompt)
        return str(response)

def run_dspy_module(module_name: str, signature: str, prompt: str) -> str:
    """Run a DSPy module with the given signature and prompt."""
    try:
        module_class = getattr(dspy, module_name)
    except AttributeError:
        raise ValueError(f"DSPy module {module_name} not found")
    
    # Set up DSPy to use LLM
    dspy.settings.configure(lm=LLMLanguageModel())
    
    # Create module instance with signature - let DSPy handle signature parsing
    module_instance = module_class(signature=signature)
    
    # Extract input field name from signature
    input_field = signature.split('->')[0].strip().split(',')[0].strip()
    
    # Get the response from DSPy - pass prompt as a keyword argument
    response = module_instance.forward(**{input_field: prompt})
    return str(response)

@llm.hookimpl
def register_commands(cli: click.Group) -> None:
    """Register the DSPy command with LLM."""
    @cli.command()
    @click.argument("module_and_signature", type=str)
    @click.argument("prompt", nargs=-1)
    def dspy(module_and_signature: str, prompt: tuple[str, ...]) -> None:
        """Run a DSPy module with a given signature and prompt.
        
        MODULE_AND_SIGNATURE should be in the format: 'ModuleName(inputs -> outputs)'
        For example: 'ChainOfThought(question -> answer)'
        
        PROMPT is the text to process
        """
        try:
            # Extract module name and signature
            match = re.match(r"(\w+)\((.*)\)", module_and_signature)
            if not match:
                raise ValueError("Invalid module signature format. Expected: ModuleName(inputs -> outputs)")
            
            module_name, signature = match.groups()
            prompt_text = " ".join(prompt)
            
            # Run the DSPy module and print result
            result = run_dspy_module(module_name, signature, prompt_text)
            click.echo(result)
        except Exception as e:
            raise click.ClickException(str(e))
