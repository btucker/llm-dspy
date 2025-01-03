import llm
import dspy
import re
import click
from typing import List

def run_dspy_module(module_name: str, signature: str, prompt: str) -> str:
    """Run a DSPy module with the given signature and prompt."""
    try:
        module_class = getattr(dspy, module_name)
    except AttributeError:
        raise ValueError(f"DSPy module {module_name} not found")
    
    # Create module instance with signature - let DSPy handle signature parsing
    module_instance = module_class(signature=signature)
    
    # Extract input field name from signature
    input_field = signature.split('->')[0].strip().split(',')[0].strip()
    
    # Get the response from DSPy - pass prompt as a keyword argument
    response = module_instance.forward(**{input_field: prompt})
    return str(response)

@llm.hookimpl
def register_commands(cli):
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
