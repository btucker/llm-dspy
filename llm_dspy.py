import llm
import dspy
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Union, ClassVar
import os
from pydantic import BaseModel

@dataclass
class DSPyConfig:
    """Configuration for a DSPy template"""
    module: str
    signature: Optional[Union[str, Dict[str, Any]]] = None
    config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if isinstance(self.signature, dict):
            # Convert dict to arrow syntax
            input_fields = self.signature.get('input_fields', [])
            output_fields = self.signature.get('output_fields', [])
            self.signature = f"{', '.join(input_fields)} -> {', '.join(output_fields)}"

class DSPyTemplate(llm.Template):
    """Template that uses DSPy modules"""
    
    class MissingVariables(Exception):
        pass
    
    def __init__(self, name: str, type: str, dspy: Dict[str, Any], prompt: Optional[str] = None, system: Optional[str] = None):
        super().__init__(name=name, type=type)
        self.dspy = DSPyConfig(**dspy)
        self.prompt = prompt
        self.system = system

    def evaluate(self, input_text: str, params: Optional[Dict[str, str]] = None) -> tuple[str, Optional[str]]:
        """Evaluate the template with the given input"""
        # Check for missing variables
        if params is None:
            params = {}
        
        if self.prompt:
            import re
            variables = set(re.findall(r'\$(\w+)', self.prompt))
            missing = variables - set(['input']) - set(params.keys())
            if missing:
                raise self.MissingVariables(f"Missing variables: {', '.join(sorted(missing))}")
        
        # Get the DSPy module class
        try:
            module_class = getattr(dspy, self.dspy.module)
        except AttributeError:
            raise ValueError(f"DSPy module {self.dspy.module} not found")
        
        # Create module instance with config and signature
        config = self.dspy.config or {}
        if self.dspy.signature:
            config['signature'] = self.dspy.signature
        
        # Create module instance
        module_instance = module_class(**config)
        
        # Get the structured prompt from DSPy
        response = module_instance.forward(input_text)
        
        # Return the structured prompt and system prompt
        return str(response.prompt), self.system
    
    def __str__(self) -> str:
        """Return a string representation of the template"""
        try:
            module_class = getattr(dspy, self.dspy.module)
            if self.dspy.signature:
                return f"{self.dspy.module}[{self.dspy.signature}]"
            return f"{self.dspy.module}(?)"
        except (AttributeError, TypeError):
            return f"{self.dspy.module}(?)"

@llm.hookimpl
def register_template_types() -> Dict[str, type]:
    """Register the DSPy template type with LLM"""
    return {"dspy": DSPyTemplate}
