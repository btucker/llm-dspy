import pytest
from llm_dspy import DSPyTemplate, DSPyConfig
import dspy
from unittest.mock import patch, MagicMock
from llm.plugins import pm
import llm_dspy

# Create a basic signature for testing
class BasicSignature(dspy.Signature):
    """Basic signature for testing"""
    input = dspy.InputField()
    output = dspy.OutputField()

# Mock response class
class MockResponse:
    def __init__(self, answer="42", explanation="Because it is"):
        self.answer = answer
        self.explanation = explanation

@pytest.fixture(autouse=True)
def mock_dspy_lm():
    """Mock DSPy's language model configuration"""
    with patch('dspy.ChainOfThought') as mock_cot:
        mock_instance = MagicMock()
        mock_instance.forward.return_value = MockResponse()
        mock_cot.return_value = mock_instance
        yield mock_cot

@pytest.fixture(autouse=True)
def register_plugin():
    """Register the DSPy plugin with the plugin manager"""
    from llm.plugins import load_plugins
    load_plugins()
    # No need to register again since load_plugins already registered it
    yield
    # No need to unregister since we didn't register it ourselves

def test_template_schema():
    """Test that template schema validation works"""
    template = DSPyTemplate(
        name="test",
        type="dspy",
        dspy={
            "module": "ChainOfThought",
            "signature": {
                "input_fields": ["question"],
                "output_fields": ["answer"],
                "description": "Basic QA signature"
            }
        },
        system="Be helpful",
        prompt="Process this: $input"
    )
    assert template.type == "dspy"
    assert template.dspy.module == "ChainOfThought"
    assert template.system == "Be helpful"
    assert template.prompt == "Process this: $input"

def test_basic_template():
    """Test basic template functionality without system prompt"""
    template = DSPyTemplate(
        name="test",
        type="dspy",
        dspy={
            "module": "ChainOfThought",
            "signature": {
                "input_fields": ["question"],
                "output_fields": ["answer"],
            }
        },
        prompt="Process this: $input"
    )
    prompt, system = template.evaluate("What is 2+2?")
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert prompt == "42"  # Mock response

def test_template_with_system():
    """Test template with system prompt"""
    template = DSPyTemplate(
        name="test",
        type="dspy",
        dspy={
            "module": "ChainOfThought",
            "signature": {
                "input_fields": ["question"],
                "output_fields": ["answer"],
            }
        },
        system="You are a helpful math tutor",
        prompt="Please solve this problem: $input"
    )
    prompt, system = template.evaluate("What is 2+2?")
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert prompt == "42"  # Mock response
    assert system == "You are a helpful math tutor"

def test_template_with_config():
    """Test template with DSPy module configuration"""
    template = DSPyTemplate(
        name="test",
        type="dspy",
        dspy={
            "module": "ChainOfThought",
            "config": {"max_steps": 5},
            "signature": {
                "input_fields": ["question"],
                "output_fields": ["answer"],
            }
        },
        prompt="Process this: $input"
    )
    assert template.dspy.config == {"max_steps": 5}
    prompt, _ = template.evaluate("What is 2+2?")
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert prompt == "42"  # Mock response

def test_template_with_predefined_signature():
    """Test template with a predefined DSPy signature"""
    template = DSPyTemplate(
        name="test",
        type="dspy",
        dspy={
            "module": "ChainOfThought",
            "signature": {
                "input_fields": ["question"],
                "output_fields": ["answer"],
            }
        },
        prompt="Process this: $input"
    )
    assert template.dspy.signature.input_fields == ["question"]
    assert template.dspy.signature.output_fields == ["answer"]
    prompt, _ = template.evaluate("What is 2+2?")
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert prompt == "42"  # Mock response

def test_template_with_custom_signature():
    """Test template with a custom DSPy signature"""
    template = DSPyTemplate(
        name="test",
        type="dspy",
        dspy={
            "module": "ChainOfThought",
            "signature": {
                "input_fields": ["question"],
                "output_fields": ["answer", "explanation"],
                "description": "A simple QA signature"
            }
        },
        prompt="Process this: $input"
    )
    assert template.dspy.signature.input_fields == ["question"]
    assert template.dspy.signature.output_fields == ["answer", "explanation"]
    assert template.dspy.signature.description == "A simple QA signature"
    prompt, _ = template.evaluate("What is 2+2?")
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert prompt == "42"  # Mock response

def test_template_with_inline_signature():
    """Test template with an inline DSPy signature string"""
    template = DSPyTemplate(
        name="test",
        type="dspy",
        dspy={
            "module": "ChainOfThought",
            "signature": {
                "input_fields": ["question"],
                "output_fields": ["answer"],
            }
        },
        prompt="Process this: $input"
    )
    prompt, _ = template.evaluate("What is 2+2?")
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert prompt == "42"  # Mock response

def test_template_with_typed_inline_signature():
    """Test template with a typed inline DSPy signature string"""
    template = DSPyTemplate(
        name="test",
        type="dspy",
        dspy={
            "module": "ChainOfThought",
            "signature": {
                "input_fields": ["sentence"],
                "output_fields": ["sentiment"],
            }
        },
        prompt="Process this: $input"
    )
    prompt, _ = template.evaluate("This is great!")
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert prompt == "42"  # Mock response

def test_template_with_invalid_signature():
    """Test error handling for invalid predefined signature"""
    template = DSPyTemplate(
        name="test",
        type="dspy",
        dspy={"module": "ChainOfThought", "signature": "NonexistentSignature"},
        prompt="Process this: $input"
    )
    with pytest.raises(ValueError, match="DSPy signature NonexistentSignature not found"):
        template.evaluate("test input")

def test_template_with_params():
    """Test template with additional parameters"""
    template = DSPyTemplate(
        name="test",
        type="dspy",
        dspy={
            "module": "ChainOfThought",
            "signature": {
                "input_fields": ["question"],
                "output_fields": ["answer"],
            }
        },
        prompt="Process this $input with style $style"
    )
    prompt, _ = template.evaluate("What is 2+2?", {"style": "step by step"})
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert prompt == "42"  # Mock response

def test_invalid_module():
    """Test error handling for invalid DSPy module"""
    template = DSPyTemplate(
        name="test",
        type="dspy",
        dspy={"module": "NonexistentModule"},
        prompt="Process this: $input"
    )
    with pytest.raises(ValueError, match="DSPy module NonexistentModule not found"):
        template.evaluate("test input")

def test_missing_param():
    """Test error handling for missing template parameter"""
    template = DSPyTemplate(
        name="test",
        type="dspy",
        dspy={
            "module": "ChainOfThought",
            "signature": {
                "input_fields": ["question"],
                "output_fields": ["answer"],
            }
        },
        prompt="Process this $input with $missing"
    )
    with pytest.raises(DSPyTemplate.MissingVariables, match="Missing variables: missing"):
        template.evaluate("test input")

def test_llm_template_registration():
    """Test that the DSPy template type is properly registered with LLM"""
    from llm_dspy import register_template_types
    template_types = register_template_types()
    assert "dspy" in template_types
    assert template_types["dspy"] == DSPyTemplate

def test_end_to_end_template_usage():
    """Test end-to-end template usage with actual DSPy module"""
    template = DSPyTemplate(
        name="math-tutor",
        type="dspy",
        dspy={
            "module": "ChainOfThought",
            "signature": {
                "input_fields": ["question"],
                "output_fields": ["answer", "explanation"],
                "description": "A math tutor that explains step by step"
            }
        },
        system="You are a helpful math tutor that explains problems step by step",
        prompt="Please solve this math problem: $input"
    )
    
    # Test with a simple math problem
    prompt, system = template.evaluate("What is 15% of 80?")
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert prompt == "42"  # Mock response
    assert system == "You are a helpful math tutor that explains problems step by step" 

def test_template_stringify():
    """Test that stringify returns the expected string representation"""
    template = DSPyTemplate(
        name="test",
        type="dspy",
        dspy={
            "module": "ChainOfThought",
            "signature": {
                "input_fields": ["question"],
                "output_fields": ["answer", "explanation"],
            }
        }
    )
    
    mock_module = MagicMock()
    mock_module.__str__.return_value = "ChainOfThought[question -> answer, explanation]"
    mock_module_class = MagicMock(return_value=mock_module)
    
    with patch('dspy.ChainOfThought', mock_module_class):
        result = template.stringify()
        assert result == "ChainOfThought[question -> answer, explanation]"

def test_template_stringify_invalid_module():
    """Test stringify with invalid module"""
    template = DSPyTemplate(
        name="test",
        type="dspy",
        dspy={
            "module": "NonexistentModule"
        }
    )
    result = template.stringify()
    assert result == "NonexistentModule(?)"

# Create a test signature class
class TestSignature(dspy.Signature):
    """Test signature class"""
    context = dspy.InputField(desc="Input context")
    question = dspy.InputField(desc="Input question")
    answer = dspy.OutputField(desc="Output answer")

def test_template_stringify_predefined_signature():
    """Test stringify with predefined signature class"""
    template = DSPyTemplate(
        name="test",
        type="dspy",
        dspy={
            "module": "ChainOfThought",
            "signature": "TestSignature"
        }
    )
    
    mock_module = MagicMock()
    mock_module.__str__.return_value = "ChainOfThought[TestSignature]"
    mock_module_class = MagicMock(return_value=mock_module)
    
    with patch('dspy.ChainOfThought', mock_module_class):
        result = template.stringify()
        assert result == "ChainOfThought[TestSignature]" 

def test_template_type_registration():
    """Test that the DSPy template type is registered with LLM."""
    # Get all registered template types
    template_types = {}
    for hook_result in pm.hook.register_template_types():
        template_types.update(hook_result)
    
    # Verify that the DSPy template type is registered
    assert "dspy" in template_types
    assert template_types["dspy"] == DSPyTemplate 

def test_dspy_uses_llm_model(mock_dspy_lm):
    """Test that DSPy uses LLM's model correctly"""
    # Create a mock LLM model
    mock_llm_model = MagicMock()
    mock_llm_model.prompt.return_value.text.return_value = "Test response"
    
    # Mock llm.get_model to return our mock
    with patch('llm.get_model', return_value=mock_llm_model):
        template = DSPyTemplate(
            name="test",
            type="dspy",
            dspy={
                "module": "ChainOfThought",
                "signature": {
                    "input_fields": ["question"],
                    "output_fields": ["answer"],
                }
            },
            prompt="Process this: $input"
        )
        
        prompt, _ = template.evaluate("What is 2+2?")
        
        # Verify that LLM's model was used
        mock_llm_model.prompt.assert_called_once()
        assert "What is 2+2?" in mock_llm_model.prompt.call_args[0][0]

def test_no_llm_model_error():
    """Test that an error is raised when no LLM model is configured"""
    # Mock llm.get_model to return None
    with patch('llm.get_model', return_value=None):
        template = DSPyTemplate(
            name="test",
            type="dspy",
            dspy={
                "module": "ChainOfThought",
                "signature": {
                    "input_fields": ["question"],
                    "output_fields": ["answer"],
                }
            },
            prompt="Process this: $input"
        )
        
        with pytest.raises(ValueError, match="No LLM model is configured"):
            template.evaluate("What is 2+2?") 