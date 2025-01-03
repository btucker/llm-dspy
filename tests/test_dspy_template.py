import pytest
from unittest.mock import MagicMock, patch
import dspy
import os
from llm_dspy import DSPyTemplate, DSPyConfig
from typing import ClassVar, List

@pytest.fixture
def mock_dspy_module():
    """Mock DSPy module"""
    mock_module = MagicMock()
    mock_module.return_value = MagicMock()
    mock_module.return_value.forward.return_value = MagicMock(
        prompt="Given the question 'What is 2+2?', provide a step by step solution."
    )
    
    with patch('dspy.ChainOfThought', mock_module):
        yield mock_module

def test_template_schema():
    """Test template schema validation"""
    template = DSPyTemplate(
        name="test",
        type="dspy",
        dspy={"module": "ChainOfThought"},
        prompt="Process this: $input"
    )
    assert template.name == "test"
    assert template.type == "dspy"
    assert template.prompt == "Process this: $input"

def test_basic_template(mock_dspy_module):
    """Test basic template functionality"""
    template = DSPyTemplate(
        name="test",
        type="dspy",
        dspy={
            "module": "ChainOfThought",
            "signature": "question -> answer"
        },
        prompt="Process this: $input"
    )
    prompt, system = template.evaluate("What is 2+2?")
    assert prompt == "Given the question 'What is 2+2?', provide a step by step solution."
    assert system is None

def test_template_with_system(mock_dspy_module):
    """Test template with system prompt"""
    template = DSPyTemplate(
        name="test",
        type="dspy",
        dspy={
            "module": "ChainOfThought",
            "signature": "question -> answer"
        },
        prompt="Process this: $input",
        system="You are a helpful assistant"
    )
    prompt, system = template.evaluate("What is 2+2?")
    assert prompt == "Given the question 'What is 2+2?', provide a step by step solution."
    assert system == "You are a helpful assistant"

def test_template_with_config(mock_dspy_module):
    """Test template with module config"""
    template = DSPyTemplate(
        name="test",
        type="dspy",
        dspy={
            "module": "ChainOfThought",
            "config": {"max_tokens": 100},
            "signature": "question -> answer"
        },
        prompt="Process this: $input"
    )
    prompt, _ = template.evaluate("What is 2+2?")
    assert prompt == "Given the question 'What is 2+2?', provide a step by step solution."

def test_template_with_invalid_signature():
    """Test error handling for invalid predefined signature"""
    template = DSPyTemplate(
        name="test",
        type="dspy",
        dspy={
            "module": "ChainOfThought",
            "signature": "invalid -> signature"
        },
        prompt="Process this: $input"
    )
    assert str(template) == "ChainOfThought[invalid -> signature]"

def test_template_with_params(mock_dspy_module):
    """Test template with parameters"""
    template = DSPyTemplate(
        name="test",
        type="dspy",
        dspy={
            "module": "ChainOfThought",
            "signature": "question -> answer"
        },
        prompt="Process this: $input with $param"
    )
    prompt, _ = template.evaluate("What is 2+2?", {"param": "test"})
    assert prompt == "Given the question 'What is 2+2?', provide a step by step solution."

def test_invalid_module():
    """Test error handling for invalid DSPy module"""
    with pytest.raises(ValueError, match="DSPy module NonexistentModule not found"):
        template = DSPyTemplate(
            name="test",
            type="dspy",
            dspy={"module": "NonexistentModule"},
            prompt="Process this: $input"
        )
        template.evaluate("What is 2+2?")

def test_missing_param():
    """Test error handling for missing parameter"""
    template = DSPyTemplate(
        name="test",
        type="dspy",
        dspy={
            "module": "ChainOfThought",
            "signature": "question -> answer"
        },
        prompt="Process this: $input with $param"
    )
    with pytest.raises(DSPyTemplate.MissingVariables, match="Missing variables: param"):
        template.evaluate("What is 2+2?")

def test_llm_template_registration():
    """Test template type registration"""
    from llm_dspy import register_template_types
    templates = register_template_types()
    assert "dspy" in templates
    assert templates["dspy"] == DSPyTemplate

def test_template_stringify():
    """Test template string representation"""
    template = DSPyTemplate(
        name="test",
        type="dspy",
        dspy={
            "module": "ChainOfThought",
            "signature": "question -> answer"
        },
        prompt="Process this: $input"
    )
    assert str(template) == "ChainOfThought[question -> answer]"

def test_template_stringify_invalid_module():
    """Test stringify with invalid module"""
    template = DSPyTemplate(
        name="test",
        type="dspy",
        dspy={"module": "NonexistentModule"},
        prompt="Process this: $input"
    )
    assert str(template) == "NonexistentModule(?)"

def test_template_type_registration():
    """Test template type registration"""
    from llm_dspy import register_template_types
    templates = register_template_types()
    assert "dspy" in templates
    assert templates["dspy"] == DSPyTemplate

def test_dspy_structures_prompt(mock_dspy_module):
    """Test that DSPy structures the prompt but doesn't make LLM calls"""
    template = DSPyTemplate(
        name="test",
        type="dspy",
        dspy={
            "module": "ChainOfThought",
            "signature": "question -> answer"
        },
        prompt="Process this: $input"
    )
    
    # When we evaluate
    prompt, _ = template.evaluate("What is 2+2?")
    
    # DSPy should have been used to structure the prompt
    mock_dspy_module.return_value.forward.assert_called_once_with("What is 2+2?")
    
    # The structured prompt should be returned
    assert prompt == "Given the question 'What is 2+2?', provide a step by step solution." 