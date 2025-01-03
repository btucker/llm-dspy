import pytest
from unittest.mock import MagicMock, patch
import dspy
import click.testing
from llm_dspy import run_dspy_module
from dspy.primitives.prediction import Prediction

@pytest.fixture
def mock_dspy_module():
    """Mock DSPy module"""
    mock_module = MagicMock()
    mock_instance = MagicMock()
    mock_instance.forward.return_value = Prediction(answer="Here's a step by step solution...")
    mock_module.return_value = mock_instance
    
    with patch('dspy.ChainOfThought', mock_module):
        yield mock_module

def test_run_dspy_module(mock_dspy_module):
    """Test running a DSPy module"""
    result = run_dspy_module("ChainOfThought", "question -> answer", "What is 2+2?")
    assert result == "Here's a step by step solution..."
    mock_dspy_module.assert_called_once_with(signature="question -> answer")
    mock_dspy_module.return_value.forward.assert_called_once_with(question="What is 2+2?")

def test_invalid_module():
    """Test error handling for invalid DSPy module"""
    with pytest.raises(ValueError, match="DSPy module NonexistentModule not found"):
        run_dspy_module("NonexistentModule", "question -> answer", "test")

@pytest.fixture
def cli_runner():
    """Create a Click CLI test runner"""
    return click.testing.CliRunner()

def test_dspy_command(mock_dspy_module, cli_runner, monkeypatch):
    """Test the dspy command"""
    # Create a mock CLI context
    @click.group()
    def cli():
        pass
    
    # Import our register_commands and run it
    from llm_dspy import register_commands
    register_commands(cli)
    
    # Run the command
    result = cli_runner.invoke(cli, ["dspy", "ChainOfThought(question -> answer)", "What is 2+2?"])
    assert result.exit_code == 0
    assert result.output.strip() == "Here's a step by step solution..."

def test_dspy_command_multiple_words(mock_dspy_module, cli_runner):
    """Test the dspy command with multiple word prompt"""
    # Create a mock CLI context
    @click.group()
    def cli():
        pass
    
    # Import our register_commands and run it
    from llm_dspy import register_commands
    register_commands(cli)
    
    # Run the command
    result = cli_runner.invoke(cli, [
        "dspy",
        "ChainOfThought(question -> answer)",
        "Why", "is", "the", "sky", "blue?"
    ])
    assert result.exit_code == 0
    assert result.output.strip() == "Here's a step by step solution..."

def test_invalid_command_format(cli_runner):
    """Test invalid command format handling"""
    # Create a mock CLI context
    @click.group()
    def cli():
        pass
    
    # Import our register_commands and run it
    from llm_dspy import register_commands
    register_commands(cli)
    
    # Run the command
    result = cli_runner.invoke(cli, ["dspy", "InvalidFormat", "test"])
    assert result.exit_code != 0
    assert "Invalid module signature format" in result.output 

def test_chain_of_thought_basic_math(mock_dspy_module):
    """Test basic math with chain of thought."""
    result = run_dspy_module("ChainOfThought", "question -> answer", "What is 2+2?")
    assert isinstance(result, str), "Result should be a string"
    assert len(result) > 0, "Result should not be empty"
    assert result == "Here's a step by step solution..."
    mock_dspy_module.assert_called_once_with(signature="question -> answer")
    mock_dspy_module.return_value.forward.assert_called_once_with(question="What is 2+2?")

def test_run_dspy_module_with_quoted_spaces(mocker):
    """Test that quoted arguments containing spaces are handled correctly."""
    # Mock the DSPy module and its forward method
    mock_module = mocker.MagicMock()
    mock_module.forward.return_value = mocker.MagicMock(answer="Test response")
    mocker.patch('dspy.ChainOfThought', return_value=mock_module)
    
    # Run the module with quoted arguments
    run_dspy_module(
        "ChainOfThought",
        "question, context -> answer",
        '"What color is it?" "The sky is blue"'
    )
    
    # Verify the module was called with correctly parsed arguments
    mock_module.forward.assert_called_once_with(
        question="What color is it?",
        context="The sky is blue"
    ) 