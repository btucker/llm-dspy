"""Tests for basic DSPy CLI command functionality."""
import pytest
from unittest.mock import MagicMock, patch
import dspy
import click.testing
from llm_dspy import run_dspy_module
from tests.mocks.dspy_mock import MockPrediction

@pytest.fixture(autouse=True)
def mock_dspy_configure():
    """Mock DSPy configure to prevent OpenAI initialization"""
    with patch('dspy.configure') as mock:
        yield mock

@pytest.fixture(autouse=True)
def mock_ensure_signature(mocker):
    """Mock ensure_signature to return object with input/output fields."""
    mock = mocker.MagicMock()
    mock.input_fields = {'question': str}  # Default for single input
    mock.output_fields = {'answer': str}  # Default for single output
    mocker.patch('dspy.signatures.signature.ensure_signature', return_value=mock)
    return mock

@pytest.fixture
def mock_dspy_module(mocker):
    """Mock DSPy module."""
    mock_module = mocker.MagicMock()
    mock_prediction = MockPrediction(answer="Simple answer")
    mock_module.return_value = mock_module  # Return self to act as both class and instance
    mock_module.forward.return_value = mock_prediction
    mocker.patch('dspy.ChainOfThought', mock_module)
    return mock_module

@pytest.fixture
def cli_runner():
    """Create a Click CLI test runner"""
    return click.testing.CliRunner()

@pytest.fixture
def cli():
    """Create a CLI with our command registered"""
    @click.group()
    def cli():
        pass
    
    from llm_dspy import register_commands
    register_commands(cli)
    return cli

class TestBasicCommand:
    """Tests for basic DSPy command functionality."""
    
    def test_basic_question(self, mock_dspy_module, cli_runner, cli):
        """Test basic question answering."""
        result = cli_runner.invoke(cli, [
            "dspy",
            "ChainOfThought(question -> answer)",
            "What is 2+2?"
        ])
        
        assert result.exit_code == 0
        assert "Simple answer" in result.output
    
    def test_invalid_module(self, cli_runner, cli):
        """Test error handling for invalid DSPy module"""
        result = cli_runner.invoke(cli, [
            "dspy",
            "NonexistentModule(question -> answer)",
            "test"
        ])
        assert result.exit_code != 0
        assert "DSPy module NonexistentModule not found" in result.output
    
    def test_invalid_command_format(self, cli_runner, cli):
        """Test invalid command format handling"""
        result = cli_runner.invoke(cli, ["dspy"])
        assert result.exit_code != 0
        assert "Missing argument" in result.output 