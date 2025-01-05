"""Tests for the DSPy command."""
import pytest
from unittest.mock import MagicMock, patch
import dspy
import click.testing
from llm_dspy import run_dspy_module
from dspy.primitives.prediction import Prediction
import llm
from tests.mocks.llm import MockModel, Collection
from tests.mocks.dspy_mock import MockPrediction

@pytest.fixture(autouse=True)
def mock_dspy_configure():
    """Mock DSPy configure to prevent OpenAI initialization"""
    with patch('dspy.configure') as mock:
        yield mock

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

def test_basic_question(mock_dspy_module, cli_runner, cli):
    """Test basic question answering."""
    result = cli_runner.invoke(cli, [
        "dspy",
        "ChainOfThought(question -> answer)",
        "What is 2+2?"
    ])
    
    assert result.exit_code == 0
    assert "Simple answer" in result.output

def test_multiple_inputs(mock_dspy_module, cli_runner, cli):
    """Test handling multiple inputs."""
    result = cli_runner.invoke(cli, [
        "dspy",
        "ChainOfThought(context, question -> answer)",
        "Here is some context",
        "What can you tell me?"
    ])
    
    assert result.exit_code == 0
    assert "Simple answer" in result.output

def test_invalid_module(cli_runner, cli):
    """Test error handling for invalid DSPy module"""
    result = cli_runner.invoke(cli, [
        "dspy",
        "NonexistentModule(question -> answer)",
        "test"
    ])
    assert result.exit_code != 0
    assert "DSPy module NonexistentModule not found" in result.output

def test_invalid_command_format(cli_runner, cli):
    """Test invalid command format handling"""
    result = cli_runner.invoke(cli, ["dspy"])
    assert result.exit_code != 0
    assert "Missing argument" in result.output

def test_rag_with_llm_embeddings(mocker, cli_runner, cli):
    """Test RAG functionality with LLM embeddings."""
    # Mock the collection
    mock_collection = Collection("my_collection", model_id="ada-002")
    mocker.patch('llm.Collection', return_value=mock_collection)
    
    # Mock the DSPy module
    mock_module = mocker.MagicMock()
    mock_prediction = MockPrediction(answer="Simple answer")
    mock_module.return_value = mock_module  # Return self to act as both class and instance
    mock_module.forward.return_value = mock_prediction
    mocker.patch('dspy.ChainOfThought', mock_module)
    
    # Mock the DSPy configure
    mocker.patch('dspy.configure')
    
    # Run the command with context and question
    result = cli_runner.invoke(cli, [
        "dspy",
        "ChainOfThought(context, question -> answer)",
        "--context", "my_collection",
        "--question", "What is mentioned in the documents?"
    ])
    
    assert result.exit_code == 0, f"Command failed with output: {result.output}"
    assert "Simple answer" in result.output 