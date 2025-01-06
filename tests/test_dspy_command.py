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

@pytest.fixture(autouse=True)
def mock_ensure_signature(mocker):
    """Mock ensure_signature to return object with input/output fields."""
    mock = mocker.MagicMock()
    mock.input_fields = {'question': str}  # Default for single input
    mock.output_fields = {'answer': str}  # Default for single output
    mocker.patch('dspy.signatures.signature.ensure_signature', return_value=mock)
    return mock

@pytest.fixture
def mock_collection(mocker):
    """Create a mock collection that returns predictable results."""
    collection = mocker.MagicMock()
    collection.similar.return_value = [{"text": "Retrieved context"}]
    
    # Mock Collection constructor
    mocker.patch('llm.Collection', return_value=collection)
    
    # Create collections dict if it doesn't exist
    if not hasattr(llm, 'collections'):
        setattr(llm, 'collections', {})
    
    # Set up collections dict
    llm.collections['collection_name'] = collection
    
    yield collection
    
    # Clean up
    if hasattr(llm, 'collections'):
        delattr(llm, 'collections')

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

def test_multiple_inputs(mock_dspy_module, mock_ensure_signature, cli_runner, cli):
    """Test handling multiple inputs."""
    # Update mock signature for multiple inputs
    mock_ensure_signature.input_fields = {'context': str, 'question': str}
    mock_ensure_signature.output_fields = {'answer': str}
    
    result = cli_runner.invoke(cli, [
        "dspy",
        "ChainOfThought(context, question -> answer)",
        "--context", "Here is some context",
        "--question", "What can you tell me?"
    ])
    
    assert result.exit_code == 0
    assert "Simple answer" in result.output
    mock_dspy_module.forward.assert_called_once_with(
        context="Here is some context",
        question="What can you tell me?"
    )

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

def test_single_input_positional(mock_dspy_module, cli_runner, cli):
    """Test single input field with positional argument."""
    result = cli_runner.invoke(cli, [
        "dspy",
        "ChainOfThought(foo -> bar)",
        "input for foo"
    ])
    
    assert result.exit_code == 0
    assert "Simple answer" in result.output
    mock_dspy_module.forward.assert_called_once_with(foo="input for foo")

def test_single_input_stdin(mock_dspy_module, cli_runner, cli):
    """Test single input field from stdin."""
    result = cli_runner.invoke(cli, [
        "dspy",
        "ChainOfThought(foo -> bar)"
    ], input="input for foo")
    
    assert result.exit_code == 0
    assert "Simple answer" in result.output
    mock_dspy_module.forward.assert_called_once_with(foo="input for foo")

def test_multiple_inputs_named_options(mock_dspy_module, cli_runner, cli):
    """Test multiple input fields with named options."""
    result = cli_runner.invoke(cli, [
        "dspy",
        "ChainOfThought(foo, baz -> bar)",
        "--foo", "input for foo",
        "--baz", "input for baz"
    ])
    
    assert result.exit_code == 0
    assert "Simple answer" in result.output
    mock_dspy_module.forward.assert_called_once_with(foo="input for foo", baz="input for baz")

def test_rag_input_collection(mock_dspy_module, mock_collection, cli_runner, cli, mocker):
    """Test RAG with collection name as input."""
    # Mock EnhancedRAGModule
    mock_rag = mocker.MagicMock()
    mock_rag_instance = mocker.MagicMock()
    mock_rag_instance.forward.return_value = mocker.MagicMock(answer="Retrieved context")
    mock_rag.return_value = mock_rag_instance
    mocker.patch('dspy.EnhancedRAGModule', mock_rag)
    
    result = cli_runner.invoke(cli, [
        "dspy",
        "ChainOfThought(foo, baz, query -> bar)",
        "--foo", "input for foo",
        "--baz", "collection_name",
        "--query", "test query"
    ])
    
    assert result.exit_code == 0
    assert "Simple answer" in result.output
    mock_rag.assert_called_once_with(collection_name="collection_name", k=5)
    mock_rag_instance.forward.assert_called_once_with(question="test query")
    mock_dspy_module.forward.assert_called_once()
    kwargs = mock_dspy_module.forward.call_args[1]
    assert kwargs["foo"] == "input for foo"
    assert kwargs["baz"] == "Retrieved context"

def test_stdin_with_multiple_inputs(mock_dspy_module, mock_collection, cli_runner, cli, mocker):
    """Test using stdin with multiple inputs."""
    # Mock EnhancedRAGModule
    mock_rag = mocker.MagicMock()
    mock_rag_instance = mocker.MagicMock()
    mock_rag_instance.forward.return_value = mocker.MagicMock(answer="Retrieved context")
    mock_rag.return_value = mock_rag_instance
    mocker.patch('dspy.EnhancedRAGModule', mock_rag)
    
    result = cli_runner.invoke(cli, [
        "dspy",
        "ChainOfThought(foo, baz, query -> bar)",
        "--foo", "stdin",
        "--baz", "collection_name",
        "--query", "test query"
    ], input="input for foo")
    
    assert result.exit_code == 0
    assert "Simple answer" in result.output
    mock_rag.assert_called_once_with(collection_name="collection_name", k=5)
    mock_rag_instance.forward.assert_called_once_with(question="test query")
    mock_dspy_module.forward.assert_called_once()
    kwargs = mock_dspy_module.forward.call_args[1]
    assert kwargs["foo"] == "input for foo"
    assert kwargs["baz"] == "Retrieved context" 