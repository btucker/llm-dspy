import pytest
from unittest.mock import MagicMock, patch
import dspy
import click.testing
from llm_dspy import run_dspy_module
from dspy.primitives.prediction import Prediction
import llm

@pytest.fixture(autouse=True)
def mock_dspy_configure():
    """Mock DSPy configure to prevent OpenAI initialization"""
    with patch('dspy.configure') as mock:
        yield mock

@pytest.fixture
def mock_dspy_module():
    """Mock DSPy module"""
    mock_module = MagicMock()
    mock_instance = MagicMock()
    mock_instance.forward.return_value = Prediction(answer="Here's a step by step solution...")
    mock_module.return_value = mock_instance
    
    with patch('dspy.ChainOfThought', mock_module):
        yield mock_module

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
        "ChainOfThought",
        "question -> answer",
        "--question", "What is 2+2?"
    ])
    
    assert result.exit_code == 0
    assert "Here's a step by step solution..." in result.output

def test_multiple_inputs(mock_dspy_module, cli_runner, cli):
    """Test handling multiple inputs."""
    result = cli_runner.invoke(cli, [
        "dspy",
        "ChainOfThought",
        "question, style -> answer",
        "--question", "What is the meaning of life?",
        "--style", "philosophical"
    ])
    
    assert result.exit_code == 0
    assert "Here's a step by step solution..." in result.output
    mock_dspy_module.return_value.forward.assert_called_once_with(
        question="What is the meaning of life?",
        style="philosophical"
    )

def test_invalid_module(cli_runner, cli):
    """Test error handling for invalid DSPy module"""
    result = cli_runner.invoke(cli, [
        "dspy",
        "NonexistentModule",
        "question -> answer",
        "--question", "test"
    ])
    assert result.exit_code != 0
    assert "DSPy module NonexistentModule not found" in result.output

def test_invalid_command_format(cli_runner, cli):
    """Test invalid command format handling"""
    result = cli_runner.invoke(cli, ["dspy", "InvalidFormat"])
    assert result.exit_code != 0
    assert "Missing module signature" in result.output

def test_rag_with_llm_embeddings(mocker, cli_runner, cli):
    """Test RAG functionality with LLM embeddings."""
    # Mock the LLM embeddings functionality
    mock_collection = mocker.MagicMock()
    mock_collection.search.return_value = ["This is some relevant context from the database"]
    
    # Mock the collection class
    mock_collection_class = mocker.MagicMock(return_value=mock_collection)
    mocker.patch('llm.Collection', mock_collection_class)
    
    # Mock the DSPy module
    mock_module = mocker.MagicMock()
    mock_module.forward.return_value = mocker.MagicMock(answer="Answer based on context")
    mocker.patch('dspy.ChainOfThought', return_value=mock_module)
    
    # Run the command with context and question
    result = cli_runner.invoke(cli, [
        "dspy",
        "ChainOfThought",
        "context, question -> answer",
        "--context", "my_collection",
        "--question", "What is mentioned in the documents?"
    ])
    
    assert result.exit_code == 0
    assert "Answer based on context" in result.output
    
    # Verify LLM embeddings were used
    mock_collection_class.assert_called_once_with("my_collection")
    mock_collection.search.assert_called_once_with("What is mentioned in the documents?")
    
    # Verify the module was called with the retrieved context
    mock_module.forward.assert_called_once_with(
        context="This is some relevant context from the database",
        question="What is mentioned in the documents?"
    )

def test_signature_based_options(mocker, cli_runner, cli):
    """Test that different signatures generate appropriate options."""
    # Mock the LLM embeddings functionality
    mock_collection = mocker.MagicMock()
    mock_collection.search.return_value = ["This is some relevant context from the database"]
    
    # Mock the collection class
    mock_collection_class = mocker.MagicMock(return_value=mock_collection)
    mocker.patch('llm.Collection', mock_collection_class)
    
    # Mock the DSPy module
    mock_module = mocker.MagicMock()
    mock_module.forward.return_value = mocker.MagicMock(answer="Answer based on context")
    mocker.patch('dspy.ChainOfThought', return_value=mock_module)
    
    # Test with different signatures
    test_cases = [
        # Basic RAG case
        {
            'cmd': [
                "dspy",
                "ChainOfThought",
                "context, query -> answer",
                "--context", "my_collection",
                "--query", "What is mentioned?"
            ],
            'expected_kwargs': {
                'context': "This is some relevant context from the database",
                'query': "What is mentioned?"
            }
        },
        # Multiple inputs case
        {
            'cmd': [
                "dspy",
                "ChainOfThought",
                "background, prompt, style -> answer",
                "--background", "my_collection",
                "--prompt", "Analyze this",
                "--style", "concise"
            ],
            'expected_kwargs': {
                'background': "This is some relevant context from the database",
                'prompt': "Analyze this",
                'style': "concise"
            }
        }
    ]

    for i, case in enumerate(test_cases):
        print(f"\nRunning test case {i + 1}:")
        print(f"Command: {' '.join(case['cmd'])}")
        print(f"Expected kwargs: {case['expected_kwargs']}")
        
        result = cli_runner.invoke(cli, case['cmd'])
        print(f"Exit code: {result.exit_code}")
        print(f"Output:\n{result.output}")
        if result.exception:
            print(f"Exception: {result.exception}")
        
        assert result.exit_code == 0
        assert "Answer based on context" in result.output
        
        # If this input has a collection field, verify search was performed
        collection_fields = ['context', 'background']  # Fields that trigger RAG
        for field in collection_fields:
            if field in case['expected_kwargs']:
                mock_collection_class.assert_called_with(case['cmd'][case['cmd'].index(f'--{field}') + 1])
                mock_collection.search.assert_called()
        
        # Verify the module was called with expected kwargs
        mock_module.forward.assert_called_with(**case['expected_kwargs']) 