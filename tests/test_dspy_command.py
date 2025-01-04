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
def mock_dspy_module(mocker):
    """Mock DSPy module."""
    mock_module = mocker.MagicMock()
    mock_prediction = mocker.MagicMock()
    mock_prediction.answer = "Simple answer"
    mock_prediction.text = "Simple answer"
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
        "dspy", "run",
        "ChainOfThought",
        "question -> answer",
        "--question", "What is 2+2?"
    ])
    
    assert result.exit_code == 0
    assert "Simple answer" in result.output

def test_multiple_inputs(mock_dspy_module, cli_runner, cli):
    """Test handling multiple inputs."""
    # Update mock for this test
    mock_prediction = MagicMock()
    mock_prediction.answer = "Deep answer"
    mock_prediction.text = "Deep answer"
    mock_dspy_module.forward.return_value = mock_prediction
    
    result = cli_runner.invoke(cli, [
        "dspy", "run",
        "ChainOfThought",
        "question, style -> answer",
        "--question", "What is the meaning of life?",
        "--style", "philosophical"
    ])
    
    assert result.exit_code == 0
    assert "Deep answer" in result.output

def test_invalid_module(cli_runner, cli):
    """Test error handling for invalid DSPy module"""
    result = cli_runner.invoke(cli, [
        "dspy", "run",
        "NonexistentModule",
        "question -> answer",
        "--question", "test"
    ])
    assert result.exit_code != 0
    assert "DSPy module NonexistentModule not found" in result.output

def test_invalid_command_format(cli_runner, cli):
    """Test invalid command format handling"""
    result = cli_runner.invoke(cli, ["dspy", "run"])
    assert result.exit_code != 0
    assert "Missing argument" in result.output

def test_rag_with_llm_embeddings(mocker, cli_runner, cli):
    """Test RAG functionality with LLM embeddings."""
    # Mock the LLM embeddings functionality
    mock_collection = mocker.MagicMock()
    mock_collection.similar.return_value = [{"text": "This is some relevant context from the database"}]
    
    # Mock the collection class
    mock_collection_class = mocker.MagicMock(return_value=mock_collection)
    mocker.patch('llm.Collection', mock_collection_class)
    
    # Mock the DSPy module
    mock_module = mocker.MagicMock()
    mock_prediction = mocker.MagicMock()
    mock_prediction.answer = "Answer based on context"
    mock_prediction.text = "Answer based on context"
    mock_module.return_value = mock_module  # Return self to act as both class and instance
    mock_module.forward.return_value = mock_prediction
    mocker.patch('dspy.ChainOfThought', mock_module)
    
    # Run the command with context and question
    result = cli_runner.invoke(cli, [
        "dspy", "run",
        "ChainOfThought",
        "context, question -> answer",
        "--context", "my_collection",
        "--question", "What is mentioned in the documents?"
    ])
    
    assert result.exit_code == 0
    assert "Answer based on context" in result.output
    
    # Verify LLM embeddings were used
    mock_collection_class.assert_called_once_with("my_collection", model_id="ada-002")
    mock_collection.similar.assert_called_once()
    
    # Verify the module was called with the retrieved context
    mock_module.forward.assert_called_once()

def test_signature_based_options(mocker, cli_runner, cli):
    """Test that different signatures generate appropriate options."""
    # Mock the LLM embeddings functionality
    mock_collection = mocker.MagicMock()
    mock_collection.similar.return_value = [{"text": "This is some relevant context from the database"}]
    
    # Mock the collection class
    mock_collection_class = mocker.MagicMock(return_value=mock_collection)
    mocker.patch('llm.Collection', mock_collection_class)
    
    # Mock the DSPy module
    mock_module = mocker.MagicMock()
    mock_prediction = mocker.MagicMock()
    mock_prediction.answer = "Answer based on context"
    mock_prediction.text = "Answer based on context"
    mock_module.return_value = mock_module  # Return self to act as both class and instance
    mock_module.forward.return_value = mock_prediction
    mocker.patch('dspy.ChainOfThought', mock_module)
    
    # Test with different signatures
    test_cases = [
        # Basic RAG case
        {
            'cmd': [
                "dspy", "run",
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
                "dspy", "run",
                "ChainOfThought",
                "context, prompt, style -> answer",
                "--context", "my_collection",
                "--prompt", "Analyze this",
                "--style", "concise"
            ],
            'expected_kwargs': {
                'context': "This is some relevant context from the database",
                'prompt': "Analyze this",
                'style': "concise"
            }
        }
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\nRunning test case {i + 1}:")
        print(f"Command: {' '.join(case['cmd'])}")
        print(f"Expected kwargs: {case['expected_kwargs']}")
        
        # Reset mock before each test case
        mock_collection_class.reset_mock()
        
        result = cli_runner.invoke(cli, case['cmd'])
        print(f"Exit code: {result.exit_code}")
        print(f"Output:\n{result.output}")
        if result.exception:
            print(f"Exception: {result.exception}")
        
        assert result.exit_code == 0
        assert "Answer based on context" in result.output
        
        # If this input has a collection field, verify search was performed
        collection_fields = ['context']  # Fields that trigger RAG
        for field in collection_fields:
            if field in case['expected_kwargs']:
                # Verify that the collection was created with the correct name
                collection_name = case['cmd'][case['cmd'].index(f'--{field}') + 1]
                assert any(
                    call.args[0] == collection_name and call.kwargs.get('model_id') == 'ada-002'
                    for call in mock_collection_class.call_args_list
                ), f"Collection was not created with name '{collection_name}' and model_id='ada-002'"

def test_enhanced_rag_functionality(mocker, cli_runner, cli):
    """Test enhanced RAG functionality with query transformation and multi-hop reasoning."""
    # Mock the LLM collection
    mock_collection = mocker.MagicMock()
    mock_collection.similar.side_effect = [
        [{"text": "Paris is the capital of France and is known for its cuisine."}],
        [{"text": "The Eiffel Tower and Louvre Museum are famous landmarks in Paris."}],
        [{"text": "French cuisine includes baguettes, croissants, and fine wines."}]
    ]
    
    # Mock the collection class
    mock_collection_class = mocker.MagicMock(return_value=mock_collection)
    mocker.patch('llm.Collection', mock_collection_class)
    
    # Mock the DSPy modules
    mock_module = mocker.MagicMock()
    mock_prediction = mocker.MagicMock()
    mock_prediction.search_query = "Tell me about Paris landmarks and cuisine"
    mock_prediction.sub_questions = ["What are the famous landmarks in Paris?", "What is special about French cuisine?"]
    mock_prediction.focused_context = "Focused context about Paris"
    mock_prediction.answer = "Paris is special because of its iconic landmarks like the Eiffel Tower and its renowned cuisine."
    mock_prediction.text = mock_prediction.answer
    mock_module.return_value = mock_module  # Return self to act as both class and instance
    mock_module.forward.return_value = mock_prediction
    mocker.patch('dspy.ChainOfThought', mock_module)
    
    # Run the command with a complex question
    result = cli_runner.invoke(cli, [
        "dspy", "run",
        "ChainOfThought",
        "context, question -> answer",
        "--context", "paris_guide",
        "--question", "What makes Paris special in terms of landmarks and food?"
    ])
    
    assert result.exit_code == 0
    assert "Paris" in result.output
    assert "Eiffel Tower" in result.output
    assert "cuisine" in result.output
    
    # Verify multiple searches were performed
    assert mock_collection.similar.call_count >= 2, "Expected multiple search calls for multi-hop reasoning"
    
    # Verify the answer contains information from all contexts
    answer = result.output.lower()
    assert any(word in answer for word in ['eiffel', 'louvre']), "Expected landmarks in answer"
    assert any(word in answer for word in ['cuisine', 'food']), "Expected food-related information in answer"
    assert 'paris' in answer, "Expected basic information about Paris" 

def test_arbitrary_parameter_names(mocker, cli_runner, cli):
    """Test that the command works with arbitrary parameter names."""
    # Mock the DSPy module
    mock_module = mocker.MagicMock()
    mock_prediction = mocker.MagicMock()
    mock_prediction.answer = "Positive sentiment"
    mock_prediction.text = "Positive sentiment"
    mock_module.return_value = mock_module  # Return self to act as both class and instance
    mock_module.forward.return_value = mock_prediction
    mocker.patch('dspy.ChainOfThought', mock_module)
    
    # Test cases with different parameter names
    test_cases = [
        {
            'cmd': [
                "dspy",
                "ChainOfThought(sentence -> classification)",
                "This is a great day!"
            ],
            'expected_kwargs': {
                'sentence': "This is a great day!"
            }
        },
        {
            'cmd': [
                "dspy",
                "ChainOfThought(input_text, style -> output_text)",
                "Process this text",
                "formal"
            ],
            'expected_kwargs': {
                'input_text': "Process this text",
                'style': "formal"
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
        assert "Positive sentiment" in result.output
        
        # Verify the module was called with the correct kwargs
        mock_module.forward.assert_called_with(**case['expected_kwargs']) 