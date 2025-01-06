import pytest
import subprocess
import json
import os
from pathlib import Path
import shlex
import sys
from unittest.mock import patch
import llm
import dspy
from llm_dspy import EnhancedRAGModule

def run_command(cmd: str) -> tuple[str, str, int]:
    """Run a shell command and return stdout, stderr, and return code."""
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parent)},
        shell=True  # Enable shell features like pipes
    )
    stdout, stderr = process.communicate()
    return stdout.strip(), stderr.strip(), process.returncode

@pytest.fixture(scope="module")
def installed_plugin():
    """Install the plugin in editable mode and clean up after tests."""
    # Get the project root directory (where pyproject.toml is)
    root_dir = Path(__file__).parent.parent
    
    # Install the plugin
    stdout, stderr, code = run_command(f"llm install -e {root_dir}")
    assert code == 0, f"Failed to install plugin: {stderr}"
    
    # Add mocks directory to Python path
    mocks_dir = Path(__file__).parent / "mocks"
    sys.path.insert(0, str(mocks_dir))
    
    yield  # Run the tests
    
    # Remove mocks directory from Python path
    sys.path.remove(str(mocks_dir))
    
    # Cleanup: Uninstall the plugin
    stdout, stderr, code = run_command("llm uninstall llm-dspy -y")
    assert code == 0, f"Failed to uninstall plugin: {stderr}"

def test_plugin_installation(installed_plugin):
    """Test that the plugin is properly installed and visible to LLM."""
    stdout, stderr, code = run_command("llm plugins")
    assert code == 0, "llm plugins command failed"
    
    plugins = json.loads(stdout)
    
    # Find our plugin in the list
    our_plugin = next((p for p in plugins if p["name"] == "llm-dspy"), None)
    assert our_plugin is not None, "Plugin not found in llm plugins output"
    assert "register_commands" in our_plugin["hooks"]

def test_basic_dspy_command(installed_plugin):
    """Test running a basic DSPy command."""
    cmd = 'llm dspy "ChainOfThought(question -> answer)" "What is 2+2?"'
    stdout, stderr, code = run_command(cmd)
    assert code == 0, f"Command failed: {stderr}"
    assert stdout, "Expected non-empty output"
    assert "4" in stdout.lower(), "Expected answer to contain '4'"

def test_complex_signature(installed_plugin):
    """Test using a more complex signature with multiple inputs/outputs."""
    cmd = 'llm dspy "ChainOfThought(context, question -> answer, confidence)" --context "Here is some context" --question "What can you tell me?"'
    stdout, stderr, code = run_command(cmd)
    assert code == 0, f"Command failed: {stderr}"
    assert stdout, "Expected non-empty output"
    assert len(stdout) > 10, "Expected a reasonably long response"

def test_invalid_module(installed_plugin):
    """Test error handling for invalid module."""
    cmd = 'llm dspy "NonexistentModule(question -> answer)" "test"'
    stdout, stderr, code = run_command(cmd)
    assert code != 0, "Expected command to fail"
    assert "NonexistentModule not found" in stderr

def test_invalid_signature_format(installed_plugin):
    """Test error handling for invalid signature format."""
    cmd = 'llm dspy "InvalidFormat" "test"'
    stdout, stderr, code = run_command(cmd)
    assert code != 0, "Expected command to fail"
    assert "Invalid module signature format" in stderr 

def test_rag_with_collection(installed_plugin, tmp_path):
    """Test RAG functionality with a real LLM collection."""
    # Create test documents
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    
    doc1 = docs_dir / "doc1.txt"
    doc1.write_text("The capital of France is Paris. It is known for the Eiffel Tower.")
    
    doc2 = docs_dir / "doc2.txt"
    doc2.write_text("Paris has many museums, including the Louvre and Musee d'Orsay.")
    
    # Add documents to collection
    stdout, stderr, code = run_command(f"llm embed test_collection doc1 --content {shlex.quote(doc1.read_text())} --model ada-002")
    assert code == 0, f"Failed to embed doc1: {stderr}"
    
    stdout, stderr, code = run_command(f"llm embed test_collection doc2 --content {shlex.quote(doc2.read_text())} --model ada-002")
    assert code == 0, f"Failed to embed doc2: {stderr}"
    
    # Use the collection in a DSPy command
    cmd = 'llm dspy "ChainOfThought(context, question -> answer)" --context test_collection --question "What can you tell me about Paris?"'
    stdout, stderr, code = run_command(cmd)
    
    # Verify the command succeeded
    assert code == 0, f"Command failed: {stderr}"
    assert stdout, "Expected non-empty output"
    
    # Verify the answer contains information from our documents
    assert any(word in stdout.lower() for word in ['paris', 'france', 'eiffel', 'louvre']), "Expected answer to contain information from the documents"
    
    # Clean up
    stdout, stderr, code = run_command("llm collections delete test_collection")
    assert code == 0, f"Failed to delete collection: {stderr}" 

def test_dspy_completion_interface(installed_plugin):
    """Test that DSPy's completion interface works correctly with our LLM adapter."""
    cmd = 'llm dspy "Predict(question -> answer)" "What is 2+2?"'
    stdout, stderr, code = run_command(cmd)
    
    # The command should succeed
    assert code == 0, f"Command failed: {stderr}"
    
    # We should get a non-empty response
    assert stdout, "Expected non-empty output"
    
    # The response should contain a number
    assert any(char.isdigit() for char in stdout), "Expected response to contain a number"
    
    # The response should specifically contain 4
    assert "4" in stdout, "Expected response to contain the correct answer (4)" 

def test_dspy_rag_functionality(installed_plugin, tmp_path):
    """Test that DSPy's RAG functionality works correctly with our LLM adapter."""
    # Create test documents
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    
    doc1 = docs_dir / "doc1.txt"
    doc1.write_text("The capital of France is Paris. It is known for the Eiffel Tower.")
    
    # Add document to collection
    stdout, stderr, code = run_command(f"llm embed test_rag_collection doc1 --content {shlex.quote(doc1.read_text())} --model ada-002")
    assert code == 0, f"Failed to embed doc1: {stderr}"
    
    # Use DSPy's RAG capabilities
    cmd = 'llm dspy "ChainOfThought(context, question -> answer)" --context test_rag_collection --question "What is the capital of France?"'
    stdout, stderr, code = run_command(cmd)
    
    # The command should succeed
    assert code == 0, f"Command failed: {stderr}"
    
    # We should get a non-empty response
    assert stdout, "Expected non-empty output"
    
    # The response should contain information from our document
    assert "Paris" in stdout, "Expected response to contain information from the document"
    
    # Clean up
    stdout, stderr, code = run_command("llm collections delete test_rag_collection")
    assert code == 0, f"Failed to delete collection: {stderr}" 

def test_enhanced_rag_integration(installed_plugin, tmp_path):
    """Integration test for the enhanced RAG implementation."""
    # Create test documents
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    
    # Create documents with information about Paris landmarks and cuisine
    doc1 = docs_dir / "landmarks.txt"
    doc1.write_text("Paris is known for its iconic landmarks. The Eiffel Tower and Louvre Museum are the most famous landmarks in Paris.")
    
    doc2 = docs_dir / "cuisine.txt"
    doc2.write_text("French cuisine is renowned for its sophistication. Paris restaurants serve dishes like coq au vin and boeuf bourguignon.")
    
    # Add documents to collection
    stdout, stderr, code = run_command(f"llm embed paris_guide landmarks --content {shlex.quote(doc1.read_text())} --model ada-002")
    assert code == 0, f"Failed to embed landmarks doc: {stderr}"
    
    stdout, stderr, code = run_command(f"llm embed paris_guide cuisine --content {shlex.quote(doc2.read_text())} --model ada-002")
    assert code == 0, f"Failed to embed cuisine doc: {stderr}"
    
    # Use the enhanced RAG module through the command line
    cmd = 'llm dspy "EnhancedRAGModule(collection_name, question -> answer)" --collection_name paris_guide --question "What makes Paris special in terms of landmarks and cuisine?"'
    stdout, stderr, code = run_command(cmd)
    
    # Verify the command succeeded
    assert code == 0, f"Command failed: {stderr}"
    assert stdout, "Expected non-empty output"
    assert "Eiffel Tower" in stdout, "Expected answer to mention Eiffel Tower"
    assert "cuisine" in stdout.lower(), "Expected answer to mention cuisine"
    
    # Clean up
    run_command("llm collections delete paris_guide")

def test_rag_error_handling(mocker):
    """Test error handling in the enhanced RAG implementation."""
    # Mock the collection to raise an exception
    mock_collection = mocker.MagicMock()
    mock_collection.similar.side_effect = Exception("Failed to connect to database")
    
    # Mock the collection class
    mock_collection_class = mocker.MagicMock(return_value=mock_collection)
    mocker.patch('llm.Collection', mock_collection_class)
    
    # Create the enhanced RAG module with max_hops=1 to limit searches
    rag_module = EnhancedRAGModule(k=1, max_hops=1)
    
    # Test with a simple question
    question = "What is the capital of France?"
    result = rag_module(collection_name="test_collection", question=question)
    
    # Verify that the module handles the error gracefully
    assert result.answer, "Should still provide an answer even when retrieval fails"
    
    # Verify that the collection was created
    mock_collection_class.assert_called_with("test_collection", model_id="ada-002")
    
    # Verify that search attempts were made and failed
    assert mock_collection.similar.call_count > 0, "Should attempt to search at least once"
    
    # Verify all calls failed with the expected error
    for call_args in mock_collection.similar.call_args_list:
        assert 'text' in call_args[1], "Each search should have a query"
        assert call_args[1]['n'] == 1, "Should request 1 result per search as specified by k=1"

def test_rag_with_empty_results(mocker):
    """Test RAG behavior when no relevant documents are found."""
    # Mock the collection to return empty results
    mock_collection = mocker.MagicMock()
    mock_collection.similar.return_value = []
    
    # Mock the collection class
    mock_collection_class = mocker.MagicMock(return_value=mock_collection)
    mocker.patch('llm.Collection', mock_collection_class)
    
    # Create the enhanced RAG module
    rag_module = EnhancedRAGModule(collection_name="empty_collection", k=1)
    
    # Test with a question
    question = "What is in the collection?"
    result = rag_module(question)
    
    # Verify that the module handles empty results gracefully
    assert result.answer, "Should provide an answer even with no retrieved documents"
    
    # Verify that the collection was searched
    assert mock_collection.similar.call_count > 0, "Should attempt to retrieve documents" 

def test_single_input_positional(installed_plugin):
    """Test single input field with positional argument."""
    cmd = 'llm dspy "ChainOfThought(foo -> bar)" "What is 2+2?"'
    stdout, stderr, code = run_command(cmd)
    assert code == 0, f"Command failed: {stderr}"
    assert stdout, "Expected non-empty output"
    assert "4" in stdout.lower(), "Expected answer to contain '4'"

def test_single_input_stdin(installed_plugin, tmp_path):
    """Test single input field from stdin."""
    # Create a temporary file with input
    input_file = tmp_path / "input.txt"
    input_file.write_text("What is 2+2?")
    
    cmd = f'cat {input_file} | llm dspy "ChainOfThought(foo -> bar)"'
    stdout, stderr, code = run_command(cmd)
    assert code == 0, f"Command failed: {stderr}"
    assert stdout, "Expected non-empty output"
    assert "4" in stdout.lower(), "Expected answer to contain '4'"

def test_multiple_inputs_named_options(installed_plugin):
    """Test multiple input fields with named options."""
    cmd = 'llm dspy "ChainOfThought(foo, baz -> bar)" --foo "What is" --baz "2+2?"'
    stdout, stderr, code = run_command(cmd)
    assert code == 0, f"Command failed: {stderr}"
    assert stdout, "Expected non-empty output"
    assert "4" in stdout.lower(), "Expected answer to contain '4'"

def test_rag_with_collection(installed_plugin, tmp_path):
    """Test RAG functionality with collection name as input."""
    # Create test documents
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    
    doc1 = docs_dir / "doc1.txt"
    doc1.write_text("The capital of France is Paris. It is known for the Eiffel Tower.")
    
    # Create and populate collection
    collection_name = "test_rag_collection_2"
    stdout, stderr, code = run_command(f"llm embed {collection_name} doc1 --content {shlex.quote(doc1.read_text())} --model ada-002")
    assert code == 0, f"Failed to embed doc1: {stderr}"
    
    # Test RAG with collection
    cmd = f'llm dspy "ChainOfThought(foo, baz -> bar)" --foo "What is the capital of France?" --baz "{collection_name}"'
    stdout, stderr, code = run_command(cmd)
    
    # Cleanup
    run_command(f"llm collections delete {collection_name}")
    
    assert code == 0, f"Command failed: {stderr}"
    assert stdout, "Expected non-empty output"
    assert "Paris" in stdout, "Expected answer to contain 'Paris'"

def test_stdin_with_multiple_inputs(installed_plugin, tmp_path):
    """Test using stdin with multiple inputs and collection."""
    # Create test documents
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    
    doc1 = docs_dir / "doc1.txt"
    doc1.write_text("The capital of France is Paris. It is known for the Eiffel Tower.")
    
    # Create input file
    input_file = tmp_path / "input.txt"
    input_file.write_text("What is the capital of France?")
    
    # Create and populate collection
    collection_name = "test_rag_collection_3"
    stdout, stderr, code = run_command(f"llm embed {collection_name} doc1 --content {shlex.quote(doc1.read_text())} --model ada-002")
    assert code == 0, f"Failed to embed doc1: {stderr}"
    
    # Test with stdin and collection
    cmd = f'cat {input_file} | llm dspy "ChainOfThought(foo, baz -> bar)" --foo stdin --baz "{collection_name}"'
    stdout, stderr, code = run_command(cmd)
    
    # Cleanup
    run_command(f"llm collections delete {collection_name}")
    
    assert code == 0, f"Command failed: {stderr}"
    assert stdout, "Expected non-empty output"
    assert "Paris" in stdout, "Expected answer to contain 'Paris'" 