import pytest
import subprocess
import json
import os
from pathlib import Path
import shlex
import sys
from unittest.mock import patch

def run_command(cmd: str) -> tuple[str, str, int]:
    """Run a shell command and return stdout, stderr, and return code."""
    process = subprocess.Popen(
        shlex.split(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parent)}
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
    cmd = 'llm dspy "ChainOfThought(context, question -> answer, confidence)" "Here is some context" "What can you tell me?"'
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