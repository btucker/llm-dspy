"""Tests for LLMRetriever component."""
import pytest
from llm_dspy.retrieval import LLMRetriever

class TestLLMRetriever:
    """Tests for the LLMRetriever class."""
    
    def test_input_validation(self):
        """Test LLMRetriever input validation."""
        retriever = LLMRetriever(collection_name="test")
        
        with pytest.raises(ValueError) as exc_info:
            retriever(None)
        assert "query cannot be None" in str(exc_info.value)
        
        with pytest.raises(ValueError) as exc_info:
            retriever("")
        assert "query cannot be empty" in str(exc_info.value) 