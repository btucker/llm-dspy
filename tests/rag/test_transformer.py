"""Tests for QueryTransformer component."""
import pytest
import dspy
from llm_dspy.rag.transformer import QueryTransformer

class MockStringResponse:
    """Mock response that returns string instead of list for sub_questions."""
    def __init__(self):
        self.search_query = "main query"
        self.sub_questions = "follow up 1, follow up 2"

class MockListResponse:
    """Mock response that returns proper list for sub_questions."""
    def __init__(self):
        self.search_query = "test query"
        self.sub_questions = ["sub question 1"]

class MockTransformString:
    """Mock transform that returns string sub_questions."""
    def __call__(self, **kwargs):
        return MockStringResponse()

class MockTransformList:
    """Mock transform that returns list sub_questions."""
    def __call__(self, **kwargs):
        return MockListResponse()

class TestQueryTransformer:
    """Tests for the QueryTransformer class."""
    
    def test_converts_string_to_list(self):
        """Test that QueryTransformer converts string sub_questions to list."""
        transformer = QueryTransformer()
        transformer.transform = MockTransformString()
        
        result = transformer.forward("test question")
        assert isinstance(result.sub_questions, list)
        assert len(result.sub_questions) == 2
        assert result.sub_questions == ["follow up 1", "follow up 2"]
    
    def test_accepts_list(self):
        """Test that QueryTransformer accepts list sub_questions."""
        transformer = QueryTransformer()
        transformer.transform = MockTransformList()
        
        result = transformer.forward("test question")
        assert isinstance(result.sub_questions, list)
        assert len(result.sub_questions) == 1
        assert result.sub_questions[0] == "sub question 1" 