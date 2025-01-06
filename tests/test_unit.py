"""Unit tests for llm_dspy components."""
import pytest
import dspy
from llm_dspy.rag.transformer import QueryTransformer
from llm_dspy.rag.enhanced import EnhancedRAGModule
from llm_dspy.retrieval import LLMRetriever

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

def test_query_transformer_converts_string():
    """Test that QueryTransformer converts string sub_questions to list."""
    transformer = QueryTransformer()
    transformer.transform = MockTransformString()
    
    result = transformer.forward("test question")
    assert isinstance(result.sub_questions, list)
    assert len(result.sub_questions) == 2
    assert result.sub_questions == ["follow up 1", "follow up 2"]

def test_query_transformer_accepts_list():
    """Test that QueryTransformer accepts list sub_questions."""
    transformer = QueryTransformer()
    transformer.transform = MockTransformList()
    
    result = transformer.forward("test question")
    assert isinstance(result.sub_questions, list)
    assert len(result.sub_questions) == 1
    assert result.sub_questions[0] == "sub question 1"

def test_enhanced_rag_fixed_collection():
    """Test that EnhancedRAGModule maintains fixed collection state."""
    module = EnhancedRAGModule(collection_name="collection1")
    assert module.collection_name == "collection1"
    
    # Replace transformer with mock
    module.query_transformer.transform = MockTransformList()
    initial_retriever = module.retriever
    
    # Collection name is fixed at initialization
    result = module.forward(question="test")
    assert module.retriever is initial_retriever
    
    # Different collection creates different retriever
    another_module = EnhancedRAGModule(collection_name="collection2")
    assert another_module.collection_name == "collection2"
    assert another_module.retriever is not initial_retriever

def test_llm_retriever_validation():
    """Test LLMRetriever input validation."""
    retriever = LLMRetriever(collection_name="test")
    
    with pytest.raises(ValueError) as exc_info:
        retriever(None)
    assert "query cannot be None" in str(exc_info.value)
    
    with pytest.raises(ValueError) as exc_info:
        retriever("")
    assert "query cannot be empty" in str(exc_info.value) 

def test_context_rewriter_preserves_entities():
    """Test that ContextRewriter preserves specific entity names and numbers."""
    from llm_dspy.rag.enhanced import ContextRewriter
    
    rewriter = ContextRewriter()
    context = """
    Key transactions:
    - Client A paid $50,000 for Enterprise License
    - Client B invested $75,000 in Custom Development
    """
    question = "What were the key transactions?"
    
    result = rewriter.forward(context=context, question=question)
    answer = result.focused_context.lower()
    
    # Check that specific entities and numbers are preserved
    assert "client a" in answer, "Should preserve client names"
    assert "client b" in answer, "Should preserve client names"
    assert "50,000" in answer, "Should preserve exact amounts"
    assert "75,000" in answer, "Should preserve exact amounts"
    assert "enterprise license" in answer, "Should preserve transaction types"
    assert "custom development" in answer, "Should preserve transaction types" 