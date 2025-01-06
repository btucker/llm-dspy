"""Tests for EnhancedRAGModule component."""
import pytest
import dspy
import llm
from llm_dspy.rag.enhanced import EnhancedRAGModule
from tests.rag.test_transformer import MockListResponse, MockTransformList

class TestEnhancedRAGModule:
    """Tests for the EnhancedRAGModule class."""
    
    def test_fixed_collection_state(self, mocker):
        """Test that EnhancedRAGModule maintains fixed collection state."""
        # Mock llm.collections
        mock_collection = mocker.MagicMock()
        mock_collections = {'collection1': mock_collection, 'collection2': mocker.MagicMock()}
        
        # Create collections attribute if it doesn't exist
        if not hasattr(llm, 'collections'):
            setattr(llm, 'collections', {})
        
        # Now patch collections
        mocker.patch.dict('llm.collections', mock_collections)
        
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
    
    def test_validation_errors(self):
        """Test validation error handling."""
        # Test empty collection name
        with pytest.raises(ValueError, match="collection_name must be provided"):
            EnhancedRAGModule(collection_name="")
        
        # Test non-existent collection
        with pytest.raises(KeyError, match="Collection 'nonexistent' not found"):
            EnhancedRAGModule(collection_name="nonexistent") 