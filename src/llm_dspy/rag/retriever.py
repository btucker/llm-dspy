import sys
import dspy
import llm
import logging

logger = logging.getLogger(__name__)

class LLMRetriever(dspy.Retrieve):
    """Retriever that uses LLM's collections for retrieval."""
    def __init__(self, collection_name: str, k: int = 3, collection=None):
        super().__init__(k=k)
        self.collection_name = collection_name
        
        # For testing: if collections are available in the global namespace, use those
        if hasattr(llm, 'test_collections') and collection_name in llm.test_collections:
            logger.debug(f"Using test collection for {collection_name}")
            self.collection = llm.test_collections[collection_name]
        # Use provided collection or existing collection from llm.collections
        elif collection:
            logger.debug(f"Using provided collection for {collection_name}")
            self.collection = collection
        elif hasattr(llm, 'collections') and collection_name in llm.collections:
            logger.debug(f"Using existing collection for {collection_name}")
            self.collection = llm.collections[collection_name]
        # Create new collection as last resort
        else:
            logger.debug(f"Creating new collection for {collection_name}")
            # Get model ID from existing collection if available
            model_id = "ada-002"
            if hasattr(llm, 'collections'):
                for existing_collection in llm.collections.values():
                    if hasattr(existing_collection, 'model_id'):
                        model_id = existing_collection.model_id
                        break
            self.collection = llm.Collection(collection_name, model_id=model_id)
        
        logger.debug(f"Collection type: {type(self.collection)}")
        logger.debug(f"Collection model: {self.collection.model_id if hasattr(self.collection, 'model_id') else 'unknown'}")
    
    def forward(self, query):
        """Retrieve similar documents for the query."""
        try:
            logger.debug(f"Retrieving documents for query: {query}")
            # Get similar documents from the collection
            results = self.collection.similar(value=query, number=self.k)
            logger.debug(f"Found {len(results)} results")
            
            # Convert results to DSPy's expected format
            passages = []
            for i, result in enumerate(results):
                logger.debug(f"Result {i+1}:")
                logger.debug(f"  Type: {type(result)}")
                logger.debug(f"  Score: {result.score if hasattr(result, 'score') else 'unknown'}")
                logger.debug(f"  Content: {result.content if hasattr(result, 'content') else 'unknown'}")
                if hasattr(result, 'content'):
                    # Handle Entry objects
                    content = result.content
                    if content is not None:
                        logger.debug(f"  Content: {content[:100]}...")
                        passages.append({"text": str(content)})
                elif isinstance(result, dict) and "text" in result:
                    # Handle dictionary format
                    logger.debug(f"  Text: {result['text'][:100]}...")
                    passages.append({"text": str(result["text"])})
                else:
                    logger.warning(f"Unexpected result format: {type(result)}")
            
            return dspy.Prediction(passages=passages)
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return dspy.Prediction(passages=[])
