import dspy
import llm
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger('llm_dspy.retrieval')

class LLMRetriever(dspy.Retrieve):
    """Retriever that integrates LLM collections with DSPy."""
    
    def __init__(self, k: int = 3, collection_name: Optional[str] = None):
        """Initialize the retriever.
        
        Args:
            k: Number of passages to retrieve
            collection_name: Name of the collection to retrieve from
        """
        super().__init__(k=k)
        self.collection_name = collection_name
        
    def forward(self, query: str) -> dspy.Prediction:
        """Retrieve passages from LLM collections based on the query.
        
        Args:
            query: The search query
            
        Returns:
            A Prediction containing retrieved passages
            
        Raises:
            ValueError: If query is None or empty
        """
        if query is None:
            raise ValueError("query cannot be None")
        if not query.strip():
            raise ValueError("query cannot be empty")
            
        try:
            # Validate collection exists
            if not self.collection_name:
                logger.warning("No collection configured")
                return dspy.Prediction(passages=[])
                
            if self.collection_name not in llm.collections:
                logger.warning(f"Collection '{self.collection_name}' not found")
                return dspy.Prediction(passages=[])
            
            collection = llm.collections[self.collection_name]
            logger.debug(f"Using collection '{self.collection_name}' for query: {query}")
            
            # Get similar passages
            results = collection.similar(value=query, n=self.k)
            
            # Format passages for DSPy
            passages = [
                {
                    "text": r.text,
                    "title": self.collection_name,
                    "score": r.score
                }
                for r in results
            ]
            
            return dspy.Prediction(passages=passages)
                
        except Exception as e:
            logger.error(f"Error retrieving from collection: {e}")
            return dspy.Prediction(passages=[]) 