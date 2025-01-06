import dspy
import llm
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger('llm_dspy.retrieval')

class LLMRetriever(dspy.Retrieve):
    """Retriever that integrates LLM collections with DSPy."""
    
    def __init__(self, k: int = 3, collection_name: Optional[str] = None):
        super().__init__(k=k)
        self.collection_name = collection_name
        
    def forward(self, query: str) -> dspy.Prediction:
        """Retrieve passages from LLM collections based on the query."""
        try:
            # Use the configured collection
            if self.collection_name and self.collection_name in llm.collections:
                collection = llm.collections[self.collection_name]
                logger.debug(f"Using collection '{self.collection_name}' for query: {query}")
                
                # Get similar passages using the actual query
                results = collection.similar(value=query, n=self.k)
                
                # Format passages for DSPy
                passages = []
                for r in results:
                    logger.debug(f"Retrieved passage: {r.text}")
                    passages.append({
                        "text": r.text,
                        "title": self.collection_name,
                        "score": r.score
                    })
                
                return dspy.Prediction(passages=passages)
            else:
                logger.debug(f"No collection configured or collection not found")
                return dspy.Prediction(passages=[])
                
        except Exception as e:
            logger.error(f"Error retrieving from collection: {e}")
            return dspy.Prediction(passages=[]) 