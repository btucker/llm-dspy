import sys
import dspy
import llm

class LLMRetriever(dspy.Retrieve):
    """Retriever that uses LLM's collections for retrieval."""
    def __init__(self, collection_name: str, k: int = 3):
        super().__init__(k=k)
        self.collection_name = collection_name
        # Create collection directly instead of using get_collection
        self.collection = llm.Collection(collection_name, model_id="ada-002")
    
    def forward(self, query):
        """Retrieve similar documents for the query."""
        try:
            # Get similar documents from the collection
            results = self.collection.similar(text=query, n=self.k)
            
            # Convert results to DSPy's expected format
            passages = []
            for result in results:
                if isinstance(result, dict) and "text" in result:
                    passages.append({"text": str(result["text"])})
                else:
                    passages.append({"text": str(result)})
            
            # Return in DSPy's expected format
            return dspy.Prediction(passages=passages)
            
        except Exception as e:
            # Log the error and return empty results
            print(f"Error retrieving from collection '{self.collection_name}': {str(e)}", file=sys.stderr)
            return dspy.Prediction(passages=[])
