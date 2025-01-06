import sys
import dspy
import llm

class LLMRetriever(dspy.Retrieve):
    """Retriever that uses LLM's collections for retrieval."""
    def __init__(self, collection_name: str, k: int = 3, collection=None):
        super().__init__(k=k)
        self.collection_name = collection_name
        # Use provided collection, existing collection from llm.collections, or create new one
        self.collection = collection or (
            llm.collections.get(collection_name) if hasattr(llm, 'collections') else None
        ) or llm.Collection(collection_name, model_id="ada-002")
    
    def forward(self, query):
        """Retrieve similar documents for the query."""
        try:
            # Get similar documents from the collection
            results = self.collection.similar(value=query, number=self.k)
            
            # Convert results to DSPy's expected format
            passages = []
            for result in results:
                if hasattr(result, 'content'):
                    # Handle Entry objects
                    content = result.content
                    if content is not None:
                        passages.append({"text": str(content)})
                elif isinstance(result, dict) and "text" in result:
                    # Handle dictionary format
                    passages.append({"text": str(result["text"])})
                else:
                    # Handle other formats
                    passages.append({"text": str(result)})
            
            # Return in DSPy's expected format
            return dspy.Prediction(passages=passages)
            
        except Exception as e:
            # Log the error and return empty results
            print(f"Error retrieving from collection '{self.collection_name}': {str(e)}", file=sys.stderr)
            return dspy.Prediction(passages=[])
