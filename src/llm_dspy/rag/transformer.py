import dspy
from typing import List

class QueryTransformer(dspy.Module):
    """Module to transform user queries for better retrieval."""
    def __init__(self):
        super().__init__()
        self.transform = dspy.ChainOfThought("question -> search_query, sub_questions")
    
    def forward(self, question: str) -> dspy.Prediction:
        """Transform a question into a search query and sub-questions.
        
        Args:
            question: The input question to transform
            
        Returns:
            A Prediction with search_query and sub_questions (list) fields
            
        Raises:
            TypeError: If sub_questions cannot be converted to a list
        """
        result = self.transform(question=question)
        
        # Convert string to list if needed
        if isinstance(result.sub_questions, str):
            # Split on commas and clean up
            sub_questions = [q.strip() for q in result.sub_questions.split(',')]
            # Update the result
            result.sub_questions = sub_questions
        elif not isinstance(result.sub_questions, list):
            raise TypeError(f"sub_questions must be a list or comma-separated string, got {type(result.sub_questions)}")
            
        return result
