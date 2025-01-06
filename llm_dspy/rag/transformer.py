import dspy

class QueryTransformer(dspy.Module):
    """Module to transform user queries for better retrieval."""
    def __init__(self):
        super().__init__()
        self.transform = dspy.ChainOfThought("question -> search_query, sub_questions")
    
    def forward(self, question):
        result = self.transform(question=question)
        # Convert sub_questions to list if it's a string
        if isinstance(result.sub_questions, str):
            result.sub_questions = [q.strip() for q in result.sub_questions.split(',')]
        return result
