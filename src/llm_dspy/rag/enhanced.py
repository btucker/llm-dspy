import dspy
from .retriever import LLMRetriever
from .transformer import QueryTransformer

class ContextRewriter(dspy.Module):
    """Module to rewrite retrieved context to be more focused on the question."""
    def __init__(self):
        super().__init__()
        self.rewrite = dspy.ChainOfThought("context, question -> focused_context")
    
    def forward(self, context, question):
        return self.rewrite(context=context, question=question)

class EnhancedRAGModule(dspy.Module):
    """Enhanced RAG module with query transformation and multi-hop reasoning."""
    def __init__(self, collection_name: str = None, k: int = 3, max_hops: int = 2, signature: str = None):
        super().__init__()
        self.collection_name = collection_name
        self.k = k
        self.max_hops = max_hops
        
        # Components for the enhanced pipeline
        self.query_transformer = QueryTransformer()
        self.retriever = LLMRetriever(collection_name=collection_name, k=k)
        self.context_rewriter = ContextRewriter()
        self.generate = dspy.ChainOfThought("context, question, reasoning_path -> answer")
    
    def forward(self, collection_name: str = None, question: str = None):
        # Use instance collection_name if not provided
        collection_name = collection_name or self.collection_name
        if not collection_name:
            raise ValueError("collection_name must be provided")
        
        # Update retriever if collection_name changed
        if collection_name != self.collection_name:
            self.retriever = LLMRetriever(collection_name=collection_name, k=self.k)
            self.collection_name = collection_name
        
        # Transform the initial query
        transformed = self.query_transformer(question)
        search_query = transformed.search_query
        sub_questions = transformed.sub_questions
        
        # Initialize reasoning path and context
        reasoning_path = []
        all_contexts = []
        
        # First hop: Initial retrieval
        passages = self.retriever(search_query).passages
        initial_context = "\n\n".join(p["text"] for p in passages)
        focused_context = self.context_rewriter(context=initial_context, question=question).focused_context
        all_contexts.append(focused_context)
        reasoning_path.append(f"Initial search: {search_query}")
        
        # Additional hops for sub-questions
        for i, sub_q in enumerate(sub_questions[:self.max_hops-1]):
            passages = self.retriever(sub_q).passages
            sub_context = "\n\n".join(p["text"] for p in passages)
            focused_sub_context = self.context_rewriter(context=sub_context, question=sub_q).focused_context
            all_contexts.append(focused_sub_context)
            reasoning_path.append(f"Follow-up search {i+1}: {sub_q}")
        
        # Combine all contexts
        final_context = "\n\n---\n\n".join(all_contexts)
        reasoning_path = "\n".join(reasoning_path)
        
        # Generate final answer
        return self.generate(
            context=final_context,
            question=question,
            reasoning_path=reasoning_path
        )
