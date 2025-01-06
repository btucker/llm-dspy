import dspy
import logging
from typing import Optional, List
from .retriever import LLMRetriever
from .transformer import QueryTransformer

logger = logging.getLogger(__name__)

class ContextRewriter(dspy.Module):
    """Module to rewrite retrieved context to be more focused on the question."""
    def __init__(self):
        super().__init__()
        self.rewrite = dspy.ChainOfThought(
            "context, question -> focused_context",
            instructions="""
            When rewriting the context:
            1. Keep all specific details like client names, dates, and amounts exactly as they appear
            2. Keep the original wording for key facts and figures
            3. Only remove or summarize parts that are not relevant to the question
            4. Preserve the structure of lists and bullet points
            """
        )
    
    def forward(self, context: str, question: str) -> dspy.Prediction:
        logger.debug(f"ContextRewriter input context: {context}")
        logger.debug(f"ContextRewriter question: {question}")
        result = self.rewrite(context=context, question=question)
        logger.debug(f"ContextRewriter output: {result.focused_context}")
        return result

class EnhancedRAGModule(dspy.Module):
    """Enhanced RAG module with query transformation and multi-hop reasoning."""
    def __init__(self, collection_name: str, k: int = 3, max_hops: int = 2):
        """Initialize the module with fixed collection and parameters.
        
        Args:
            collection_name: Name of collection to use for retrieval
            k: Number of passages to retrieve
            max_hops: Maximum number of reasoning hops
        """
        super().__init__()
        if not collection_name:
            raise ValueError("collection_name must be provided")
            
        self.k = k
        self.max_hops = max_hops
        self.collection_name = collection_name
        
        # Initialize components
        self.query_transformer = QueryTransformer()
        self.retriever = LLMRetriever(collection_name=collection_name, k=k)
        self.context_rewriter = ContextRewriter()
        self.generate = dspy.ChainOfThought("context, question, reasoning_path -> answer")
    
    def forward(self, question: str) -> dspy.Prediction:
        """Process a question through the RAG pipeline.
        
        Args:
            question: The question to answer
            
        Returns:
            A Prediction containing the answer
        """
        logger.debug(f"Processing question: {question}")
        
        # Determine if question is asking for specific data
        needs_specifics = any(term in question.lower() for term in [
            'how much', 'what amount', 'how many', 'revenue', 'cost',
            'transaction', 'number', 'total', 'price', 'value'
        ])
        logger.debug(f"Needs specifics: {needs_specifics}")
        
        # Transform the initial query to be more specific
        transformed = self.query_transformer(question)
        search_query = transformed.search_query
        sub_questions = transformed.sub_questions
        logger.debug(f"Transformed query: {search_query}")
        logger.debug(f"Sub-questions: {sub_questions}")
        
        # Initialize reasoning path and context
        reasoning_path = []
        all_contexts = []
        
        # First hop: Initial retrieval with enhanced query
        if needs_specifics:
            # For financial questions, try both overview and details
            enhanced_queries = [
                f"{search_query} overview total revenue",
                f"{search_query} specific transactions details",
                f"{search_query} including specific numbers, amounts, and dates"
            ]
            for query in enhanced_queries:
                logger.debug(f"Enhanced query: {query}")
                passages = self.retriever(query).passages
                logger.debug(f"Retrieved {len(passages)} passages")
                context = "\n\n".join(p["text"] for p in passages)
                focused_context = self.context_rewriter(context=context, question=question).focused_context
                all_contexts.append(focused_context)
                reasoning_path.append(f"Search: {query}")
        else:
            enhanced_query = f"{search_query} looking for relevant details and context"
            logger.debug(f"Enhanced query: {enhanced_query}")
            passages = self.retriever(enhanced_query).passages
            logger.debug(f"Retrieved {len(passages)} passages")
            initial_context = "\n\n".join(p["text"] for p in passages)
            focused_context = self.context_rewriter(context=initial_context, question=question).focused_context
            all_contexts.append(focused_context)
            reasoning_path.append(f"Initial search: {enhanced_query}")
        
        # Additional hops for sub-questions
        for i, sub_q in enumerate(sub_questions[:self.max_hops-1]):
            # Enhance sub-questions based on the type of information needed
            if needs_specifics:
                enhanced_sub_q = f"{sub_q} focusing on numbers, amounts, and specific details"
            else:
                enhanced_sub_q = f"{sub_q} looking for supporting information and context"
            logger.debug(f"Enhanced sub-question {i+1}: {enhanced_sub_q}")
                
            passages = self.retriever(enhanced_sub_q).passages
            logger.debug(f"Retrieved {len(passages)} passages for sub-question {i+1}")
            sub_context = "\n\n".join(p["text"] for p in passages)
            focused_sub_context = self.context_rewriter(context=sub_context, question=sub_q).focused_context
            all_contexts.append(focused_sub_context)
            reasoning_path.append(f"Follow-up search {i+1}: {enhanced_sub_q}")
        
        # Combine all contexts with clear separation
        final_context = "\n\n---\n\n".join(all_contexts)
        reasoning_path = "\n".join(reasoning_path)
        logger.debug(f"Final context: {final_context}")
        logger.debug(f"Reasoning path: {reasoning_path}")
        
        # Generate final answer with appropriate instruction
        if needs_specifics:
            enhanced_question = f"{question} (Please include all specific numbers, amounts, and dates from the context in your answer)"
        else:
            enhanced_question = f"{question} (Please provide a comprehensive answer based on the context)"
        logger.debug(f"Enhanced question: {enhanced_question}")
            
        result = self.generate(
            context=final_context,
            question=enhanced_question,
            reasoning_path=reasoning_path
        )
        logger.debug(f"Final answer: {result.answer}")
        return result
