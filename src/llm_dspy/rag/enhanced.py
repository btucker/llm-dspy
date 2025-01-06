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
            
        Raises:
            KeyError: If collection_name is not found in llm.collections
            ValueError: If collection_name is empty
        """
        super().__init__()
        if not collection_name:
            raise ValueError("collection_name must be provided")
        
        # Validate collection exists
        import llm
        if collection_name not in llm.collections:
            raise KeyError(f"Collection '{collection_name}' not found")
            
        self.k = k
        self.max_hops = max_hops
        self.collection_name = collection_name
        
        # Initialize components
        self.query_transformer = QueryTransformer()
        self.retriever = LLMRetriever(collection_name=collection_name, k=k)
        self.context_rewriter = ContextRewriter()
        self.generate = dspy.ChainOfThought(
            "context, question, reasoning_path -> answer",
            instructions="""
            When generating answers:
            1. Include ALL specific details from the context that are relevant to the question
            2. Keep exact numbers, dates, and amounts as they appear in the context
            3. If listing items, make sure to include ALL items from the context
            4. Maintain the original wording for technical terms and proper nouns
            5. If the question asks for a list, ensure ALL items are included in the response
            6. If the question asks for chronological order, list items by date from earliest to latest
            7. For transactions, always include the date, amount, client, and type of transaction
            """
        )
    
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
            'transaction', 'number', 'total', 'price', 'value', 'list'
        ])
        needs_chronological = 'chronological' in question.lower() or 'order' in question.lower()
        logger.debug(f"Needs specifics: {needs_specifics}")
        logger.debug(f"Needs chronological: {needs_chronological}")
        
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
            # For questions needing specific details, try multiple focused queries
            enhanced_queries = [
                search_query,  # Original query first
                f"{search_query} specific details dates amounts",
                f"{search_query} all items complete list",
                f"{search_query} including all information"
            ]
            if needs_chronological:
                enhanced_queries.append(f"{search_query} in chronological order by date")
            
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
                enhanced_sub_q = f"{sub_q} including all specific details and complete information"
                if needs_chronological:
                    enhanced_sub_q += " in chronological order"
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
            enhanced_question = f"{question} (Please include ALL specific details, numbers, amounts, and dates from the context in your answer. Make sure to list EVERY item mentioned"
            if needs_chronological:
                enhanced_question += " in chronological order by date"
            enhanced_question += ".)"
        else:
            enhanced_question = f"{question} (Please provide a comprehensive answer based on ALL information in the context)"
        logger.debug(f"Enhanced question: {enhanced_question}")
            
        result = self.generate(
            context=final_context,
            question=enhanced_question,
            reasoning_path=reasoning_path
        )
        logger.debug(f"Final answer: {result.answer}")
        return result
