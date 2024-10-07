# kaag/models/knowledge_integrator.py

from typing import Dict, Any, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class KnowledgeIntegrator:
    def __init__(self, knowledge_base: Dict[str, str]):
        self.knowledge_base = knowledge_base
        self.vectorizer = TfidfVectorizer()
        self.knowledge_vectors = self.vectorizer.fit_transform(list(knowledge_base.values()))

    def integrate(self, query: str, context: Dict[str, Any], top_k: int = 3) -> str:
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.knowledge_vectors)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        relevant_knowledge = [list(self.knowledge_base.values())[i] for i in top_indices]
        
        # Synthesize knowledge with context
        synthesized_knowledge = self._synthesize_knowledge(relevant_knowledge, context)
        
        return synthesized_knowledge

    def _synthesize_knowledge(self, relevant_knowledge: List[str], context: Dict[str, Any]) -> str:
        current_stage = context.get('current_stage', '')
        metrics = context.get('metrics', {})
        
        # Implement logic to synthesize knowledge based on current stage and metrics
        synthesized = f"Given the current stage '{current_stage}' and metrics {metrics}, here's the relevant information:\n"
        synthesized += "\n".join(relevant_knowledge)
        
        return synthesized

    def update_knowledge(self, new_knowledge: Dict[str, str]):
        self.knowledge_base.update(new_knowledge)
        self.knowledge_vectors = self.vectorizer.fit_transform(list(self.knowledge_base.values()))