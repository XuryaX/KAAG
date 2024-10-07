from typing import Dict, Any, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class KnowledgeIntegrator:
    def __init__(self, knowledge_base: Dict[str, str]):
        self.knowledge_base = knowledge_base
        self.vectorizer = TfidfVectorizer()
        self.knowledge_vectors = self.vectorizer.fit_transform(list(knowledge_base.values()))

    def integrate(self, query: str, top_k: int = 3) -> str:
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.knowledge_vectors)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        relevant_knowledge = [list(self.knowledge_base.values())[i] for i in top_indices]
        return "\n".join(relevant_knowledge)

    def update_knowledge(self, new_knowledge: Dict[str, str]):
        self.knowledge_base.update(new_knowledge)
        self.knowledge_vectors = self.vectorizer.fit_transform(list(self.knowledge_base.values()))