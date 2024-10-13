import re
from typing import List, Tuple
from .base import BaseKnowledgeRetriever
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class TextFileKnowledgeRetriever(BaseKnowledgeRetriever):
    def __init__(self, file_path: str, top_k: int = 3):
        with open(file_path, 'r') as file:
            self.content = file.read()
        self.sentences = re.split(r'(?<=[.!?])\s+', self.content)
        self.vectorizer = TfidfVectorizer()
        self.sentence_vectors = self.vectorizer.fit_transform(self.sentences)
        self.top_k = top_k

    def retrieve(self, query: str) -> str:
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.sentence_vectors)
        
        top_k_indices = np.argsort(similarities[0])[-self.top_k:][::-1]
        relevant_sentences = [self.sentences[i] for i in top_k_indices]
        
        return ' '.join(relevant_sentences)