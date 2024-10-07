# kaag//kaag/analyzers/topic_analyzer.py

from .base_analyzer import BaseAnalyzer
from typing import Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TopicAnalyzer(BaseAnalyzer):
    def __init__(self):
        self.previous_text = ""
        self.previous_vector = None

    def analyze(self, user_input: str, ai_response: str, context: Dict[str, Any]) -> Dict[str, float]:
        current_text = user_input + " " + ai_response

        vectorizer = TfidfVectorizer(lowercase=True)
        if self.previous_text:
            corpus = [self.previous_text, current_text]
            vectors = vectorizer.fit_transform(corpus)
            previous_vector = vectors[0]
            current_vector = vectors[1]
            topic_coherence = cosine_similarity(previous_vector, current_vector)[0][0]
        else:
            topic_coherence = 1.0  # Initial coherence is 1

        self.previous_text = current_text

        return {"topic_coherence": topic_coherence}
