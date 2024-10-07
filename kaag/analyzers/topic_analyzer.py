# kaag/analyzers/topic_analyzer.py

from .base_analyzer import BaseAnalyzer
from typing import Dict, Any, List
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

class TopicAnalyzer(BaseAnalyzer):
    def __init__(self, num_topics: int = 5):
        self.num_topics = num_topics
        self.lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        self.vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        self.conversation_history: List[str] = []

    def analyze(self, user_input: str, ai_response: str, context: Dict[str, Any]) -> Dict[str, float]:
        self.conversation_history.append(user_input + " " + ai_response)
        
        if len(self.conversation_history) < 2:
            return {"topic_coherence": 1.0}

        dtm = self.vectorizer.fit_transform(self.conversation_history)
        lda_output = self.lda_model.fit_transform(dtm)

        topic_coherence = self._calculate_topic_coherence(lda_output)
        return {"topic_coherence": topic_coherence}

    def _calculate_topic_coherence(self, lda_output: np.ndarray) -> float:
        # Implement topic coherence calculation
        # This is a simplified version, you might want to use a more sophisticated method
        topic_differences = np.diff(lda_output, axis=0)
        coherence = 1 - np.mean(np.abs(topic_differences))
        return coherence