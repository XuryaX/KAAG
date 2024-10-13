from typing import Dict, Any, Tuple
from .base import BaseAnalyzer
from .nlp_utils import get_sentiment, get_noun_phrases
import numpy as np

class BudgetAlignmentAnalyzer(BaseAnalyzer):
    def __init__(self):
        self.budget_keywords = set(['affordable', 'expensive', 'cost', 'price', 'budget', 'value'])

    def analyze(self, current_state: Dict[str, Any]) -> Tuple[str, float]:
        last_message = current_state.get('last_message', '')
        current_alignment = current_state.get('budget_alignment', 50)
        
        sentiment = get_sentiment(last_message)
        noun_phrases = set(get_noun_phrases(last_message))
        
        budget_relevance = len(self.budget_keywords.intersection(noun_phrases)) / len(self.budget_keywords)
        
        # Positive sentiment with budget terms increases alignment, negative sentiment decreases it
        alignment_change = budget_relevance * sentiment * 20
        
        new_alignment = np.clip(current_alignment + alignment_change, 0, 100)
        return 'budget_alignment', new_alignment