from typing import Dict, Any, Tuple
from .base import BaseAnalyzer
from .nlp_utils import get_entities, get_verb_phrases
import numpy as np

class UrgencyAnalyzer(BaseAnalyzer):
    def __init__(self):
        self.urgency_keywords = set(['urgent', 'immediate', 'quick', 'soon', 'deadline'])

    def analyze(self, current_state: Dict[str, Any]) -> Tuple[str, float]:
        last_message = current_state.get('last_message', '')
        current_urgency = current_state.get('urgency', 10)
        
        entities = set(get_entities(last_message))
        verb_phrases = set(get_verb_phrases(last_message))
        
        relevant_terms = entities.union(verb_phrases)
        urgency_relevance = len(self.urgency_keywords.intersection(relevant_terms)) / len(self.urgency_keywords)
        
        urgency_change = urgency_relevance * 30
        new_urgency = np.clip(current_urgency + urgency_change, 0, 100)
        return 'urgency', new_urgency