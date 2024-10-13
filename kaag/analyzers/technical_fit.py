from typing import Dict, Any, Tuple
from .base import BaseAnalyzer
from .nlp_utils import get_entities, get_noun_phrases
import numpy as np

class TechnicalFitAnalyzer(BaseAnalyzer):
    def __init__(self):
        self.tech_keywords = set(['blockchain', 'cryptocurrency', 'smart contract', 'scalability', 'security'])

    def analyze(self, current_state: Dict[str, Any]) -> Tuple[str, float]:
        last_message = current_state.get('last_message', '')
        current_fit = current_state.get('technical_fit', 50)
        
        entities = set(get_entities(last_message))
        noun_phrases = set(get_noun_phrases(last_message))
        
        relevant_terms = entities.union(noun_phrases)
        tech_relevance = len(self.tech_keywords.intersection(relevant_terms)) / len(self.tech_keywords)
        
        fit_change = tech_relevance * 20
        new_fit = np.clip(current_fit + fit_change, 0, 100)
        return 'technical_fit', new_fit