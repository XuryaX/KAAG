from typing import Dict, Any, Tuple
from .base import BaseAnalyzer
from .nlp_utils import get_entities, get_noun_phrases, get_verb_phrases
import numpy as np

class ComprehensionAnalyzer(BaseAnalyzer):
    def analyze(self, current_state: Dict[str, Any]) -> Tuple[str, float]:
        last_message = current_state.get('last_message', '')
        current_comprehension = current_state.get('comprehension', 20)
        
        entities = get_entities(last_message)
        noun_phrases = get_noun_phrases(last_message)
        verb_phrases = get_verb_phrases(last_message)
        
        # Comprehension is higher when the message contains more entities, noun phrases, and verb phrases
        comprehension_change = (len(entities) + len(noun_phrases) + len(verb_phrases)) * 2
        
        new_comprehension = np.clip(current_comprehension + comprehension_change, 0, 100)
        return 'comprehension', new_comprehension