from typing import Dict, Any, Tuple
from .base import BaseAnalyzer
from .nlp_utils import nlp
import numpy as np

class LanguageComfortAnalyzer(BaseAnalyzer):
    def analyze(self, current_state: Dict[str, Any]) -> Tuple[str, float]:
        last_message = current_state.get('last_message', '')
        current_comfort = current_state.get('language_comfort', 80)
        
        doc = nlp(last_message)
        sentence_length = len(doc)
        complex_words = len([token for token in doc if len(token.text) > 6])
        
        if sentence_length > 0:
            complexity_score = complex_words / sentence_length
        else:
            complexity_score = 0
        
        comfort_change = (1 - complexity_score) * 20 - 10  # Slight decay if no change
        new_comfort = np.clip(current_comfort + comfort_change, 0, 100)
        return 'language_comfort', new_comfort