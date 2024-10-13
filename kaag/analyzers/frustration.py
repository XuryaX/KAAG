from typing import Dict, Any, Tuple
from .base import BaseAnalyzer
from .nlp_utils import get_sentiment, get_verb_phrases
import numpy as np

class FrustrationAnalyzer(BaseAnalyzer):
    def analyze(self, current_state: Dict[str, Any]) -> Tuple[str, float]:
        last_message = current_state.get('last_message', '')
        current_frustration = current_state.get('frustration', 0)
        
        sentiment = get_sentiment(last_message)
        verb_phrases = get_verb_phrases(last_message)
        
        negative_verbs = ['hate', 'dislike', 'frustrated', 'annoyed']
        negative_verb_count = sum(verb in ' '.join(verb_phrases).lower() for verb in negative_verbs)
        
        frustration_change = (1 - sentiment) * 25 + negative_verb_count * 5
        new_frustration = np.clip(current_frustration + frustration_change, 0, 100)
        return 'frustration', new_frustration