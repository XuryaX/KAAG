from  typing import Dict, Any, Tuple
from .base import BaseAnalyzer
from .nlp_utils import get_sentiment, get_subjectivity, get_noun_phrases
import numpy as np

class InterestAnalyzer(BaseAnalyzer):
    def analyze(self, current_state: Dict[str, Any]) -> Tuple[str, float]:
        last_message = current_state.get('last_message', '')
        current_interest = current_state.get('interest', 30)
        
        sentiment = get_sentiment(last_message)
        subjectivity = get_subjectivity(last_message)
        noun_phrases = get_noun_phrases(last_message)
        
        # Interest is higher when sentiment is positive, subjectivity is high, and there are many noun phrases
        interest_change = (sentiment + 1) * 10 + subjectivity * 10 + min(len(noun_phrases), 5) * 2
        
        new_interest = np.clip(current_interest + interest_change, 0, 100)
        return 'interest', new_interest