from .base_analyzer import BaseAnalyzer
from typing import Dict, Any
from textblob import TextBlob

class SentimentAnalyzer(BaseAnalyzer):
    def analyze(self, user_input: str, ai_response: str, context: Dict[str, Any]) -> Dict[str, float]:
        user_input = str(user_input)
        ai_response = str(ai_response)
        user_sentiment = TextBlob(user_input).sentiment.polarity
        ai_sentiment = TextBlob(ai_response).sentiment.polarity
        return {
            "user_sentiment": user_sentiment,
            "ai_sentiment": ai_sentiment,
            "sentiment_delta": ai_sentiment - user_sentiment
        }