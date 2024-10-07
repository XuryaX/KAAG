# kaag/analyzers/sentiment_analyzer.py

from .base_analyzer import BaseAnalyzer
from typing import Dict, Any, List
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer

class SentimentAnalyzer(BaseAnalyzer):
    def __init__(self):
        self.naive_analyzer = NaiveBayesAnalyzer()
        self.sentiment_history: List[float] = []

    def analyze(self, user_input: str, ai_response: str, context: Dict[str, Any]) -> Dict[str, float]:
        user_sentiment = self._analyze_sentiment(user_input)
        ai_sentiment = self._analyze_sentiment(ai_response)
        
        self.sentiment_history.append(user_sentiment['polarity'])
        
        return {
            "user_sentiment": user_sentiment['polarity'],
            "ai_sentiment": ai_sentiment['polarity'],
            "sentiment_delta": ai_sentiment['polarity'] - user_sentiment['polarity'],
            "user_emotion": user_sentiment['classification'],
            "sentiment_trend": self._calculate_sentiment_trend()
        }

    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        blob = TextBlob(text, analyzer=self.naive_analyzer)
        return {
            "polarity": blob.sentiment.polarity,
            "classification": blob.sentiment.classification
        }

    def _calculate_sentiment_trend(self) -> float:
        if len(self.sentiment_history) < 2:
            return 0
        return (self.sentiment_history[-1] - self.sentiment_history[0]) / len(self.sentiment_history)