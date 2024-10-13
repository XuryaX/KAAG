import spacy
from typing import List
from textblob import TextBlob

nlp = spacy.load("en_core_web_sm")

def get_sentiment(text: str) -> float:
    return TextBlob(text).sentiment.polarity

def get_subjectivity(text: str) -> float:
    return TextBlob(text).sentiment.subjectivity

def get_entities(text: str) -> List[str]:
    doc = nlp(text)
    return [ent.text for ent in doc.ents]

def get_noun_phrases(text: str) -> List[str]:
    doc = nlp(text)
    return [chunk.text for chunk in doc.noun_chunks]

def get_verb_phrases(text: str) -> List[str]:
    doc = nlp(text)
    return [token.text for token in doc if token.pos_ == "VERB"]