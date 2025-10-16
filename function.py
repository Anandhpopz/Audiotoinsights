
# ----------------------------
# Imports
# ----------------------------
import spacy
from transformers import pipeline
from keybert import KeyBERT
import pandas as pd

# ----------------------------
# 1. Sentiment Analysis
# ----------------------------
def analyze_sentiment(text):
    sentiment_analyzer = pipeline("sentiment-analysis")
    result = sentiment_analyzer(text)
    return result[0]['label'], result[0]['score']

# ----------------------------
# 2. Intent Classification
# ----------------------------
def classify_intent(text):
    text_lower = text.lower()
    if "vacancy" in text_lower or "room" in text_lower:
        intent = "Booking Inquiry"
    elif "price" in text_lower:
        intent = "Price Check"
    else:
        intent = "General Inquiry"
    return intent

# ----------------------------
# 3. Named Entity Recognition
# ----------------------------
nlp = spacy.load("en_core_web_sm")  # Load once globally

def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# ----------------------------
# 4. Keyword Extraction
# ----------------------------
kw_model = KeyBERT()  # Initialize once globally

def extract_keywords(text, top_n=10):
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1,2),
        stop_words='english',
        top_n=top_n
    )
    return keywords

# ----------------------------
# 5. Topic Modeling (using KeyBERT keywords)
# ----------------------------
def extract_topic(text, top_n=5):
    keywords = extract_keywords(text, top_n=top_n)
    if keywords:
        topic_keywords = [word for word, score in keywords]
        topic_summary = ", ".join(topic_keywords)
    else:
        topic_summary = "No topic found"
    return topic_summary

# ----------------------------
# 6. Combined Pipeline
# ----------------------------
def nlp_pipeline(text):
    sentiment_label, sentiment_score = analyze_sentiment(text)
    intent = classify_intent(text)
    entities = extract_entities(text)
    keywords = extract_keywords(text)
    topic = extract_topic(text)
    
    insights = {
        "sentiment": sentiment_label,
        "sentiment_score": sentiment_score,
        "intent": intent,
        "entities": entities,
        "keywords": keywords,
        "topic": topic
    }
    return insights

# ----------------------------
# Example Usage
# ----------------------------
call_text = """
A female caller, identifying herself as Ashwin, inquired about hostel vacancies.
She specified her interest in the Palayam area. The manager informed her that while
they have hostels in multiple locations, including Palayam, Nandavanam, and Murinjapalam,
single rooms are currently unavailable. He stated that only two-sharing rooms are available,
priced at 9500 with a shared bathroom. The caller asked if a two-sharing room could be occupied
as a single, to which the manager confirmed it could, but the price would remain 9500. He clarified
that single rooms are not available in Palayam, only in Murinjapalam if taken as a two-sharing room
for one person. The manager also addressed a previous inquiry under the name Revathy Sunil,
clarifying that older pricing (like 5000) was for a different type or location. The conversation
concluded with the manager offering to send details via WhatsApp.
"""

insights = nlp_pipeline(call_text)
print(insights)
