import spacy
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

# Load Spacy model
nlp = spacy.load('en_core_web_sm')

def analyze_concepts(tweet):
    doc = nlp(tweet)
    semantic_analysis_results = []

    # Process the analyzed tokens and extract relevant information
    for token in doc:
        concept = token.lemma_
        score = token.prob
        semantic_analysis_results.append((concept, score))

    # Return the semantic analysis results
    return semantic_analysis_results

# Function to determine sentiment score using SentiStrength and SenticNet
def get_sentiment_score(word):
    # Use TextBlob to calculate sentiment polarity
    blob = TextBlob(word)
    sentiment_score = blob.sentiment.polarity
    return sentiment_score

def create_feature_vector(tweet):
    features = []

    # N-grams features
    vectorizer = TfidfVectorizer(ngram_range = (1, 3))
    ngrams = vectorizer.fit_transform([tweet])
    features.extend(ngrams.toarray()[0])

    # Contradiction features
    sentiment_scores = [get_sentiment_score(word) for word in nltk.word_tokenize(tweet)]
    contradiction = abs(sum(sentiment_scores))
    features.append(contradiction)

    if isinstance(sentiment_scores[0], (list, tuple)):
        positive_score = sum([score[0] for score in sentiment_scores])
        negative_score = sum([score[1] for score in sentiment_scores])
    else:
        positive_score = sentiment_scores
        negative_score = 0.0
    features.append(positive_score)
    features.append(negative_score)

    # Punctuation and special symbols features
    emoticon_count = len(re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', tweet))
    capitalization_count = sum([1 for char in tweet if char.isupper()])
    exclamation_count = tweet.count('!')
    features.append(emoticon_count)
    features.append(capitalization_count)
    features.append(exclamation_count)

    return features

