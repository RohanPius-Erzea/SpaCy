# crafted by RohanPius 

# NLP Pipeline for Sentiment Analysis using NLTK
# This script demonstrates tokenization, text normalization (stop word removal and lemmatization),
# and lexicon-based sentiment scoring using VADER.

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure necessary NLTK packages are downloaded (run once if needed)
try:
    # crafted by Rohan: Download VADER, tokenizers, and wordnet for full functionality
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('vader_lexicon')

# --- 1. Data Setup ---
# Example text to test the sentiment analysis pipeline.
# Note: This text is informal and includes a contraction and an emoticon, which VADER handles well.
text = "The new software update is amazing, but it was slightly buggy before! I'm happy now. :)"

# --- 2. Tokenization and Normalization Functions (The Preprocessing Stage) ---

def preprocess_text(input_text):
    # Step 1: Convert to lower case to standardize the text (Normalization)
    lower_text = input_text.lower()

    # Step 2: Tokenization - Breaking the text into individual words
    tokens = word_tokenize(lower_text)

    # Step 3: Remove Stop Words (Normalization)
    stop_words = set(stopwords.words('english'))
    # crafted by Rohan: Filter out common words that don't add semantic value
    filtered_tokens = [
        word for word in tokens if word.isalnum() and word not in stop_words
    ]

    # Step 4: Lemmatization (Normalization)
    lemmatizer = WordNetLemmatizer()
    # crafted by Rohan: Reducing words to their base or dictionary form (e.g., 'running' -> 'run')
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    return lemmatized_tokens

# --- 3. Sentiment Analysis Function ---

def get_vader_sentiment(lemmatized_tokens):
    # VADER requires the text to be a single string, so we join the tokens back
    processed_text = " ".join(lemmatized_tokens)

    # Initialize the Sentiment Analyzer
    analyzer = SentimentIntensityAnalyzer()

    #Generate the sentiment scores (neg, neu, pos, compound)
    sentiment_scores = analyzer.polarity_scores(processed_text)

    #The compound score is the normalized, main metric for overall sentiment
    compound_score = sentiment_scores['compound']

    #Simple classification based on the compound score
    if compound_score >= 0.05:
        sentiment = "Positive"
    elif compound_score <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return sentiment, sentiment_scores

# --- 4. Execution ---
print("--- Crafted by Rohan: NLP Sentiment Pipeline Report ---")
print(f"Original Text: {text}\n")

#Run the preprocessing steps
processed_tokens = preprocess_text(text)

print(f"Tokens after Normalization (Stop word removal & Lemmatization):")
print(processed_tokens)
print("-" * 50)

#Run the sentiment analysis on the normalized tokens
final_sentiment, scores = get_vader_sentiment(processed_tokens)

print(f"VADER Polarity Scores:")
#The scores are Neg (Negative), Neu (Neutral), Pos (Positive), and Compound (Overall)
print(f"  Negative: {scores['neg']:.2f}")
print(f"  Neutral: {scores['neu']:.2f}")
print(f"  Positive: {scores['pos']:.2f}")
print(f"  Compound Score: {scores['compound']:.2f} (The main metric)")
print("-" * 50)
print(f"Final Sentiment Classification: {final_sentiment}")

# crafted by Rohan: End of script for Lexicon-Based Sentiment Analysis
