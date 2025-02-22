import os
import nltk
import re
import spacy
import pandas as pd
from unidecode import unidecode
from nltk.corpus import stopwords

# Download the French stopwords list from NLTK
nltk.download('stopwords')

# Load French stopwords
french_stopwords = set(stopwords.words('french'))

# Load French language model for spaCy
nlp = spacy.load("fr_core_news_sm")

# Function to normalize text (lowercase, remove accents, and special characters)
def normalize_text(text):
    print("Normalizing text...")
    text = text.lower()
    text = unidecode(text)  # Remove accents
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces
    return text

# Function to tokenize and remove stopwords
def tokenize_and_remove_stopwords(text):
    print("Tokenizing and removing stopwords...")
    # Tokenize the text by splitting it into words
    words = text.split()
    # Remove stopwords
    words = [word for word in words if word not in french_stopwords]
    return words

# Function for lemmatization using spaCy
def lemmatize_words(words):
    print("Lemmatizing words...")
    # Process words through spaCy
    doc = nlp(" ".join(words))
    return [token.lemma_ for token in doc]

# Function to apply preprocessing to activity and description
def process_dataframe(df):
    print("Processing the DataFrame...")
    try:
        # Apply preprocessing to activity column
        print("Processing 'activity' column...")
        df['processed_activity'] = df['activity'].apply(lambda x: ' '.join(lemmatize_words(tokenize_and_remove_stopwords(normalize_text(str(x))))))

        # Apply preprocessing to description column
        print("Processing 'description' column...")
        df['processed_description'] = df['description'].apply(lambda x: ' '.join(lemmatize_words(tokenize_and_remove_stopwords(normalize_text(str(x))))))

        print("Data processed successfully.")
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        raise  # Re-raise the error so it can be caught by the outer try-except block

    return df

# Function to process the DataFrame with caching option
def process_dataframe_with_cache(df, use_cache=True):
    print("Processing the DataFrame...")
    
    # Define the cached file path
    cached_file_path = "processed_df.csv"
    
    if use_cache and os.path.exists(cached_file_path):
        # If caching is enabled and the file exists, load it from the cache
        print("Cache found! Loading processed DataFrame from cache...")
        processed_df = pd.read_csv(cached_file_path)
    else:
        # If caching is disabled or cache doesn't exist, process the DataFrame
        print("No cache found or caching disabled. Processing the DataFrame...")
        processed_df = process_dataframe(df)  # Process the DataFrame
        
        # Save the processed DataFrame to the cache file
        print(f"Saving processed DataFrame to cache at {cached_file_path}...")
        processed_df.to_csv(cached_file_path, index=False)
        
    return processed_df

# Example usage: Load the Excel file into DataFrame and process with caching option
def process_excel_file(file_path, use_cache=True):
    print(f"Loading file from {file_path}...")
    df_ff = pd.read_excel(file_path)
    
    # Call the function with the caching option
    processed_df = process_dataframe_with_cache(df_ff, use_cache)
    
    return processed_df
