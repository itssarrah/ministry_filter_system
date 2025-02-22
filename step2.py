import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np
import os

# Load multilingual Sentence-BERT model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Load French language model for lemmatization (though lemmatization is not needed here)
nlp = spacy.load("fr_core_news_sm")

# Caching mechanism: Check if embeddings are cached
def load_embeddings_from_cache():
    if os.path.exists("df_ff_embeddings.npy") and os.path.exists("df_ff_descriptions.npy"):
        print("Loading embeddings from cache...")
        try:
            df_ff_embeddings = np.load("df_ff_embeddings.npy", allow_pickle=True)
            df_ff_descriptions = np.load("df_ff_descriptions.npy", allow_pickle=True)
        except Exception as e:
            print(f"Error loading df_ff embeddings: {e}")
            df_ff_embeddings, df_ff_descriptions = None, None
    else:
        df_ff_embeddings, df_ff_descriptions = None, None
    
    if os.path.exists("df_official_embeddings.npy") and os.path.exists("df_official_activities.npy"):
        print("Loading official embeddings from cache...")
        try:
            df_official_embeddings = np.load("df_official_embeddings.npy", allow_pickle=True)
            df_official_activities = np.load("df_official_activities.npy", allow_pickle=True)
        except Exception as e:
            print(f"Error loading df_official embeddings: {e}")
            df_official_embeddings, df_official_activities = None, None
    else:
        df_official_embeddings, df_official_activities = None, None

    # Check and print the shape of embeddings
    if df_ff_embeddings is not None:
        print(f"df_ff embeddings shape: {df_ff_embeddings.shape}")
    else:
        print("df_ff embeddings are None")
    
    if df_official_embeddings is not None:
        print(f"df_official embeddings shape: {df_official_embeddings.shape}")
    else:
        print("df_official embeddings are None")
    
    return df_ff_embeddings, df_ff_descriptions, df_official_embeddings, df_official_activities




# Function to get embeddings from Sentence-BERT
def get_embeddings(text):
    try:
        return model.encode(text, convert_to_numpy=True)
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return np.zeros(384)  # Return a default zero-vector if something fails

# Function to generate embeddings and save them (with caching)
def generate_and_save_embeddings(df_ff, df_official):
    print("Generating embeddings...")

    # Check if columns are present in the DataFrames
    if 'processed_activity' not in df_ff or 'processed_description' not in df_ff:
        print("Missing columns in df_ff.")
        return

    df_ff['activity_description'] = df_ff['processed_activity'] + " " + df_ff['processed_description']

    # Check for missing embeddings and generate
    if df_ff['embeddings'].isnull().any():
        print("Generating embeddings for df_ff...")
        df_ff['embeddings'] = df_ff['activity_description'].apply(get_embeddings)

    if 'processed_activity' not in df_official:
        print("Missing columns in df_official.")
        return

    # Generate embeddings for df_official
    if df_official['field_embeddings'].isnull().any():
        print("Generating embeddings for df_official...")
        df_official['field_embeddings'] = df_official['processed_activity'].apply(get_embeddings)

    # Save df_ff embeddings
    try:
        print("Saving df_ff embeddings...")
        np.save("df_ff_embeddings.npy", np.stack(df_ff['embeddings'].values))
        np.save("df_ff_descriptions.npy", df_ff['activity_description'].values)
    except Exception as e:
        print(f"Error saving df_ff embeddings: {e}")

    # Save df_official embeddings
    try:
        print("Saving df_official embeddings...")
        np.save("df_official_embeddings.npy", np.stack(df_official['field_embeddings'].values))
        np.save("df_official_activities.npy", df_official['processed_activity'].values)
    except Exception as e:
        print(f"Error saving df_official embeddings: {e}")

    print("✅ Embeddings saved successfully!")

# Example function to process DataFrames and check cache
def process_dataframes(df_ff, df_official, use_cache=True):
    if use_cache:
        # Load embeddings from cache if available
        print("df_ff head:")
        print(df_ff.head())

        print("df_official head:")
        print(df_official.head())

        df_ff_embeddings, df_ff_descriptions, df_official_embeddings, df_official_activities = load_embeddings_from_cache()

        if df_ff_embeddings is not None and df_official_embeddings is not None:
            print("Cache used, skipping embedding generation.")

            # Ensure embeddings are assigned as individual vectors for each row
            df_ff['embeddings'] = df_ff_embeddings.tolist()  # Convert embeddings to a list of vectors (one per row)
            df_ff['activity_description'] = df_ff_descriptions

            df_official['field_embeddings'] = df_official_embeddings.tolist()
            df_official['processed_activity'] = df_official_activities

            print(f"df_ff embeddings: {df_ff['embeddings'].head()}")
            print(df_ff.head())
        else:
            print("Cache not found or invalid. Generating embeddings...")
            generate_and_save_embeddings(df_ff, df_official)
    else:
        print("No cache requested. Generating embeddings...")
        generate_and_save_embeddings(df_ff, df_official)

