import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from step1 import process_excel_file  # Import the NLP processing function
from step2 import generate_and_save_embeddings, process_dataframes  # Import Step 2 functions
from step3 import filter_commercial_activities,filter_commercial_activities_advanced
from activity_clustering import cluster_business_activities
from groq_helper import generate_cluster_titles


app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Store processed dataframe temporarily in memory (this could be adjusted for a more scalable solution)
processed_data = None

official_embeddings = np.load('df_official_embeddings.npy')  # Example, adjust as per actual usage

# Function to calculate cosine similarity between the proposed activity and the official activities
def compute_similarity(row):
    similarities = cosine_similarity([row['activity_embeddings']], official_embeddings)
    max_similarity = np.max(similarities)
    return max_similarity

@app.route('/api/upload', methods=['POST'])
def upload_file():
    global processed_data  # Using a global variable for simplicity

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    try:
        # Process the file using the NLP steps (Step 1)
        processed_data = process_excel_file(filepath)

        # For now, just send a preview of the first few rows of the processed data
        preview = processed_data[['processed_activity', 'processed_description']].head().to_dict(orient='records')

        return jsonify({
            "message": "File uploaded and processed successfully!",
            "preview": preview
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/generate_embeddings', methods=['POST'])
def generate_embeddings():
    global processed_data  # Get processed data from the previous step

    if processed_data is None:
        return jsonify({"error": "No processed data found. Please upload a file first."}), 400

    try:
        # Step 1: Load the processed data
        df_ff = pd.read_csv("fuzzy_cleaned.csv", encoding="utf-8")  # This is the processed data from Step 1
        
        # Create the 'activity_description' column by concatenating 'activity' and 'description'
        if 'activity' in df_ff.columns and 'description' in df_ff.columns:
            df_ff['activity_description'] = df_ff['activity'] + " " + df_ff['description']
        else:
            return jsonify({"error": "Missing 'activity' or 'description' columns in fuzzy_cleaned.csv."}), 400
        
        # Remove rows where 'activity_description' is NaN
        df_ff = df_ff.dropna(subset=['activity_description'])
        
        # Step 2: Load df_official from CSV file
        df_official = pd.read_csv("df_official.csv")
        print(df_ff)

        # Step 4: Generate and save embeddings (Check cache first)
        print("Starting embedding generation...")
        process_dataframes(df_ff, df_official, use_cache=True)

        # Step 5: If successful, return success response
        return jsonify({"message": "Embeddings generated and saved successfully!"}), 200
    except Exception as e:
        # Catch any error and return it in the response
        print(f"Error generating embeddings: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/load_cleaned_embeddings', methods=['GET'])
def load_cleaned_embeddings():
    try:
        # Load the original df_ff (we assume it is available)
        df_ff = pd.read_csv("fuzzy_cleaned.csv", encoding="utf-8")
        
        # Load cleaned embeddings from cache
        cleaned_embeddings_file = "cleaned_embeddings.csv"
        if os.path.exists(cleaned_embeddings_file):
            cleaned_df = pd.read_csv(cleaned_embeddings_file)
            
            # Calculate the number of rows removed
            rows_removed = len(df_ff) - len(cleaned_df)
            return jsonify({
                "message": f"Performed FAISS similarity. Now the number of rows are: {len(cleaned_df)}. Removed {rows_removed} rows."
            }), 200
        else:
            return jsonify({"error": "Cleaned embeddings not found."}), 404
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/filter_commercial_activities', methods=['GET'])
def filter_commercial():
    try:
        # Load the cleaned embeddings (from step2)
        cleaned_embeddings_file = "cleaned_embeddings.csv"
        if os.path.exists(cleaned_embeddings_file):
            cleaned_df = pd.read_csv(cleaned_embeddings_file)

            # Filter commercial activities
            non_commercial_cleaned = filter_commercial_activities(cleaned_df)
            
            # Save the filtered dataframe or return success message
            non_commercial_cleaned.to_csv("non_commercial_cleaned.csv", index=False)
            # filter_commercial_activities_advanced(non_commercial_cleaned) it will give filtered_df 
            filtered_df = pd.read_csv("filtered_activities.csv")
            return jsonify({
                "message": f"Commercial activities filtered. Now the number of rows are: {len(filtered_df)}"
            }), 200
        else:
            return jsonify({"error": "Cleaned embeddings not found."}), 404
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/filter_unmatched_activities', methods=['GET'])
def filter_unmatched_activities():
    try:
        # Load the filtered non-commercial activities (already done in the previous step)
        filtered_df = pd.read_csv("filtered_activities.csv")

        # Compute the cosine similarity for each row in filtered_df
        # filtered_df['similarity'] = filtered_df.apply(compute_similarity, axis=1)

        # Set a threshold for similarity (can be adjusted as per requirement)
        threshold = 0.8
        # filtered_df_unmatched = filtered_df[filtered_df['similarity'] < threshold]

        # Save the result to a CSV file
        # filtered_df_unmatched.to_csv('filtered_unmatched_activities.csv', index=False)
        filtered_df_unmatched =  pd.read_csv("filtered_unmatched_activities.csv")
        # Return success message with the number of unmatched activities
        return jsonify({
            "message": f"Activities with similarity below threshold have been saved. Remaining rows: {filtered_df_unmatched.shape[0]}"
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/api/cluster', methods=['GET'])
def cluster_activities():
    try:
        # Load CSV data
        df = pd.read_csv("filtered_unmatched_activities.csv")
        print(f"Data loaded with {len(df)} records.")

        # Get number of clusters
        n_clusters = int(request.args.get('n_clusters', 15))
        print(f"Starting clustering with {n_clusters} clusters.")

        # Perform clustering
        df_clustered, summaries, similarity_matrix = cluster_business_activities(df, n_clusters)

        # Generate titles using Groq
        cluster_titles = generate_cluster_titles(summaries)
        
        # Prepare cluster data
        clusters_dataframes = {}
        for cluster_id in range(n_clusters):
            cluster_df = df_clustered[df_clustered['cluster'] == cluster_id]
            cluster_df = cluster_df[['wilaya', 'field', 'activity', 'description']].replace({np.nan: None})
            clusters_dataframes[cluster_id] = cluster_df
        
        # Return JSON response with cluster titles
        return jsonify({
            'message': 'Clustering completed successfully!',
            'cluster_titles': {cluster_id: title for cluster_id, title in enumerate(cluster_titles)},
            'cluster_dataframes': {cluster_id: cluster_df.to_dict(orient='records') for cluster_id, cluster_df in clusters_dataframes.items()},
        })

    except Exception as e:
        print(f"Error during clustering: {e}")
        return jsonify({
            'message': 'Error during clustering process.',
            'error': str(e),
        })

    
if __name__ == '__main__':
    app.run(debug=True)
