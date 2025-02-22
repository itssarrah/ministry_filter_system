import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import re
from scipy.cluster.hierarchy import dendrogram

# Function to convert string representation of arrays to numpy arrays
def convert_string_to_array(string_array):
    try:
        # Check if it's already a numpy array or similar
        if isinstance(string_array, (np.ndarray, list)):
            return np.array(string_array)
        
        # Clean the string and extract numerical values using regex
        # This pattern matches floating point numbers with optional scientific notation
        pattern = r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'
        values = re.findall(pattern, string_array)
        
        # Convert to float and return as numpy array
        return np.array([float(val) for val in values])
    except Exception as e:
        print(f"Error processing embedding: {e}")
        # Return zero array with estimated dimension (adjust based on your data)
        return np.zeros(300)

# Modified function to use only 'embeddings' (not 'activity_embeddings')
def process_dataframe_embeddings(df):
    # Create a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Get a sample to determine embedding dimensions
    sample_str = processed_df['embeddings'].iloc[0]
    sample_vals = convert_string_to_array(sample_str)
    embedding_dim = len(sample_vals)
    print(f"Detected embedding dimension: {embedding_dim}")
    
    # Convert string embeddings to numpy arrays - only process 'embeddings' column
    print("Converting embeddings from strings to arrays...")
    processed_df['embeddings_array'] = processed_df['embeddings'].apply(convert_string_to_array)
    
    # Validate dimensions
    processed_df['embedding_dim'] = processed_df['embeddings_array'].apply(len)
    
    # Check for inconsistent dimensions
    if processed_df['embedding_dim'].nunique() > 1:
        print("Warning: Inconsistent embedding dimensions detected!")
        print(f"Embedding dimensions: {processed_df['embedding_dim'].value_counts()}")
        
        # Fill inconsistent arrays to match the modal dimension
        modal_dim = processed_df['embedding_dim'].mode()[0]
        
        def pad_or_truncate(arr, target_len):
            if len(arr) < target_len:
                return np.pad(arr, (0, target_len - len(arr)), 'constant')
            else:
                return arr[:target_len]
        
        processed_df['embeddings_array'] = processed_df['embeddings_array'].apply(
            lambda x: pad_or_truncate(x, modal_dim))
    
    # Use only 'embeddings' for clustering
    embedding_column = 'embeddings_array'
    
    # Extract embeddings into a matrix
    print("Creating embedding matrix...")
    embedding_matrix = np.vstack(processed_df[embedding_column].values)
    
    # Check for NaN values
    if np.isnan(embedding_matrix).any():
        print("Warning: NaN values found in embeddings. Replacing with zeros.")
        embedding_matrix = np.nan_to_num(embedding_matrix)
    
    return processed_df, embedding_matrix

# Function to calculate cosine similarity matrix
def calculate_similarity_matrix(embedding_matrix):
    print("Calculating cosine similarity matrix...")
    similarity_matrix = cosine_similarity(embedding_matrix)
    # Convert to distance matrix (1 - similarity)
    distance_matrix = 1 - similarity_matrix
    return similarity_matrix, distance_matrix

# Function to perform hierarchical clustering
def perform_clustering(distance_matrix, n_clusters=10):
    print(f"Performing hierarchical clustering with {n_clusters} clusters...")
    # Use Agglomerative Clustering with cosine distance
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='average'        # Use average linkage strategy
    )
    
    cluster_labels = clustering.fit_predict(distance_matrix)
    return clustering, cluster_labels

# Function to analyze and summarize clusters
def analyze_clusters(df, cluster_labels, top_n=5):
    # Add cluster labels to the dataframe
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = cluster_labels
    
    # Get number of clusters
    n_clusters = len(np.unique(cluster_labels))
    
    # Print summary of each cluster
    print(f"\nCluster Analysis (Total clusters: {n_clusters}):")
    cluster_summaries = []
    
    for cluster_id in range(n_clusters):
        # Get entries for this cluster
        cluster_df = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
        cluster_size = len(cluster_df)
        
        # Skip empty clusters
        if cluster_size == 0:
            continue
            
        # Get most common activities in this cluster
        top_activities = cluster_df['activity'].value_counts().head(top_n)
        
        # Get most common fields in this cluster
        top_fields = cluster_df['field'].value_counts().head(top_n)
        
        # Calculate average similarity within cluster if possible
        avg_similarity = "N/A"
        try:
            if cluster_size > 1:
                embeddings = np.vstack(cluster_df['embeddings_array'].values)  # Using embeddings_array
                similarity = cosine_similarity(embeddings)
                # Exclude self-similarity (diagonal)
                np.fill_diagonal(similarity, 0)
                avg_similarity = similarity.sum() / (cluster_size * (cluster_size - 1))
            else:
                avg_similarity = 1.0  # Single element cluster
        except Exception as e:
            print(f"Could not calculate average similarity for cluster {cluster_id}: {e}")
            
        # Get sample entries from this cluster
        sample_entries = cluster_df.sample(min(3, cluster_size))[['activity', 'description']]
        
        # Create cluster summary
        summary = {
            'cluster_id': cluster_id,
            'size': cluster_size,
            'avg_similarity': avg_similarity,
            'top_activities': top_activities,
            'top_fields': top_fields,
            'sample_entries': sample_entries
        }
        
        cluster_summaries.append(summary)
        
        # Print summary
        # print(f"\nCluster {cluster_id} (Size: {cluster_size}, Avg Similarity: {avg_similarity if isinstance(avg_similarity, float) else 'N/A'})")
        # print("Top Activities:")
        # for activity, count in top_activities.items():
        #     print(f"  - {activity} ({count})")
        # print("Top Fields:")
        # for field, count in top_fields.items():
        #     print(f"  - {field} ({count})")
        # print("Sample Entries:")
        # for idx, entry in sample_entries.iterrows():
        #     print(f"  - {entry['activity']}: {entry['description'][:100]}...")
    
    return df_with_clusters, cluster_summaries

# Main execution function
def cluster_business_activities(df, n_clusters=15):
    # Process dataframe and get embedding matrix
    processed_df, embedding_matrix = process_dataframe_embeddings(df)
    
    # Calculate similarity matrix
    similarity_matrix, distance_matrix = calculate_similarity_matrix(embedding_matrix)
    
    # Perform clustering
    clustering_model, cluster_labels = perform_clustering(distance_matrix, n_clusters)
    
    # Analyze clusters
    df_with_clusters, cluster_summaries = analyze_clusters(processed_df, cluster_labels)
    
    # Return the enriched dataframe with cluster assignments
    return df_with_clusters, cluster_summaries, similarity_matrix


