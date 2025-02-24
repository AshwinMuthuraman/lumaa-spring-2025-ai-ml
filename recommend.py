#!/usr/bin/env python
"""
Content-Based Recommendation System
--------------------------------------------
This script loads a movie dataset, converts plot summaries to TF-IDF vectors,
and computes cosine similarity between a user-provided query and movie plots.
It then outputs the top 5 movie recommendations.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Constants
DATA_FILE = "movies.csv"  # Ensure this CSV file is in your project folder
TOP_N = 5  # Number of recommendations to return
MAX_ROWS = 500  # Limit the dataset to first 500 rows for efficiency

def load_data(file_path):
    """
    Loads the dataset from a CSV file and preprocesses it.
    Expects the CSV to have at least 'Title' and 'Plot' columns.
    """
    if not os.path.exists(file_path):
        sys.exit(f"Error: File '{file_path}' not found. Please download the dataset and place it in the project folder.")
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        sys.exit(f"Error reading the CSV file: {e}")
    
    # Check for required columns
    if 'Title' not in df.columns or 'Plot' not in df.columns:
        sys.exit("Error: CSV file must contain 'Title' and 'Plot' columns.")
    
    # Drop rows with missing plot data and limit to MAX_ROWS
    df = df.dropna(subset=['Plot'])
    df = df.head(MAX_ROWS)
    
    # Optional: Reset index after filtering
    df = df.reset_index(drop=True)
    
    return df

def build_tfidf_matrix(corpus):
    """
    Transforms the text corpus (movie plots) into a TF-IDF matrix.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix

def get_recommendations(query, vectorizer, tfidf_matrix, df, top_n=TOP_N):
    """
    Computes cosine similarity between the query and each movie plot.
    Returns the top_n movie recommendations.
    """
    if not query.strip():
        sys.exit("Error: Query text is empty. Please provide a valid input description.")
    
    # Transform the query to the same vector space
    query_vec = vectorizer.transform([query])
    
    # Compute cosine similarity between query vector and all movie plot vectors
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Handle the edge case where all similarities are zero
    if np.count_nonzero(similarities) == 0:
        sys.exit("No similar movies found. Please try a different query description.")
    
    # Get indices of top_n similar movies
    top_indices = similarities.argsort()[::-1][:top_n]
    recommendations = df.iloc[top_indices].copy()
    recommendations['Similarity'] = similarities[top_indices]
    
    return recommendations[['Title', 'Similarity']]

def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python recommend.py \"<movie description>\"")
    
    query = sys.argv[1]
    
    # Load and preprocess data
    df = load_data(DATA_FILE)
    
    # Build TF-IDF vectorizer and matrix from movie plots
    vectorizer, tfidf_matrix = build_tfidf_matrix(df['Plot'].tolist())
    
    # Get recommendations based on the query
    recommendations = get_recommendations(query, vectorizer, tfidf_matrix, df)
    
    # Print the recommendations
    print("\nRecommendations:")
    for idx, row in recommendations.iterrows():
        print(f"{row['Title']} (Score: {row['Similarity']:.2f})")

if __name__ == "__main__":
    main()
