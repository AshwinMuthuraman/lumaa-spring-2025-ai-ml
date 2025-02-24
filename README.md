# Simple Content-Based Recommendation System

## Overview
This project implements a content-based recommendation system for movies. Given a short text description of a user's preferences, the system returns the top 3–5 similar movies based on their plot summaries.

## Dataset
We use a subset of the [Wikipedia Movie Plots dataset](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots) from Kaggle.  
**Instructions:**
- Download the dataset (e.g., `wiki_movie_plots_deduped.csv`).
- For simplicity, copy the CSV into your project directory and rename it to `movies.csv` (or modify the file path in the code).
- The code automatically uses the first 500 rows, which is within the required size.

## Setup
- **Python Version:** 3.8+
- **Virtual Environment:** (Optional but recommended)
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows use: venv\Scripts\activate
