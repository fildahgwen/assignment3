import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn import metrics
import pickle

# Load the saved models
with open('scikit_learn_sgd.pickle', 'rb') as f:
    sgd_pipe = pickle.load(f)
    
with open('scikit_learn_svm.pickle', 'rb') as f:
    svc_pipe = pickle.load(f)

# Function to predict clusters and URLs
def predict_clusters_and_urls(text):
    # Predict cluster numbers
    sgd_cluster = sgd_pipe.predict([text])[0]
    svc_cluster = svc_pipe.predict([text])[0]
    
    # Map cluster numbers to category names
    cluster_mapping = {0: 'sport', 1: 'politics', 2: 'business', 3: 'entertainment'}
    
    # Define URLs for each cluster
    urls = {
        0: 'https://www.bbc.com/sport',
        1: 'https://www.bbc.com/news/world',
        2: 'https://www.bbc.com/business',
        3: 'https://www.bbc.com/culture/entertainment-news'
    }
    
    return {
        'Top cluster number (SGD)': sgd_cluster,
        'Top cluster number (SVC)': svc_cluster,
        'Top cluster category (SGD)': cluster_mapping.get(sgd_cluster, 'Unknown'),
        'Top cluster category (SVC)': cluster_mapping.get(svc_cluster, 'Unknown'),
        'URL (SGD)': urls.get(sgd_cluster, 'Unknown'),
        'URL (SVC)': urls.get(svc_cluster, 'Unknown')
    }

# Streamlit app
st.title('News Clustering App')

text_input = st.text_area("Enter your news text here:", "")

if st.button("Cluster News"):
    if text_input:
        result = predict_clusters_and_urls(text_input)
        st.write(result)
    else:
        st.write("Please enter some text to cluster the news.")
