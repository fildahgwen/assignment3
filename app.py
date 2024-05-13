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


def display_instructions():
    instructions = """
    Instructions:
    1. Open the app in your web browser.
    2. Enter your news text in the text area provided.
    3. Click on the "Cluster News" button.
    4. The app will predict the cluster number and URL for the entered news text.
    5. The predicted cluster number and URL will be displayed below the button.
    6. If the predicted cluster number is not recognized, the URL will be displayed as "Unknown".
    
    Please note that the app may take some time to process the news text and display the results.
    
    If you encounter any issues or have specific questions about the app, please let me know, and I'll be happy to assist you further.
    """
    print(instructions)

def predict_clusters_and_urls(text):
    # Predict cluster numbers
    sgd_cluster = sgd_pipe.predict([text])[0]
    svc_cluster = svc_pipe.predict([text])[0]
    
    # Define URLs based on cluster categories
    url_mapping = {
        'sport': 'https://www.bbc.com/sport',
        'politics': 'https://www.bbc.com/news/world',
        'business': 'https://www.bbc.com/business',
        'entertainment': 'https://www.bbc.com/culture/entertainment-news'
    }
    
    # Get the cluster categories
    sgd_category = {v: k for k, v in list(sgd_pipe.named_steps['clf'].classes_)}[sgd_cluster]
    svc_category = {v: k for k, v in list(svc_pipe.named_steps['clf'].classes_)}[svc_cluster]
   
    
    # Map cluster categories to URLs
    sgd_url = url_mapping.get(sgd_category, None)
    svc_url = url_mapping.get(svc_category, None)
    
    return {
        'Top cluster number (SGD)': sgd_cluster,
        'Top cluster number (SVC)': svc_cluster,
        'URL (SGD)': sgd_url,
        'URL (SVC)': svc_url
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
