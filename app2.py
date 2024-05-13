import streamlit as st
import pickle

# Load the saved model
with open('scikit_learn_sgd.pickle', 'rb') as f:
    sgd_pipe = pickle.load(f)

# Define the URL mappings for each cluster
url_mappings = {
    'sports': 'https://www.bbc.com/sport',
    'politics': 'https://www.bbc.com/news/world',
    'business': 'https://www.bbc.com/business',
    'entertainment': 'https://www.bbc.com/culture/entertainment-news'
}

# Function to predict the cluster
def predict_cluster(text):
    cluster = sgd_pipe.predict([text])[0]
    return cluster

# Streamlit app
st.title('Story Clustering App')

text_input = st.text_input('Enter a story:')
if st.button('Predict Cluster'):
    cluster = predict_cluster(text_input)
    st.write(f'The predicted cluster for the story is: {cluster}')
    st.write(f'Related stories URL: {url_mappings.get(cluster, "No URL available")}')
