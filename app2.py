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
# Set the background color to Benjamin Moore Hale Navy
st.markdown(
    """
    <style>
    body {
        background-color: #4D648D; /* Benjamin Moore Hale Navy */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add the newspaper icon to the app titlest.title("<i class='fas fa-newspaper'></i> Your App Title", unsafe_allow_html=True)

# Include Font Awesome library
st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">', unsafe_allow_html=True)

# Add the newspaper icon to the app title
st.title("<i class='fas fa-newspaper'></i> Your App Title", unsafe_allow_html=True)
##


##st.title('Story Clustering App')

text_input = st.text_input('Enter a story:')
if st.button('Predict Cluster'):
    cluster = predict_cluster(text_input)
    st.write(f'The predicted cluster for the story is: {cluster}')
    st.write(f'Related stories URL: {url_mappings.get(cluster, "No URL available")}')
