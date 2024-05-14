import streamlit as st
import pickle

# Load the saved model
with open('scikit_learn_sgd.pickle', 'rb') as f:
    sgd_pipe = pickle.load(f)


##
def main():
    st.set_page_config(
    page_title="Assistive app for the visually impaired",
    page_icon="👁️‍🗨️",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

    st.sidebar.markdown("---")
    st.sidebar.markdown("Developed by Alim Tleuliyev")
    st.sidebar.markdown("Contact: [alim.tleuliyev@nu.edu.kz](mailto:alim.tleuliyev@nu.edu.kz)")
    st.sidebar.markdown("GitHub: [Repo](https://github.com/AlimTleuliyev/image-to-audio)")

    st.markdown(
        """
        <style>
        .container {
            max-width: 800px;
        }
        .title {
            text-align: center;
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .description {
            margin-bottom: 30px;
        }
        .instructions {
            margin-bottom: 20px;
            padding: 10px;
            background-color: #f5f5f5;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Title
    st.markdown("<div class='title'>Assistive app for the visually impaired</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])

    with col1:
        st.write("")

    with col2:
        st.image("images/1.jpg", use_column_width=True, caption="Bridging the Gap")

    with col3:
        st.write("")

    # Model Description
    st.markdown("<div class='description'>" + model_description + "</div>", unsafe_allow_html=True)

    # Instructions
    with st.expander("Instructions"):
        st.markdown("1. Upload an image or provide the URL of an image.")
        st.markdown("3. Click the 'Generate Caption and Speech' button.")
        st.markdown("2. Upload a video")
        st.markdown("4. The generated caption will be displayed, and the speech will start playing.")


##
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
st.title("News Clustering App")

text_input = st.text_input('Enter a story:')
if st.button('Predict Cluster'):
    cluster = predict_cluster(text_input)
    st.write(f'The predicted cluster for the story is: {cluster}')
    st.write(f'Related stories URL: {url_mappings.get(cluster, "No URL available")}')
