import streamlit as st
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

# Set page configuration
st.set_page_config(
    page_title="News Classifier",
    page_icon="📰",
    layout="centered"
)

# Title and description
st.title("📰 News Classifier")
st.markdown("Classify news articles into categories: **World**, **Sports**, **Business**, or **Sci/Tech**.")

# Load resources (cached to avoid reloading on every interaction)
@st.cache_resource
def load_resources():
    # Load the embedding model
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    
    # Load the classifier
    classifier = joblib.load("models/news_classifier_logreg.pkl")
        
    # Load the label encoder
    label_encoder = joblib.load("models/label_encoder.pkl")
        
    return embedding_model, classifier, label_encoder

try:
    embedding_model, classifier, label_encoder = load_resources()
    st.success("Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# User input
news_text = st.text_area("Enter the news article text here:", height=200)

if st.button("Classify"):
    if news_text.strip():
        with st.spinner("Classifying..."):
            # Generate embedding
            embedding = embedding_model.encode(news_text, normalize_embeddings=True)
            
            # Reshape for prediction (1 sample, n features)
            embedding_reshaped = embedding.reshape(1, -1)
            
            # Predict   
            prediction_index = classifier.predict(embedding_reshaped)[0]
            prediction_label_code = label_encoder.inverse_transform([prediction_index])[0]
            
            # Map code to label
            label_mapping = {
                0: "World",
                1: "Sports",
                2: "Business",
                3: "Sci/Tech"
            }
            prediction_label = label_mapping.get(prediction_label_code, "Unknown")
            
            # Display result
            st.subheader("Prediction:")
            
            # Color coding based on category
            color_map = {
                "World": "blue",
                "Sports": "green",
                "Business": "orange",
                "Sci/Tech": "red"
            }
            color = color_map.get(prediction_label, "black")
            
            st.markdown(f"<h2 style='color: {color};'>{prediction_label}</h2>", unsafe_allow_html=True)
            
            # Show probabilities if available
            if hasattr(classifier, "predict_proba"):
                probabilities = classifier.predict_proba(embedding_reshaped)[0]
                st.subheader("Confidence Scores:")
                
                # Create a dictionary of class: probability
                # Map classes (which are likely ints) to strings
                class_probs = {label_mapping.get(label, str(label)): prob for label, prob in zip(label_encoder.classes_, probabilities)}
                
                # Sort by probability
                sorted_probs = dict(sorted(class_probs.items(), key=lambda item: item[1], reverse=True))
                
                for label, prob in sorted_probs.items():
                    st.progress(float(prob))
                    st.write(f"{label}: {prob:.2%}")
            
    else:
        st.warning("Please enter some text to classify.")

# Sidebar info
st.sidebar.header("About")
st.sidebar.info(
    "This app uses a Logistic Regression model trained on news articles. "
    "It uses 'paraphrase-multilingual-MiniLM-L12-v2' for generating embeddings."
)
