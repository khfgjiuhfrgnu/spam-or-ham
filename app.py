import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd

# -----------------------------
# Télécharger NLTK resources
# -----------------------------
nltk.download('stopwords')

# -----------------------------
# Load model and vectorizer (training outputs)
# -----------------------------
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('tfidf.pkl')

# -----------------------------
# Preprocessing function
# -----------------------------
def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

# -----------------------------
# Inject CSS
# -----------------------------
def inject_css(file_path="style.css"):
    try:
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("⚠️ style.css not found, using default style.")

inject_css()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("\nréalisé par  kh
