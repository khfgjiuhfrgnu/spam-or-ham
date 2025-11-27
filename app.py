import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd

# NLTK setup
nltk.download('stopwords', quiet=True)

# Load model & vectorizer
try:
    model = joblib.load('spam_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
except Exception as e:
    st.error(f"Erreur loading model/vectorizer: {e}")
    st.stop()

# Preprocessing
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words]
    return ' '.join(words)

# Global CSS (professional UI + cybersecurity-style alerts)
st.markdown('<div class="footer">📌 Réalisé par Khaled __ Omar __ Ahmed — Projet Spam Detector</div>', unsafe_allow_html=True)
st.markdown(
    """
    <style>
    /* Base app background (elegant gradient) */
    .stApp {
        background: linear-gradient(135deg, #f0f4f8, #d9e4f5);
        font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Title styling */
    .app-title {
        color: red;
        text-align: center;
        font-size: 2rem;
        font-weight: 800;
        margin: 0.5rem 0 0.25rem 0;
    }
    .app-subtitle {
        color: #3b3b3b;
        text-align: center;
        font-size: 1rem;
        margin-bottom: 1rem;
    }

    /* Result cards */
    .result-card {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 6px 14px rgba(0,0,0,0.12);
    }
    .ham-result {
        background-color: #d1fae5; /* soft green */
        color: #065f46;
        animation: fadeIn 700ms ease-in-out;
    }
    .spam-result {
        background-color: #ff0000; /* strong red */
        color: #ffffff;
        animation: pulse 1s infinite, fadeIn 300ms ease-in-out;
        text-shadow: 0 1px 0 #8b0000;
        box-shadow: 0 0 20px rgba(255,0,0,0.8);
        border: 1px solid rgba(255,255,255,0.35);
    }

    /* Confidence text */
    span.confiance {
        font-size: 0.95em;
        font-weight: 600;
        margin-left: 6px;
        display: inline-block;
        opacity: 0.95;
    }

    /* Buttons (professional) */
    div.stButton > button {
        background-color: #1E90FF;
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-size: 16px;
        font-weight: 700;
        cursor: pointer;
        transition: all 0.25s ease;
        box-shadow: 0 4px 10px rgba(0,0,0,0.18);
    }
    div.stButton > button:hover {
        background-color: #0d6efd;
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(0,0,0,0.22);
    }
    div.stButton > button:active {
        background-color: #0b5ed7;
        transform: translateY(0);
        box-shadow: 0 3px 8px rgba(0,0,0,0.18);
    }

    /* Optional colored variants (use with custom HTML buttons if needed) */
    .btn-green { background-color: #22c55e !important; color: white !important; }
    .btn-red { background-color: #ef4444 !important; color: white !important; }
    .btn-orange { background-color: #f97316 !important; color: white !important; }

    /* Page turns full red on SPAM (cybersecurity alert) */
    .spam-background {
        background-color: #9b0000 !important; /* darker red for contrast */
        background-image: radial-gradient(circle at 15% 20%, #ff2a2a 0%, #9b0000 60%);
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 12px rgba(255,0,0,0.65); }
        50% { box-shadow: 0 0 32px rgba(255,0,0,1); }
        100% { box-shadow: 0 0 12px rgba(255,0,0,0.65); }
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 28px;
        font-size: 0.9em;
        color: #555;
    }

    @media print {
        .spam-result, .ham-result { animation: none !important; }
        .spam-background { background: #ffffff !important; }
    }
    </style>


    
    """,
    unsafe_allow_html=True
)

# Headers
st.markdown('<h1 class="app-title">📩 Détecteur Spam ou Ham</h1>', unsafe_allow_html=True)
# Input
user_input = st.text_area("Entrez un message pour vérifier s'il est spam ou ham", max_chars=1000, placeholder="Tapez votre message ici...")

# Predict button
if st.button("Predict Message"):
    if user_input.strip():
        processed = preprocess_text(user_input)
        try:
            X_new = vectorizer.transform([processed])
            prediction = model.predict(X_new)[0]
            proba = model.predict_proba(X_new)[0]
            ham_conf = float(proba[0] * 100)
            spam_conf = float(proba[1] * 100)

            if prediction == 1:
                # SPAM: turn full background red and show alert-style card
                st.markdown(
                    f"""
                    <script>
                    const app = document.querySelector('.stApp');
                    if (app) {{
                        app.classList.add('spam-background');
                    }}
                    </script>
                    <div class="result-card spam-result">❌ SPAM — <span class="confiance">{spam_conf:.2f}%</span></div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                # HAM: ensure background returns to normal and show green card
                st.markdown(
                    """
                    <script>
                    const app = document.querySelector('.stApp');
                    if (app) {
                        app.classList.remove('spam-background');
                    }
                    </script>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div class="result-card ham-result">✔ Ham — <span class="confiance">{ham_conf:.2f}%</span></div>',
                    unsafe_allow_html=True
                )

            # Show both probabilities
            st.write(f"Fiabilité → Ham: {ham_conf:.2f}% | Spam: {spam_conf:.2f}%")

        except Exception as e:
            st.error(f"Erreur prediction: {e}")
    else:
        st.warning("⚠️ Please enter a message!")

# CSV upload preview
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)
    except Exception as e:
        st.error(f"Erreur lecture CSV: {e}")
# Footer / حقوق

st.markdown('<div class="footer"> © 2025 Projet Spam Detector — Tous droits réservés</div>', unsafe_allow_html=True)
