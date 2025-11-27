import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

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


st.set_page_config(page_title="Détecteur Spam ou Ham",
                   page_icon="📧",
                   layout="centered")

# --- CSS PERSONNALISÉ ---
st.markdown("""
<style>
body {
    font-family: 'Inter', sans-serif;
}
.main {
    background: linear-gradient(135deg, #155799, #159957);
}
h1, h2, h3, label {
    color: white !important;
}
textarea {
    border-radius: 12px !important;
    border: 2px solid rgba(255,255,255,0.35) !important;
    background-color: rgba(255,255,255,0.15) !important;
    color: white !important;
    font-size: 16px !important;
}
div[data-testid="stFileUploader"] section {
    background-color: rgba(255,255,255,0.12);
    border-radius: 12px;
    padding: 10px;
}
.stButton > button {
    background-color: #0c6efd !important;
    border-radius: 8px !important;
    padding: 10px 20px;
    color: white;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

st.title("📧 Détecteur Spam ou Ham")
st.write("Entrez un message pour vérifier s’il est spam ou ham.")

message = st.text_area("Message :", max_chars=1000)

if st.button("🔍 Prédire"):
    st.success("Exemple : ham (ici vous mettrez votre modèle)")
    
uploaded = st.file_uploader("Ou importer un fichier CSV :", type=["csv"])
if uploaded:
    st.success("CSV bien reçu")


# Predict single message
user_input = st.text_area("", max_chars=1000)
if st.button("Predict Message"):
    if user_input.strip():
        processed = preprocess_text(user_input)
        try:
            X_new = vectorizer.transform([processed])
            prediction = model.predict(X_new)[0]
            proba = model.predict_proba(X_new)[0]
            ham_conf = proba[0] * 100
            spam_conf = proba[1] * 100

            if prediction == 0:
                st.markdown(
                    f'<div class="ham-result">✔ Ham — <span class="confiance">{ham_conf:.2f}%</span></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="spam-result">❌ SPAM — <span class="confiance">{spam_conf:.2f}%</span></div>',
                    unsafe_allow_html=True
                )

            # Show both probabilities
            st.write(f"Fiabilité  → Ham: {ham_conf:.2f}% | Spam: {spam_conf:.2f}%")
        except Exception as e:
            st.error(f"Erreur prediction: {e}")
    else:
        st.warning("⚠️ Please enter a message!")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    import pandas as pd
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)
