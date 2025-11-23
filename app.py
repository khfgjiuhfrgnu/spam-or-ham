import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# -----------------------------
# Télécharger NLTK resources
# -----------------------------
nltk.download('stopwords', quiet=True)

# -----------------------------
# Load model and vectorizer
# -----------------------------
try:
    model = joblib.load('spam_model.pkl')
    vectorizer = joblib.load('tfidf.pkl')
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle: {e}")
    st.stop()

# -----------------------------
# Preprocessing function
# -----------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

# -----------------------------
# Inject CSS
# -----------------------------
css_code = """
.ham-result {
    background-color: #a7f3d0;
    color: #065f46;
    padding: 10px;
    border-radius: 8px;
    margin: 8px 0;
    font-weight: bold;
}
.spam-result {
    background-color: #fee2e2;
    color: #b91c1c;
    padding: 10px;
    border-radius: 8px;
    margin: 8px 0;
    font-weight: bold;
    animation: shake 1s ease-in-out infinite;
}
@keyframes shake {
    0% { transform: translateX(0); }
    20% { transform: translateX(-5px); }
    40% { transform: translateX(5px); }
    60% { transform: translateX(-5px); }
    80% { transform: translateX(5px); }
    100% { transform: translateX(0); }
}
"""
st.markdown(f"<style>{css_code}</style>", unsafe_allow_html=True)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📩 Détecteur Spam ou Ham Debug")
user_input = st.text_area("Message:")

if st.button("Predict Message"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a message!")
    else:
        processed_text = preprocess_text(user_input)
        st.write("✅ Processed text:", processed_text)  # debug
        X_new = vectorizer.transform([processed_text])
        st.write("✅ Vector shape:", X_new.shape)  # debug
        st.write("✅ Vector sample:", X_new.toarray()[0][:20])  # debug (أول 20 قيمة فقط)

        prediction = model.predict(X_new)[0]
        confidence = model.predict_proba(X_new).max() * 100

        if prediction == 0:
            st.markdown(f'<div class="ham-result">✔ Ham — Confiance: {confidence:.2f}%</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="spam-result">❌ SPAM — Confiance: {confidence:.2f}%</div>', unsafe_allow_html=True)

