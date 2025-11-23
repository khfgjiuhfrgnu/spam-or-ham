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
# Load model and vectorizer avec try/except
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
/* Background général */
body, .stApp {
    background-color: #d1fae5;  /* أخضر فاتح */
}

.ham-result {
    background-color: #d1fae5;
    color: #065f46;
    padding: 10px;
    border-radius: 15px;
    margin: 8px 0;
    font-weight: bold;
}

.spam-result {
    background-color: #fee2e2;
    color: #991b1b;
    padding: 10px;
    border-radius: 15px;
    margin: 8px 0;
    font-weight: bold;
   animation: shake 1s ease-in-out infinite;  /* ← الآن يهتز بشكل مستمر */
}

@keyframes shake {
    0% { transform: translateX(0); }
    20% { transform: translateX(-5px); }
    40% { transform: translateX(5px); }
    60% { transform: translateX(-5px); }
    80% { transform: translateX(5px); }
    100% { transform: translateX(0); }
}

.ham-result span, 
.spam-result span {
    font-size: 0.9em;
    font-weight: normal;
    margin-left: 5px;
    color: #374151;
}

.warning-text {
    color: #b91c1c; /* أحمر للنصوص التحذيرية */
    font-weight: bold;
}
"""
st.markdown(f"<style>{css_code}</style>", unsafe_allow_html=True)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("\nréalisé par  khaled | Omar  | Ahmed")
st.title("📩 Détecteur Spam ou Ham")
st.write("Entrez un message pour vérifier s'il est spam ou ham.")

# -----------------------------
# Individual message prediction
# -----------------------------
user_input = st.text_area("Message:")
predict_btn = st.button("Predict Message")  # زر Predict حقيقي

if predict_btn:
    if not user_input.strip():
        st.markdown('<div class="warning-text">⚠️ Please enter a message!</div>', unsafe_allow_html=True)
    else:
        processed_text = preprocess_text(user_input)
        X_new = vectorizer.transform([processed_text])

        prediction = model.predict(X_new)[0]
        confidence = model.predict_proba(X_new).max() * 100

        if prediction == 0:
            st.markdown(f'<div class="ham-result">✔ Ham — <span>Confiance: {confidence:.2f}%</span></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="spam-result">❌ SPAM — <span>Confiance: {confidence:.2f}%</span></div>', unsafe_allow_html=True)
