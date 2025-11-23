import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# -----------------------------
# Télécharger NLTK resources
# -----------------------------
nltk.download('stopwords')

# -----------------------------
# Load model and vectorizer
# -----------------------------
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('tfidf.pkl')

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
# Inject CSS (اختياري)
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
st.title("\nréalisé par  khaled | Omar  | Ahmed")
st.title("📩 Détecteur Spam ou Ham")
st.write("Entrez un message pour vérifier s'il est spam ou ham.")

# -----------------------------
# Individual message prediction
# -----------------------------
user_input = st.text_area("Message:")
predict_btn = st.button("Predict Message")  # زر Predict حقيقي

if predict_btn:
    if user_input.strip():
        processed_text = preprocess_text(user_input)
        X_new = vectorizer.transform([processed_text])

        prediction = model.predict(X_new)[0]
        confidence = model.predict_proba(X_new).max() * 100

        if prediction == 0:
            st.markdown(f'<div class="ham-result">✔ Ham — Confiance: {confidence:.2f}%</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="spam-result">❌ SPAM — Confiance: {confidence:.2f}%</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a message.")
