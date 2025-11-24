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



# لون الخلفية كامل الصفحة
st.markdown(
    """
    <style>
    /* لون الخلفية كامل الصفحة */
    .stApp {
        background-color: #dbeeff;  /* أزرق فاتح */
    }

    /* صندوق HAM */
    .ham-result {
        background-color: #4ade80;  /* أخضر فاتح */
        color: #065f46;
        padding: 10px;
        border-radius: 8px;
        margin: 8px 0;
        font-weight: bold;
        text-align: center;
    }

    /* صندوق SPAM */
    .spam-result {
        background-color: #f87171;  /* أحمر فاتح */
        color: #991b1b;
        padding: 10px;
        border-radius: 8px;
        margin: 8px 0;
        font-weight: bold;
        animation: shake 0.5s ease-in-out infinite;
        text-align: center;
    }

    @keyframes shake {
        0% { transform: translateX(0); }
        20% { transform: translateX(-5px); }
        40% { transform: translateX(5px); }
        60% { transform: translateX(-5px); }
        80% { transform: translateX(5px); }
        100% { transform: translateX(0); }
    }

    span.confiance {
        font-size: 0.9em;
        font-weight: normal;
        margin-left: 5px;
        display: inline-block;
    }

    @media print {
        .spam-result, .ham-result {
            animation: none !important;
        }
    }
    </style>
    """, unsafe_allow_html=True
)


# عنوان باللون الأزرق
st.markdown('<h1 style="color: orange; text-align: center;">réalisé par Khaled  __   Omar  __ Ahmed</h1>', unsafe_allow_html=True)


# عنوان باللون الأزرق
st.markdown('<h1 style="color: #1E90FF; text-align: center;"> 📩 Détecteur Spam ou Ham </h1>', unsafe_allow_html=True)


# Predict single message
user_input = st.text_area("Entrez un message pour vérifier s'il est spam ou ham:", max_chars=1000)
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
    st.dataframe(df.head())
    st.dataframe(df.body())
