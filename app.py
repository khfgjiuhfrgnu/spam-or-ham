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


st.markdown(
    """
    <style>
    /* Animation Fade-in */
    .fade-in {
        animation: fadeIn 6s ease-in-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    

    /* صناديق النتائج مع دمج fade-in */
    .ham-result {
        background-color: ;
        color: #065f46;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .spam-result {
        background-color: #ff0000;  /* أحمر قوي */
        color: #fff;                /* النص أبيض */
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        font-weight: bold;
        text-align: center;
        box-shadow: 0 0 20px rgba(255,0,0,0.8); /* ظل أحمر قوي */
        animation: pulse 1s infinite; /* حركة تحذير */
    }
      /* Animation pulse */
    @keyframes pulse {
        0% { box-shadow: 0 0 10px rgba(255,0,0,0.6); }
        50% { box-shadow: 0 0 30px rgba(255,0,0,1); }
        100% { box-shadow: 0 0 10px rgba(255,0,0,0.6); }
    }
    </style>
    """,
    unsafe_allow_html=True
)




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
