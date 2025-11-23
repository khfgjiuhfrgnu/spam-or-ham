import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
import pandas as pd

nltk.download('stopwords', quiet=True)

# -----------------------------
# Load model & vectorizer
# -----------------------------
try:
    model = joblib.load('spam_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
except Exception as e:
    st.error(f"Erreur loading model/vectorizer: {e}")
    st.stop()

# -----------------------------
# Preprocessing function
# -----------------------------
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# -----------------------------
# CSS
# -----------------------------
css = """
body, .stApp { background-color: #d1fae5; }

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
    color: #991b1b;
    padding: 10px;
    border-radius: 8px;
    margin: 8px 0;
    font-weight: bold;
    animation: shake 0.5s ease-in-out infinite;
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
}
"""
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📩 Détecteur Spam ou Ham")
st.write("Entrez un message ou uploadez un CSV pour vérifier spam/ham.")

# -----------------------------
# Predict single message (avec confidence)
# -----------------------------
user_input = st.text_area("Message:")
if st.button("Predict Message"):
    if user_input.strip():
        processed = preprocess_text(user_input)
        X_new = vectorizer.transform([processed])
        prediction = model.predict(X_new)[0]
        confidence = model.predict_proba(X_new).max() * 100

        if prediction == 0:
            st.markdown(f'<div class="ham-result">✔ Ham — <span class="confiance">{confidence:.2f}%</span></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="spam-result">❌ SPAM — <span class="confiance">{confidence:.2f}%</span></div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter a message!")

# -----------------------------
# Predict CSV (sans confidence)
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
if uploaded_file:
    if st.button("Predict CSV"):
        try:
            df = pd.read_csv(uploaded_file)
            col_name = 'sms' if 'sms' in df.columns else 'message' if 'message' in df.columns else None
            if not col_name:
                st.error("CSV doit contenir une colonne 'sms' ou 'message'")
            else:
                df['processed'] = df[col_name].astype(str).apply(preprocess_text)
                X_vec = vectorizer.transform(df['processed'])
                df['prediction'] = model.predict(X_vec)
                df['label'] = df['prediction'].map({0:'Ham',1:'Spam'})

                for _, row in df.iterrows():
                    if row['label']=='Ham':
                        st.markdown(f'<div class="ham-result">✔ Ham — {row[col_name]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="spam-result">❌ SPAM — {row[col_name]}</div>', unsafe_allow_html=True)

                csv_out = df.drop(columns=['processed','prediction']).to_csv(index=False).encode('utf-8')
                st.download_button("Download Predictions CSV", csv_out, "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Erreur CSV: {e}")
