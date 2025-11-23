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
st.title("\nréalisé par  khaled | Omar  | Ahmed")

st.title("📩 Détecteur Spam ou Ham")
st.write("Entrez un message ou uploadez un CSV pour vérifier s'il est spam ou ham.")

# -----------------------------
# Individual message prediction
# -----------------------------
user_input = st.text_area("Message:")

# زر Predict بكلاس خاص
if st.markdown('<button class="predict-btn">Predict</button>', unsafe_allow_html=True):
    if user_input.strip():
        processed_text = preprocess_text(user_input)
        X_new = vectorizer.transform([processed_text])

        prediction = model.predict(X_new)[0]
        confidence = model.predict_proba(X_new).max() * 100   # ← إضافة الثقة

        if prediction == 0:
            st.markdown(
                f'<div class="ham-result">✔ Ham — Confiance: {confidence:.2f}%</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="spam-result">❌ SPAM — Confiance: {confidence:.2f}%</div>',
                unsafe_allow_html=True
            )
    else:
        st.warning("⚠️ Please enter a message.")

# -----------------------------
# CSV batch prediction
# -----------------------------
st.subheader("Upload a CSV ")
uploaded_file = st.file_uploader("", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        if 'sms' not in df.columns and 'message' not in df.columns:
            st.error("CSV must contain a column named 'sms' or 'message'.")
        else:
            col_name = 'sms' if 'sms' in df.columns else 'message'

            df['processed'] = df[col_name].astype(str).apply(preprocess_text)
            X_vec = vectorizer.transform(df['processed'])

            df['prediction'] = model.predict(X_vec)
            df['confidence'] = model.predict_proba(X_vec).max(axis=1) * 100  # ← إضافة الثقة
            df['label'] = df['prediction'].map({0: 'Ham', 1: 'Spam'})

            st.success("upload completed!")

            for _, row in df.iterrows():
                if row['label'] == 'Ham':
                    st.markdown(
                        f'<div class="ham-result">✔ Ham — {row[col_name]} — Confiance: {row["confidence"]:.2f}%</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="spam-result">❌ SPAM — {row[col_name]} — Confiance: {row["confidence"]:.2f}%</div>',
                        unsafe_allow_html=True
                    )

            # Download button
            csv_out = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", csv_out, "predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
