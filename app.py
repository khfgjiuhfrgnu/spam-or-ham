import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import pandas as pd

# -----------------------------
# Config page
# -----------------------------
st.set_page_config(page_title="Détecteur Spam or Ham", page_icon="📩", layout="centered")




# -----------------------------
# Inject CSS
# -----------------------------
def inject_css(path="style.css"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            css = f.read()
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("⚠️ style.css introuvable, using default styles.")

inject_css()

# -----------------------------
# NLTK
# -----------------------------
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# -----------------------------
# Charger modèle + vectorizer
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("tfidf.pkl")
    return model, vectorizer

try:
    model, vectorizer = load_model()
except:
    st.error("❌ Fichiers du modèle introuvables !")
    st.stop()

# -----------------------------
# Nettoyage texte (same as training)
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# -----------------------------
# UI
# -----------------------------
st.title("📩 Détecteur Spam or Ham")
st.caption("Développé par Ahmed | Khaled | Omar")

message = st.text_area("Écris ton message ici :", placeholder="Colle ton SMS ou email...")

col1, col2 = st.columns([1,1])
analyze = col1.button("Analyser")
reset = col2.button("Effacer")

if reset:
    st.experimental_set_query_params()
    st.rerun()

if analyze:
    if not message.strip():
        st.warning("⚠️ Veuillez entrer un message.")
    else:
        cleaned = clean_text(message)
        if cleaned.strip() == "":
            cleaned = "empty"
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        label = "Ham" if pred==0 else "Spam"

        confidence = ""
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(vec)[0][pred]
            confidence = f" — Confiance: {prob:.2%}"

        if pred == 0:
            st.success(f"✔ {label}{confidence}")
        else:
            st.error(f"❌ {label}{confidence}")

# -----------------------------
# Analyse CSV
# -----------------------------
st.subheader("📁 Analyse d’un fichier CSV")
uploaded = st.file_uploader("Importer un fichier CSV", type=["csv"])

def read_csv_safe(file):
    encodings = ["utf-8", "latin1", "iso-8859-1", "cp1252"]
    for enc in encodings:
        try:
            return pd.read_csv(file, encoding=enc)
        except:
            pass
    raise ValueError("⚠️ Type d'encodage non supporté")

if uploaded:
    try:
        df = read_csv_safe(uploaded)
        col = None
        for c in ["sms", "text", "message", "content"]:
            if c in df.columns:
                col = c
                break
        if col is None:
            st.error("⚠️ Le CSV doit contenir une colonne 'sms' ou 'text'.")
        else:
            df["cleaned"] = df[col].astype(str).apply(clean_text)
            df["cleaned"] = df["cleaned"].replace("", "empty")
            X = vectorizer.transform(df["cleaned"])
            df["prediction"] = model.predict(X)
            df["Class"] = df["prediction"].map({0: "Ham", 1: "Spam"})

            st.success("Analyse terminée !")
            st.dataframe(df[[col, "Class"]], use_container_width=True)

            csv_out = df.to_csv(index=False).encode("utf-8")
            st.download_button("Télécharger Résultats", csv_out, "predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Erreur : {e}")
