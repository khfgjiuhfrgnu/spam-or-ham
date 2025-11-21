import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
import pandas as pd

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Détecteur Spam or Ham", page_icon="📩", layout="centered")

# -----------------------------
# Inject CSS (sécurisé)
# -----------------------------
def inject_css(path="style.css"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        # Optionnel: thème minimal si le fichier n’existe pas
        st.markdown("""
        <style>
        body {background: #0f172a; color: #e5e7eb;}
        .stTextArea textarea {background:#111827; color:#e5e7eb; border-radius:10px;}
        .stButton button {background:#3b82f6; color:white; border-radius:8px;}
        </style>
        """, unsafe_allow_html=True)

inject_css()

# -----------------------------
# NLTK stopwords
# -----------------------------
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

# -----------------------------
# Charger modèle + vectorizer
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("tfidf.pkl")
    return model, vectorizer

try:
    model, vectorizer = load_model()
except Exception:
    st.error("❌ Les fichiers du modèle sont introuvables. Lance d'abord train_model.py")
    st.stop()

# -----------------------------
# Nettoyage texte
# -----------------------------
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    tokens = [w for w in text.split() if w not in stop_words]
    return " ".join(tokens)

# -----------------------------
# Header
# -----------------------------
st.title("Réalisé par Ahmed | Khaled | Omar")
st.title("📩 Détecteur de Spam or Ham")

# -----------------------------
# UI principal
# -----------------------------
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
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        label = "✔ Ham" if pred == 0 else "❌ SPAM"

        # Optionnel: afficher la confiance si disponible
        confiance = ""
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(vec)[0]
            score = proba[pred]
            confiance = f" — Confiance: {score:.2%}"

        if pred == 0:
            st.success(f"Résultat : {label}{confiance}")
        else:
            st.error(f"Résultat : {label}{confiance}")

# -----------------------------
# Prédiction CSV
# -----------------------------
st.subheader("Prédiction sur un fichier CSV")
uploaded = st.file_uploader("Importer un fichier CSV ", type=["csv"])

def read_csv_safely(file) -> pd.DataFrame:
    encodings = ["utf-8", "latin1", "iso-8859-1", "cp1252"]
    for enc in encodings:
        try:
            return pd.read_csv(file, encoding=enc)
        except Exception:
            continue
    raise ValueError("⚠️ Impossible de lire le fichier CSV — encodage non supporté.")

if uploaded:
    try:
        df = read_csv_safely(uploaded)

        # Vérifier la colonne
        target_col = None
        for candidate in ["sms", "message", "text"]:
            if candidate in df.columns:
                target_col = candidate
                break

        if target_col is None:
            st.error("⚠️ Le CSV doit contenir une colonne 'sms' (ou 'message' / 'text').")
        else:
            df["cleaned"] = df[target_col].astype(str).apply(clean_text)
            X = vectorizer.transform(df["cleaned"])
            df["prediction"] = model.predict(X)

            # CORRECTION: mapping cohérent avec l'entraînement (ham=0, spam=1)
            df["class"] = df["prediction"].map({0: "Ham", 1: "Spam"})

            st.success("Analyse terminée !")
            st.dataframe(df[[target_col, "class"]], use_container_width=True)

            csv_out = df.to_csv(index=False).encode("utf-8")
            st.download_button("Télécharger Résultats", csv_out, "predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"Erreur: {e}")
