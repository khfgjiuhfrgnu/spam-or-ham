import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib

# Télécharger NLTK resources (run once)
nltk.download('stopwords')

# -----------------------------
# 1️⃣ Charger dataset
# -----------------------------
df = pd.read_csv('spam (or) ham.csv', encoding='latin-1')  # <-- dataset ici
df = df.iloc[:, :2]  # garder أول جوج كولونات فقط
df.columns = ['label', 'message']  # renommer les colonnes

# Nettoyer NaN et garder uniquement ham/spam
df = df.dropna(subset=['label', 'message'])
df['label'] = df['label'].str.lower()
df = df[df['label'].isin(['ham', 'spam'])]

# Encoder les labels: 0 = ham, 1 = spam
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# -----------------------------
# 2️⃣ Split train/test
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# -----------------------------
# 3️⃣ Nettoyage + tokenization + stemming
# -----------------------------
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)  # supprimer ponctuation
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words]
    return ' '.join(words)

X_train_processed = X_train.apply(preprocess_text)
X_test_processed = X_test.apply(preprocess_text)

# -----------------------------
# 4️⃣ Vectorizer TF-IDF
# -----------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train_processed)
X_test_vectorized = vectorizer.transform(X_test_processed)

# -----------------------------
# 5️⃣ Modèle Logistic Regression
# -----------------------------
model = LogisticRegression(random_state=42)
model.fit(X_train_vectorized, y_train)

# -----------------------------
# 6️⃣ Évaluation
# -----------------------------
y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')

print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Score Macro: {f1_macro:.4f}")

# -----------------------------
# 7️⃣ Sauvegarde modèle et vectorizer
# -----------------------------
joblib.dump(model, 'spam_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("Model and vectorizer saved!")
