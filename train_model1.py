import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib

# -----------------------------
# Q1 : Lecture du dataset
# -----------------------------
df = pd.read_csv("spam (or) ham.csv", encoding="latin-1")[['Class','sms']]
df.columns = ['Class','sms']

# -----------------------------
# Q2 : Nettoyage du texte
# -----------------------------
nltk.download("stopwords")
nltk.download("punkt")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)   # supprimer les URLs
    text = re.sub(r"[^a-zA-Z]", " ", text)       # supprimer chiffres et ponctuation
    tokens = [w for w in text.split() if w not in stop_words]
    return " ".join(tokens)

df['cleaned'] = df['sms'].apply(clean_text)

# -----------------------------
# Q3 : Tokenisation + Stemming
# -----------------------------
stemmer = PorterStemmer()
df['tokens'] = df['cleaned'].apply(word_tokenize)
df['stemmed'] = df['tokens'].apply(lambda words: [stemmer.stem(w) for w in words])

# -----------------------------
# Q4 : Vectorisation
# -----------------------------
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2)  
# ngram_range=(1,2) → prend unigrams et bigrams
# min_df=2 → ignore mots trop rares
X = vectorizer.fit_transform(df['cleaned'])

# -----------------------------
# Q5 : Mapping des classes
# -----------------------------
y = df['Class'].map({'ham':0, 'spam':1})

# -----------------------------
# Q6 : Entraînement du modèle
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=500, class_weight='balanced')  
# class_weight='balanced' → corrige déséquilibre spam/ham
model.fit(X_train, y_train)

# -----------------------------
# Q7 : Évaluation
# -----------------------------
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-score macro:", f1_score(y_test, y_pred, average='macro'))

# -----------------------------
# Sauvegarde
# -----------------------------
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "tfidf.pkl")

print("✅ Modèle sauvegardé avec succès!")

