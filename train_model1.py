import pandas as pd

# Q 1 : Lecture du dataset
df = pd.read_csv("spam (or) ham.csv", encoding="latin-1")[['Class','sms']]
df.columns = ['Class','sms']
print(df.head())

# Q 2 : Nettoyage du texte
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)   # حذف الروابط
    text = re.sub(r"[^a-zA-Z]", " ", text)       # حذف الرموز والأرقام
    tokens = [w for w in text.split() if w not in stop_words]
    return " ".join(tokens)

df['cleaned'] = df['sms'].apply(clean_text)

# Q 3 : Tokenisation
from nltk.tokenize import word_tokenize
nltk.download("punkt")

df['tokens'] = df['cleaned'].apply(word_tokenize)
print(df['tokens'].head())

# Q 4 : Stemming
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

df['stemmed'] = df['tokens'].apply(lambda words: [stemmer.stem(w) for w in words])

# Q 5 : Vectorisation
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned'])

# ⚠️ Correction mapping : ham = 0, spam = 1
y = df['Class'].map({'ham':0, 'spam':1})

# Q 6 : Entraînement du modèle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)  # max_iter pour éviter warning
model.fit(X_train, y_train)

# Q 7 : Évaluation
from sklearn.metrics import accuracy_score, f1_score

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-score macro:", f1_score(y_test, y_pred, average='macro'))

# ---- Sauvegarde ----
import joblib

joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "tfidf.pkl")

print("✅ Modèle sauvegardé avec succès!")
