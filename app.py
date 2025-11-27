import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

nltk.download('stopwords', quiet=True)

# 1️⃣ Charger dataset
df = pd.read_csv('spam_ham_dataset.csv  ', encoding='latin-1')
df = df[['label', 'text']].dropna()
df['label'] = df['label'].str.lower()
df = df[df['label'].isin(['ham', 'spam'])]
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 2️⃣ Split
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.20, random_state=42
)

# 3️⃣ Preprocessing
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

X_train_processed = X_train.apply(preprocess_text)
X_test_processed = X_test.apply(preprocess_text)

# 4️⃣ TF-IDF (adapté aux gros datasets)
vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
X_train_vectorized = vectorizer.fit_transform(X_train_processed)
X_test_vectorized = vectorizer.transform(X_test_processed)

# 5️⃣ Logistic Regression optimisé
model = LogisticRegression(random_state=42, solver='saga', max_iter=1000, n_jobs=-1)
model.fit(X_train_vectorized, y_train)

# 6️⃣ Evaluation
y_pred = model.predict(X_test_vectorized)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-Score Macro: {f1_score(y_test, y_pred, average='macro'):.4f}")
print(classification_report(y_test, y_pred))

# 7️⃣ Sauvegarde
joblib.dump(model, 'spam_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("✅ Model and vectorizer saved!")
