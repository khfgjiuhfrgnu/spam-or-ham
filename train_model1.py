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

# Télécharger ressources
nltk.download("stopwords")
nltk.download("punkt")

# -------------------------
# 1) Charger dataset
# -------------------------
df = pd.read_csv("spam (or) ham.csv", encoding="latin-1")

# garder غير أول جوج كولونات
df = df.iloc[:, :2]
df.columns = ['Class', 'sms']

# 🔥 حذف أي صف فيه NaN:
df = df.dropna(subset=['Class', 'sms'])

# 🔥 تحويل labels ل lower-case (باش نتفادو أخطاء SPAM / Spam)
df['Class'] = df['Class'].str.lower()

# 🔥 الإبقاء فقط على spam / ham
df = df[df['Class'].isin(['spam', 'ham'])]

print("Shape après nettoyage :", df.shape)

# -------------------------
# 2) Nettoyage
# -------------------------
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

df["cleaned"] = df["sms"].apply(clean_text)

# -------------------------
# 3) Vectorizer + encode label
# -------------------------
vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    min_df=2,
    sublinear_tf=True
)

X = vectorizer.fit_transform(df["cleaned"])
y = df["Class"].map({'ham': 0, 'spam': 1})

# -------------------------
# 4) Train test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# 5) Model
# -------------------------
model = LogisticRegression(
    max_iter=800,
    class_weight='balanced',
    C=1.8
)

model.fit(X_train, y_train)

# -------------------------
# 6) Metrics
# -------------------------
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='macro'))

# -------------------------
# 7) Save
# -------------------------
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "tfidf.pkl")

print("✅ Model entraîné et sauvegardé")
