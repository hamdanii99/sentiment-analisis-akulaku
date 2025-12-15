import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("dataset_akulaku_balanced.csv")

X = df["text"].astype(str)
y = df["label"]

tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(X)

model = MultinomialNB()
model.fit(X_tfidf, y)

joblib.dump(tfidf, "tfidf.pkl")
joblib.dump(model, "model_nb.pkl")

print("MODEL BERHASIL DISIMPAN")
