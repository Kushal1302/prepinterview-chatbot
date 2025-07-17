# importing necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib

df = pd.read_csv("dataset/data.csv")
df.dropna()


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["question"]) # convert texts to numerical vectors eg. [0.5,0.3]

model = NearestNeighbors(n_neighbors=1,metric="cosine")
model.fit(X)

joblib.dump(vectorizer,"model/vectorizer.pkl")
joblib.dump(model,"model/nn_model.pkl")
df.to_csv("model/qa_data.csv",index=False)

