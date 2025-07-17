# 📁 lambda/predict/handler.py
import joblib
import pandas as pd
import boto3
import tempfile
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def lambda_handler(event, context):
    question = event["question"]

    s3 = boto3.client("s3")
    bucket = "chatbot-ml-predictor"

    with tempfile.TemporaryDirectory() as tmp:
        s3.download_file(bucket, "model/vectorizer.pkl", f"{tmp}/vectorizer.pkl")
        s3.download_file(bucket, "model/nn_model.pkl", f"{tmp}/nn_model.pkl")
        s3.download_file(bucket, "model/qa_data.csv", f"{tmp}/qa_data.csv")

        vectorizer = joblib.load(f"{tmp}/vectorizer.pkl")
        model = joblib.load(f"{tmp}/nn_model.pkl")
        df = pd.read_csv(f"{tmp}/qa_data.csv")

        vec = vectorizer.transform([question])
        distance, index = model.kneighbors(vec)
        idx = index[0][0]

        return {
            "your_question": question,
            "matched_question": df.iloc[idx]['question'],
            "answer": df.iloc[idx]['answer'],
            "confidence": float(1 - distance[0][0])
        }
