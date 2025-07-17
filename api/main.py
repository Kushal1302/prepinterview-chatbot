from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import boto3
import tempfile
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# ✅ Allow frontend domains (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 👈 You can restrict to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return "Fastapi Server is running on port 8000"

@app.get("/predict")
async def predict_answer(question:str):
    print(question)
    s3 = boto3.client("s3")
    bucket = os.getenv("S3_CHATBOT_BUCKET_NAME")
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