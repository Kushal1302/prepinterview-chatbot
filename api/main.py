from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import joblib
import pandas as pd
import boto3
import tempfile
from dotenv import load_dotenv
import os

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Loading models from S3...")

    s3 = boto3.client("s3")
    bucket = os.getenv("S3_CHATBOT_BUCKET_NAME")
    versionFolder = os.getenv("MODEL_VERSION_FOLDER")

    with tempfile.TemporaryDirectory() as tmp:
        s3.download_file(bucket, f"model/{versionFolder}/vectorizer.pkl", f"{tmp}/vectorizer.pkl")
        s3.download_file(bucket, f"model/{versionFolder}/nn_model.pkl", f"{tmp}/nn_model.pkl")
        s3.download_file(bucket, f"model/{versionFolder}/qa_data.csv", f"{tmp}/qa_data.csv")

        app.state.vectorizer = joblib.load(f"{tmp}/vectorizer.pkl")
        app.state.model = joblib.load(f"{tmp}/nn_model.pkl")
        app.state.qa_data = pd.read_csv(f"{tmp}/qa_data.csv")

    print("✅ Models loaded successfully")
    yield
    print("🧹 Clean-up if needed")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return "FastAPI Server is running on port 8000"

@app.get("/predict")
async def predict_answer(question: str, request: Request):
    vectorizer = request.app.state.vectorizer
    model = request.app.state.model
    df = request.app.state.qa_data

    vec = vectorizer.transform([question])
    distance, index = model.kneighbors(vec)
    idx = index[0][0]

    return {
        "your_question": question,
        "matched_question": df.iloc[idx]['question'],
        "answer": df.iloc[idx]['answer'],
        "confidence": float(1 - distance[0][0])
    }
