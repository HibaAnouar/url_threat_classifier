from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import re
import time
from urllib.parse import urlparse

app = FastAPI(title="URL Threat Classifier", version="1.0")

model = joblib.load("model/model.joblib")
label_encoder = joblib.load("model/label_encoder.joblib")

def extract_features(url):
    parsed = urlparse(url)
    url_length = len(url)
    hostname_length = len(parsed.netloc)
    path_length = len(parsed.path)
    num_dots = url.count('.')
    num_hyphens = url.count('-')
    num_underscores = url.count('_')
    num_slashes = url.count('/')
    num_at = url.count('@')
    num_question = url.count('?')
    num_equals = url.count('=')
    num_ampersand = url.count('&')
    num_percent = url.count('%')
    num_digits = sum(c.isdigit() for c in url)
    has_ip = 1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url) else 0
    has_https = 1 if url.startswith('https') else 0
    subdomain_count = max(0, len(parsed.netloc.split('.')) - 2)
    def entropy(s):
        if len(s) == 0: return 0
        probs = [s.count(c)/len(s) for c in set(s)]
        return -sum(p * np.log2(p) for p in probs)
    url_entropy = entropy(url)
    suspicious_words = ['login','signin','account','secure','update',
                        'verify','confirm','banking','paypal','free',
                        'click','download','install','exe','zip']
    has_suspicious = 1 if any(w in url.lower() for w in suspicious_words) else 0
    return [url_length, hostname_length, path_length,
            num_dots, num_hyphens, num_underscores, num_slashes,
            num_at, num_question, num_equals, num_ampersand, num_percent,
            num_digits, has_ip, has_https, subdomain_count,
            url_entropy, has_suspicious]

class URLRequest(BaseModel):
    url: str

class PredictionResponse(BaseModel):
    url: str
    prediction: str
    confidence: float
    latency_ms: float

@app.get("/")
def root():
    return {"message": "URL Threat Classifier API", "status": "running"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: URLRequest):
    start = time.perf_counter()
    features = np.array([extract_features(request.url)])
    pred_idx = model.predict(features)[0]
    pred_proba = model.predict_proba(features)[0]
    latency_ms = (time.perf_counter() - start) * 1000
    return PredictionResponse(
        url=request.url,
        prediction=label_encoder.classes_[pred_idx],
        confidence=round(float(pred_proba.max()), 4),
        latency_ms=round(latency_ms, 2)
    )