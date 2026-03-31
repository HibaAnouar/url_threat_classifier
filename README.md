# URL Threat Classifier

Classifies URLs as benign, phishing, malware, or defacement using
18 hand-crafted security features + Logistic Regression.

## Results
- Validation Macro F1: 0.70
- Inference latency: ~2ms per URL

## Run the API
pip install -r requirements.txt
uvicorn app.app:app --reload

## Dataset
Malicious URLs Dataset (Kaggle, CC0) — 651,191 URLs, 4 classes.
