from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
import numpy as np
import joblib
import os
from datetime import datetime
from feature_extractor import URLFeatureExtractor

app = FastAPI(title="PhishingGuard AI", description="Real-Time Phishing Detection with XAI", version="2.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "models", "phishing_model.pkl"))
explainer = joblib.load(os.path.join(BASE_DIR, "models", "shap_explainer.pkl"))
feature_names = joblib.load(os.path.join(BASE_DIR, "models", "feature_names.pkl"))
extractor = URLFeatureExtractor()

class URLRequest(BaseModel):
    url: str

class FeatureExplanation(BaseModel):
    feature: str
    value: float
    contribution: float
    direction: str
    description: str

class PredictionResponse(BaseModel):
    url: str
    prediction: str
    confidence: float
    probability: float
    features: List[float]
    feature_names: List[str]
    explanations: List[FeatureExplanation]
    shap_values: List[float]
    timestamp: str
    risk_level: str

FEATURE_DESCRIPTIONS = {
    'having_IP_Address': 'URL contains an IP address instead of domain name',
    'URL_Length': 'URL length is unusually long (common in phishing)',
    'Shortining_Service': 'URL uses a link shortening service',
    'having_At_Symbol': 'URL contains @ symbol (can trick users)',
    'double_slash_redirecting': 'URL contains // after protocol (redirect trick)',
    'Prefix_Suffix': 'Domain contains hyphen (often mimics real sites)',
    'having_Sub_Domain': 'Excessive number of subdomains detected',
    'SSLfinal_State': 'SSL certificate is invalid, missing, or untrusted',
    'Domain_registeration_length': 'Domain registered for less than 1 year',
    'Favicon': 'Favicon loaded from external domain',
    'Port': 'Non-standard port being used',
    'HTTPS_token': 'HTTPS keyword found in domain name (deception)',
    'Request_URL': 'Most objects loaded from external domains',
    'URL_of_Anchor': 'Suspicious anchor tags or javascript:void links',
    'Links_in_tags': 'Meta/script/link tags reference external domains',
    'SFH': 'Suspicious form handler (blank or external)',
    'Submitting_to_email': 'Form submits to email address',
    'Abnormal_URL': 'Domain does not match WHOIS records',
    'Redirect': 'Multiple redirects detected',
    'on_mouseover': 'JavaScript changes status bar on mouseover',
    'RightClick': 'Right-click disabled on page',
    'popUpWidnow': 'Popup window with form input detected',
    'Iframe': 'Hidden iframe detected on page',
    'age_of_domain': 'Domain is less than 6 months old',
    'DNSRecord': 'No DNS records found for domain',
    'web_traffic': 'Low or no web traffic data available',
    'Page_Rank': 'Low or no page rank data',
    'Google_Index': 'Domain not indexed by Google',
    'Links_pointing_to_page': 'Very few or no backlinks to this page',
    'Statistical_report': 'Domain flagged in statistical/phishing reports'
}

@app.get("/", response_class=HTMLResponse)
async def root():
    with open(os.path.join(BASE_DIR, "static", "index.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": True, "version": "2.0.0", "timestamp": datetime.now().isoformat()}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: URLRequest):
    try:
        url = request.url
        features = extractor.extract(url)
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)[0]
        probabilities = model.predict_proba(features_array)[0]
        confidence = probabilities[prediction]

        shap_values = explainer.shap_values(features_array)
        shap_for_pred = shap_values[1][0] if prediction == 1 else shap_values[0][0]

        explanations = []
        for feat_name, feat_val, shap_val in zip(feature_names, features, shap_for_pred):
            direction = "Increases phishing risk" if shap_val > 0 else ("Decreases phishing risk" if shap_val < 0 else "Neutral impact")
            explanations.append(FeatureExplanation(
                feature=feat_name, value=float(feat_val), contribution=float(shap_val),
                direction=direction, description=FEATURE_DESCRIPTIONS.get(feat_name, "")
            ))
        explanations.sort(key=lambda x: abs(x.contribution), reverse=True)

        risk = "CRITICAL" if confidence >= 0.85 else ("HIGH" if confidence >= 0.65 else "MEDIUM")
        if prediction == 0:
            risk = "SAFE" if confidence >= 0.85 else ("LOW" if confidence >= 0.65 else "MEDIUM")

        return PredictionResponse(
            url=url, prediction="PHISHING" if prediction == 1 else "LEGITIMATE",
            confidence=float(confidence), probability=float(probabilities[1]),
            features=features, feature_names=feature_names,
            explanations=explanations[:10], shap_values=shap_for_pred.tolist(),
            timestamp=datetime.now().isoformat(), risk_level=risk
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/model/info")
async def model_info():
    return {"algorithm": "Random Forest", "n_estimators": model.n_estimators, "max_depth": model.max_depth,
            "features": len(feature_names), "xai_method": "SHAP", "version": "2.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)