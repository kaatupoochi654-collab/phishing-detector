from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List
import numpy as np
import joblib
import os
import sys
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PhishingGuard AI", description="Real-Time Phishing Detection with XAI", version="2.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# === ROBUST PATH SETUP ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logger.info(f"BASE_DIR: {BASE_DIR}")

# List what's actually on disk
logger.info(f"Root files: {os.listdir(BASE_DIR)}")

# Ensure models folder exists
MODELS_DIR = os.path.join(BASE_DIR, "models")
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR, exist_ok=True)
    logger.info("Created models directory")

# If models don't exist, train them now (fallback)
MODEL_PATH = os.path.join(MODELS_DIR, "phishing_model.pkl")
EXPLAINER_PATH = os.path.join(MODELS_DIR, "shap_explainer.pkl")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.pkl")

if not os.path.exists(MODEL_PATH):
    logger.warning("Model not found! Running train_model.py...")
    import subprocess
    result = subprocess.run([sys.executable, "train_model.py"], capture_output=True, text=True)
    logger.info(result.stdout)
    if result.returncode != 0:
        logger.error(result.stderr)

# Load model
try:
    model = joblib.load(MODEL_PATH)
    explainer = joblib.load(EXPLAINER_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None
    explainer = None
    feature_names = []

# Load feature extractor
try:
    from feature_extractor import URLFeatureExtractor
    extractor = URLFeatureExtractor()
except Exception as e:
    logger.error(f"Failed to load extractor: {e}")
    extractor = None

# === INLINE HTML FALLBACK ===
INLINE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PhishingGuard AI</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,sans-serif;background:linear-gradient(135deg,#0f172a 0%,#1e293b 100%);color:#e2e8f0;min-height:100vh}
.container{max-width:900px;margin:0 auto;padding:40px 20px}
header{text-align:center;margin-bottom:40px}
header h1{font-size:2.5rem;background:linear-gradient(90deg,#38bdf8,#818cf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:10px}
header p{color:#94a3b8;font-size:1.1rem}
.card{background:rgba(30,41,59,0.7);backdrop-filter:blur(10px);border:1px solid rgba(148,163,184,0.1);border-radius:16px;padding:30px;margin-bottom:24px;box-shadow:0 10px 40px rgba(0,0,0,0.3)}
.input-group{display:flex;gap:12px;margin-bottom:20px}
input[type="url"]{flex:1;padding:14px 20px;border-radius:12px;border:1px solid #334155;background:#0f172a;color:#fff;font-size:1rem;outline:none}
input[type="url"]:focus{border-color:#38bdf8}
button{padding:14px 32px;border-radius:12px;border:none;background:linear-gradient(90deg,#38bdf8,#818cf8);color:#0f172a;font-weight:700;font-size:1rem;cursor:pointer}
button:hover{transform:translateY(-2px)}
button:disabled{opacity:0.6;cursor:not-allowed}
.result{display:none;animation:fadeIn 0.5s ease}
@keyframes fadeIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.badge{display:inline-block;padding:8px 20px;border-radius:999px;font-weight:700;font-size:0.9rem;text-transform:uppercase}
.badge-phishing{background:rgba(239,68,68,0.15);color:#ef4444;border:1px solid rgba(239,68,68,0.3)}
.badge-safe{background:rgba(34,197,94,0.15);color:#22c55e;border:1px solid rgba(34,197,94,0.3)}
.confidence-bar{height:8px;background:#334155;border-radius:4px;overflow:hidden;margin:12px 0}
.confidence-fill{height:100%;border-radius:4px;transition:width 0.8s ease}
.explanation-list{list-style:none;margin-top:20px}
.explanation-item{display:flex;align-items:center;justify-content:space-between;padding:14px 16px;margin-bottom:10px;background:rgba(15,23,42,0.5);border-radius:10px;border-left:4px solid}
.explanation-item.positive{border-left-color:#ef4444}
.explanation-item.negative{border-left-color:#22c55e}
.explanation-item.neutral{border-left-color:#94a3b8}
.feat-name{font-weight:600;color:#f1f5f9}
.feat-desc{font-size:0.85rem;color:#94a3b8;margin-top:4px}
.feat-value{font-family:monospace;font-size:0.9rem;color:#cbd5e1}
.shap-bar{width:100px;height:6px;background:#334155;border-radius:3px;overflow:hidden;margin-top:6px}
.shap-fill{height:100%;border-radius:3px}
.stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin-top:20px}
.stat-box{text-align:center;padding:20px;background:rgba(15,23,42,0.5);border-radius:12px}
.stat-box h3{font-size:1.8rem;color:#38bdf8;margin-bottom:4px}
.stat-box p{color:#94a3b8;font-size:0.9rem}
.loading{display:none;text-align:center;padding:20px}
.spinner{width:40px;height:40px;border:3px solid #334155;border-top-color:#38bdf8;border-radius:50%;animation:spin 1s linear infinite;margin:0 auto 10px}
@keyframes spin{to{transform:rotate(360deg)}}
.error{background:rgba(239,68,68,0.1);color:#ef4444;padding:14px;border-radius:10px;margin-top:12px;display:none}
.status-box{background:rgba(56,189,248,0.1);border:1px solid rgba(56,189,248,0.3);color:#38bdf8;padding:16px;border-radius:12px;margin-bottom:20px;text-align:center}
</style>
</head>
<body>
<div class="container">
<header>
<h1>PhishingGuard AI</h1>
<p>Real-Time Phishing Detection with Explainable AI</p>
</header>
<div class="card">
<div class="input-group">
<input type="url" id="urlInput" placeholder="Enter URL to analyze (e.g., https://example.com)" value="https://secure-bank-login.tk/verify">
<button id="analyzeBtn" onclick="analyze()">Analyze URL</button>
</div>
<div class="loading" id="loading">
<div class="spinner"></div>
<p>Extracting features & computing SHAP values...</p>
</div>
<div class="error" id="error"></div>
</div>
<div class="result card" id="result">
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;">
<div>
<span class="badge" id="predictionBadge">PHISHING</span>
<span class="badge" id="riskBadge" style="margin-left:8px;">CRITICAL</span>
</div>
<div style="text-align:right;">
<div style="font-size:0.85rem;color:#94a3b8;">Confidence</div>
<div style="font-size:1.5rem;font-weight:700;" id="confidenceText">94.2%</div>
</div>
</div>
<div class="confidence-bar"><div class="confidence-fill" id="confidenceBar"></div></div>
<div class="stats-grid">
<div class="stat-box"><h3 id="phishingProb">0%</h3><p>Phishing Probability</p></div>
<div class="stat-box"><h3 id="featCount">30</h3><p>Features Analyzed</p></div>
<div class="stat-box"><h3 id="topThreat">SSL</h3><p>Top Risk Factor</p></div>
</div>
<h3 style="margin:28px 0 12px;font-size:1.2rem;">XAI Explanation (SHAP)</h3>
<p style="color:#94a3b8;margin-bottom:16px;font-size:0.9rem;">Top features contributing to this prediction.</p>
<ul class="explanation-list" id="explanationList"></ul>
</div>
</div>
<script>
async function analyze(){
const url=document.getElementById('urlInput').value;
if(!url)return alert('Please enter a URL');
document.getElementById('loading').style.display='block';
document.getElementById('result').style.display='none';
document.getElementById('error').style.display='none';
document.getElementById('analyzeBtn').disabled=true;
try{
const res=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url})});
const data=await res.json();
if(!res.ok)throw new Error(data.detail||'Analysis failed');
render(data);
}catch(e){
document.getElementById('error').textContent=e.message;
document.getElementById('error').style.display='block';
}finally{
document.getElementById('loading').style.display='none';
document.getElementById('analyzeBtn').disabled=false;
}
}
function render(data){
const isPhishing=data.prediction==='PHISHING';
const badge=document.getElementById('predictionBadge');
badge.textContent=data.prediction;
badge.className='badge '+(isPhishing?'badge-phishing':'badge-safe');
const riskBadge=document.getElementById('riskBadge');
riskBadge.textContent=data.risk_level;
riskBadge.style=isPhishing?'margin-left:8px;background:rgba(239,68,68,0.15);color:#ef4444;border:1px solid rgba(239,68,68,0.3);padding:8px 20px;border-radius:999px;font-weight:700;font-size:0.9rem;text-transform:uppercase;':'margin-left:8px;background:rgba(34,197,94,0.15);color:#22c55e;border:1px solid rgba(34,197,94,0.3);padding:8px 20px;border-radius:999px;font-weight:700;font-size:0.9rem;text-transform:uppercase;';
document.getElementById('confidenceText').textContent=(data.confidence*100).toFixed(1)+'%';
document.getElementById('confidenceBar').style.width=(data.confidence*100)+'%';
document.getElementById('confidenceBar').style.background=isPhishing?'#ef4444':'#22c55e';
document.getElementById('phishingProb').textContent=(data.probability*100).toFixed(1)+'%';
document.getElementById('featCount').textContent=data.features.length;
document.getElementById('topThreat').textContent=data.explanations[0].feature.replace(/_/g,' ');
const list=document.getElementById('explanationList');
list.innerHTML='';
data.explanations.slice(0,8).forEach(exp=>{
const li=document.createElement('li');
li.className='explanation-item '+(exp.contribution>0?'positive':exp.contribution<0?'negative':'neutral');
const barWidth=Math.min(Math.abs(exp.contribution)*500,100);
const barColor=exp.contribution>0?'#ef4444':'#22c55e';
li.innerHTML=`<div style="flex:1;"><div class="feat-name">${exp.feature.replace(/_/g,' ')}</div><div class="feat-desc">${exp.description}</div><div class="shap-bar"><div class="shap-fill" style="width:${barWidth}%;background:${barColor};"></div></div></div><div style="text-align:right;margin-left:16px;"><div class="feat-value">${exp.value>0?'+':''}${exp.value.toFixed(0)}</div><div style="font-size:0.8rem;color:${barColor};margin-top:4px;">${exp.direction}</div></div>`;
list.appendChild(li);
});
document.getElementById('result').style.display='block';
}
</script>
</body>
</html>"""

# === API MODELS ===
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

# === ENDPOINTS ===
@app.get("/", response_class=HTMLResponse)
async def root():
    # Try to read from file first, fallback to inline HTML
    html_path = os.path.join(BASE_DIR, "static", "index.html")
    if os.path.exists(html_path):
        try:
            with open(html_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading index.html: {e}")
    logger.info("Serving inline HTML fallback")
    return INLINE_HTML

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "extractor_loaded": extractor is not None,
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: URLRequest):
    if model is None or extractor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")
    
    try:
        url = request.url
        logger.info(f"Analyzing URL: {url}")
        
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
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/model/info")
async def model_info():
    return {
        "algorithm": "Random Forest",
        "n_estimators": getattr(model, 'n_estimators', 0) if model else 0,
        "features": len(feature_names) if feature_names else 0,
        "xai_method": "SHAP",
        "version": "2.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
