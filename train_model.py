import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')

# Create dataset
np.random.seed(42)
n_samples = 10000
feature_names = [
    'having_IP_Address', 'URL_Length', 'Shortining_Service', 'having_At_Symbol',
    'double_slash_redirecting', 'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State',
    'Domain_registeration_length', 'Favicon', 'Port', 'HTTPS_token', 'Request_URL',
    'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL',
    'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe', 'age_of_domain',
    'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index', 'Links_pointing_to_page',
    'Statistical_report'
]

X = np.random.choice([-1, 0, 1], size=(n_samples, 30), p=[0.4, 0.2, 0.4])
y = np.random.binomial(1, 0.55, n_samples)

# Add realistic correlations
for i in range(n_samples):
    if y[i] == 1:
        X[i, 2] = np.random.choice([-1, 1], p=[0.2, 0.8])
        X[i, 3] = np.random.choice([-1, 1], p=[0.3, 0.7])
        X[i, 5] = np.random.choice([-1, 1], p=[0.2, 0.8])
        X[i, 7] = np.random.choice([-1, 0, 1], p=[0.2, 0.2, 0.6])
        X[i, 16] = np.random.choice([-1, 1], p=[0.3, 0.7])
    else:
        X[i, 2] = np.random.choice([-1, 1], p=[0.95, 0.05])
        X[i, 3] = np.random.choice([-1, 1], p=[0.95, 0.05])
        X[i, 5] = np.random.choice([-1, 1], p=[0.9, 0.1])

df = pd.DataFrame(X, columns=feature_names)
df['Result'] = y
df.to_csv('models/phishing_dataset.csv', index=False)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=200, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
print("=" * 50)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))

# Save SHAP explainer
explainer = shap.TreeExplainer(model)
joblib.dump(model, 'models/phishing_model.pkl')
joblib.dump(explainer, 'models/shap_explainer.pkl')
joblib.dump(feature_names, 'models/feature_names.pkl')
print("\n✅ Model trained and saved in /models folder!")