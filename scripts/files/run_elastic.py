# This script is to run elastic net regression on matrices created for AUD risk prediction
# When running this ensure input files include your TRAINING data (*{model}_train.csv)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import numpy as np
import pickle

# Load data
filename = input("Enter filename: ")
modeltype = input("Enter your model type (e.g., rare440_elastic): ")
burden = pd.read_csv(filename)
pheno = pd.read_csv('alc_broad_aud_ph_recoded.txt', sep='\t')

# Merge IDs and remove missing phenos
data = burden.merge(pheno[['IID', 'broad_aud_ph']], on='IID')
data = data.dropna(subset=['broad_aud_ph'])
data['broad_aud_ph'] = data['broad_aud_ph'] - 1

# Features and labels
X = data.drop(['#FID', 'IID', 'broad_aud_ph'], axis=1)
y = data['broad_aud_ph']

# Prinst feature info
print(f"Total samples: {len(y)}")
print(f"Total features: {X.shape[1]}")
print(f"Cases: {sum(y)} ({100*sum(y)/len(y):.1f}%)")
print(f"Controls: {len(y)-sum(y)} ({100*(len(y)-sum(y))/len(y):.1f}%)")

# Create training and internal test split before scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Sanity check to print split info, uncomment if needed
# print(f"Training: {len(y_train)} samples ({100*sum(y_train)/len(y_train):.1f}% cases)")
# print(f"Testing: {len(y_test)} samples ({100*sum(y_test)/len(y_test):.1f}% cases)")

# Scale features on training data X_train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  

# Train model (also handles class imbalance)
model = LogisticRegression(
    penalty='elasticnet', 
    solver='saga', 
    l1_ratio=0.5, 
    max_iter=1000,
    random_state=42,
    class_weight='balanced'  
)
model.fit(X_train_scaled, y_train)

# Evaluate on train and internal test
y_train_pred = model.predict_proba(X_train_scaled)[:, 1]
y_test_pred = model.predict_proba(X_test_scaled)[:, 1]

train_auc = roc_auc_score(y_train, y_train_pred)
test_auc = roc_auc_score(y_test, y_test_pred)

# Gets coefficients
coefs = pd.DataFrame({
    'gene': X.columns,
    'coefficient': model.coef_[0]
})
coefs = coefs.sort_values('coefficient', ascending=False)

# Find non-zero coefficients (selected by elastic net)
selected_features = coefs[coefs['coefficient'] != 0]

# Saves coeff outputs
coefs.to_csv(f'{modeltype}_coefficients.csv', index=False)

# Save model performance in .txt file
with open(f'{modeltype}_performance.txt', 'w') as f:
    f.write(f"{modeltype} TRAINING PERFORMANCE\n")
    f.write(f"Samples: {len(y_train)}\n")
    f.write(f"Cases: {sum(y_train)}\n")
    f.write(f"Controls: {len(y_train) - sum(y_train)}\n")
    f.write(f"AUC: {train_auc:.3f}\n\n")
    
    f.write(f"{modeltype} INTERNAL TEST PERFORMANCE\n")
    f.write(f"Samples: {len(y_test)}\n")
    f.write(f"Cases: {sum(y_test)}\n")
    f.write(f"Controls: {len(y_test) - sum(y_test)}\n")
    f.write(f"AUC: {test_auc:.3f}\n\n")
    
    f.write(f"Overfit gap: {train_auc - test_auc:.3f}\n")
    f.write(f"Features selected: {len(selected_features)} / {len(coefs)}\n")

# Save model and scaler for validation run with validation-test.py
pickle.dump(model, open(f'{modeltype}_model.pkl', 'wb'))
pickle.dump(scaler, open(f'{modeltype}_scaler.pkl', 'wb'))
