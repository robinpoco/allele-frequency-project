# This script can be run on your {model}_test.csv files - or any file that includes you held-out validation set
# You also need: scalers and trained model (.pkl files)

import pandas as pd
import pickle
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import numpy as np

# Input files
modeltype = input("Enter the model to validate (NB! needs to have same prefix as .pkl files): ")
test_filename = input("Enter test filename (e.g., new_matrix_rare440_test.csv): ")

# Load the trained model and scaler
model = pickle.load(open(f'{modeltype}_model.pkl', 'rb'))
scaler = pickle.load(open(f'{modeltype}_scaler.pkl', 'rb'))

# Load test data and phenotype data
test_burden = pd.read_csv(test_filename)
pheno = pd.read_csv('alc_broad_aud_ph_recoded.txt', sep='\t')

# Merge test data with phenotypes and get features and labels
test_data = test_burden.merge(pheno[['IID', 'broad_aud_ph']], on='IID')
test_data = test_data.dropna(subset=['broad_aud_ph'])
test_data['broad_aud_ph'] = test_data['broad_aud_ph'] - 1  # Convert 1/2 to 0/1
X_test = test_data.drop(['#FID', 'IID', 'broad_aud_ph'], axis=1)
y_test = test_data['broad_aud_ph']

# Scale features using TRAINED scaler (not fit, just transform!)
X_test_scaled = scaler.transform(X_test)  # CRITICAL: Only transform, don't refit!

# Make predictions using trained model
y_test_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
y_test_pred_class = model.predict(X_test_scaled)

# Calculate performance metrics
test_auc = roc_auc_score(y_test, y_test_pred_proba)
cm = confusion_matrix(y_test, y_test_pred_class)
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn)  # True positive rate
specificity = tn / (tn + fp)  # True negative rate
ppv = tp / (tp + fp)  # Positive predictive value
npv = tn / (tn + fn)  # Negative predictive value

# For sanity, uncomment to print results to terminal
# print(f"\n=== EXTERNAL VALIDATION RESULTS ===")
# print(f"Model: {modeltype}")
# print(f"Test AUC: {test_auc:.3f}")
# print(f"Sensitivity: {sensitivity:.3f}")
# print(f"Specificity: {specificity:.3f}")
# print(f"PPV: {ppv:.3f}")
# print(f"NPV: {npv:.3f}")

# Save  results
with open(f'{modeltype}_external_validation.txt', 'w') as f:
    f.write(f" RESULTS \n")
    f.write(f"Model: {modeltype}\n")
    f.write(f"Test file: {test_filename}\n")
    f.write(f"Test samples: {len(y_test)}\n")
    f.write(f"Test cases: {sum(y_test)}\n")
    f.write(f"Test controls: {len(y_test) - sum(y_test)}\n\n")
    
    f.write(f"PERFORMANCE METRICS:\n")
    f.write(f"AUC: {test_auc:.3f}\n")
    f.write(f"Sensitivity: {sensitivity:.3f}\n")
    f.write(f"Specificity: {specificity:.3f}\n")
    f.write(f"PPV: {ppv:.3f}\n")
    f.write(f"NPV: {npv:.3f}\n\n")
    
    f.write(f"CONFUSION MATRIX:\n")
    f.write(f"True Negatives: {tn}\n")
    f.write(f"False Positives: {fp}\n")
    f.write(f"False Negatives: {fn}\n")
    f.write(f"True Positives: {tp}\n")
