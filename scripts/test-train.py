# This script is to get train/test split saved to .txt files

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)

# Load phenotype file to ensure stratified split by AUD status
pheno = pd.read_csv('alc_broad_aud_ph_recoded.txt', sep='\t')

# Load unsplit matrix with all IIDs
first_matrix = pd.read_csv('new_matrix_rare440.csv')
merged = first_matrix[['IID']].merge(pheno[['IID', 'broad_aud_ph']], on='IID')
merged = merged.dropna(subset=['broad_aud_ph'])

# Create stratified split of IIDs
# 80% for train+val, 20% for test
train_val_ids, test_ids = train_test_split(
    merged['IID'], 
    test_size=0.2, 
    random_state=42, 
    stratify=merged['broad_aud_ph']
)
# Save IID lists - save these! needed for reproducibility
train_val_ids.to_csv('train_val_ids.txt', index=False, header=False)
test_ids.to_csv('test_ids.txt', index=False, header=False)

# Apply to all matrices
matrix_files = [
    'new_matrix_rare440.csv',
    'new_matrix_rare44.csv',  
    'new_matrix_prs_only.csv',
    'new_matrix_rare440_prs.csv',  
    'new_matrix_rare44_prs.csv'  
]

for file in matrix_files:
    df = pd.read_csv(file)
    # Split based on IIDs
    train_val_df = df[df['IID'].isin(train_val_ids)]
    test_df = df[df['IID'].isin(test_ids)]
    # Save split files by model name
    base_name = file.replace('.csv', '')
    train_val_df.to_csv(f'{base_name}_trainval.csv', index=False)
    test_df.to_csv(f'{base_name}_test.csv', index=False)