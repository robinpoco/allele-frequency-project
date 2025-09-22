# This script creates genome-wide scores for RVGs and creates new matrices (RVGW only and RVGW + PRS)
# using the unsplit matrix of 440 RVG as input file.

import pandas as pd
import numpy as np

# Set random seed for reproducibility, can also use the train/test IIDs if looking for specific indiv
np.random.seed(42)

# Load existing main matrix (this was the 440 RVG matrix in our case) you want to calculate genome-wide score for
rare_440 = pd.read_csv('new_matrix_rare440.csv')

# Get gene columns (exclude IID and #FID)
id_cols = ['#FID', 'IID']
gene_cols = [col for col in rare_440.columns if col not in id_cols]

# Calculate genome-wide rare variant score (sum across all 440 genes)
rare_440['rare_genome_wide'] = rare_440[gene_cols].sum(axis=1)

# Sanity check, uncomment to check score stats
# print(f"  Mean: {rare_440['rare_genome_wide'].mean():.2f}")
# print(f"  SD: {rare_440['rare_genome_wide'].std():.2f}")
# print(f"  Min: {rare_440['rare_genome_wide'].min()}")
# print(f"  Max: {rare_440['rare_genome_wide'].max()}")

# Create genome-wide only matrix (RVGW only)
rare_gw_only = rare_440[['#FID', 'IID', 'rare_genome_wide']].copy()
rare_gw_only.to_csv('new_matrix_rare_gw.csv', index=False)

# Load PRS for combined model
prs = pd.read_csv('new_prs_scaled.csv')
if '#FID' in prs.columns:
    prs = prs.drop('#FID', axis=1)

# Create genome-wide + PRS matrix (RVGW + PRS)
rare_gw_prs = rare_gw_only.merge(prs, on='IID', how='left')
rare_gw_prs.to_csv('new_matrix_rare_gw_prs.csv', index=False)

# Create training, test, validation splits, assumes you have participant IIDs saved seperately
# Load saved train/val and test IIDs from your previous split
train_val_ids = pd.read_csv('train_val_ids.txt', header=None, names=['IID'])
test_ids = pd.read_csv('test_ids.txt', header=None, names=['IID'])

# Add the new matrices to the list
matrix_files = [
    'new_matrix_rare_gw.csv',
    'new_matrix_rare_gw_prs.csv'
]

for file in matrix_files:
    df = pd.read_csv(file)
    # Split based on IIDs
    train_val_df = df[df['IID'].isin(train_val_ids['IID'])]
    test_df = df[df['IID'].isin(test_ids['IID'])]
    
    # Save split files and uses model name
    base_name = file.replace('.csv', '')
    train_val_df.to_csv(f'{base_name}_trainval.csv', index=False)
    test_df.to_csv(f'{base_name}_test.csv', index=False)

