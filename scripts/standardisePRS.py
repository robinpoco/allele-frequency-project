# Use this script to standardise your PRS scores for matrix inclusion

import pandas as pd

# PRS file with same samples as in rare variant list
prs = pd.read_csv('new_prs_training_scores.txt', sep='\t')

# Standardise
mean = prs['PRS_COMMON'].mean()
std = prs['PRS_COMMON'].std()
prs['PRS_scaled'] = (prs['PRS_COMMON'] - mean) / std

# save file to .csv
prs[['#FID', 'IID', 'PRS_scaled']].to_csv('new_prs_training_scaled.csv', index=False)

# Optional sanity check, uncomment if needed 
# print(f"Mean: {mean:.2e}, Std: {std:.2e}")
# print(f"Scaled range: {prs['PRS_scaled'].min():.1f} to {prs['PRS_scaled'].max():.1f}")