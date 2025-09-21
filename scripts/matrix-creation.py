# This script creates unsplit matrices for first five models
# !!! These matrices still need to be split for train/test/val !!!
# This can be done with train-test-val.py
import pandas as pd

# Load files, in our case the unsplit 440 RVG matrix was the main matrix for everything
# all_iid.txt file cannot have a header
rare_440 = pd.read_csv('new_rare_variants_burden_matrix.csv')
all_ids = pd.read_csv('all_iids.txt', header=None, names=['IID'])

# Get training 
rare_440_matrix = rare_440[rare_440['IID'].isin(all_ids['IID'])]
rare_440_matrix.to_csv('new_matrix_rare440.csv', index=False)

# Load top genes - for matrix with 44 top gene list
top44_genes = pd.read_csv('44_rvgs_list.txt', header=None, names=['gene'])
top44_cols = ['#FID', 'IID'] + ['gene_' + str(g) for g in top44_genes['gene'].values]

# Keep IDs from 440 RVG and plug in scores 
rare_44_train = rare_440_matrix[top44_cols]
rare_44_train.to_csv('new_matrix_rare44.csv', index=False)

# Load PRS scores - drop #FID if it exists to avoid duplicates
prs = pd.read_csv('new_prs_scaled.csv')
if '#FID' in prs.columns:
    prs = prs.drop('#FID', axis=1)

# 440 RVG + PRS - merge on IID only
rare440_prs = rare_440_matrix.merge(prs, on='IID', how='left')
rare440_prs.to_csv('new_matrix_rare440_prs.csv', index=False)

# 44 RVG + PRS - merge on IID only
rare44_prs = rare_44_train.merge(prs, on='IID', how='left')
rare44_prs.to_csv('new_matrix_rare44_prs.csv', index=False)

# PRS only - keep the FID from 440 RVG training set
prs_only = rare_440_matrix[['#FID', 'IID']].merge(prs, on='IID', how='left')
prs_only.to_csv('new_matrix_prs_only.csv', index=False)

# Sanity check, uncomment to check that all matrices have correct columns
# for file in ['new_matrix_rare440.csv', 'new_matrix_rare44.csv',
#              'new_matrix_prs_only.csv', 'new_matrix_rare440_prs.csv',
#              'new_matrix_rare44_prs.csv']:
#     df = pd.read_csv(file, nrows=0)
#     print(f"{file}: First 3 cols = {list(df.columns[:3])}")