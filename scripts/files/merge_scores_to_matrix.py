# This script creates a matrix from PLINK scored burden score files (440 genes x 150k participants)
# This file does not include any splitting of individuals - use train-test.py for that

import pandas as pd
import glob

# Get individual IIDs
first_file = glob.glob('new_score_rare_gene_*_chr*.sscore')[0]
df = pd.read_csv(first_file, sep='\t')[['#FID', 'IID']]

# Put scores in columns
for file in glob.glob('new_score_rare_gene_*_chr*.sscore'):
    gene_id = file.split('gene_')[1].split('_chr')[0]
    
    # Reads into columns
    df_temp = pd.read_csv(file, sep='\t')[['IID', 'SCORE1_AVG']]
    df_temp.rename(columns={'SCORE1_AVG': f'gene_{gene_id}'}, inplace=True)
    
    # Merges everything
    df = df.merge(df_temp, on='IID', how='left')

# Saves to csv - !!! important !!! this matrix was used as main matrix to add/substract features to calculate genome-wide burden scores, split into train,test,val
df.to_csv('burden_matrix.csv', index=False) 
