import pandas as pd
import numpy as np
import torch
from scipy.spatial.distance import cosine, euclidean, cityblock
from collections import defaultdict
import ast

# ==================== Load Data ====================

# Base directory
base_path = ""

# Load cancer type thresholds
thresholds_df = pd.read_csv(base_path + 'cancer_type_thresholds.csv')

# Load CUIs for each cancer type from cancer_types.csv
cancer_types_df = pd.read_csv(base_path + 'cancer_types.csv')

# Store CUIs in multiple cancer types
cancer_type_map = defaultdict(list)

for _, row in cancer_types_df.iterrows():
    cancer_class = row['CANCER_CLASS']
    cuis = row['SUBJECT_CUI'].strip('[]').split(', ')
    for cui in cuis:
        if cancer_class not in cancer_type_map[cui]:  # Prevent duplicate cancer class entries
            cancer_type_map[cui].append(cancer_class)

# Debug: Check total unique CUIs
print(f"Total unique CUIs: {len(cancer_type_map)}")

# Load original embeddings using PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
original_numpy = np.load(base_path + "original_node_embeddings.npy")
original_embeddings = torch.tensor(original_numpy).to(device)

denoised_numpy = np.load(base_path + "denoised_node_embeddings.npy")
denoised_embeddings = torch.tensor(denoised_numpy).to(device)

original_embedding_dict = {cui: original_embeddings[i] for i, cui in enumerate(cancer_type_map.keys())}
denoised_embedding_dict = {cui: denoised_embeddings[i] for i, cui in enumerate(cancer_type_map.keys())}

cancer_cui_dict = defaultdict(list)
for cui, cancer_types in cancer_type_map.items():
    for cancer in cancer_types:
        cancer_cui_dict[cancer].append(cui)

print(f"Total unique cancer types: {len(cancer_cui_dict)}")

# ==================== Cross-Cancer Comparison ====================

results = []

for cancer_class, original_cuis in cancer_cui_dict.items():
    if cancer_class not in thresholds_df['CANCER_CLASS'].values:
        print(f"Warning: No threshold found for {cancer_class}. Skipping.")
        continue  

    threshold_row = thresholds_df[thresholds_df['CANCER_CLASS'] == cancer_class].iloc[0]
    cosine_thresh = threshold_row['mean_cosine_similarity']
    euclidean_thresh = threshold_row['mean_euclidean_distance']
    manhattan_thresh = threshold_row['mean_manhattan_distance']

    other_cancer_cuis = [cui for other_cancer, cuis in cancer_cui_dict.items() if other_cancer != cancer_class for cui in cuis]

    for subject_cui in original_cuis:
        if subject_cui not in original_embedding_dict:
            continue
        original_vec = original_embedding_dict[subject_cui]

        passing_cuis = []
        passing_cancer_types = []

        for denoised_cui in other_cancer_cuis:
            if denoised_cui not in denoised_embedding_dict:
                continue
            denoised_vec = denoised_embedding_dict[denoised_cui]

            den_np, org_np = denoised_vec.cpu().numpy(), original_vec.cpu().numpy()
            cosine_sim = 1 - cosine(den_np, org_np)
            euclidean_dist = euclidean(den_np, org_np)
            manhattan_dist = cityblock(den_np, org_np)

            if (cosine_sim >= cosine_thresh and euclidean_dist <= euclidean_thresh and manhattan_dist <= manhattan_thresh):
                passing_cuis.append(denoised_cui)
                if denoised_cui in cancer_type_map:
                    passing_cancer_types.append(cancer_type_map[denoised_cui])

        results.append({
            'cancer_class': cancer_class,
            'subject_cui': subject_cui,
            'pass_all_3': list(set(passing_cuis)) if passing_cuis else None,
            'pass_all_3_cancer_types': list(set(tuple(sub) if isinstance(sub, list) else sub for sub in passing_cancer_types)) if passing_cancer_types else None
        })

# ==================== Load Cancer Dataset and Add OBJECT_CUI ====================

cancer_dataset_df = pd.read_csv(base_path + 'cancer_dataset.csv')
subject_to_object_map = dict(zip(cancer_dataset_df['SUBJECT_CUI'], cancer_dataset_df['OBJECT_CUI']))

df_results = pd.DataFrame(results)

df_results['object_cui'] = df_results['subject_cui'].map(subject_to_object_map)

df_results['pass_all_3_count'] = df_results['pass_all_3'].apply(lambda x: len(x) if isinstance(x, list) else 0)

df_results.to_csv(base_path + 'cross_cancer_comparisons.csv', index=False)

print(f"Total processed cancer types: {df_results['cancer_class'].nunique()}")
print(f"Total rows in results: {len(df_results)}")
print(f"Total successful comparisons: {df_results['pass_all_3_count'].sum()}")
print(df_results.head())

