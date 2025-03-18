import pandas as pd
import numpy as np
import torch
from scipy.spatial.distance import cosine, euclidean, cityblock

# Load cancer categories with CUIs
df = pd.read_csv('cancer_types.csv')  # Assumes CANCER_CLASS, SUBJECT_CUI

# Load embeddings using PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
original_numpy = np.load("original_node_embeddings.npy")
original_embeddings = torch.tensor(original_numpy).to(device)

# Load CUI list from positie_dataset.csv
positie_df = pd.read_csv('positive_dataset.csv', usecols=['SUBJECT_CUI'])
cui_list = positie_df['SUBJECT_CUI'].dropna().astype(str).tolist()

# Create a dictionary: {CUI: embedding}
embedding_dict = {cui: original_embeddings[i] for i, cui in enumerate(cui_list) if i < len(original_embeddings)}

# Function to calculate pairwise similarities and distances
def calculate_metrics(cuis):
    # Filter for available embeddings
    cui_embeddings = {cui: embedding_dict[cui] for cui in cuis if cui in embedding_dict}
    if len(cui_embeddings) < 2:
        return None  # Skip if less than two embeddings

    comparisons = []
    cosine_similarities, euclidean_distances, manhattan_distances = [], [], []
    cui_keys = list(cui_embeddings.keys())

    # Pairwise comparison
    for i in range(len(cui_keys)):
        for j in range(i + 1, len(cui_keys)):
            cui1, cui2 = cui_keys[i], cui_keys[j]
            vec1, vec2 = cui_embeddings[cui1], cui_embeddings[cui2]

            # Convert to numpy for scipy distance calculations
            vec1_np, vec2_np = vec1.cpu().numpy(), vec2.cpu().numpy()

            cosine_similarity = 1 - cosine(vec1_np, vec2_np)
            euclidean_distance = euclidean(vec1_np, vec2_np)
            manhattan_distance = cityblock(vec1_np, vec2_np)

            # Store all pairwise metrics in a dictionary
            comparisons.append({
                'pair': (cui1, cui2),
                'cosine_similarity': cosine_similarity,
                'euclidean_distance': euclidean_distance,
                'manhattan_distance': manhattan_distance
            })

            # Store metrics for mean calculations
            cosine_similarities.append(cosine_similarity)
            euclidean_distances.append(euclidean_distance)
            manhattan_distances.append(manhattan_distance)

    print(f"Number of pairs for cancer type: {len(comparisons)}")
    # Compute mean for each metric
    return {
        'mean_cosine_similarity': np.mean(cosine_similarities),
        'mean_euclidean_distance': np.mean(euclidean_distances),
        'mean_manhattan_distance': np.mean(manhattan_distances),
        'comparisons': comparisons
    }

# Process each cancer type
results = []
for _, row in df.iterrows():
    cuis = row['SUBJECT_CUI'].strip('[]').split(', ')
    metrics = calculate_metrics(cuis)
    if metrics:
        metrics['CANCER_CLASS'] = row['CANCER_CLASS']
        results.append(metrics)

# Save results into a dataframe
df_results = pd.DataFrame(results)

# Display dataframe
print(df_results)

# Optional: Save the results to a CSV file (excluding the comparisons dict since it's not CSV-friendly)
df_results.to_csv('cancer_type_thresholds_summary.csv', index=False)
