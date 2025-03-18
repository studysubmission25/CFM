import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances

# Load embeddings from .npy files
original_embeddings = np.load("original_node_embeddings.npy")
denoised_embeddings = np.load("denoised_node_embeddings.npy")

denoised_numpy = denoised_embeddings
original_numpy = original_embeddings

# Debug: Check first rows
#print("Original (first row):", original_numpy[:1])
#print("Denoised (first row):", denoised_numpy[:1])

# ========================
# 1. Cosine Similarity (Mean)
# ========================
cosine_score = np.mean(np.sum(original_numpy * denoised_numpy, axis=1) /
                       (np.linalg.norm(original_numpy, axis=1) * np.linalg.norm(denoised_numpy, axis=1)))

# ========================
# 2. Mean Squared Error (MSE)
# ========================
mse_score = np.mean((original_numpy - denoised_numpy) ** 2)

# ========================
# 3. Euclidean Distance (L2 Norm)
# ========================
euclidean_score = np.mean(np.linalg.norm(original_numpy - denoised_numpy, axis=1))

# ========================
# 4. Manhattan Distance (L1 Norm)
# ========================
manhattan_score = np.mean(np.sum(np.abs(original_numpy - denoised_numpy), axis=1))

# ========================
# 5. Cosine Similarity Matrix (Mean of Diagonal)
# ========================
similarity_matrix = cosine_similarity(original_numpy, denoised_numpy)
diagonal_mean = np.mean(np.diag(similarity_matrix))

# ========================
# PRINT RESULTS
# ========================
print(f"\nEvaluation Metrics:")
print(f"Mean Cosine Similarity:         {cosine_score:.4f}")
print(f"Mean Squared Error (MSE):       {mse_score:.6f}")
print(f"Mean Euclidean Distance (L2):   {euclidean_score:.6f}")
print(f"Mean Manhattan Distance (L1):   {manhattan_score:.6f}")
print(f"Mean Diagonal Cosine Similarity:{diagonal_mean:.4f}")

