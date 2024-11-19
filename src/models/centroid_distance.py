import numpy as np
import torch
from scipy.spatial.distance import euclidean, cosine
from src.models.adapter import reduce_embeddings_to_fixed_dim
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def min_max(embeddings):
    # Check for NaNs and Infs
    if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
        raise ValueError("Input array contains NaNs or Infs. Please clean the data.")
    
    min_vals = embeddings.min(axis=0)
    max_vals = embeddings.max(axis=0)
    
    range_vals = max_vals - min_vals
    range_vals = np.where(range_vals == 0, 1, range_vals)  # Prevent division by zero
    
    normalized_embeddings = (embeddings - min_vals) / range_vals
    return normalized_embeddings

def std_dev(embeddings):
    # Check for NaNs and Infs
    if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
        raise ValueError("Input array contains NaNs or Infs. Please clean the data.")
    # embeddings: (n, d)
    # n: number of images
    # d: size of embedding space

    stdev = np.std(embeddings, axis=0) # d-dimensional
    stdev = np.where(stdev == 0, 1, stdev)  # Prevent division by zero

    # Broadcasts because embeddings is (n, d) and stdev is (d,)
    embeddings /= stdev

    return embeddings

def centroid_distance(encoder, distance_metric, normalization, fix_dimension):
    file_path = "/home/ubuntu/semrep/embeddings/encoders/"+encoder+".npz"
    dataset = np.load(file_path)
    embeddings = dataset["embeddings"]
    dataset_flags = dataset["dataset_flag"]

    if fix_dimension == "adapter":
        embeddings = reduce_embeddings_to_fixed_dim(dataset["embeddings"], 384)
    if fix_dimension == "PCA":
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)
        pca = PCA(n_components=384)
        embeddings = pca.fit_transform(embeddings)
    if fix_dimension == "zero_pad":
        padding_size = 2048 - embeddings.shape[0]
        if padding_size > 0:
            embeddings = np.pad(embeddings, (0, padding_size), mode='constant', constant_values=0)

    #Normalize
    if normalization == "std_dev":
        std_dev(embeddings)
    elif normalization == "min_max":
        embeddings = min_max(embeddings)

    #Create masks for sim and real data
    sim_mask = dataset_flags == "s"
    real_mask = dataset_flags == "r"
    sim_embeddings = (embeddings[sim_mask])
    real_embeddings = (embeddings[real_mask])

    #find centroids
    sim_centroid = np.mean(sim_embeddings, axis=0)
    real_centroid = np.mean(real_embeddings, axis=0)
    print(encoder, "=", sim_centroid.shape[0])

    #calculate normalized euclidean distance
    if distance_metric == "euclidean":
        euclidean_distance = euclidean(sim_centroid, real_centroid)
        if fix_dimension == "sqrt":
            euclidean_distance /= np.sqrt(embeddings.shape[1])
        print("Normalized Euclidean Distance:", euclidean_distance)
        return euclidean_distance
    elif distance_metric == "cosine":
        cosine_distance = cosine(sim_centroid, real_centroid)
        print("Normalized Cosine Distance:", cosine_distance)
        return cosine_distance