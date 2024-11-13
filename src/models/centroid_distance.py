import numpy as np
import torch
from scipy.spatial.distance import euclidean, cosine

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
    # std_devs = []
    # for i in range (len(embeddings)):
    #     std_devs.append(np.std(embeddings[i], axis=0, ddof=1))
    # norm = np.mean(std_devs)
    # embeddings /= norm
    for i in range(len(embeddings)):
        embeddings[i] = embeddings[i] / np.std(embeddings[i], axis=0, ddof=1)
    return embeddings

def centroid_distance(encoder, distance_metric, normalization):
    file_path = "/home/ubuntu/semrep/embeddings/encoders/"+encoder+".npz"
    dataset = np.load(file_path)
    embeddings = dataset["embeddings"]
    dataset_flags = dataset["dataset_flag"]
    #Normalize
    norm = 1
    if normalization == "std_dev":
        #norm = np.mean(np.std(embeddings, axis=0, ddof=1))
        std_dev(embeddings)
    elif normalization == "mean":
        norm = np.sqrt(embeddings.shape[1])
    elif normalization == "min_max":
        embeddings = min_max(embeddings)
    print(normalization, "Normalization Value:", norm, "(1 indicates min_max)")

    #Create masks for sim and real data
    sim_mask = dataset_flags == "s"
    real_mask = dataset_flags == "r"
    sim_embeddings = (embeddings[sim_mask])/norm
    real_embeddings = (embeddings[real_mask])/norm

    #find centroids
    sim_centroid = np.mean(sim_embeddings, axis=0)
    real_centroid = np.mean(real_embeddings, axis=0)
    print("Sim Centroid:", sim_centroid.shape)
    print("Real Centroid:", real_centroid.shape)

    #calculate normalized euclidean distance
    if distance_metric == "euclidean":
        euclidean_distance = euclidean(sim_centroid, real_centroid)
        print("Normalized Euclidean Distance:", euclidean_distance)
        return euclidean_distance
    elif distance_metric == "cosine":
        cosine_distance = cosine(sim_centroid, real_centroid)
        print("Normalized Cosine Distance:", cosine_distance)
        return cosine_distance