import numpy as np
from scipy.spatial.distance import euclidean, cosine

def euclidean_distance(embeddings, dataset_flags):
    #find normalization value
    norm = np.mean(np.std(embeddings, axis=0, ddof=1))
    print("Normalization Value:", norm)
    
    # Create masks for sim and real data
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
    euclidean_distance = euclidean(sim_centroid, real_centroid)
    print("Normalized Euclidean Distance:", euclidean_distance)
    return euclidean_distance

def cosine_distance(embeddings, dataset_flags):
    #find normalization value
    norm = np.mean(np.std(embeddings, axis=0, ddof=1))
    print("Normalization Value:", norm)

    # Create masks for sim and real data
    sim_mask = dataset_flags == "s"
    real_mask = dataset_flags == "r"
    sim_embeddings = embeddings[sim_mask]/norm
    real_embeddings = embeddings[real_mask]/norm

    #find centroids
    sim_centroid = np.mean(sim_embeddings, axis=0)
    real_centroid = np.mean(real_embeddings, axis=0)
    print("Sim Centroid:", sim_centroid.shape)
    print("Real Centroid:", real_centroid.shape)

    #calculate normalized euclidean distance
    cosine_distance = cosine(sim_centroid, real_centroid)
    print("Normalized Cosine Distance:", cosine_distance)
    return cosine_distance

