import numpy as np
from scipy.spatial.distance import euclidean, cosine

def euclidean_distance(embeddings, dataset_flags):
    # Create masks for sim and real data
    sim_mask = dataset_flags == "s"
    real_mask = dataset_flags == "r"
    sim_embeddings = embeddings[sim_mask]
    real_embeddings = embeddings[real_mask]

    #find centroids
    sim_centroid = np.mean(sim_embeddings, axis=0)
    real_centroid = np.mean(real_embeddings, axis=0)
    print("Sim Centroid:", sim_centroid.shape)
    print("Real Centroid:", real_centroid.shape)

    #find normalization value
    norm = np.mean(np.std(embeddings, axis=0, ddof=1))
    print("Normalization Value:", norm)

    #calculate normalized euclidean distance
    euclidean_distance = euclidean(sim_centroid, real_centroid)
    print("Euclidean Distance:", euclidean_distance)
    print("Normalized Euclidean Distance:", euclidean_distance/norm)
    return euclidean_distance/norm

def cosine_distance(embeddings, dataset_flags):
    # Create masks for sim and real data
    sim_mask = dataset_flags == "s"
    real_mask = dataset_flags == "r"
    sim_embeddings = embeddings[sim_mask]
    real_embeddings = embeddings[real_mask]

    #find centroids
    sim_centroid = np.mean(sim_embeddings, axis=0)
    real_centroid = np.mean(real_embeddings, axis=0)
    print("Sim Centroid:", sim_centroid.shape)
    print("Real Centroid:", real_centroid.shape)

    #find normalization value
    norm = np.mean(np.std(embeddings, axis=0, ddof=1))
    print("Normalization Value:", norm)

    #calculate normalized euclidean distance
    cosine_distance = cosine(sim_centroid, real_centroid)
    print("Cosine Distance:", cosine_distance)
    print("Normalized Cosine Distance:", cosine_distance/norm)
    return cosine_distance/norm

