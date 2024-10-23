import numpy as np
from scipy.spatial.distance import euclidean, cosine

def euclidean_distance(embeddings, dataset_flags):
    # Create masks for sim and real data
    sim_mask = dataset_flags == "s"
    real_mask = dataset_flags == "r"
    sim_embeddings = embeddings[sim_mask]
    real_embeddings = embeddings[real_mask]
    sim_centroid = np.mean(sim_embeddings, axis=0)
    real_centroid = np.mean(real_embeddings, axis=0)

    print("Sim Centroid:", sim_centroid)
    print("Real Centroid:", real_centroid)
    euclidean_distance = euclidean(sim_centroid, real_centroid)
    print("Euclidean Distance:", euclidean_distance)
    return euclidean_distance

def cosine_distance(embeddings, dataset_flags):
    # Create masks for sim and real data
    sim_mask = dataset_flags == "s"
    real_mask = dataset_flags == "r"
    sim_embeddings = embeddings[sim_mask]
    real_embeddings = embeddings[real_mask]
    sim_centroid = np.mean(sim_embeddings, axis=0)
    real_centroid = np.mean(real_embeddings, axis=0)

    print("Sim Centroid:", sim_centroid)
    print("Real Centroid:", real_centroid)
    cosine_distance = cosine(sim_centroid, real_centroid)
    print("Cosine Distance:", cosine_distance)
    return cosine_distance

