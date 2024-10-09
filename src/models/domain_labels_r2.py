from pathlib import Path
import zarr
from tqdm import trange
import numpy as np
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

from src.models.generate_embeddings import generate_embeddings

# Our alternative to the Domain Labels Probe 

def domain_r_squared(embeddings, dataset_flags):
    # Create masks for sim and real data
    sim_mask = dataset_flags == "s"
    real_mask = dataset_flags == "r"
    sim_embeddings = embeddings[sim_mask].flatten()
    real_embeddings = embeddings[real_mask].flatten()

    length = sim_embeddings.shape[0] 
    print(length)

    real_embeddings = real_embeddings[:length]

    correlation_matrix = np.corrcoef(sim_embeddings, real_embeddings)
    correlation_coefficient = correlation_matrix[0, 1]

    # Calculate R^2 value
    r_squared = correlation_coefficient ** 2

    print(f"R^2 value: {r_squared}")
