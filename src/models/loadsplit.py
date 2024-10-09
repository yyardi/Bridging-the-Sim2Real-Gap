from pathlib import Path
import zarr
from tqdm import trange


import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

from src.models.generate_embeddings import generate_embeddings
import numpy as np


def load_and_use_existing_split(file_path):
    # Load the NPZ file
    data = np.load(file_path)

    # Extract the arrays
    embeddings = data["embeddings"] # embeddings for model
    labels = data["labels"] # action labels 
    train_flag = data["train_flag"] # is training or val data
    dataset_flag = data["dataset_flag"] # domain labels (is sim or real)

    # Use the existing train_flag to split the data
    train_mask = train_flag == 1
    val_mask = train_flag == 0

    # Create the final splits
    train_data = {
        "embeddings": embeddings[train_mask],
        "labels": labels[train_mask],
        "dataset_flag": dataset_flag[train_mask],
    }

    val_data = {
        "embeddings": embeddings[val_mask],
        "labels": labels[val_mask],
        "dataset_flag": dataset_flag[val_mask],
    }

    return train_data, val_data