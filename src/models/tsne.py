import torch
import numpy as np
from tqdm import trange
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


def plot_tsne(embeddings, dataset_flags, perplexity=30, n_iter=1000):
    # Convert embeddings to numpy array if it's a torch tensor
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()

    # Fit t-SNE on the embeddings
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create masks for sim and real data
    sim_mask = dataset_flags == "s"
    real_mask = dataset_flags == "r"

    # Plot the 2D embeddings
    plt.figure(figsize=(12, 8))
    plt.scatter(
        embeddings_2d[sim_mask, 0], embeddings_2d[sim_mask, 1], c="blue", label="Sim", alpha=0.7
    )
    plt.scatter(
        embeddings_2d[real_mask, 0], embeddings_2d[real_mask, 1], c="red", label="Real", alpha=0.7
    )

    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.title("t-SNE visualization of sim vs real embeddings")
    plt.show()
