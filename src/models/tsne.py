import torch
import numpy as np
from tqdm import trange
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


def plot_tsne(embeddings, domain_labels, num_samples=1000):
    # Convert embeddings to numpy array if it's a torch tensor
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    # Fit t-SNE on the embeddings
    tsne = TSNE(n_components=2)
    embeddings_2d = tsne.fit_transform(embeddings)

    sim_mask = domain_labels == "s"
    real_mask = domain_labels == "r"
    
    sim_embeddings_2d = embeddings_2d[sim_mask]
    real_embeddings_2d = embeddings_2d[real_mask]

    # Plot the 2D embeddings
    plt.figure(figsize=(12, 8))
    plt.scatter(
        sim_embeddings_2d[:, 0], sim_embeddings_2d[:, 1], c="orange",
        label="Sim", alpha=0.7
    )
    plt.scatter(
        real_embeddings_2d[:, 0], real_embeddings_2d[:, 1], c="blue",
        label="Real", alpha=0.7
    )
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.title("t-SNE visualization of sim vs real embeddings")
    plt.show()
