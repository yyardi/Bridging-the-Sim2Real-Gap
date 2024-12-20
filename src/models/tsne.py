import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


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

    return sim_embeddings_2d, real_embeddings_2d