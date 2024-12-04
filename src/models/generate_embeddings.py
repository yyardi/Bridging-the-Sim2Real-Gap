import torch
import numpy as np
from tqdm import trange
import torchvision.transforms as transforms
import os

resize_transform = transforms.Compose(
    [
        transforms.ToPILImage(),  # Convert tensor to PIL image
        transforms.Resize((224, 224)),  # Resize image to 224x224
        transforms.ToTensor(),  # Convert PIL image back to tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # Normalize image
    ]
)


@torch.no_grad()
def generate_embeddings(
    model,
    images,
    num_samples=None,  # None for all
    batch_size=200,
):

    # List to store embeddings for each group
    embedding_batches = []

    if num_samples is None:
        num_samples = images.shape[0]
        indices = np.arange(num_samples)
    else:
        indices = np.random.choice(images.shape[0], size=num_samples, replace=False)

    # Embedding generation loop
    for i in trange(0, num_samples, batch_size, desc="Processing"):
        batch_indices = indices[i : i + batch_size]
        batch = (
            torch.stack([resize_transform(img) for img in images[batch_indices]]).float().cuda()
        )

        embeddings = model(batch).cpu()
        embedding_batches.append(embeddings)

    # Concatenate all embeddings
    embeddings = torch.cat(embedding_batches)

    return embeddings
