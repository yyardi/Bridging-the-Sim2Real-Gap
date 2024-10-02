import torch
import numpy as np
from tqdm import trange
import torchvision.transforms as transforms
import os


def generate_embeddings(
    model,
    sim1_imgs,
    sim2_imgs,
    real_imgs,
    model_output_dim,
    num_samples=1000,
    batch_size=200,
    selected_model_name="ResNet18",
    output_dir="embeddings",
):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define resize transformation for MVP and VIP models
    resize_transform = (
        transforms.Compose(
            [
                transforms.ToPILImage(),  # Convert tensor to PIL image
                transforms.Resize((224, 224)),  # Resize image to 224x224
                transforms.ToTensor(),  # Convert PIL image back to tensor
            ]
        )
        if selected_model_name in ["MVP", "VIP"]
        else None
    )

    # Sample indices for each dataset
    sim1_indices = np.random.choice(sim1_imgs.shape[0], size=num_samples, replace=False)
    sim2_indices = np.random.choice(sim2_imgs.shape[0], size=num_samples, replace=False)
    real_indices = np.random.choice(real_imgs.shape[0], size=num_samples, replace=False)

    # Create tensors to store embeddings
    sim1_embeddings = torch.zeros(num_samples, model_output_dim.shape[1])
    sim2_embeddings = torch.zeros(num_samples, model_output_dim.shape[1])
    real_embeddings = torch.zeros(num_samples, model_output_dim.shape[1])

    # Embedding generation loop
    with torch.no_grad():
        for i in trange(0, num_samples, batch_size):
            # Process sim1 images
            batch_indices = sim1_indices[i : i + batch_size]
            if resize_transform:
                sim_batch = (
                    torch.stack([resize_transform(img) for img in sim1_imgs[batch_indices]])
                    .float()
                    .cuda()
                )
            else:
                sim_batch = (
                    torch.tensor(sim1_imgs[batch_indices]).permute(0, 3, 1, 2).float().cuda()
                )
            sim1_embeddings[i : i + batch_size] = model(sim_batch).cpu()

            # Process sim2 images
            batch_indices = sim2_indices[i : i + batch_size]
            if resize_transform:
                sim_batch = (
                    torch.stack([resize_transform(img) for img in sim2_imgs[batch_indices]])
                    .float()
                    .cuda()
                )
            else:
                sim_batch = (
                    torch.tensor(sim2_imgs[batch_indices]).permute(0, 3, 1, 2).float().cuda()
                )
            sim2_embeddings[i : i + batch_size] = model(sim_batch).cpu()

            # Process real images
            batch_indices = real_indices[i : i + batch_size]
            if resize_transform:
                real_batch = (
                    torch.stack([resize_transform(img) for img in real_imgs[batch_indices]])
                    .float()
                    .cuda()
                )
            else:
                real_batch = (
                    torch.tensor(real_imgs[batch_indices]).permute(0, 3, 1, 2).float().cuda()
                )
            real_embeddings[i : i + batch_size] = model(real_batch).cpu()

    # Save embeddings to separate files
    np.save(
        os.path.join(output_dir, f"{selected_model_name}_sim1_embeddings.npy"),
        sim1_embeddings.cpu().numpy(),
    )
    np.save(
        os.path.join(output_dir, f"{selected_model_name}_sim2_embeddings.npy"),
        sim2_embeddings.cpu().numpy(),
    )
    np.save(
        os.path.join(output_dir, f"{selected_model_name}_real_embeddings.npy"),
        real_embeddings.cpu().numpy(),
    )

    print(f"Embeddings saved to {output_dir} with model name: {selected_model_name}")

    return sim1_embeddings, sim2_embeddings, real_embeddings
