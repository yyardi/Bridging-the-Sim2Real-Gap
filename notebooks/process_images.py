# %% [markdown]
# ResNet18, 34, 50, VIP, MVP, DINO

# %% [markdown]
# **We need to move the loading into the Compiled Models Notebook From Here**

# %%
from pathlib import Path
import zarr
from tqdm import trange


import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

from src.models.generate_embeddings import generate_embeddings

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

# %%
visualize = False

# %%
from src.models.encoders import models

# %% [markdown]
# **All Files Loading Info**
#

# %%
# NOTE: Change this to wherever the project is located
project_root = Path(__file__).resolve().parents[1]

data_path = project_root / "data" / "processed"
models_path = project_root / "models"

# Output path
output_path = project_root / "embeddings" / "encoders"
output_path.mkdir(exist_ok=True, parents=True)

# %%
# Load the data
sim = zarr.open(data_path / "one_leg_med_sim.zarr", mode="r")
real = zarr.open(data_path / "one_leg_low_real.zarr", mode="r")

datasets = {"sim": sim, "real": real}

for name, dataset in datasets.items():
    imgs = dataset["color_image2"]
    labels = dataset["action/pos"]

    print(
        f"Loaded {len(dataset['episode_ends'])} trajectories containing {imgs.shape[0]} frames of {name} data"
    )

# %%
if visualize:
    # Sample 8 images from each dataset
    sim_indices = np.random.choice(datasets["sim"]["color_image2"].shape[0], size=8, replace=False)
    real_indices = np.random.choice(
        datasets["real"]["color_image2"].shape[0], size=8, replace=False
    )

    # Create a figure and axes
    fig, axes = plt.subplots(2, 8, figsize=(20, 5))

    # Display the sampled images
    for i, idx in enumerate(sim_indices):
        axes[0, i].imshow(datasets["sim"]["color_image2"][idx])
        axes[0, i].axis("off")

    for i, idx in enumerate(real_indices):
        axes[1, i].imshow(datasets["real"]["color_image2"][idx])
        axes[1, i].axis("off")

plt.tight_layout()
plt.show()


# %%
def process_dataset(m, dataset, dataset_type, num_samples=None, batch_size=1024):
    # Generate embeddings
    embeddings = generate_embeddings(
        m,
        dataset["color_image2"],
        num_samples=num_samples,
        batch_size=batch_size,
    ).numpy()

    print(f"Generated {embeddings.shape[0]} embeddings for {dataset_type} data")
    print(f"Embedding shape: {embeddings.shape}")
    print(embeddings.min(), embeddings.max(), embeddings.mean(), embeddings.std())

    # Make a split array for the embeddings into train and eval according to 90/10 split of trajectories
    split_index = dataset["episode_ends"][-5]

    # Split the embeddings into train and eval
    train_flag = np.zeros(embeddings.shape[0], dtype=bool)
    train_flag[:split_index] = True

    # Create a flag to indicate the dataset type (sim or real)
    dataset_flag = np.full(embeddings.shape[0], dataset_type, dtype=str)

    return embeddings, dataset["action/pos"], train_flag, dataset_flag


# %%
def process_all_models_and_datasets(models, datasets, batch_size=1024, overwrite=False):
    results = {}

    print("Available models:", models.keys())

    for model_name, model_class in models.items():
        print(f"Processing with {model_name}")

        # Check if the output file already exists
        output_file = output_path / f"{model_name}.npz"

        if output_file.exists() and not overwrite:
            print(f"Skipping {model_name} as the output file already exists")
            continue

        try:
            m: torch.nn.Module = model_class()
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            continue

        # Process sim data
        sim_embeddings, sim_labels, sim_train_flag, sim_dataset_flag = process_dataset(
            m, datasets["sim"], "sim", batch_size=batch_size
        )

        # Process real data
        real_embeddings, real_labels, real_train_flag, real_dataset_flag = process_dataset(
            m, datasets["real"], "real", batch_size=batch_size
        )

        # Combine sim and real data
        combined_embeddings = np.vstack((sim_embeddings, real_embeddings))
        combined_labels = np.vstack((sim_labels, real_labels))
        combined_train_flag = np.concatenate((sim_train_flag, real_train_flag))
        combined_dataset_flag = np.concatenate((sim_dataset_flag, real_dataset_flag))

        # Store results for this model
        results = {
            "embeddings": combined_embeddings,
            "labels": combined_labels,
            "dataset_flag": combined_dataset_flag,
            "train_flag": combined_train_flag,
        }

        # Save the results
        np.savez(
            output_file,
            **results,
        )

        print(f"Finished processing {model_name}")

    return results


# %%
len(models)

# %%
process_models = dict(
    # "mcr": models["mcr"]
    # "Swin": models["Swin"],
    # "BEiT": models["BEiT"],
    # "CoAtNet": models["CoAtNet"],
    # "vgg16": models["vgg16"],
    # "ResNet18": models["ResNet18"],
    # "ViT": models["ViT"],
    # "HybridViT": models["HybridViT"],
    # "VIP": models["VIP"],
    # "Swin": models["Swin"],
    # "CLIP-Base-16": models["CLIP-Base-16"]
    # list(models.items())[0:12]
    # MVP=models["MVP"]
    # **models
    [(key, value) for key, value in models.items() if key != "DinoV2-B"]
)

# %%
# Usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
overwrite = False

all_results = process_all_models_and_datasets(
    process_models, datasets, batch_size=batch_size, overwrite=overwrite
)
