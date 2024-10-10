import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.models.loadsplit import load_and_use_existing_split


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Debugging helper
def debug_info(variable, name):
    print(
        f"{name} - Type: {type(variable)}, Dtype: {getattr(variable, 'dtype', 'N/A')}, Sample: {variable[:5]}"
    )


# Define the Linear Probe model
class LinearProbe(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


def train_domain_probe(file_path, epochs=30, batch_size=200, lr=0.001, device="cpu"):
    # Load train and validation splits
    train_data, val_data = load_and_use_existing_split(file_path)

    # Convert data to PyTorch tensors
    train_embeddings = torch.tensor(train_data["embeddings"], dtype=torch.float32)
    debug_info(train_embeddings, "train_embeddings")

    try:
        train_labels = torch.tensor(
            train_data["dataset_flag"] == "r", dtype=torch.float32
        ).unsqueeze(
            1
        )  # Binary labels
        debug_info(train_labels, "train_labels")
    except Exception as e:
        print(f"Error in train_labels: {e}")

    val_embeddings = torch.tensor(val_data["embeddings"], dtype=torch.float32)
    val_labels = torch.tensor(val_data["dataset_flag"] == "r", dtype=torch.float32).unsqueeze(
        1
    )  # Binary labels

    # Create TensorDatasets
    train_dataset = TensorDataset(train_embeddings, train_labels)
    val_dataset = TensorDataset(val_embeddings, val_labels)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define the Linear Probe model
    input_dim = train_embeddings.shape[1]  # Based on embedding size
    domain_probe = LinearProbe(input_dim).to(device)

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss
    optimizer = optim.Adam(domain_probe.parameters(), lr=lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training function
    for epoch in range(epochs):
        domain_probe.train()
        train_loss = 0.0

        # Training phase
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = domain_probe(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        domain_probe.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = domain_probe(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Binary prediction
                predicted = torch.round(torch.sigmoid(outputs))
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_accuracy = correct / total
        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {train_loss/len(train_loader):.4f} | "
            f"Val Loss: {val_loss/len(val_loader):.4f} | "
            f"Val Accuracy: {val_accuracy * 100:.2f}%"
        )


# Example usage:
# train_domain_probe("path_to_your_npz_file.npz", epochs=30, batch_size=200, lr=0.001)
