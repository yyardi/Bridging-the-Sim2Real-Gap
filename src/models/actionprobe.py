from tqdm import tqdm 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd  # For rolling window calculations
import os
from src.models.loadsplit import load_and_use_existing_split

# Debugging helper
def debug_info(variable, name):
    print(
        f"{name} - Type: {type(variable)}, Dtype: {getattr(variable, 'dtype', 'N/A')}, Sample: {variable[:5]}"
    )

class LinearProbe(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

def train_action_probe(file_path, epochs=30, batch_size=200, lr=0.001, rolling_window=10, model_name="Model"):
    train_data, val_data = load_and_use_existing_split(file_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_embeddings = torch.tensor(train_data["embeddings"], dtype=torch.float32).to(device)
    val_embeddings = torch.tensor(val_data["embeddings"], dtype=torch.float32).to(device)
    train_labels = torch.tensor(train_data["labels"], dtype=torch.float32).to(device)
    val_labels = torch.tensor(val_data["labels"], dtype=torch.float32).to(device)

    debug_info(train_embeddings, "train_embeddings")
    debug_info(train_labels, "train_labels")

    train_dataset = TensorDataset(train_embeddings, train_labels)
    val_dataset = TensorDataset(val_embeddings, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_dim = train_embeddings.shape[1]
    action_probe = LinearProbe(input_dim).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(action_probe.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        action_probe.train()
        train_loss = 0.0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = action_probe(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        action_probe.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = action_probe(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

    if epochs >= 10000:
        if len(train_losses) >= 100:
            avg_last_100_train_loss = sum(train_losses[-100:]) / 100
            tqdm.write(f"Average Training Loss (Last 100 Epochs) for {model_name}: {avg_last_100_train_loss:.4f}")

            averages_path = "/home/ubuntu/semrep/src/models/averages.txt"
            with open(averages_path, "a") as f:
                f.write(f"{model_name}: {avg_last_100_train_loss:.4f}\n")

    return train_losses, val_losses
