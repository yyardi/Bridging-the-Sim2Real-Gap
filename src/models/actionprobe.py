import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.models.loadsplit import load_and_use_existing_split


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


def train_action_probe(file_path, epochs=30, batch_size=200, lr=0.001):

    train_data, val_data = load_and_use_existing_split(file_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_embeddings = torch.tensor(train_data["embeddings"], dtype=torch.float32).to(device)
    val_embeddings = torch.tensor(val_data["embeddings"], dtype=torch.float32).to(device)

    train_labels = torch.tensor(train_data["labels"], dtype=torch.float32).unsqueeze(1).to(device)
    val_labels = torch.tensor(val_data["labels"], dtype=torch.float32).unsqueeze(1).to(device)

    # Debugging information
    debug_info(train_embeddings, "train_embeddings")
    debug_info(train_labels, "train_labels")

    train_dataset = TensorDataset(train_embeddings, train_labels)
    val_dataset = TensorDataset(val_embeddings, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    input_dim = train_embeddings.shape[1]  # Based on embedding size
    action_probe = LinearProbe(input_dim).to(device)  # Make sure the model is on the correct device

    criterion = nn.MSELoss()  # Mean Squared Error for action prediction
    optimizer = optim.Adam(action_probe.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        action_probe.train()
        train_loss = 0.0

        # Training phase
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = action_probe(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        action_probe.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = action_probe(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Val Loss: {val_loss/len(val_loader):.4f}"
        )
