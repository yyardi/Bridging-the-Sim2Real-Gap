import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.models.loadsplit import load_and_use_existing_split


# Define the Linear Probe model for multi-class classification
class LinearProbeMultiClass(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearProbeMultiClass, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


def train_action_probe(file_path, epochs=30, batch_size=200, lr=0.001, num_classes=5):
    # Load train and validation splits
    train_data, val_data = load_and_use_existing_split(file_path)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Convert data to PyTorch tensors
    train_embeddings = torch.tensor(train_data["embeddings"], dtype=torch.float32).to(device)
    val_embeddings = torch.tensor(val_data["embeddings"], dtype=torch.float32).to(device)

    # Labels (move them to the correct device; no one-hot encoding)
    train_labels = torch.tensor(train_data["labels"], dtype=torch.long).to(device)
    val_labels = torch.tensor(val_data["labels"], dtype=torch.long).to(device)

    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(train_embeddings, train_labels)
    val_dataset = TensorDataset(val_embeddings, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define the Linear Probe model
    input_dim = train_embeddings.shape[1]  # Based on embedding size
    action_probe = LinearProbeMultiClass(input_dim, num_classes).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Cross entropy loss
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
            loss = criterion(outputs, labels)  # No need to modify labels
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        action_probe.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = action_probe(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Get predictions (argmax for the highest score)
                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == labels).sum().item()  # Direct comparison
                total += labels.size(0)

        val_accuracy = correct / total
        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {train_loss/len(train_loader):.4f} | "
            f"Val Loss: {val_loss/len(val_loader):.4f} | "
            f"Val Accuracy: {val_accuracy * 100:.2f}%"
        )
