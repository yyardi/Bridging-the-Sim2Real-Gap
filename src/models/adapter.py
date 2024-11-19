import torch
import torch.nn as nn

# Define a simple neural network model that maps the input dimension to the target dimension
class RandomNNAdapter(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RandomNNAdapter, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

# Function to reduce embeddings to a fixed dimension

def reduce_embeddings_to_fixed_dim(embeddings, target_dim):
    # Stack all embeddings into a single tensor (assuming each embedding is a 1D array)
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    
    # Get the input dimension of the embeddings
    input_dim = embeddings_tensor.shape[1]  # Assuming embeddings are in shape (batch_size, input_dim)
    
    # Initialize the model
    model = RandomNNAdapter(input_dim, target_dim)
    
    # Apply the model to the entire batch of embeddings
    reduced_embeddings = model(embeddings_tensor)
    
    # Convert the reduced embeddings to numpy array and return them
    return reduced_embeddings.detach().numpy()