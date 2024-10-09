import torch
from torchvision import transforms

#  this file can be moved to encoders.py

class DinoV2Encoder(torch.nn.Module):
    def __init__(self, model_name="dinov2_vits14", freeze=True, device="cuda"):
        super().__init__()
        assert model_name in [
            "dinov2_vits14",
            "dinov2_vitb14",
            "dinov2_vitl14",
            "dinov2_vitg14",
        ]
        self.device = device

        # Model wants a batch of images of shape (batch_size, 3, 224, 224) and normalized
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)

        self.encoding_dim = self.model.norm.normalized_shape[0]

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

            self.model.eval()

        self.model = self.model.to(device)

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    # Expect input to be a batch of images of shape (batch_size, 3, 224, 224) in range [0, 255]
    def forward(self, x):
        # Normalize images
        x = x / 255.0
        x = self.normalize(x)
        x = self.model(x)
        return x
