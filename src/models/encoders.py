from r3m import load_r3m

# import mvp
from vip import load_vip
import torch

# from src.models.loading_dino import DinoV2Encoder
import mcr
import timm
from transformers import (
    AutoModelForImageClassification,
    AutoImageProcessor,
    AutoModel,
    HybridViTModel,
)
import clip
import numpy as np
from PIL import Image
from abc import ABC, abstractmethod
from torchvision import transforms

from r3m import R3MEncoder


class BaseEncoder(ABC):
    """Abstract base class for all encoders"""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.preprocess = None

    def preprocess_numpy(self, image_np):
        """Convert numpy image [0-255] to preprocessed tensor"""
        # Convert numpy array to PIL Image
        if isinstance(image_np, np.ndarray):
            image_pil = Image.fromarray(image_np.astype("uint8"))
        else:
            raise ValueError("Input must be a numpy array")

        # Apply model-specific preprocessing
        image_tensor = self.preprocess(image_pil).unsqueeze(0).to(self.device)
        return image_tensor

    def process_batch(self, batch):
        """Process a batch of images or single image
        Args:
            batch: numpy array (H, W, 3) or (B, H, W, 3) with values [0-255]
        Returns:
            tensor: preprocessed torch tensor ready for model
        """
        if len(batch.shape) == 3:
            # Single image
            return self.preprocess_numpy(batch)
        elif len(batch.shape) == 4:
            # Batch of images
            return torch.stack([self.preprocess_numpy(img) for img in batch])
        else:
            raise ValueError(f"Invalid input shape: {batch.shape}")

    def __call__(self, x):
        """
        Args:
            x: numpy array of shape (H, W, 3) with values [0-255]
               or batch of arrays (B, H, W, 3)
        Returns:
            embeddings: torch tensor
        """
        x = self.process_batch(x)
        with torch.no_grad():
            return self.model(x)


class TimmEncoder(BaseEncoder):
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self._load_model()
        self._setup_preprocessing()

    def _load_model(self):
        self.model = timm.create_model(self.model_name, pretrained=self.pretrained)
        self.model.eval()
        self.model.to(self.device)

    def _setup_preprocessing(self):
        data_cfg = timm.data.resolve_data_config(self.model.pretrained_cfg)
        self.preprocess = timm.data.create_transform(**data_cfg)


class TransformerEncoder(BaseEncoder):
    """Base class for HuggingFace transformer models"""

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self._load_model()
        self._setup_preprocessing()

    def _load_model(self):
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(self.device)
        self.model.config.return_dict = False  # For compatibility with torch.jit.trace

    def _setup_preprocessing(self):
        # Use AutoImageProcessor for preprocessing
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)

    def preprocess_numpy(self, image_np):
        """Convert numpy image [0-255] to preprocessed tensor"""
        # Convert numpy array to PIL Image
        if isinstance(image_np, np.ndarray):
            image_pil = Image.fromarray(image_np.astype("uint8"))
        else:
            raise ValueError("Input must be a numpy array")

        # Use processor to handle preprocessing
        inputs = self.processor(images=image_pil, return_tensors="pt")
        return inputs.pixel_values.to(self.device)

    def process_batch(self, batch):
        """Override because transformers use torch.cat instead of stack"""
        if len(batch.shape) == 3:
            return self.preprocess_numpy(batch)
        elif len(batch.shape) == 4:
            return torch.cat([self.preprocess_numpy(img) for img in batch], dim=0)
        else:
            raise ValueError(f"Invalid input shape: {batch.shape}")

    def __call__(self, x):
        """
        Args:
            x: numpy array of shape (H, W, 3) with values [0-255]
               or batch of arrays (B, H, W, 3)
        Returns:
            embeddings: torch tensor of shape (1, embedding_dim)
                       or (B, embedding_dim) for batches
        """
        x = self.process_batch(x)
        with torch.no_grad():
            return self.model(x)[0]  # Return last hidden states


class R3MEncoder(BaseEncoder):
    """Wrapper for R3M models to maintain consistent interface"""

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self._load_model()
        self._setup_preprocessing()

    def _load_model(self):
        self.model = load_r3m(self.model_name)
        self.model.eval()
        self.model.to(self.device)

    def _setup_preprocessing(self):
        # R3M handles normalization internally, so we just need to convert to tensor
        self.preprocess = transforms.Compose([transforms.ToTensor()])

    def __call__(self, x):
        x = self.process_batch(x)
        with torch.no_grad():
            return self.model(x * 255.0)  # R3M expects [0-255] range


class CLIPEncoder(BaseEncoder):
    """Wrapper for CLIP models to maintain consistent interface"""

    def __init__(self, model, preprocess):
        super().__init__()
        self.model = model
        self.preprocess = preprocess

    def __call__(self, x):
        x = self.process_batch(x)
        with torch.no_grad():
            return self.model.encode_image(x)


class VIPEncoder(BaseEncoder):
    """Wrapper for VIP models to maintain consistent interface"""

    def __init__(self):
        super().__init__()
        self._load_model()
        self._setup_preprocessing()

    def _load_model(self):
        self.model = load_vip()
        self.model.eval()
        self.model.to(self.device)

    def _setup_preprocessing(self):
        # VIP handles normalization internally, so we just need to convert to tensor
        self.preprocess = transforms.Compose([transforms.ToTensor()])

    def __call__(self, x):
        x = self.process_batch(x)
        with torch.no_grad():
            return self.model(x * 255.0)  # VIP expects [0-255] range


class MCREncoder(BaseEncoder):
    """Wrapper for MCR models to maintain consistent interface"""

    def __init__(self, ckpt_path):
        super().__init__()
        self.ckpt_path = ckpt_path
        self._load_model()
        self._setup_preprocessing()

    def _load_model(self):
        self.model = mcr.load_mcr(ckpt_path=self.ckpt_path)
        self.model.eval()
        self.model.to(self.device)

    def _setup_preprocessing(self):
        # MCR expects input in [0-255] range and handles normalization internally
        self.preprocess = transforms.Compose([transforms.ToTensor()])

    def __call__(self, x):
        x = self.process_batch(x)
        with torch.no_grad():
            # MCR's forward method expects input in [0-255] range
            return self.model(x * 255.0)


class HybridViTEncoder(BaseEncoder):
    """Wrapper for HuggingFace's HybridViT models"""

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self._load_model()
        self._setup_preprocessing()

    def _load_model(self):
        self.model = HybridViTModel.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(self.device)

    def _setup_preprocessing(self):
        # Use AutoImageProcessor for preprocessing
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)

    def preprocess_numpy(self, image_np):
        """Convert numpy image [0-255] to preprocessed tensor"""
        if isinstance(image_np, np.ndarray):
            image_pil = Image.fromarray(image_np.astype("uint8"))
        else:
            raise ValueError("Input must be a numpy array")

        inputs = self.processor(images=image_pil, return_tensors="pt")
        return inputs.pixel_values.to(self.device)

    def __call__(self, x):
        x = self.process_batch(x)
        with torch.no_grad():
            return self.model(x)[0]  # Return last hidden states


# Model configurations
MODEL_CONFIGS = {
    # TIMM models
    "ResNet18": {"type": "timm", "name": "resnet18"},
    "ResNet34": {"type": "timm", "name": "resnet34"},
    "ResNet50": {"type": "timm", "name": "resnet50"},
    "ResNet101": {"type": "timm", "name": "resnet101"},
    "EfficientNetB0": {"type": "timm", "name": "efficientnet_b0"},
    "MobileNetv3": {"type": "timm", "name": "mobilenetv3_large_100"},
    "vgg16": {"type": "timm", "name": "vgg16"},
    "vgg19": {"type": "timm", "name": "vgg19"},
    "ViT": {"type": "timm", "name": "vit_base_patch16_224"},
    "Swin": {"type": "timm", "name": "swin_base_patch4_window7_224"},
    "BEiT": {"type": "timm", "name": "beit_large_patch16_224"},
    "CoAtNet": {"type": "timm", "name": "coatnet_1_224"},
    "dinov2": {"type": "timm", "name": "vit_base_patch14_dinov2"},
    # Transformers models
    "HybridViT": {"type": "hybridvit", "name": "google/vit-hybrid-base-bit-384"},
    # Robot learning models
    "VIP": {"type": "vip"},
    "R3M18": {"type": "r3m", "name": "resnet18"},
    "R3M34": {"type": "r3m", "name": "resnet34"},
    "R3M50": {"type": "r3m", "name": "resnet50"},
    "MVP": {"type": "mvp", "name": "vitb-mae-egosoup"},
    "mcr": {"type": "mcr", "path": "/home/ubuntu/robots-pretrain-robots/mcr_resnet50.pth"},
    # CLIP models
    "CLIP-Base-32": {"type": "clip", "name": "ViT-B/32"},
    "CLIP-Base-16": {"type": "clip", "name": "ViT-B/16"},
    "CLIP-Large-14": {"type": "clip", "name": "ViT-L/14"},
    "CLIP-Large-336": {"type": "clip", "name": "ViT-L/14@336px"},
}


def load_model(model_name):
    """Factory function to load models based on configuration"""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")

    config = MODEL_CONFIGS[model_name]
    model_type = config["type"]

    if model_type == "timm":
        model = timm.create_model(config["name"], pretrained=config.get("pretrained", True))
    elif model_type == "clip":
        model, preprocess = clip.load(
            config["name"], device="cuda" if torch.cuda.is_available() else "cpu"
        )
        model = CLIPEncoder(model, preprocess)
    elif model_type == "hf":
        model = TransformerEncoder(config["name"])
    elif model_type == "hybridvit":
        model = HybridViTEncoder(config["name"])
    elif model_type == "vip":
        model = VIPEncoder()
    elif model_type == "r3m":
        model = R3MEncoder()
    elif model_type == "mvp":
        model = mvp.load(config["name"])
        model.freeze()
    elif model_type == "mcr":
        model = MCREncoder(config["path"])

    model.eval()
    return model


# Update the models dictionary to use the factory function
models = {name: lambda n=name: load_model(n) for name in MODEL_CONFIGS}
