from pathlib import Path
from mcr.models.models_mcr import MCR
from r3m import load_r3m
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import mvp
from vip import load_vip
import torch

# from src.models.loading_dino import DinoV2Encoder
import mcr
import timm
from transformers import (
    AutoImageProcessor,
    AutoModel,
    ViTHybridModel,
)
import clip
import numpy as np
from PIL import Image
from abc import ABC, abstractmethod
from torchvision import transforms

from r3m.models.models_r3m import R3M
from vip import VIP

from src.models.gradcam_module import (
    reshape_transform_swin,
    reshape_transform_ViT,
    reshape_transform_clip,
)
from vc_models.models.vit import model_utils

from data4robotics import load_vit, load_resnet18

from torch import nn


class BaseEncoder(ABC, torch.nn.Module):
    """Abstract base class for all encoders"""

    model_name: str

    def __init__(self):
        super().__init__()
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
            return torch.cat([self.preprocess_numpy(img) for img in batch], dim=0)
        else:
            raise ValueError(f"Invalid input shape: {batch.shape}")

    def __call__(self, x):
        """
        Args:
            x: numpy array of shape (H, W, 3) with values [0-255]
               or batch of arrays (B, H, W, 3)
            But if input is tensor we assume it's already preprocessed
        Returns:
            embeddings: torch tensor
        """
        if not isinstance(x, torch.Tensor):
            x = self.process_batch(x)
        h = self.model(x)
        return h

    def get_gradcam_target_layers(self):
        """Return target layers for GradCAM"""
        raise NotImplementedError("Subclasses must implement this method")

    def get_gradcam_transform(self):
        """Return reshape transform for GradCAM"""
        return None


class TimmEncoder(BaseEncoder):
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self._load_model()
        self._setup_preprocessing()

    def _load_model(self):
        self.model = timm.create_model(self.model_name, pretrained=self.pretrained, num_classes=0)
        self.model.eval()
        self.model.to(self.device)

    def _setup_preprocessing(self):
        data_cfg = timm.data.resolve_data_config(self.model.pretrained_cfg)
        self.preprocess = timm.data.create_transform(**data_cfg)

    def get_gradcam_target_layers(self):
        if isinstance(self.model, timm.models.mobilenetv3.MobileNetV3):
            return [self.model.blocks[-1][-1]]
        elif isinstance(self.model, timm.models.resnet.ResNet):
            return [self.model.layer4[-1]]
        elif isinstance(self.model, timm.models.efficientnet.EfficientNet):
            return [self.model.blocks[-1][-1]]
        elif isinstance(self.model, timm.models.vgg.VGG):
            return [self.model.features[-1]]
        elif isinstance(
            self.model, (timm.models.vision_transformer.VisionTransformer, timm.models.beit.Beit)
        ):
            return [self.model.blocks[-1].norm1]
        elif isinstance(self.model, timm.models.swin_transformer.SwinTransformer):
            return [self.model.layers[-1].blocks[-1].norm1]
        elif isinstance(self.model, timm.models.maxxvit.MaxxVit):
            return [self.model.stages[-1].blocks[-1].norm1]
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")

    def get_gradcam_transform(self):
        if isinstance(
            self.model, (timm.models.vision_transformer.VisionTransformer, timm.models.beit.Beit)
        ):
            return reshape_transform_ViT
        elif isinstance(self.model, timm.models.swin_transformer.SwinTransformer):
            return reshape_transform_swin
        else:
            return None


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
        if not isinstance(x, torch.Tensor):
            x = self.process_batch(x)
        return self.model(x)[0]  # Return last hidden states

    def get_gradcam_target_layers(self):
        return [self.model.encoder.layer[-1].norm1]

    def get_gradcam_transform(self):
        return reshape_transform_ViT


class R3MEncoder(BaseEncoder):
    """Wrapper for R3M models to maintain consistent interface"""

    def __init__(self, model_name):
        super().__init__()
        self.model_name = f"R3M-{model_name}"
        self.backbone = model_name
        self._load_model()
        self._setup_preprocessing()

    def _load_model(self):
        self.model: R3M = load_r3m(self.backbone).module
        self.model.eval()
        self.model.to(self.device)

    def _setup_preprocessing(self):
        r3m_norm = self.model.normlayer
        # This transform takes in a PIL image, resizes it to 224x224, converts it to a tensor,
        # and then normalizes it using R3M's normalization parameters
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                r3m_norm,
            ]
        )

    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            x = self.process_batch(x)
        h = self.model.convnet(x)
        return h

    def get_gradcam_target_layers(self):
        return [self.model.convnet.layer4[-1]]


class MVPEncoder(BaseEncoder):
    """Wrapper for MVP models to maintain consistent interface"""

    def __init__(self, model_name):
        super().__init__()
        self.model_name = f"MVP-{model_name}"
        self.backbone = model_name
        self._load_model()
        self._setup_preprocessing()

    def _load_model(self):
        self.model: timm.models.vision_transformer.VisionTransformer = mvp.load(self.backbone)

        self.model.eval()
        self.model.to(self.device)

    def _setup_preprocessing(self):
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        )

    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            x = self.process_batch(x)
        h = self.model(x)
        return h

    def get_gradcam_target_layers(self):
        return [self.model.blocks[-1].norm1]


class CLIPEncoder(BaseEncoder):
    """Wrapper for CLIP models to maintain consistent interface"""

    def __init__(self, model_name):
        super().__init__()
        self.model, self.preprocess = clip.load(
            model_name, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_name = model_name

    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            x = self.process_batch(x)
        h = self.model.encode_image(x)
        return h

    def get_gradcam_target_layers(self):
        return [self.model.visual.transformer.resblocks[-1].ln_1]
        # return [self.model.visual.ln_post]

    def get_gradcam_transform(self):
        return reshape_transform_clip


class VIPEncoder(BaseEncoder):
    """Wrapper for VIP models to maintain consistent interface"""

    def __init__(self):
        super().__init__()
        self._load_model()
        self._setup_preprocessing()
        self.model_name = "VIP-resnet50"

    def _load_model(self):
        self.model: VIP = load_vip().module
        self.model.eval()
        self.model.to(self.device)

    def _setup_preprocessing(self):
        vip_norm = self.model.normlayer
        # This transform takes in a PIL image, resizes it to 224x224, converts it to a tensor,
        # and then normalizes it using VIP's normalization parameters
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                vip_norm,
            ]
        )

    def get_gradcam_target_layers(self):
        return [self.model.convnet.layer4[-1]]

    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            x = self.process_batch(x)
        h = self.model.convnet(x)
        return h


class MCREncoder(BaseEncoder):
    """Wrapper for MCR models to maintain consistent interface"""

    def __init__(self, model_name):
        super().__init__()
        self.ckpt_path = Path(__file__).parent.parent.parent / "models/mcr/mcr_resnet50.pth"
        self.model_name = "MCR-resnet50"
        self._load_model()
        self._setup_preprocessing()

    def _load_model(self):
        self.model: MCR = mcr.load_mcr(ckpt_path=self.ckpt_path)
        self.model.eval()
        self.model.to(self.device)

    def _setup_preprocessing(self):
        # This transform takes in a PIL image, resizes it to 224x224, converts it to a tensor,
        # and then normalizes it using mcr's normalization parameters
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            x = self.process_batch(x)
        h = self.model(x)
        return h

    def get_gradcam_target_layers(self):
        return [self.model.layer4[-1]]


class VC1Encoder(BaseEncoder):
    """Wrapper for CV1 models to maintain consistent interface"""

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self._load_model()
        self._setup_preprocessing()

    def _load_model(self):

        self.model, *_ = model_utils.load_model(self.model_name)
        self.model.eval()
        self.model.to(self.device)

    def _setup_preprocessing(self):
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            x = self.process_batch(x)
        h = self.model(x)
        return h

    def get_gradcam_target_layers(self):
        # As a ViT return norm1 of the last block
        return [self.model.blocks[-1].norm1]

    def get_gradcam_transform(self):
        return reshape_transform_ViT


class HRPEncoder(BaseEncoder):
    """Wrapper for HRP models to maintain consistent interface"""

    def __init__(self, model_name):
        super().__init__()
        self.model_name = f"HRP-{model_name}"
        self._load_model()
        self._setup_preprocessing()

    def _load_model(self):
        _, self.model = load_resnet18() if self.model_name.endswith("resnet18") else load_vit()
        if hasattr(self.model, "_model"):
            self.model = self.model._model
        self.model.eval()
        self.model.to(self.device)

    def _setup_preprocessing(self):
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def get_gradcam_target_layers(self):
        if self.model_name.endswith("resnet18"):

            return [self.model.layer4[-1]]
        else:
            return [self.model.blocks[-1].norm1]

    def get_gradcam_transform(self):
        if self.model_name.endswith("resnet18"):
            return None
        else:
            return reshape_transform_ViT

    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            x = self.process_batch(x)
        h = self.model(x)
        return h


class DinoV2Encoder(BaseEncoder):
    """Wrapper for DinoV2 models to maintain consistent interface"""

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self._load_model()
        self._setup_preprocessing()

    def _load_model(self):
        self.model: nn.Module = torch.hub.load("facebookresearch/dinov2", self.model_name)
        self.model.eval()
        self.model.to(self.device)

    def _setup_preprocessing(self):
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        )

    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            x = self.process_batch(x)
        h = self.model(x)
        return h

    def get_gradcam_target_layers(self):
        return [self.model.blocks[-1].norm1]

    def get_gradcam_transform(self):
        return reshape_transform_ViT


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
    # "DinoV2": {"type": "timm", "name": "vit_base_patch14_dinov2"},
    # "CoAtNet": {"type": "timm", "name": "coatnet_3_rw_224"},
    # HuggingFace models
    # Robot learning models
    "VIP": {"type": "vip"},
    "R3M18": {"type": "r3m", "name": "resnet18"},
    "R3M34": {"type": "r3m", "name": "resnet34"},
    "R3M50": {"type": "r3m", "name": "resnet50"},
    "MCR": {"type": "mcr", "name": "mcr"},
    "VC1-B": {"type": "vc1", "name": "vc1_vitb"},
    "HRP-ResNet18": {"type": "hrp", "name": "resnet18"},
    "HRP-ViT": {"type": "hrp", "name": "vit"},
    # CLIP models
    "CLIP-Base-16": {"type": "clip", "name": "ViT-B/16"},
    "CLIP-Base-32": {"type": "clip", "name": "ViT-B/32"},
    "CLIP-Large-14": {"type": "clip", "name": "ViT-L/14"},
    # "BEiT": {"type": "timm", "name": "beit_large_patch16_224"},
    # "DinoV2-B": {"type": "hf", "name": "facebook/dinov2-base"},
    "MVP": {"type": "mvp", "name": "vitb-mae-egosoup"},
    "VC1-L": {"type": "vc1", "name": "vc1_vitl"},
    "DinoV2-B": {"type": "dinov2", "name": "dinov2_vitb14"},
}


def load_model(model_name):
    """Factory function to load models based on configuration"""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")

    config = MODEL_CONFIGS[model_name]
    model_type = config["type"]

    if model_type == "timm":
        model = TimmEncoder(config["name"])
    elif model_type == "clip":
        model = CLIPEncoder(config["name"])
    elif model_type == "hf":
        model = TransformerEncoder(config["name"])
    elif model_type == "vip":
        model = VIPEncoder()
    elif model_type == "r3m":
        model = R3MEncoder(config["name"])
    elif model_type == "mvp":
        # raise NotImplementedError("MVP models are not supported as the weights are not available")
        model = MVPEncoder(config["name"])
    elif model_type == "mcr":
        model = MCREncoder(config["name"])
    elif model_type == "vc1":
        model = VC1Encoder(config["name"])
    elif model_type == "hrp":
        model = HRPEncoder(config["name"])
    elif model_type == "dinov2":
        model = DinoV2Encoder(config["name"])

    return model


# Update the models dictionary to use the factory function
models = {name: lambda n=name: load_model(n) for name in MODEL_CONFIGS}
