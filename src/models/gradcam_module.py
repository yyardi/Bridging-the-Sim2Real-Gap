import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import resnet18

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def prepare_image_resnet(img_array):
    """Prepare image for model input"""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img_array).unsqueeze(0)
    return img_tensor
