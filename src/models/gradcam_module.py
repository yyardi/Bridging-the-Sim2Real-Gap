import torch
import torchvision.transforms as transforms


def prepare_image_resnet(img_array, size=224, device="cuda"):
    """Prepare image for model input"""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img_array).unsqueeze(0)
    return img_tensor.to(device)

def prepare_image_dino(img_array, size=224, device="cuda"):
    """Prepare image for DinoV2 model input with slight modifications"""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        # Using the same normalization as prepare_image_resnet
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img_array).unsqueeze(0)
    return img_tensor.to(device)

def reshape_transform(tensor, height=14, width=14):
    """Reshape transform for Vision Transformer models"""
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result