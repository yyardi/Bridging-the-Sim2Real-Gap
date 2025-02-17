import math
import torch
import torchvision.transforms as transforms


def prepare_image(img_array, size=224, device="cuda"):
    """Prepare image for model input"""
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img_tensor = transform(img_array).unsqueeze(0)
    return img_tensor.to(device)


def reshape_transform_ViT(tensor):
    result = tensor[:, 1:, :]
    height = width = int(math.isqrt(result.shape[1]))
    result = result.reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def reshape_transform_swin(tensor, height=7, width=7):
    result = tensor.reshape(tensor.shape[0], height, width, tensor.shape[-1])

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def reshape_transform_clip(tensor):
    result = tensor[1:, :, :]

    height = width = int(math.isqrt(result.shape[0]))

    result = result.reshape(1, height, width, tensor.shape[-1])

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.permute(0, 3, 1, 2)
    return result

