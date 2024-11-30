from r3m import load_r3m
import mvp
from vip import load_vip
from src.models.loading_dino import DinoV2Encoder
import mcr
import timm

def load_effnet():
    model = timm.create_model("efficientnet_b0", pretrained=True)
    model.eval()
    return model

def load_mobilenet():
    model = timm.create_model("mobilenetv3_large_100", pretrained=True)
    model.eval()
    return model

def load_vgg16():
    model = timm.create_model("vgg16", pretrained=True)
    model.eval()
    return model

def load_vgg19():
    model = timm.create_model("vgg19", pretrained=True)
    model.eval()
    return model

def load_resnet18():
    model = timm.create_model("resnet18", pretrained=True)
    model.eval()
    return model

def load_resnet34():
    model = timm.create_model("resnet34", pretrained=True)
    model.eval()
    return model

def load_resnet50():
    model = timm.create_model("resnet50", pretrained=True)
    model.eval()
    return model

def load_resnet101():
    model = timm.create_model("resnet101", pretrained=True)
    model.eval()
    return model


def load_vip_model():
    vip = load_vip().cuda().eval()
    vip = vip.module
    return vip


def load_R3M18():
    r3m = load_r3m("resnet18").cuda().eval()
    r3m = r3m.module
    return r3m


def load_R3M34():
    r3m = load_r3m("resnet34").cuda().eval()
    r3m = r3m.module
    return r3m


def load_R3M50():
    r3m = load_r3m("resnet50").cuda().eval()
    r3m = r3m.module
    return r3m


def load_mvp():
    # Load the encoder with pretrained weights
    mvpmodel = mvp.load("vitb-mae-egosoup")
    mvpmodel.freeze()

    return mvpmodel

def load_mcr():
    mcrencoder = mcr.load_mcr(ckpt_path="/home/ubuntu/robots-pretrain-robots/mcr_resnet50.pth")
    return mcrencoder

# Dictionary to store the models
models = {
    "VIP": load_vip_model,
    "R3M18": load_R3M18,
    "R3M34": load_R3M34,
    "R3M50": load_R3M50,
    "MVP": load_mvp,
    "dinov2": lambda: DinoV2Encoder("dinov2_vits14"),
    "mcr": load_mcr,
    "ResNet18": load_resnet18,
    "ResNet34": load_resnet34,
    "ResNet50": load_resnet50,
    "ResNet101": load_resnet101,
    "EfficientNetB0": load_effnet,
    "MobileNetv3": load_mobilenet,
    "vgg16": load_vgg16,
    "vgg19": load_vgg19,

}
