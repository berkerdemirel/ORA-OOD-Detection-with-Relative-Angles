import os
import pdb

import timm
import torch
from open_clip import create_model_from_pretrained
from pytorch_pretrained_vit import ViT

# import models.densenet as dn
# import models.wideresnet as wn


# Model registry - maps model architectures to their specific configurations
MODEL_REGISTRY = {
    # Standard models
    "resnet18": {
        "featdim": 512,
        "img_size": 32,
        "feature_method": "penult_feature",
        "load_method": "standard",
    },
    "resnet18-supcon": {
        "featdim": 512,
        "img_size": 32,
        "feature_method": "penult_feature",
        "load_method": "standard",
    },
    "resnet50": {
        "featdim": 2048,
        "img_size": 224,
        "feature_method": "features",
        "load_method": "standard",
    },
    "resnet50-supcon": {
        "featdim": 2048,
        "img_size": 224,
        "feature_method": "encoder",
        "load_method": "standard",
    },
    # Vision Transformer models
    "vit": {
        "featdim": 768,
        "img_size": 384,  # ViT uses 384x384 input
        "feature_method": "custom_forward",
        "load_method": "vit",
    },
    # Mobile models
    "mobile": {
        "featdim": 1280,
        "img_size": 224,
        "feature_method": "features",
        "load_method": "mobile",
    },
    # ConvNext models
    "convnext": {
        "featdim": 1024,
        "img_size": 224,
        "feature_method": "forward_features",
        "load_method": "timm_convnext",
    },
    "convnextpre": {
        "featdim": 1024,
        "img_size": 224,
        "feature_method": "forward_features",
        "load_method": "timm_convnext_pre",
    },
    # Swin Transformer models
    "swin": {
        "featdim": 1024,
        "img_size": 256,
        "feature_method": "forward_features",
        "load_method": "timm_swin",
    },
    "swinpre": {
        "featdim": 1024,
        "img_size": 256,
        "feature_method": "forward_features",
        "load_method": "timm_swin_pre",
    },
    # DeiT models
    "deit": {
        "featdim": 768,
        "img_size": 224,
        "feature_method": "forward_features",
        "load_method": "timm_deit",
    },
    "deitpre": {
        "featdim": 768,
        "img_size": 224,
        "feature_method": "forward_features",
        "load_method": "timm_deit_pre",
    },
    # DenseNet models
    "densenet": {
        "featdim": 1024,
        "img_size": 224,
        "feature_method": "forward_features",
        "load_method": "timm_densenet",
    },
    # EVA models
    "eva": {
        "featdim": 768,
        "img_size": 448,
        "feature_method": "forward_features",
        "load_method": "timm_eva",
    },
    # CLIP models
    "clip": {
        "featdim": 1024,  # CLIP ViT-B/32 embedding size
        "img_size": 224,
        "feature_method": "encode_image",
        "load_method": "clip",
    },
}


def load_model(args, num_classes):
    """
    Load a model based on the architecture specified in args
    """
    model_arch = args.model_arch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model architecture: {model_arch}")

    config = MODEL_REGISTRY[model_arch]
    load_method = config["load_method"]

    # Use the model_loader utility when possible
    if load_method == "standard":
        if args.in_dataset == "imagenet":
            if args.model_arch == "resnet50":
                from models.resnet import resnet50

                model = resnet50(num_classes=num_classes, pretrained=True)
            elif args.model_arch == "resnet50-supcon":
                from models.resnet_supcon import SupConResNet

                model = SupConResNet(num_classes=num_classes)
                checkpoint = torch.load("./ckpt/ImageNet_resnet50_supcon.pth")
                state_dict = {
                    str.replace(k, "module.", ""): v
                    for k, v in checkpoint["model"].items()
                }
                checkpoint_linear = torch.load(
                    "./ckpt/ImageNet_resnet50_supcon_linear.pth"
                )
                state_dict["fc.weight"] = checkpoint_linear["model"]["fc.weight"]
                state_dict["fc.bias"] = checkpoint_linear["model"]["fc.bias"]
                model.load_state_dict(state_dict)
        else:
            # create model for cifar
            if args.model_arch == "resnet18":
                from models.resnet import resnet18_cifar

                model = resnet18_cifar(
                    num_classes=num_classes, method=args.method, p=args.p
                )
                checkpoint = torch.load("./ckpt/CIFAR10_resnet18.pth.tar")
                checkpoint = {
                    "state_dict": {
                        key.replace("module.", ""): value
                        for key, value in checkpoint["state_dict"].items()
                    }
                }
                model.load_state_dict(checkpoint["state_dict"])
            elif args.model_arch == "resnet18-supcon":
                from models.resnet_ss import resnet18_cifar

                model = resnet18_cifar(num_classes=num_classes, method=args.method)
                checkpoint = torch.load("./ckpt/CIFAR10_resnet18_supcon.pth.tar")
                checkpoint = {
                    "state_dict": {
                        key.replace("module.", ""): value
                        for key, value in checkpoint["state_dict"].items()
                    }
                }
                checkpoint_linear = torch.load(
                    "ckpt/CIFAR10_resnet18_supcon_linear.pth"
                )
                checkpoint["state_dict"]["fc.weight"] = checkpoint_linear["model"][
                    "fc.weight"
                ]
                checkpoint["state_dict"]["fc.bias"] = checkpoint_linear["model"][
                    "fc.bias"
                ]
                model.load_state_dict(checkpoint["state_dict"])
            elif args.model_arch == "densenet":
                from models.densenet_ss import DenseNet3

                model = DenseNet3(num_classes=num_classes)
                checkpoint = torch.load("./ckpt/CIFAR10_densenet121.pth.tar")
                checkpoint = {
                    "state_dict": {
                        key.replace("module.", ""): value
                        for key, value in checkpoint["state_dict"].items()
                    }
                }
                model.load_state_dict(checkpoint["state_dict"])
            else:
                assert False, "Not supported model arch: {}".format(args.model_arch)

    # Model-specific loading methods
    elif load_method == "vit":
        from pytorch_pretrained_vit import ViT

        model = ViT("B_16_imagenet1k", pretrained=True)

    elif load_method == "mobile":
        model = torch.hub.load(
            "pytorch/vision:v0.10.0", "mobilenet_v2", pretrained=True
        )

    elif load_method == "timm_convnext":
        model = timm.create_model("convnext_base", pretrained=True)

    elif load_method == "timm_convnext_pre":
        model = timm.create_model("convnext_base_in22ft1k", pretrained=True)

    elif load_method == "timm_swin":
        model = timm.create_model("swinv2_base_window16_256", pretrained=True)

    elif load_method == "timm_swin_pre":
        model = timm.create_model(
            "swinv2_base_window12to16_192to256_22kft1k", pretrained=True
        )

    elif load_method == "timm_deit":
        model = timm.create_model("deit3_base_patch16_224", pretrained=True)

    elif load_method == "timm_deit_pre":
        model = timm.create_model("deit3_base_patch16_224_in21ft1k", pretrained=True)

    elif load_method == "timm_densenet":
        model = timm.create_model("densenet121", pretrained=True)

    elif load_method == "timm_eva":
        model = timm.create_model(
            "eva02_base_patch14_448.mim_in22k_ft_in1k", pretrained=True
        )

    elif load_method == "clip":
        from open_clip import create_model_from_pretrained

        model, _ = create_model_from_pretrained("hf-hub:apple/DFN5B-CLIP-ViT-H-14-384")

    else:
        raise ValueError(f"Unknown model loading method: {load_method}")

    model.to(device)
    model.eval()
    # get the number of model parameters
    print(
        "Number of model parameters: {}".format(
            sum([p.data.nelement() for p in model.parameters()])
        )
    )
    return model
