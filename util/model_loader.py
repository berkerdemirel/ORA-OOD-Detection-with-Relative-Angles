import os
import pdb

# import models.densenet as dn
# import models.wideresnet as wn


import torch
import timm

timm_models = {
    "BiT_m": {"config": {"model_name": "resnetv2_101x1_bitm", "pretrained": True}},
    "BiT_s": {
        "config": {
            "model_name": "resnetv2_101x1_bitm",
            "checkpoint_path": "./model_weights/checkpoints/BiT-S-R101x1.npz",
        }
    },
    "vit_base_patch16_224_21kpre": {
        "config": {"model_name": "vit_base_patch16_224", "pretrained": True}
    },
    "vit_base_patch16_384_21kpre": {
        "config": {"model_name": "vit_base_patch16_384", "pretrained": True}
    },
    "convnext_base_in22ft1k": {
        "config": {"model_name": "convnext_base_in22ft1k", "pretrained": True}
    },
    "convnext_base": {"config": {"model_name": "convnext_base", "pretrained": True}},
    "convnext_tiny-22k": {
        "config": {"model_name": "convnext_tiny_384_in22ft1k", "pretrained": True}
    },
    "deit3_base_patch16_224": {
        "config": {"model_name": "deit3_base_patch16_224", "pretrained": True}
    },
    "deit3_base_patch16_224_in21ft1k": {
        "config": {"model_name": "deit3_base_patch16_224_in21ft1k", "pretrained": True}
    },
    "tf_efficientnetv2_m": {
        "config": {"model_name": "tf_efficientnetv2_m", "pretrained": True}
    },
    "tf_efficientnetv2_m_in21ft1k": {
        "config": {"model_name": "tf_efficientnetv2_m_in21ft1k", "pretrained": True}
    },
    "swinv2-22k": {
        "config": {
            "model_name": "swinv2_base_window12to16_192to256_22kft1k",
            "pretrained": True,
        }
    },
    "swinv2-1k": {
        "config": {"model_name": "swinv2_base_window16_256", "pretrained": True}
    },
    "deit3-384-22k": {
        "config": {"model_name": "deit3_base_patch16_384_in21ft1k", "pretrained": True}
    },
    "deit3-384-1k": {
        "config": {"model_name": "deit3_base_patch16_384", "pretrained": True}
    },
    "tf_efficientnet_b7_ns": {
        "config": {"model_name": "tf_efficientnet_b7_ns", "pretrained": True}
    },
    "tf_efficientnet_b7": {
        "config": {"model_name": "tf_efficientnet_b7", "pretrained": True}
    },
    "resnet50": {"config": {"model_name": "resnet50", "pretrained": True}},
    "efficientnet_b0": {
        "config": {"model_name": "efficientnet_b0", "pretrained": True}
    },
    "vit_base_patch16_384_laion2b_in12k_in1k": {
        "config": {
            "model_name": "vit_base_patch16_clip_384.laion2b_ft_in12k_in1k",
            "pretrained": True,
        }
    },
    "vit_base_patch16_384_laion2b_in1k": {
        "config": {
            "model_name": "vit_base_patch16_clip_384.laion2b_ft_in1k",
            "pretrained": True,
        }
    },
    "vit_base_patch16_384_openai_in12k_in1k": {
        "config": {
            "model_name": "vit_base_patch16_clip_384.openai_ft_in12k_in1k",
            "pretrained": True,
        }
    },
    "vit_base_patch16_384_openai_in1k": {
        "config": {
            "model_name": "vit_base_patch16_clip_384.openai_ft_in1k",
            "pretrained": True,
        }
    },
    "vit_base_patch16_384": {
        "config": {
            "model_name": "vit_base_patch16_384.augreg_in1k",
            "pretrained": True,
        },
        "batch_size": 128,
        "server": "curie",
    },
    "xcit_medium_24_p16_224_dist": {
        "config": {"model_name": "xcit_medium_24_p16_224_dist", "pretrained": True}
    },
    "xcit_medium_24_p16_224": {
        "config": {"model_name": "xcit_medium_24_p16_224", "pretrained": True}
    },
    "eva_base": {
        "config": {
            "model_name": "eva02_base_patch14_448.mim_in22k_ft_in1k",
            "pretrained": True,
        }
    },
}


def get_model(args, num_classes, load_ckpt=True, load_epoch=None):
    if args.in_dataset == "imagenet":
        if args.model_arch == "resnet50":
            from models.resnet import resnet50

            model = resnet50(num_classes=num_classes, pretrained=True)
        elif args.model_arch == "resnet50-supcon":
            from models.resnet_supcon import SupConResNet

            model = SupConResNet(num_classes=num_classes)
            checkpoint = torch.load("./ckpt/ImageNet_resnet50_supcon.pth")
            state_dict = {
                str.replace(k, "module.", ""): v for k, v in checkpoint["model"].items()
            }
            checkpoint_linear = torch.load("./ckpt/ImageNet_resnet50_supcon_linear.pth")
            state_dict["fc.weight"] = checkpoint_linear["model"]["fc.weight"]
            state_dict["fc.bias"] = checkpoint_linear["model"]["fc.bias"]
            model.load_state_dict(state_dict)
        elif args.model_arch == "vit":
            from pytorch_pretrained_vit import ViT

            model = ViT("B_16_imagenet1k", pretrained=True)
        elif args.model_arch == "mobile":
            from transformers import MobileNetV2ForImageClassification

            # model = MobileNetV2ForImageClassification.from_pretrained("google/mobilenet_v2_1.0_224")
            model = torch.hub.load(
                "pytorch/vision:v0.10.0", "mobilenet_v2", pretrained=True
            )
        elif args.model_arch == "convnext":
            model_name = "convnext_base"
            model = timm.create_model(**timm_models[model_name]["config"])

        elif args.model_arch == "convnextpre":
            model_name = "convnext_base_in22ft1k"
            model = timm.create_model(**timm_models[model_name]["config"])

        elif args.model_arch == "swin":
            model_name = "swinv2-1k"
            model = timm.create_model(**timm_models[model_name]["config"])

        elif args.model_arch == "swinpre":
            model_name = "swinv2-22k"
            model = timm.create_model(**timm_models[model_name]["config"])
        elif args.model_arch == "deit":
            model_name = "deit3_base_patch16_224"
            model = timm.create_model(**timm_models[model_name]["config"])
        elif args.model_arch == "deitpre":
            model_name = "deit3_base_patch16_224_in21ft1k"
            model = timm.create_model(**timm_models[model_name]["config"])
        elif args.model_arch == "eva":
            model_name = "eva_base"
            model = timm.create_model(**timm_models[model_name]["config"])
        elif args.model_arch == "densenet":
            model = timm.create_model("densenet121", pretrained=True)

    else:
        # create model
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
            checkpoint_linear = torch.load("ckpt/CIFAR10_resnet18_supcon_linear.pth")
            checkpoint["state_dict"]["fc.weight"] = checkpoint_linear["model"][
                "fc.weight"
            ]
            checkpoint["state_dict"]["fc.bias"] = checkpoint_linear["model"]["fc.bias"]
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

        # if load_ckpt:
        #    epoch = args.epochs
        #    if load_epoch is not None:
        #        epoch = load_epoch
        # checkpoint = torch.load("./checkpoints/{in_dataset}/{model_arch}/checkpoint_{epochs}.pth.tar".format(in_dataset=args.in_dataset, model_arch=args.name, epochs=epoch))
        # checkpoint = torch.load("./checkpoints/{in_dataset}/{model_arch}/checkpoint_{epochs}.pth.tar".format(in_dataset=args.in_dataset, model_arch=args.name, epochs=epoch), map_location='cpu')
        # checkpoint = torch.load("/home/litianl/Neural-Collapse/model_weights/resnet_adam_sota_cifar10/epoch_{epochs}.pth".format(epochs=epoch), map_location='cpu')
        # pdb.set_trace()
        # checkpoint = {'state_dict': {key.replace("module.", ""): value for key, value in checkpoint['state_dict'].items()}}
        # model.load_state_dict(checkpoint['state_dict'])
        #    model.load_state_dict(checkpoint)

    model.cuda()
    model.eval()
    # get the number of model parameters
    print(
        "Number of model parameters: {}".format(
            sum([p.data.nelement() for p in model.parameters()])
        )
    )
    return model
