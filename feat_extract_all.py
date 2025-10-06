#!/usr/bin/env python3
"""
Universal Feature Extractor for various model architectures

This script extracts features from both in-distribution and out-of-distribution
datasets using a variety of model architectures. It's a consolidated version of
multiple model-specific extraction scripts.

Usage:
  python feat_extract_all.py --model-arch [MODEL_TYPE] --in-dataset [DATASET]
                         --out-datasets [OOD_DATASET1 OOD_DATASET2 ...]
                         --name [EXPERIMENT_NAME] --batch-size [BATCH_SIZE]
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import timm
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from util.args_loader import get_args
from util.data_loader import get_loader_in, get_loader_out
from util.model_loader import get_model

# Set up seeds for reproducibility
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model registry - maps model architectures to their specific configurations
MODEL_REGISTRY = {
    # Standard models
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


def vit_forward_features(model, x):
    """Custom feature extraction for ViT models"""
    b, c, fh, fw = x.shape
    x = model.patch_embedding(x)
    x = x.flatten(2).transpose(1, 2)

    if hasattr(model, "class_token"):
        x = torch.cat((model.class_token.expand(b, -1, -1), x), dim=1)

    if hasattr(model, "positional_embedding"):
        x = model.positional_embedding(x)

    x = model.transformer(x)

    if hasattr(model, "pre_logits"):
        x = model.pre_logits(x)
        x = torch.tanh(x)

    if hasattr(model, "fc"):
        x = model.norm(x)[:, 0]

    return x


def load_model(args, num_classes):
    """
    Load a model based on the architecture specified in args
    """
    model_arch = args.model_arch

    if model_arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model architecture: {model_arch}")

    config = MODEL_REGISTRY[model_arch]
    load_method = config["load_method"]

    # Use the model_loader utility when possible
    if load_method == "standard":
        model = get_model(args, num_classes, load_ckpt=True)

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

    return model.to(device)


def extract_features(model, inputs, model_arch):
    """
    Extract features from inputs using the appropriate method based on model architecture
    """
    config = MODEL_REGISTRY[model_arch]
    feature_method = config["feature_method"]

    # Resize inputs if needed
    if model_arch == "vit" or model_arch == "eva":
        resize_transform = transforms.Resize((config["img_size"], config["img_size"]))
        inputs = torch.stack([resize_transform(image) for image in inputs])

    # Extract features using the appropriate method
    if feature_method == "encoder":
        out = model.encoder(inputs)
    elif feature_method == "features":
        out = model.features(inputs)
    elif feature_method == "forward_features":
        out = model.forward_features(inputs)
    elif feature_method == "custom_forward":
        out = vit_forward_features(model, inputs)
    elif feature_method == "encode_image":
        out = model.encode_image(inputs)
    else:
        raise ValueError(f"Unknown feature extraction method: {feature_method}")

    # Handle feature shape if necessary
    if len(out.shape) > 2:
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)

    # Get classification scores
    try:
        # Some models like CLIP might not have a classifier
        if hasattr(model, "fc"):
            score = model.fc(out)
        elif model_arch == "densenet":
            out = model.global_pool(out)
            out = model.head_drop(out)
            score = model.classifier(out)
        elif model_arch == "mobile":
            score = model.classifier(out)
        elif model_arch == "convnext" or model_arch == "convnextpre":
            score = model(inputs)  # Use full forward pass for score
        else:
            # For models without a classifier (like CLIP), create dummy scores
            score = torch.zeros((out.size(0), 1000)).to(device)
    except Exception as e:
        print(f"Warning: Could not get classification scores: {e}")
        score = torch.zeros((out.size(0), 1000)).to(device)

    return out, score


def main():
    args = get_args()

    # Validate arguments
    if args.model_arch not in MODEL_REGISTRY:
        print(f"Error: Model architecture '{args.model_arch}' not supported.")
        print(f"Supported architectures: {list(MODEL_REGISTRY.keys())}")
        return

    # Get model configuration
    model_config = MODEL_REGISTRY[args.model_arch]
    featdim = model_config["featdim"]

    # Get data loaders
    # Update the imagenet_root if it's the default value
    if args.in_dataset == "imagenet" and args.imagenet_root == "./datasets/imagenet/":
        args.imagenet_root = "../data/imagenet/"

    loader_in_dict = get_loader_in(args, config_type="eval", split=("train", "val"))
    trainloaderIn, testloaderIn = loader_in_dict.train_loader, loader_in_dict.val_loader
    num_classes = loader_in_dict.num_classes

    # Load model
    model = load_model(args, num_classes)
    model.eval()

    # Configuration
    batch_size = args.batch_size
    FORCE_RUN = True  # Set to True to force recomputation of features
    ID_RUN = True  # Set to True to run on in-distribution data
    OOD_RUN = True  # Set to True to run on out-of-distribution data

    # Process in-distribution data
    if ID_RUN:
        # if False:
        for split, in_loader in [("val", testloaderIn), ("train", trainloaderIn)]:
            cache_dir = f"cache/{args.in_dataset}_{split}_{args.name}_in"

            if FORCE_RUN or not os.path.exists(cache_dir):
                print(f"Processing in-distribution data for {split} split...")
                os.makedirs(cache_dir, exist_ok=True)

                # Create memory-mapped files for features, scores and labels
                feat_log = np.memmap(
                    f"{cache_dir}/feat.mmap",
                    dtype=float,
                    mode="w+",
                    shape=(len(in_loader.dataset), featdim),
                )
                score_log = np.memmap(
                    f"{cache_dir}/score.mmap",
                    dtype=float,
                    mode="w+",
                    shape=(len(in_loader.dataset), num_classes),
                )
                label_log = np.memmap(
                    f"{cache_dir}/label.mmap",
                    dtype=float,
                    mode="w+",
                    shape=(len(in_loader.dataset),),
                )

                # Extract features
                with torch.no_grad():
                    for batch_idx, (inputs, targets) in enumerate(in_loader):
                        inputs, targets = inputs.to(device), targets.to(device)
                        start_ind = batch_idx * batch_size
                        end_ind = min(
                            (batch_idx + 1) * batch_size, len(in_loader.dataset)
                        )

                        # Extract features and scores
                        out, score = extract_features(model, inputs, args.model_arch)

                        # Save results
                        feat_log[start_ind:end_ind, :] = out.data.cpu().numpy()
                        label_log[start_ind:end_ind] = targets.data.cpu().numpy()
                        score_log[start_ind:end_ind] = score.data.cpu().numpy()

                        if batch_idx % 100 == 0:
                            print(f"  Processed {batch_idx}/{len(in_loader)} batches")
            else:
                print(f"Cache exists for {split} split, skipping processing.")

    # Process out-of-distribution data
    if OOD_RUN and args.out_datasets:
        for ood_dataset in args.out_datasets:
            cache_dir = f"cache/{ood_dataset}vs{args.in_dataset}_{args.name}_out"

            # Skip if cache exists
            if os.path.exists(f"{cache_dir}/score.mmap") and not FORCE_RUN:
                print(f"Cache exists for {ood_dataset}, skipping processing.")
                continue

            print(f"Processing OOD data for {ood_dataset}...")

            # Get OOD data loader
            loader_test_dict = get_loader_out(
                args, dataset=(None, ood_dataset), split=("val")
            )
            out_loader = loader_test_dict.val_ood_loader

            if FORCE_RUN or not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)

                # Create memory-mapped files for features and scores
                ood_feat_log = np.memmap(
                    f"{cache_dir}/feat.mmap",
                    dtype=float,
                    mode="w+",
                    shape=(len(out_loader.dataset), featdim),
                )
                ood_score_log = np.memmap(
                    f"{cache_dir}/score.mmap",
                    dtype=float,
                    mode="w+",
                    shape=(len(out_loader.dataset), num_classes),
                )

                # Extract features
                with torch.no_grad():
                    for batch_idx, (inputs, _) in enumerate(out_loader):
                        inputs = inputs.to(device)
                        start_ind = batch_idx * batch_size
                        end_ind = min(
                            (batch_idx + 1) * batch_size, len(out_loader.dataset)
                        )

                        # Extract features and scores
                        out, score = extract_features(model, inputs, args.model_arch)

                        # Save results
                        ood_feat_log[start_ind:end_ind, :] = out.data.cpu().numpy()
                        ood_score_log[start_ind:end_ind] = score.data.cpu().numpy()

                        if batch_idx % 100 == 0:
                            print(f"  Processed {batch_idx}/{len(out_loader)} batches")
            else:
                print(f"Cache exists for {ood_dataset}, skipping processing.")


if __name__ == "__main__":
    main()
