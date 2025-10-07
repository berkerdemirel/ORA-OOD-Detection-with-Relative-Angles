import os

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from omegaconf import DictConfig
from util.data_loader import get_loader_in, get_loader_out
from util.model_loader import MODEL_REGISTRY, load_model

# Set up seeds for reproducibility
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)


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


def extract_features(model, inputs, model_arch, device):
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


def process_and_cache_features(
    model,
    loader,
    cache_dir,
    featdim,
    num_classes,
    batch_size,
    model_arch,
    device,
    is_id=False,
):
    """
    Processes a dataset, extracts features, and caches them to memory-mapped files.
    """
    print(f"Processing data and caching to {cache_dir}...")
    os.makedirs(cache_dir, exist_ok=True)

    # Create memory-mapped files
    feat_log = np.memmap(
        f"{cache_dir}/feat.mmap",
        dtype=float,
        mode="w+",
        shape=(len(loader.dataset), featdim),
    )
    score_log = np.memmap(
        f"{cache_dir}/score.mmap",
        dtype=float,
        mode="w+",
        shape=(len(loader.dataset), num_classes),
    )
    if is_id:
        label_log = np.memmap(
            f"{cache_dir}/label.mmap",
            dtype=float,
            mode="w+",
            shape=(len(loader.dataset),),
        )

    # Extract features
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            inputs = data[0].to(device)
            start_ind = batch_idx * batch_size
            end_ind = min((batch_idx + 1) * batch_size, len(loader.dataset))

            # Extract features and scores
            out, score = extract_features(model, inputs, model_arch, device)

            # Save results
            feat_log[start_ind:end_ind, :] = out.data.cpu().numpy()
            score_log[start_ind:end_ind, :] = score.data.cpu().numpy()
            if is_id:
                targets = data[1].to(device)
                label_log[start_ind:end_ind] = targets.data.cpu().numpy()

            if batch_idx % 100 == 0:
                print(f"  Processed {batch_idx}/{len(loader)} batches")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(args: DictConfig) -> None:
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    # Validate arguments
    if args.model_arch not in MODEL_REGISTRY:
        print(f"Error: Model architecture '{args.model_arch}' not supported.")
        print(f"Supported architectures: {list(MODEL_REGISTRY.keys())}")
        return

    # Get model configuration
    model_config = MODEL_REGISTRY[args.model_arch]
    featdim = model_config["featdim"]

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
        for split, in_loader in [("val", testloaderIn), ("train", trainloaderIn)]:
            cache_dir = f"cache/{args.in_dataset}_{split}_{args.name}_in"
            if FORCE_RUN or not os.path.exists(cache_dir):
                process_and_cache_features(
                    model,
                    in_loader,
                    cache_dir,
                    featdim,
                    num_classes,
                    batch_size,
                    args.model_arch,
                    device,
                    is_id=True,
                )
            else:
                print(f"Cache exists for {split} split, skipping processing.")

    # Process out-of-distribution data
    if OOD_RUN and args.out_datasets:
        for ood_dataset in args.out_datasets:
            cache_dir = f"cache/{ood_dataset}vs{args.in_dataset}_{args.name}_out"

            if os.path.exists(f"{cache_dir}/score.mmap") and not FORCE_RUN:
                print(f"Cache exists for {ood_dataset}, skipping processing.")
                continue

            # Get OOD data loader
            loader_test_dict = get_loader_out(
                args, dataset=(None, ood_dataset), split=("val")
            )
            out_loader = loader_test_dict.val_ood_loader

            process_and_cache_features(
                model,
                out_loader,
                cache_dir,
                featdim,
                num_classes,
                batch_size,
                args.model_arch,
                device,
                is_id=False,
            )


if __name__ == "__main__":
    main()
