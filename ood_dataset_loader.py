"""
Helper script to load actual OOD datasets for MNIST experiments.
You'll need to implement the actual dataset loading based on your data structure.
"""

import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class OODDataset(Dataset):
    """Generic OOD dataset class"""

    def __init__(self, data_path, transform=None, target_size=(28, 28)):
        self.data_path = data_path
        self.transform = transform
        self.target_size = target_size
        self.images = self._load_images()

    def _load_images(self):
        """Load images from the dataset path - implement based on your data structure"""
        # This is a placeholder - you'll need to implement based on your actual data structure
        images = []
        if os.path.exists(self.data_path):
            for file in os.listdir(self.data_path):
                if file.endswith((".png", ".jpg", ".jpeg")):
                    images.append(os.path.join(self.data_path, file))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("L")  # Convert to grayscale
        image = image.resize(self.target_size)

        if self.transform:
            image = self.transform(image)

        return image, 0  # Return dummy label


def load_actual_ood_datasets(data_root="~/data"):
    """
    Load actual OOD datasets from your data directory.
    Modify the paths based on your actual data structure.
    """

    # Transform to match MNIST preprocessing
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    ood_loaders = {}

    # Dataset paths - modify these based on your actual data structure
    dataset_paths = {
        "inat": f"{data_root}/iNaturalist",
        "sun50": f"{data_root}/SUN",
        "places50": f"{data_root}/Places",
        "dtd": f"{data_root}/dtd",
        "ssbhard": f"{data_root}/ssbhard",  # You might need to check the actual folder name
        "ninco": f"{data_root}/ninco",
    }

    ood_dataset_size = {
        "inat": 10000,
        "sun50": 10000,
        "places50": 10000,
        "dtd": 5640,
        "ssbhard": 49000,
        "ninco": 5878,
    }

    for dataset_name, dataset_path in dataset_paths.items():
        if os.path.exists(dataset_path):
            print(f"Loading {dataset_name} from {dataset_path}")
            dataset = OODDataset(dataset_path, transform=transform)

            # Limit to specified size if dataset is larger
            if len(dataset) > ood_dataset_size[dataset_name]:
                # Create subset
                indices = torch.randperm(len(dataset))[: ood_dataset_size[dataset_name]]
                dataset = torch.utils.data.Subset(dataset, indices)

            ood_loaders[dataset_name] = DataLoader(
                dataset, batch_size=128, shuffle=False, num_workers=2
            )
        else:
            print(
                f"Warning: {dataset_name} path {dataset_path} not found. Creating dummy data."
            )
            # Fallback to dummy data
            dummy_data = torch.randn(ood_dataset_size[dataset_name], 1, 28, 28)
            dummy_dataset = torch.utils.data.TensorDataset(
                dummy_data, torch.zeros(ood_dataset_size[dataset_name])
            )
            ood_loaders[dataset_name] = DataLoader(
                dummy_dataset, batch_size=128, shuffle=False, num_workers=2
            )

    return ood_loaders


def load_specific_ood_datasets():
    """
    More specific loading functions for each dataset type.
    Implement these based on your actual data formats.
    """

    def load_inat():
        # Implement iNaturalist loading
        pass

    def load_sun50():
        # Implement SUN50 loading
        pass

    def load_places50():
        # Implement Places50 loading
        pass

    def load_dtd():
        # Implement DTD (Describable Textures Dataset) loading
        pass

    def load_ssbhard():
        # Implement SSB-hard loading
        pass

    def load_ninco():
        # Implement NINCO loading
        pass

    # Return dictionary of loading functions
    return {
        "inat": load_inat,
        "sun50": load_sun50,
        "places50": load_places50,
        "dtd": load_dtd,
        "ssbhard": load_ssbhard,
        "ninco": load_ninco,
    }


if __name__ == "__main__":
    # Test loading
    ood_loaders = load_actual_ood_datasets()
    print(f"Loaded {len(ood_loaders)} OOD datasets")

    for name, loader in ood_loaders.items():
        print(f"{name}: {len(loader.dataset)} samples")
        print(f"{name}: {len(loader.dataset)} samples")
