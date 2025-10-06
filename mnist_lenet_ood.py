"""
MNIST LeNet Out-of-Distribution Detection using Decision Boundary Distances

This script:
1. Trains a LeNet-5 model on MNIST (10 classes: digits 0-9)
2. Tests the model on MNIST test set
3. Computes decision boundary distances for MNIST test samples
4. Computes decision boundary distances for Fashion-MNIST samples (used as OOD)
5. Analyzes the separation between in-distribution and out-of-distribution samples
6. Generates visualizations and statistics

For each sample, we compute the distance to decision boundaries for all 9 alternative
classes (excluding the predicted class) and take the maximum distance as the OOD score.
Higher distances should indicate samples that are further from decision boundaries,
potentially indicating out-of-distribution samples.
"""

import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset


class LeNet(nn.Module):
    """LeNet-5 architecture for MNIST classification"""

    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.tanh(self.fc1(x))
        features = torch.tanh(self.fc2(x))  # Extract features before final layer
        logits = self.fc3(features)
        return features, logits


def load_mnist_data():
    """Load MNIST training and test datasets"""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    return train_loader, test_loader


def load_ood_datasets():
    """Load Fashion-MNIST as OOD dataset"""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # Load Fashion-MNIST as OOD dataset
    fashion_mnist_dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )

    fashion_mnist_loader = DataLoader(
        fashion_mnist_dataset, batch_size=128, shuffle=False, num_workers=2
    )

    ood_loaders = {"fashion_mnist": fashion_mnist_loader}

    return ood_loaders


def train_model(model, train_loader, device, num_epochs=10):
    """Train the LeNet model on MNIST"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            features, logits = model(data)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = logits.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, "
                    f"Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%"
                )

        epoch_acc = 100.0 * correct / total
        print(
            f"Epoch {epoch+1} completed: Loss: {running_loss/len(train_loader):.4f}, "
            f"Accuracy: {epoch_acc:.2f}%"
        )


def test_model(model, test_loader, device):
    """Test the model on MNIST test set"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            features, logits = model(data)
            _, predicted = logits.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


def compute_decision_boundary_distances(model, data_loader, device, num_classes=10):
    """
    Compute distances to decision boundaries for each sample.
    For each sample, compute distance to decision boundary for each alternative class
    and return the maximum distance.
    """
    model.eval()
    all_max_distances = []

    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            features, logits = model(data)  # features: [B, 84], logits: [B, 10]

            # Get predicted class for each sample
            preds = logits.argmax(dim=1)  # [B]
            max_logits = logits.max(dim=1).values  # [B]

            # Get model weights (final layer)
            weight = model.fc3.weight  # [10, 84]

            batch_size = features.size(0)
            distances_per_sample = torch.zeros(batch_size, num_classes, device=device)

            for class_id in range(num_classes):
                # For each alternative class, compute distance to decision boundary
                logit_diff = max_logits - logits[:, class_id]  # [B]
                weight_diff = weight[preds] - weight[class_id]  # [B, 84]
                weight_diff_norm_sq = torch.sum(weight_diff**2, dim=1)  # [B]

                # Avoid division by zero
                weight_diff_norm_sq = torch.clamp(weight_diff_norm_sq, min=1e-8)

                # Distance to decision boundary
                distance = torch.abs(logit_diff) / torch.sqrt(
                    weight_diff_norm_sq
                )  # [B]
                distances_per_sample[:, class_id] = distance

            # Take maximum distance across all alternative classes for each sample
            max_distances = torch.max(distances_per_sample, dim=1).values  # [B]
            all_max_distances.append(max_distances.cpu())

    return torch.cat(all_max_distances).numpy()


def compute_lafo_score(model, data_loader, device, class_means, num_classes=10):
    """
    Compute LAFO (Linear Approximation for Feature Optimization) scores.
    This is adapted from the original LAFO method for decision boundary analysis.
    """
    model.eval()
    all_scores = []

    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            features, logits = model(data)  # features: [B, 84], logits: [B, 10]

            # Get predicted class for each sample
            preds = logits.argmax(dim=1)  # [B]
            max_logits = logits.max(dim=1).values  # [B]

            # Get model weights (final layer)
            weight = model.fc3.weight  # [10, 84]

            batch_size = features.size(0)
            trajectory_list = torch.zeros(batch_size, num_classes, device=device)

            for class_id in range(num_classes):
                # Compute logit difference and weight difference
                logit_diff = max_logits - logits[:, class_id]  # [B]
                weight_diff = weight[preds] - weight[class_id]  # [B, 84]
                weight_diff_norm_sq = torch.sum(weight_diff**2, dim=1)  # [B]

                # Avoid division by zero
                weight_diff_norm_sq = torch.clamp(weight_diff_norm_sq, min=1e-8)

                # Project features to decision boundary
                features_db = (
                    features
                    - (logit_diff / weight_diff_norm_sq).unsqueeze(1) * weight_diff
                )

                # Center features around global mean
                global_mean = torch.mean(class_means, dim=0)
                centered_features = features - global_mean
                centered_features_db = features_db - global_mean

                # Normalize and compute cosine similarity
                norm_centered_features = F.normalize(centered_features, p=2, dim=1)
                norm_centered_features_db = F.normalize(
                    centered_features_db, p=2, dim=1
                )

                cos_sim = torch.sum(
                    norm_centered_features * norm_centered_features_db, dim=1
                )
                cos_sim = torch.clamp(
                    cos_sim, -1.0, 1.0
                )  # Ensure valid range for arccos

                # Compute angle (LAFO score component)
                angles = torch.arccos(cos_sim) / torch.pi
                trajectory_list[:, class_id] = angles

            # Handle NaN values and compute final score
            trajectory_list[torch.isnan(trajectory_list)] = 0.0
            ood_score = torch.max(trajectory_list, dim=1).values  # [B]
            all_scores.append(ood_score.cpu())

    return torch.cat(all_scores).numpy()


def calculate_metrics(id_scores, ood_scores):
    """
    Calculate FPR95 and AUROC metrics for OOD detection.

    Args:
        id_scores: In-distribution scores (lower is more ID-like)
        ood_scores: Out-of-distribution scores (higher is more OOD-like)

    Returns:
        fpr95: False Positive Rate at 95% True Positive Rate
        auroc: Area Under ROC Curve
    """
    # Create labels: 0 for ID, 1 for OOD
    labels = np.concatenate(
        [np.zeros(len(id_scores)), np.ones(len(ood_scores))]  # ID labels  # OOD labels
    )
    scores = np.concatenate([id_scores, ood_scores])

    # Calculate AUROC
    try:
        from sklearn.metrics import roc_auc_score, roc_curve

        auroc = roc_auc_score(labels, scores)

        # Calculate FPR95
        fpr, tpr, thresholds = roc_curve(labels, scores)

        # Find the threshold where TPR is closest to 0.95
        tpr95_idx = np.argmin(np.abs(tpr - 0.95))
        fpr95 = fpr[tpr95_idx]

        return fpr95, auroc
    except ImportError:
        print("Warning: sklearn not available for metric calculations")
        return None, None


def analyze_ood_detection_performance(id_scores, ood_scores_dict, method_name=""):
    """Analyze OOD detection performance using given scores"""
    print(f"\nOOD DETECTION ANALYSIS: MNIST vs Fashion-MNIST ({method_name})")
    print("=" * 60)

    # Statistics for MNIST (in-distribution)
    id_mean = np.mean(id_scores)
    id_std = np.std(id_scores)
    print(f"\nMNIST (In-Distribution) Statistics:")
    print(f"  Mean score: {id_mean:.4f}")
    print(f"  Std score:  {id_std:.4f}")
    print(f"  Min score:  {np.min(id_scores):.4f}")
    print(f"  Max score:  {np.max(id_scores):.4f}")

    # Statistics for Fashion-MNIST (OOD)
    print(f"\nFashion-MNIST (OOD) Statistics:")

    for dataset_name, ood_scores in ood_scores_dict.items():
        ood_mean = np.mean(ood_scores)
        ood_std = np.std(ood_scores)

        print(f"\n  {dataset_name.replace('_', '-').title()}:")
        print(f"    Mean score: {ood_mean:.4f}")
        print(f"    Std score:  {ood_std:.4f}")
        print(f"    Min score:  {np.min(ood_scores):.4f}")
        print(f"    Max score:  {np.max(ood_scores):.4f}")
        print(f"    Separation from MNIST: {ood_mean - id_mean:.4f}")

        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(id_scores) - 1) * id_std**2 + (len(ood_scores) - 1) * ood_std**2)
            / (len(id_scores) + len(ood_scores) - 2)
        )
        cohens_d = (ood_mean - id_mean) / pooled_std
        print(f"    Cohen's d (effect size): {cohens_d:.4f}")

        # Calculate FPR95 and AUROC
        fpr95, auroc = calculate_metrics(id_scores, ood_scores)
        if fpr95 is not None and auroc is not None:
            print(f"    FPR95: {fpr95:.4f}")
            print(f"    AUROC: {auroc:.4f}")

    # Simple threshold-based detection performance
    threshold = id_mean + 2 * id_std  # Simple threshold
    print(f"\nThreshold-based Detection Analysis:")
    print(f"  Threshold (μ + 2σ): {threshold:.4f}")

    # MNIST false positive rate
    id_fp = np.sum(id_scores > threshold) / len(id_scores)
    print(f"  MNIST False Positive Rate: {id_fp:.4f}")

    # Fashion-MNIST true positive rates
    for dataset_name, ood_scores in ood_scores_dict.items():
        ood_tp = np.sum(ood_scores > threshold) / len(ood_scores)
        print(
            f"  {dataset_name.replace('_', '-').title()} True Positive Rate: {ood_tp:.4f}"
        )


def plot_distance_distributions(
    mnist_scores, ood_scores_dict, method_name="", filename_suffix=""
):
    """Plot histogram of score distributions"""
    plt.figure(figsize=(12, 8))

    # Plot MNIST distribution
    plt.hist(
        mnist_scores,
        bins=50,
        alpha=0.7,
        label="MNIST (In-Distribution)",
        density=True,
        color="blue",
        edgecolor="black",
        linewidth=0.5,
    )

    # Plot Fashion-MNIST distribution
    for dataset_name, scores in ood_scores_dict.items():
        plt.hist(
            scores,
            bins=50,
            alpha=0.7,
            label=f"{dataset_name.replace('_', '-').title()} (Out-of-Distribution)",
            density=True,
            color="red",
            edgecolor="black",
            linewidth=0.5,
        )

    plt.xlabel(f"{method_name} Score", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.title(
        f"Distribution of {method_name}\nMNIST vs Fashion-MNIST",
        fontsize=16,
    )
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Add statistics text box
    mnist_mean = np.mean(mnist_scores)
    fashion_mean = np.mean(list(ood_scores_dict.values())[0])

    # Calculate metrics for text box
    fpr95, auroc = calculate_metrics(mnist_scores, list(ood_scores_dict.values())[0])

    stats_text = f"MNIST μ = {mnist_mean:.3f}\nFashion-MNIST μ = {fashion_mean:.3f}\nSeparation = {fashion_mean - mnist_mean:.3f}"
    if fpr95 is not None and auroc is not None:
        stats_text += f"\nAUROC = {auroc:.3f}\nFPR95 = {fpr95:.3f}"

    plt.text(
        0.02,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    filename = f"mnist_fashion_mnist_{filename_suffix}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    return filename


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading MNIST data...")
    train_loader, test_loader = load_mnist_data()

    print("Loading Fashion-MNIST as OOD dataset...")
    ood_loaders = load_ood_datasets()

    # Initialize model
    model = LeNet(num_classes=10).to(device)
    print(f"Model architecture:\n{model}")

    # Train model
    print("\nTraining LeNet on MNIST...")
    train_model(model, train_loader, device, num_epochs=10)

    # Test model
    print("\nTesting model...")
    test_accuracy = test_model(model, test_loader, device)

    # Save model
    torch.save(model.state_dict(), "lenet_mnist.pth")
    print("Model saved as 'lenet_mnist.pth'")

    # Compute decision boundary distances for MNIST test set
    print("\nComputing decision boundary distances for MNIST test set...")
    mnist_distances = compute_decision_boundary_distances(model, test_loader, device)

    # Compute class means for LAFO computation
    print("Computing class means for LAFO score...")
    class_means = []
    with torch.no_grad():
        # Collect features for each class from training data
        class_features = [[] for _ in range(10)]
        for data, target in train_loader:
            data = data.to(device)
            features, _ = model(data)
            for i in range(10):
                mask = target == i
                if mask.any():
                    class_features[i].append(features[mask.to(device)])

        # Compute mean for each class
        for i in range(10):
            if class_features[i]:
                class_mean = torch.cat(class_features[i], dim=0).mean(dim=0)
                class_means.append(class_mean)
            else:
                # Fallback if no samples for this class
                class_means.append(torch.zeros(84, device=device))

        class_means = torch.stack(class_means)

    # Compute LAFO scores for MNIST test set
    print("Computing LAFO scores for MNIST test set...")
    mnist_lafo_scores = compute_lafo_score(model, test_loader, device, class_means)

    # Compute decision boundary distances for Fashion-MNIST (OOD)
    print("\nComputing decision boundary distances for Fashion-MNIST (OOD)...")
    ood_distances_dict = {}
    ood_lafo_dict = {}

    for dataset_name, ood_loader in ood_loaders.items():
        print(f"  Processing {dataset_name} - Decision Boundary Distances...")
        distances = compute_decision_boundary_distances(model, ood_loader, device)
        ood_distances_dict[dataset_name] = distances

        print(f"  Processing {dataset_name} - LAFO Scores...")
        lafo_scores = compute_lafo_score(model, ood_loader, device, class_means)
        ood_lafo_dict[dataset_name] = lafo_scores

    # Analyze results
    print("\n" + "=" * 80)
    print("DECISION BOUNDARY DISTANCE ANALYSIS")
    print("=" * 80)
    analyze_ood_detection_performance(
        mnist_distances, ood_distances_dict, "Decision Boundary Distances"
    )

    print("\n" + "=" * 80)
    print("LAFO SCORE ANALYSIS")
    print("=" * 80)
    analyze_ood_detection_performance(mnist_lafo_scores, ood_lafo_dict, "LAFO Scores")

    # Plot distributions
    print("\nGenerating distribution plots...")
    db_plot_file = plot_distance_distributions(
        mnist_distances,
        ood_distances_dict,
        "Decision Boundary Distance",
        "decision_boundary_distances",
    )
    lafo_plot_file = plot_distance_distributions(
        mnist_lafo_scores, ood_lafo_dict, "LAFO Score", "lafo_scores"
    )

    # Save results
    np.save("mnist_distances.npy", mnist_distances)
    np.save("mnist_lafo_scores.npy", mnist_lafo_scores)
    for dataset_name, distances in ood_distances_dict.items():
        np.save(f"{dataset_name}_distances.npy", distances)
    for dataset_name, lafo_scores in ood_lafo_dict.items():
        np.save(f"{dataset_name}_lafo_scores.npy", lafo_scores)

    print(f"\nResults saved:")
    print(f"  - lenet_mnist.pth (trained model)")
    print(f"  - mnist_distances.npy (MNIST decision boundary distances)")
    print(f"  - mnist_lafo_scores.npy (MNIST LAFO scores)")
    print(
        f"  - fashion_mnist_distances.npy (Fashion-MNIST decision boundary distances)"
    )
    print(f"  - fashion_mnist_lafo_scores.npy (Fashion-MNIST LAFO scores)")
    print(f"  - {db_plot_file} (decision boundary distance visualization)")
    print(f"  - {lafo_plot_file} (LAFO score visualization)")
    print(f"\nExperiment completed successfully!")


if __name__ == "__main__":
    main()
