import os
import pickle
from typing import Optional

import faiss
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def calculate_acc_val(score_log_val, label_log_val):
    num_correct = (score_log_val.argmax(1) == label_log_val).sum().item()
    print(f"Accuracy on validation set: {num_correct/len(label_log_val)}")
    return num_correct / len(label_log_val)


def plot_helper(
    score_in: np.ndarray,
    scores_out_test: np.ndarray,
    in_dataset: str,
    ood_dataset_name: str,
    function_name: str = "ORA",
) -> None:
    plt.hist(score_in, bins=70, alpha=0.5, label="in", density=True)
    plt.hist(scores_out_test, bins=70, alpha=0.5, label="out", density=True)
    plt.legend(loc="upper right", fontsize=9)
    if ood_dataset_name == "dtd":
        ood_dataset_name = "Texture"
    print(ood_dataset_name)
    plt.xlabel("ORA(x)", fontsize=16)
    # plt.xlabel(r'$\sin(\theta) / \sin(\alpha)$', fontsize=16)
    # plt.xlabel(r'$\sin(\theta)$', fontsize=16)
    # plt.xlabel(r'$\sin(\alpha)$', fontsize=16)
    # plt.xlabel(r'$\theta$', fontsize=16)
    plt.ylabel("Density", fontsize=16)
    plt.title(f"{in_dataset} vs {ood_dataset_name}", fontsize=16)
    plt.savefig(f"./{in_dataset}_{ood_dataset_name}_ORA.png", dpi=600)
    plt.close()


def react_filter(feats, thold):
    feats = torch.where(feats > thold, thold, feats)
    return feats


def react_thold(id_feats, percentile=90):
    id_feats = id_feats.cpu().numpy()
    tholds = np.percentile(id_feats.reshape(-1), percentile, axis=0)
    return tholds


def React_energy(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    num_classes: int,
    class_means: torch.Tensor,
    device: str,
    thold: Optional[torch.Tensor] = None,
) -> np.ndarray:
    model = model.to(device)
    model.eval()
    all_scores = []
    with torch.inference_mode():
        for feats_batch_initial, logits_batch_initial in test_loader:
            feats_batch_initial = feats_batch_initial.to(device)
            logits_batch_initial = logits_batch_initial.to(device)
            if thold is not None:
                feats_batch_initial = react_filter(feats_batch_initial, thold).float()
                logits_batch_initial = model.fc(feats_batch_initial)
            energy_score = torch.logsumexp(logits_batch_initial, dim=1)
            all_scores.append(energy_score)
    scores = np.asarray(torch.cat(all_scores).detach().cpu().numpy(), dtype=np.float32)
    return scores


def fdbd_score(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    num_classes: int,
    class_means: torch.Tensor,
    device: str,
) -> np.ndarray:
    model = model.to(device)
    model.eval()
    all_scores = []
    class_idx = np.arange(num_classes)
    with torch.inference_mode():
        for feats_batch_initial, logits_batch_initial in test_loader:
            feats_batch_initial = feats_batch_initial.to(device)
            logits_batch_initial = logits_batch_initial.to(device)
            preds_initial = logits_batch_initial.argmax(1)
            max_logits = logits_batch_initial.max(dim=1).values
            trajectory_list = torch.zeros(
                feats_batch_initial.size(0), num_classes, device=device
            )
            for class_id in class_idx:
                logit_diff = max_logits - logits_batch_initial[:, class_id]
                weight_diff = model.fc.weight[preds_initial] - model.fc.weight[class_id]
                # weight_diff = model.classifier.weight[preds_initial] - model.classifier.weight[class_id]

                weight_diff_norm = torch.linalg.norm(weight_diff, dim=1)
                feats_batch_db = (
                    feats_batch_initial
                    - torch.divide(logit_diff, weight_diff_norm**2).view(-1, 1)
                    * weight_diff
                )
                # # fdbd original
                distance_to_db = torch.linalg.norm(
                    feats_batch_initial - feats_batch_db, dim=1
                )
                fdbd_score = distance_to_db / torch.linalg.norm(
                    feats_batch_initial - torch.mean(class_means, dim=0), dim=1
                )

                # # fdbd our derivation
                centered_feats = feats_batch_initial - torch.mean(class_means, dim=0)
                centered_feats_db = feats_batch_db - torch.mean(class_means, dim=0)
                norm_centered_feats = F.normalize(centered_feats, p=2, dim=1)
                norm_centered_feats_db = F.normalize(centered_feats_db, p=2, dim=1)
                cos_sim_origin_perspective = torch.sum(
                    norm_centered_feats * norm_centered_feats_db, dim=1
                )
                angles_origin = torch.arccos(cos_sim_origin_perspective) / torch.pi

                feats_centered_db = feats_batch_initial - feats_batch_db
                mean_centered_db = torch.mean(class_means, dim=0) - feats_batch_db
                cos_sim = F.cosine_similarity(
                    feats_centered_db, mean_centered_db, dim=1
                )
                angles_db = torch.arccos(cos_sim) / torch.pi
                our_derivation = torch.sin(angles_origin * torch.pi) / torch.sin(
                    angles_db * torch.pi
                )
                # our_derivation = torch.sin(angles_origin * torch.pi)
                # our_derivation = torch.sin(angles_db * torch.pi)

                # check our derivation is same as fdbd
                # fdbd_score[torch.isnan(fdbd_score)] = 0
                # our_derivation[torch.isnan(our_derivation)] = 0
                # # print(torch.allclose(fdbd_score, our_derivation))

                trajectory_list[:, class_id] = our_derivation
            trajectory_list[torch.isnan(trajectory_list)] = 0
            ood_score = torch.mean(trajectory_list, dim=1)
            # ood_score = torch.max(trajectory_list, dim=1).values
            all_scores.append(ood_score)
    scores = np.asarray(torch.cat(all_scores).detach().cpu().numpy(), dtype=np.float32)
    return scores


def ora_score(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    num_classes: int,
    class_means: torch.Tensor,
    device: str,
    thold: Optional[torch.Tensor] = None,
) -> np.ndarray:
    model = model.to(device)
    model.eval()

    all_scores = []
    total_size = 0

    class_idx = np.arange(num_classes)

    with torch.inference_mode():
        for feats_batch_initial, logits_batch_initial in test_loader:
            feats_batch_initial = feats_batch_initial.to(device)
            logits_batch_initial = logits_batch_initial.to(device)
            # if thold is not None:
            #     feats_batch_initial = react_filter(feats_batch_initial, thold).float()
            #     logits_batch_initial = model.fc(feats_batch_initial)
            preds_initial = logits_batch_initial.argmax(1)
            max_logits = logits_batch_initial.max(dim=1).values
            total_size += feats_batch_initial.size(0)
            trajectory_list = torch.zeros(
                feats_batch_initial.size(0), num_classes, device=device
            )
            for class_id in class_idx:
                logit_diff = max_logits - logits_batch_initial[:, class_id]
                weight_diff = model.fc.weight[preds_initial] - model.fc.weight[class_id]
                # weight_diff = model.classifier.weight[preds_initial] - model.classifier.weight[class_id]
                weight_diff_norm = torch.linalg.norm(weight_diff, dim=1)

                feats_batch_db = (
                    feats_batch_initial
                    - torch.divide(logit_diff, weight_diff_norm**2).view(-1, 1)
                    * weight_diff
                )

                centered_feats = feats_batch_initial - torch.mean(class_means, dim=0)
                centered_feats_db = feats_batch_db - torch.mean(class_means, dim=0)
                # centered_feats = feats_batch_initial - torch.median(class_means, dim=0).values
                # centered_feats_db = feats_batch_db - torch.median(class_means, dim=0).values
                # centered_feats = feats_batch_initial - class_means[0]
                # centered_feats_db = feats_batch_db - class_means[0]

                norm_centered_feats = F.normalize(centered_feats, p=2, dim=1)
                norm_centered_feats_db = F.normalize(centered_feats_db, p=2, dim=1)

                cos_sim_origin_perspective = torch.sum(
                    norm_centered_feats * norm_centered_feats_db, dim=1
                )
                angles_origin = torch.arccos(cos_sim_origin_perspective) / torch.pi
                trajectory_list[:, class_id] = angles_origin

            trajectory_list[torch.isnan(trajectory_list)] = 0
            ood_score = torch.max(trajectory_list, dim=1).values
            # ood_score = torch.topk(trajectory_list, 2, largest=False, dim=1).values[
            #     :, 1
            # ]
            # ood_score = torch.mean(trajectory_list, dim=1)
            all_scores.append(ood_score)
    scores = np.asarray(torch.cat(all_scores).detach().cpu().numpy(), dtype=np.float32)
    return scores


def energy_score(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    num_classes: int,
    class_means: torch.Tensor,
    device: str,
    thold: Optional[torch.Tensor] = None,
) -> np.ndarray:
    model = model.to(device)
    model.eval()
    all_scores = []
    with torch.inference_mode():
        for feats_batch_initial, logits_batch_initial in test_loader:
            feats_batch_initial = feats_batch_initial.to(device)
            logits_batch_initial = logits_batch_initial.to(device)
            energies = torch.logsumexp(logits_batch_initial, dim=1)
            all_scores.append(energies)
    scores = np.asarray(torch.cat(all_scores).detach().cpu().numpy(), dtype=np.float32)
    return scores


def msp_score(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    num_classes: int,
    class_means: torch.Tensor,
    device: str,
    thold: Optional[torch.Tensor] = None,
) -> np.ndarray:
    model = model.to(device)
    model.eval()
    all_scores = []
    with torch.inference_mode():
        for feats_batch_initial, logits_batch_initial in test_loader:
            feats_batch_initial = feats_batch_initial.to(device)
            logits_batch_initial = logits_batch_initial.to(device)
            probs = F.softmax(logits_batch_initial, dim=1)
            max_probs = torch.max(probs, dim=1).values
            all_scores.append(max_probs)
    scores = np.asarray(torch.cat(all_scores).detach().cpu().numpy(), dtype=np.float32)
    return scores


def maxlogit_score(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    num_classes: int,
    class_means: torch.Tensor,
    device: str,
    thold: Optional[torch.Tensor] = None,
) -> np.ndarray:
    model = model.to(device)
    model.eval()
    all_scores = []
    with torch.inference_mode():
        for _, logits_batch_initial in test_loader:
            logits_batch_initial = logits_batch_initial.to(device)
            max_logits = torch.max(logits_batch_initial, dim=1).values
            all_scores.append(max_logits)
    scores = np.asarray(torch.cat(all_scores).detach().cpu().numpy(), dtype=np.float32)
    return scores


def knn_score(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    num_classes: int,
    class_means: torch.Tensor,
    device: str,
    train_feats: torch.Tensor,
    k: Optional[torch.Tensor] = 1000,
) -> np.ndarray:
    model = model.to(device)
    model.eval()

    all_scores = []
    train_feats = F.normalize(train_feats, p=2, dim=1)
    with torch.inference_mode():
        for feats_batch_initial, logits_batch_initial in test_loader:
            feats_batch_initial = feats_batch_initial.to(device)
            feats_batch_initial = F.normalize(feats_batch_initial, p=2, dim=1)
            # calculate the kth nearest distance to the train_feats
            d = torch.cdist(
                feats_batch_initial,
                train_feats,
                p=2,
                compute_mode="donot_use_mm_for_euclid_dist",
            )
            d = torch.topk(d, k, largest=False, sorted=True).values[:, -1]
            all_scores.append(-d)
    scores = np.asarray(torch.cat(all_scores).detach().cpu().numpy(), dtype=np.float32)
    return scores


def rcos_score(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    num_classes: int,
    class_means: torch.Tensor,
    device: str,
    tholds: Optional[torch.Tensor] = None,
) -> np.ndarray:
    model = model.to(device)
    model.eval()

    all_scores = []
    total_size = 0

    class_idx = np.arange(num_classes)

    with torch.inference_mode():
        for feats_batch_initial, logits_batch_initial in test_loader:
            feats_batch_initial = feats_batch_initial.to(device)
            logits_batch_initial = logits_batch_initial.to(device)
            total_size += feats_batch_initial.size(0)
            trajectory_list = torch.zeros(
                feats_batch_initial.size(0), num_classes, device=device
            )
            for class_id in class_idx:
                class_mean_id = class_means[class_id]
                # compute the cosine similarity between the feature and the class mean
                cos_sim = F.cosine_similarity(feats_batch_initial, class_mean_id, dim=1)
                trajectory_list[:, class_id] = cos_sim
            # apply softmax on dim1
            trajectory_list = F.softmax(trajectory_list, dim=1)
            ood_score = torch.max(trajectory_list, dim=1).values
            all_scores.append(ood_score)
    scores = np.asarray(torch.cat(all_scores).detach().cpu().numpy(), dtype=np.float32)
    return scores


def plaincos_score(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    num_classes: int,
    class_means: torch.Tensor,
    device: str,
    tholds: Optional[torch.Tensor] = None,
) -> np.ndarray:
    model = model.to(device)
    model.eval()

    all_scores = []
    total_size = 0

    class_idx = np.arange(num_classes)

    with torch.inference_mode():
        for feats_batch_initial, logits_batch_initial in test_loader:
            feats_batch_initial = feats_batch_initial.to(device)
            logits_batch_initial = logits_batch_initial.to(device)
            total_size += feats_batch_initial.size(0)
            trajectory_list = torch.zeros(
                feats_batch_initial.size(0), num_classes, device=device
            )
            for class_id in class_idx:
                class_mean_id = class_means[class_id]
                # compute the cosine similarity between the feature and the class mean
                cos_sim = F.cosine_similarity(feats_batch_initial, class_mean_id, dim=1)
                trajectory_list[:, class_id] = cos_sim
            # apply softmax on dim1
            ood_score = torch.max(trajectory_list, dim=1).values
            all_scores.append(ood_score)
    scores = np.asarray(torch.cat(all_scores).detach().cpu().numpy(), dtype=np.float32)
    return scores


def rel_mahalonobis_distance_score(
    model: torch.nn.Module,
    id_feats: torch.Tensor,
    in_labels: torch.Tensor,
    test_loader: torch.utils.data.DataLoader,
    num_classes: int,
    class_means: torch.Tensor,
    device: str,
    tholds: Optional[torch.Tensor] = None,
) -> np.ndarray:

    # -------------------------------------------------------------------------
    # Move model to device and set eval mode
    # -------------------------------------------------------------------------
    model = model.to(device)
    model.eval()

    # -------------------------------------------------------------------------
    # 1. Compute class-wise precision matrix (prec) based on ID features
    #    - For each sample in id_feats, subtract the corresponding class mean
    #    - Fit EmpiricalCovariance on the centered features
    # -------------------------------------------------------------------------
    # Collect all centered features for class-wise covariance
    # (in_labels and class_means are assumed to be consistent)
    train_feat_centered = []
    with torch.no_grad():
        # If these are already on CPU, you can skip .cpu()
        # but we do so to ensure numpy usage below.
        id_feats_cpu = id_feats.cpu().numpy()
        in_labels_cpu = in_labels.cpu().numpy()
        class_means_cpu = class_means.cpu().numpy()

        for c in range(num_classes):
            # Extract features belonging to class c
            class_mask = in_labels_cpu == c
            if not np.any(class_mask):
                # If for some reason this class doesn't exist, skip
                continue
            feats_c = id_feats_cpu[class_mask]
            mean_c = class_means_cpu[c]
            train_feat_centered.append(feats_c - mean_c)

    # Concatenate all centered features
    if len(train_feat_centered) == 0:
        raise ValueError("No features found for computing class-wise covariance.")

    train_feat_centered = np.concatenate(train_feat_centered, axis=0)

    # Fit Empirical Covariance to get class-wise precision
    ec_classwise = EmpiricalCovariance(assume_centered=True)
    ec_classwise.fit(train_feat_centered.astype(np.float64))
    prec_classwise = ec_classwise.precision_  # shape (D, D)

    # Convert to torch on device
    prec_classwise_t = torch.from_numpy(prec_classwise).to(
        device=device, dtype=torch.double
    )

    # -------------------------------------------------------------------------
    # 2. Compute global precision matrix (prec_global)
    #    - Single global mean from all id_feats
    #    - Fit EmpiricalCovariance on id_feats - global_mean
    # -------------------------------------------------------------------------
    global_mean = id_feats_cpu.mean(axis=0)
    train_feat_centered_global = id_feats_cpu - global_mean

    ec_global = EmpiricalCovariance(assume_centered=True)
    ec_global.fit(train_feat_centered_global.astype(np.float64))
    prec_global = ec_global.precision_

    # Convert to torch on device
    global_mean_t = torch.from_numpy(global_mean).to(device=device, dtype=torch.double)
    prec_global_t = torch.from_numpy(prec_global).to(device=device, dtype=torch.double)

    # -------------------------------------------------------------------------
    # 3. For each batch in test_loader, compute the relative Mahalanobis score
    #    If test_loader already provides features, we skip model(...)
    #    If test_loader provides inputs, you'd do feats = model(inputs).
    #
    #    Score steps:
    #      - classwise_score(x) = - min_c [ (x - mean_c)^T @ prec_classwise @ (x - mean_c) ]
    #      - global_score(x)    = - [ (x - global_mean)^T @ prec_global @ (x - global_mean) ]
    #      - final_score(x)     = classwise_score(x) - global_score(x)
    #
    # -------------------------------------------------------------------------
    all_scores = []

    with torch.no_grad():
        for feats_batch, _ in test_loader:
            # If you need to compute features from model, uncomment:
            # feats_batch = model(feats_batch.to(device))
            # Otherwise, assume feats_batch is already the features:
            feats_batch = feats_batch.to(device, dtype=torch.double)

            # Expand feats to (B, 1, D) and class_means to (1, C, D) for vectorized distance
            B = feats_batch.size(0)
            feats_expanded = feats_batch.unsqueeze(1)  # (B, 1, D)
            means_expanded = class_means.to(device, dtype=torch.double).unsqueeze(
                0
            )  # (1, C, D)
            diff_classwise = feats_expanded - means_expanded  # (B, C, D)

            # (x - mean_c) @ prec_classwise -> shape (B, C, D)
            # Then elementwise * diff_classwise and sum over D => (B, C)
            # We'll do a manual matmul: (diff_classwise @ prec_classwise) => (B, C, D)
            temp = torch.matmul(diff_classwise, prec_classwise_t)  # (B, C, D)
            # Then multiply elementwise by diff_classwise and sum along D
            # This is the Mahalanobis distance for each sample to each class
            mahalanobis_classwise = (temp * diff_classwise).sum(dim=-1)  # (B, C)

            # We take the minimum across classes => shape (B,)
            min_maha_classwise = mahalanobis_classwise.min(dim=1).values  # (B,)

            # Negative sign to keep consistent with "score = - Mdist(...)"
            classwise_score = -min_maha_classwise

            # Now compute global Mahalanobis
            # (x - mean_global) -> (B, D)
            diff_global = feats_batch - global_mean_t
            # matmul => (B, D)
            temp_global = torch.matmul(diff_global, prec_global_t)  # (B, D)
            mahalanobis_global = (temp_global * diff_global).sum(dim=-1)  # (B,)
            global_score = -mahalanobis_global

            # Relative score = classwise_score - global_score
            rel_scores = classwise_score - global_score

            all_scores.append(rel_scores.float().cpu())

    # Concatenate all scores and return as np.float32
    final_scores = torch.cat(all_scores).numpy().astype(np.float32)
    return final_scores


def rel_mahalonobis_distance_score_2(
    model: torch.nn.Module,
    id_feats: torch.Tensor,
    in_labels: torch.Tensor,
    test_loader: torch.utils.data.DataLoader,
    num_classes: int,
    class_means: torch.Tensor,
    device: str,
    tholds: Optional[torch.Tensor] = None,
) -> np.ndarray:
    # 1) Move model to device, set eval mode.
    model = model.to(device)
    model.eval()

    # 2) Ensure class_means and id_feats are on the same device.
    id_feats = id_feats.to(device, dtype=torch.float32)
    in_labels = in_labels.to(device)
    class_means = class_means.to(device, dtype=torch.float32)

    # -------------------------------------------------------------------------
    # 3) Build classwise-centered features on GPU
    #    We gather (z_i - mu_{y_i}) for each sample, for a single shared cov
    # -------------------------------------------------------------------------
    with torch.no_grad():
        # This is a list of GPU tensors, each from one class.
        centered_feats_gpu = []
        for c in range(num_classes):
            class_mask = in_labels == c
            feats_c = id_feats[class_mask]  # shape (Nc, D)
            mean_c = class_means[c]  # shape (D,)
            centered_feats_gpu.append(feats_c - mean_c)

        all_centered_feats_gpu = torch.cat(centered_feats_gpu, dim=0)

    # -------------------------------------------------------------------------
    # 4) Compute EmpiricalCovariance (CPU / scikit-learn)
    #    So we do one big transfer to CPU
    # -------------------------------------------------------------------------
    centered_feats_cpu = all_centered_feats_gpu.cpu().numpy()  # shape (N, D)
    ec_classwise = EmpiricalCovariance(assume_centered=True)
    ec_classwise.fit(centered_feats_cpu.astype(np.float64))

    prec_classwise = ec_classwise.precision_  # (D, D)
    prec_classwise_t = torch.from_numpy(prec_classwise).to(device, dtype=torch.double)

    # -------------------------------------------------------------------------
    # 5) Compute global mean and global covariance
    # -------------------------------------------------------------------------
    with torch.no_grad():
        global_mean_gpu = id_feats.mean(dim=0)  # (D,) on GPU

        # Center globally on GPU
        all_centered_global_gpu = id_feats - global_mean_gpu  # (N, D)

    # Move global-centered feats to CPU for EmpiricalCovariance
    centered_feats_global_cpu = all_centered_global_gpu.cpu().numpy()
    ec_global = EmpiricalCovariance(assume_centered=True)
    ec_global.fit(centered_feats_global_cpu.astype(np.float64))
    prec_global = ec_global.precision_

    # Move global mean & prec back to GPU
    global_mean_t = global_mean_gpu.to(device, dtype=torch.double)
    prec_global_t = torch.from_numpy(prec_global).to(device, dtype=torch.double)

    # -------------------------------------------------------------------------
    # 6) For each batch in test_loader, compute relative Mahalanobis distance
    # -------------------------------------------------------------------------
    all_scores = []
    with torch.no_grad():
        for feats_batch, _ in test_loader:
            # If test_loader returns inputs, you would do:
            # feats_batch = model(feats_batch.to(device, dtype=torch.float32))

            feats_batch = feats_batch.to(device, dtype=torch.double)

            # Expand feats to (B, 1, D) and class_means to (1, C, D)
            B = feats_batch.size(0)
            feats_expanded = feats_batch.unsqueeze(1)  # (B, 1, D)
            means_expanded = class_means.to(device, dtype=torch.double).unsqueeze(0)
            # means_expanded => (1, C, D)
            diff_classwise = feats_expanded - means_expanded  # (B, C, D)

            # 6a) Classwise Mahalanobis: (x - mu_c)^T @ prec_classwise @ (x - mu_c)
            temp = torch.matmul(diff_classwise, prec_classwise_t)  # (B, C, D)
            mahalanobis_classwise = (temp * diff_classwise).sum(dim=-1)  # (B, C)

            min_maha_classwise = mahalanobis_classwise.min(dim=1).values  # (B,)
            classwise_score = -min_maha_classwise

            # 6b) Global Mahalanobis: (x - mu_global)^T @ prec_global @ (x - mu_global)
            diff_global = feats_batch - global_mean_t  # (B, D)
            temp_global = torch.matmul(diff_global, prec_global_t)  # (B, D)
            mahalanobis_global = (temp_global * diff_global).sum(dim=-1)  # (B,)
            global_score = -mahalanobis_global

            # 6c) Relative score = classwise_score - global_score
            rel_scores = classwise_score - global_score
            all_scores.append(rel_scores.float().cpu())

    # Concatenate and return
    final_scores = torch.cat(all_scores).numpy().astype(np.float32)
    return final_scores


def mds_score(
    model: torch.nn.Module,
    id_feats: torch.Tensor,
    in_labels: torch.Tensor,
    test_loader: torch.utils.data.DataLoader,
    num_classes: int,
    class_means: torch.Tensor,
    device: str,
    tholds: Optional[torch.Tensor] = None,
) -> np.ndarray:
    # -------------------------------------------------------------------------
    # 1) Move model to device and set it to eval
    #    (Only needed if you plan to pass raw inputs through model)
    # -------------------------------------------------------------------------
    model = model.to(device)
    model.eval()

    # -------------------------------------------------------------------------
    # 2) Compute the shared covariance matrix from classwise-centered features
    #    Because scikit-learn EmpiricalCovariance requires CPU+NumPy, we do:
    #       - Center each ID feature by its class mean on GPU
    #       - Concatenate
    #       - Move once to CPU, fit EmpiricalCovariance
    # -------------------------------------------------------------------------
    id_feats = id_feats.to(device, dtype=torch.float32)  # (N, D)
    in_labels = in_labels.to(device)
    class_means = class_means.to(device, dtype=torch.float32)

    with torch.no_grad():
        centered_gpu_list = []
        for c in range(num_classes):
            class_mask = in_labels == c
            if not torch.any(class_mask):
                # If a class doesn't appear in your ID set, skip it
                continue
            feats_c = id_feats[class_mask]  # (Nc, D)
            mean_c = class_means[c]  # (D,)
            centered_gpu_list.append(feats_c - mean_c)  # center on GPU

        if len(centered_gpu_list) == 0:
            raise ValueError("No features found to compute classwise covariance.")

        # Concatenate all classwise-centered features => (N, D)
        all_centered_feats = torch.cat(centered_gpu_list, dim=0)

    # Move to CPU for EmpiricalCovariance
    all_centered_feats_cpu = all_centered_feats.cpu().numpy()

    # Fit EmpiricalCovariance
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(all_centered_feats_cpu.astype(np.float64))
    prec = ec.precision_  # (D, D)

    # Convert precision matrix to torch on device
    prec_t = torch.from_numpy(prec).to(device=device, dtype=torch.double)

    # -------------------------------------------------------------------------
    # 3) For each test sample, compute min_{c} Mahalanobis distance, then *-1
    #    (We treat the negative distance as a "score"—the more negative,
    #     the further from ID.)
    # -------------------------------------------------------------------------
    all_scores = []

    with torch.no_grad():
        for feats_batch, _ in test_loader:
            # If test_loader gives raw inputs, do: feats_batch = model(feats_batch.to(device))
            feats_batch = feats_batch.to(device, dtype=torch.double)

            # Expand feats => (B, 1, D), class_means => (1, C, D)
            B = feats_batch.size(0)
            feats_expanded = feats_batch.unsqueeze(1)  # (B, 1, D)
            means_expanded = class_means.to(device, dtype=torch.double).unsqueeze(
                0
            )  # (1, C, D)
            diff_classwise = feats_expanded - means_expanded  # (B, C, D)

            # Compute Mdist => (diff @ prec) * diff, sum over D => (B, C)
            temp = torch.matmul(diff_classwise, prec_t)  # (B, C, D)
            maha_dists = (temp * diff_classwise).sum(dim=-1)  # (B, C)

            # For each sample, take the minimum distance across classes
            min_maha = maha_dists.min(dim=1).values  # (B,)

            # Our Mahalanobis score is -min_maha
            maha_scores = -min_maha
            all_scores.append(maha_scores.float().cpu())

    final_scores = torch.cat(all_scores).numpy().astype(np.float32)
    return final_scores


def nnguide_score(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    num_classes: int,
    class_means: torch.Tensor,
    device: str,
    train_feats: torch.Tensor,
    k: Optional[torch.Tensor] = 10,
) -> np.ndarray:
    model = model.to(device)
    model.eval()
    energy_scores = torch.from_numpy(
        energy_score(model, train_loader, num_classes, class_means, device)
    ).to(device)
    all_scores = []
    train_feats = F.normalize(train_feats, p=2, dim=1)
    with torch.inference_mode():
        for feats_batch_initial, logits_batch_initial in test_loader:
            feats_batch_initial = feats_batch_initial.to(device)  # B x 2048
            logits_batch_initial = logits_batch_initial.to(device)
            energy_score_batch = torch.logsumexp(logits_batch_initial, dim=1)
            feats_batch_initial = F.normalize(
                feats_batch_initial, p=2, dim=1
            )  # B x 2048
            # calculate the kth nearest distance to the train_feats
            d = torch.matmul(
                feats_batch_initial,  # B x 2048
                (train_feats * energy_scores.view(-1, 1)).T,  # N x 2048
            )  # B x N
            d = torch.topk(d, k, largest=True, sorted=True).values.mean(dim=1)
            all_scores.append(d * energy_score_batch)
    scores = np.asarray(torch.cat(all_scores).detach().cpu().numpy(), dtype=np.float32)
    return scores


@torch.inference_mode()
def nci_score(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,  # yields (features, logits)
    train_mean: torch.Tensor,  # 1‑D tensor (d,) – mean of ID‑training features
    device: str,
    alpha: float = 0.0001,  # NCI hyper‑parameter
) -> np.ndarray:
    """
    “Nearest‑Class Influence” (NCI) OOD score.

    Parameters
    ----------
    model : torch.nn.Module
        Classification network with a fully‑connected head called `fc`
        (change to `classifier` if that’s what your model uses).
    test_loader : DataLoader
        Each iteration should return a tuple `(features, logits)` where:
            • features – tensor of shape (B, d)
            • logits   – tensor of shape (B, num_classes)
    train_mean : torch.Tensor
        Feature mean of in‑distribution training data (shape: (d,)).
    device : str
        "cuda" or "cpu".
    alpha : float, default 0.0
        Weight of the ‖feature‖₁ regulariser (as in the original NCI paper).

    Returns
    -------
    np.ndarray
        1‑D array of OOD scores for every sample in `test_loader`
        (higher ⇒ more in‑distribution‑like, exactly as in NCI).
    """
    model = model.to(device).eval()
    w = model.fc.weight.to(device)  # (num_classes, d)
    train_mean = train_mean.to(device)  # (d,)

    all_scores = []

    for feats_b, logits_b in test_loader:
        feats_b = feats_b.to(device)  # (B, d)
        logits_b = logits_b.to(device)

        preds = logits_b.argmax(1)  # (B,)
        delta = feats_b - train_mean  # (B, d)

        # core NCI expression:
        score_b = torch.sum(w[preds] * delta, dim=1) / F.normalize(
            delta, p=2, dim=1
        ).norm(dim=1) + alpha * torch.norm(feats_b, p=1, dim=1)

        all_scores.append(score_b)

    return torch.cat(all_scores).cpu().numpy().astype(np.float32)


@torch.inference_mode()
def mahalanobisplus_score(
    train_feats: torch.Tensor,  # (N_train, d)
    train_labels: torch.Tensor,  # (N_train,)
    test_loader: torch.utils.data.DataLoader,  # yields (features, logits) – logits ignored
    num_classes: int,
    device: str = "cuda",
) -> np.ndarray:
    """
    Numerically identical to `evaluate_Mahalanobis_norm`, but works with a
    DataLoader and keeps GPU memory low.
    Returns: np.ndarray (N_test,)   — higher ⇒ more ID‑like.
    """

    # ---------------------------------------------------------------- 1) normalise train feats (CPU, float64)
    feats_np = train_feats.detach().cpu().numpy().astype(np.float64)
    labels_np = train_labels.detach().cpu().numpy().astype(np.int64)
    feats_np /= np.linalg.norm(feats_np, axis=1, keepdims=True)

    d = feats_np.shape[1]

    # ---------------------------------------------------------------- 2) per‑class means (exact loop)
    class_means_np = np.zeros((num_classes, d), dtype=np.float64)
    centered_list = []

    for cls in range(num_classes):
        fs = feats_np[labels_np == cls]
        if fs.size == 0:
            raise ValueError(f"No samples for class {cls}")
        mu = fs.mean(axis=0)
        class_means_np[cls] = mu
        centered_list.append(fs - mu)

    centered_np = np.concatenate(centered_list, axis=0)

    # ---------------------------------------------------------------- 3) empirical covariance (sklearn)
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(centered_np)
    precision_np = ec.precision_  # (d,d)  float64

    # ---------------------------------------------------------------- 4) move to GPU (still float64)
    precision = torch.from_numpy(precision_np).to(device, torch.float64)
    class_means = torch.from_numpy(class_means_np).to(device, torch.float64)  # (C,d)

    # ---------------------------------------------------------------- 5) scorer (no (B,C,d) tensor)
    def _score_batch(feats_b: torch.Tensor) -> torch.Tensor:
        feats_b = feats_b.to(device, torch.float64)
        feats_b = feats_b / feats_b.norm(dim=1, keepdim=True)  # row‑wise L2

        score_batch = -np.array(
            [
                (((f - class_means) @ precision) * (f - class_means))
                .sum(dim=-1)
                .min()
                .cpu()
                .item()
                for f in feats_b.to(device).double()
            ]
        )

        return score_batch

    scores = []
    for feats_b, _ in test_loader:
        scores.extend(_score_batch(feats_b))  # add the floats, not the array

    return np.asarray(scores, dtype=np.float64)  # shape (N_test,)


@torch.inference_mode()
def neco_score(
    train_feats: torch.Tensor,  # (N_train, d)
    test_loader: torch.utils.data.DataLoader,  # yields (features, logits)
    path: str,  # folder to cache scaler / pca
    use_logit: bool = True,
    device: str = "cuda",
) -> np.ndarray:
    """
    **Exact** replica of `evaluate_neco`, but DataLoader‑friendly.
    Returns scores for *all* samples in `test_loader` (ID and OOD mixed),
    in the same order they are yielded.
    """

    os.makedirs(path, exist_ok=True)

    # ------------------------------------------------------------------- 1) fit / load StandardScaler
    ss_path = os.path.join(path, "ss_neco_deitpre.pkl")
    if os.path.exists(ss_path):
        with open(ss_path, "rb") as f:
            ss: StandardScaler = pickle.load(f)
        complete_train = ss.transform(train_feats.cpu().numpy())
    else:
        ss = StandardScaler()
        complete_train = ss.fit_transform(train_feats.cpu().numpy())
        with open(ss_path, "wb") as f:
            pickle.dump(ss, f)

    # ------------------------------------------------------------------- 2) fit / load PCA
    pca_path = os.path.join(path, "pca_neco_deitpre.pkl")
    if os.path.exists(pca_path):
        with open(pca_path, "rb") as f:
            pca: PCA = pickle.load(f)
    else:
        pca = PCA(svd_solver="randomized", random_state=0)
        pca.fit(complete_train)
        with open(pca_path, "wb") as f:
            pickle.dump(pca, f)

    # cumulative explained variance → neco_dim (≥ 90 %)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    neco_dim = np.where(cum_var >= 0.90)[0][0] + 1  # identical logic

    # ------------------------------------------------------------------- 3) stream over test_loader
    all_scores = []

    for feats_b, logits_b in test_loader:  # logits_b needed for max‑logit
        feats_np = feats_b.cpu().numpy()
        logits_np = logits_b.cpu().numpy()

        # standardize
        complete_vecs = ss.transform(feats_np)  # (B, d)

        # PCA transform, then slice first neco_dim components
        reduced = pca.transform(complete_vecs)[:, :neco_dim]  # (B, neco_dim)

        # norms & ratio
        full_norm = np.linalg.norm(complete_vecs, axis=1)
        red_norm = np.linalg.norm(reduced, axis=1)
        ratio = red_norm / full_norm  # (B,)

        if use_logit:
            max_logits = logits_np.max(axis=1)  # (B,)
            ratio *= max_logits

        all_scores.append(ratio)

    return np.concatenate(all_scores, axis=0)  # (N_test,)
