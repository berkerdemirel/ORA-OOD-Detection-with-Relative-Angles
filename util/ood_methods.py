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
                weight_diff_norm = torch.linalg.norm(weight_diff, dim=1)

                feats_batch_db = (
                    feats_batch_initial
                    - torch.divide(logit_diff, weight_diff_norm**2).view(-1, 1)
                    * weight_diff
                )

                centered_feats = feats_batch_initial - torch.mean(class_means, dim=0)
                centered_feats_db = feats_batch_db - torch.mean(class_means, dim=0)

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

    model = model.to(device)
    model.eval()

    train_feat_centered = []
    with torch.no_grad():
        id_feats_cpu = id_feats.cpu().numpy()
        in_labels_cpu = in_labels.cpu().numpy()
        class_means_cpu = class_means.cpu().numpy()

        for c in range(num_classes):
            # Extract features belonging to class c
            class_mask = in_labels_cpu == c
            feats_c = id_feats_cpu[class_mask]
            mean_c = class_means_cpu[c]
            train_feat_centered.append(feats_c - mean_c)

    if len(train_feat_centered) == 0:
        raise ValueError("No features found for computing class-wise covariance.")

    train_feat_centered = np.concatenate(train_feat_centered, axis=0)

    ec_classwise = EmpiricalCovariance(assume_centered=True)
    ec_classwise.fit(train_feat_centered.astype(np.float64))
    prec_classwise = ec_classwise.precision_  # shape (D, D)

    prec_classwise_t = torch.from_numpy(prec_classwise).to(
        device=device, dtype=torch.double
    )

    global_mean = id_feats_cpu.mean(axis=0)
    train_feat_centered_global = id_feats_cpu - global_mean

    ec_global = EmpiricalCovariance(assume_centered=True)
    ec_global.fit(train_feat_centered_global.astype(np.float64))
    prec_global = ec_global.precision_

    global_mean_t = torch.from_numpy(global_mean).to(device=device, dtype=torch.double)
    prec_global_t = torch.from_numpy(prec_global).to(device=device, dtype=torch.double)

    all_scores = []

    with torch.no_grad():
        for feats_batch, _ in test_loader:
            feats_batch = feats_batch.to(device, dtype=torch.double)

            B = feats_batch.size(0)
            feats_expanded = feats_batch.unsqueeze(1)  # (B, 1, D)
            means_expanded = class_means.to(device, dtype=torch.double).unsqueeze(
                0
            )  # (1, C, D)
            diff_classwise = feats_expanded - means_expanded  # (B, C, D)
            temp = torch.matmul(diff_classwise, prec_classwise_t)  # (B, C, D)
            mahalanobis_classwise = (temp * diff_classwise).sum(dim=-1)  # (B, C)

            min_maha_classwise = mahalanobis_classwise.min(dim=1).values  # (B,)
            classwise_score = -min_maha_classwise

            diff_global = feats_batch - global_mean_t
            temp_global = torch.matmul(diff_global, prec_global_t)  # (B, D)
            mahalanobis_global = (temp_global * diff_global).sum(dim=-1)  # (B,)
            global_score = -mahalanobis_global

            rel_scores = classwise_score - global_score

            all_scores.append(rel_scores.float().cpu())

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
    model = model.to(device)
    model.eval()

    id_feats = id_feats.to(device, dtype=torch.float32)
    in_labels = in_labels.to(device)
    class_means = class_means.to(device, dtype=torch.float32)

    with torch.no_grad():
        centered_feats_gpu = []
        for c in range(num_classes):
            class_mask = in_labels == c
            feats_c = id_feats[class_mask]  # shape (Nc, D)
            mean_c = class_means[c]  # shape (D,)
            centered_feats_gpu.append(feats_c - mean_c)

        all_centered_feats_gpu = torch.cat(centered_feats_gpu, dim=0)

    centered_feats_cpu = all_centered_feats_gpu.cpu().numpy()  # shape (N, D)
    ec_classwise = EmpiricalCovariance(assume_centered=True)
    ec_classwise.fit(centered_feats_cpu.astype(np.float64))

    prec_classwise = ec_classwise.precision_  # (D, D)
    prec_classwise_t = torch.from_numpy(prec_classwise).to(device, dtype=torch.double)

    with torch.no_grad():
        global_mean_gpu = id_feats.mean(dim=0)  # (D,) on GPU
        all_centered_global_gpu = id_feats - global_mean_gpu  # (N, D)

    centered_feats_global_cpu = all_centered_global_gpu.cpu().numpy()
    ec_global = EmpiricalCovariance(assume_centered=True)
    ec_global.fit(centered_feats_global_cpu.astype(np.float64))
    prec_global = ec_global.precision_

    global_mean_t = global_mean_gpu.to(device, dtype=torch.double)
    prec_global_t = torch.from_numpy(prec_global).to(device, dtype=torch.double)

    all_scores = []
    with torch.no_grad():
        for feats_batch, _ in test_loader:
            feats_batch = feats_batch.to(device, dtype=torch.double)
            B = feats_batch.size(0)
            feats_expanded = feats_batch.unsqueeze(1)  # (B, 1, D)
            means_expanded = class_means.to(device, dtype=torch.double).unsqueeze(0)
            diff_classwise = feats_expanded - means_expanded  # (B, C, D)

            temp = torch.matmul(diff_classwise, prec_classwise_t)  # (B, C, D)
            mahalanobis_classwise = (temp * diff_classwise).sum(dim=-1)  # (B, C)

            min_maha_classwise = mahalanobis_classwise.min(dim=1).values  # (B,)
            classwise_score = -min_maha_classwise

            diff_global = feats_batch - global_mean_t  # (B, D)
            temp_global = torch.matmul(diff_global, prec_global_t)  # (B, D)
            mahalanobis_global = (temp_global * diff_global).sum(dim=-1)  # (B,)
            global_score = -mahalanobis_global

            rel_scores = classwise_score - global_score
            all_scores.append(rel_scores.float().cpu())

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
    model = model.to(device)
    model.eval()

    id_feats = id_feats.to(device, dtype=torch.float32)  # (N, D)
    in_labels = in_labels.to(device)
    class_means = class_means.to(device, dtype=torch.float32)

    with torch.no_grad():
        centered_gpu_list = []
        for c in range(num_classes):
            class_mask = in_labels == c
            feats_c = id_feats[class_mask]  # (Nc, D)
            mean_c = class_means[c]  # (D,)
            centered_gpu_list.append(feats_c - mean_c)  # center on GPU

        if len(centered_gpu_list) == 0:
            raise ValueError("No features found to compute classwise covariance.")

        all_centered_feats = torch.cat(centered_gpu_list, dim=0)

    all_centered_feats_cpu = all_centered_feats.cpu().numpy()
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(all_centered_feats_cpu.astype(np.float64))
    prec = ec.precision_  # (D, D)

    prec_t = torch.from_numpy(prec).to(device=device, dtype=torch.double)

    all_scores = []

    with torch.no_grad():
        for feats_batch, _ in test_loader:
            feats_batch = feats_batch.to(device, dtype=torch.double)

            B = feats_batch.size(0)
            feats_expanded = feats_batch.unsqueeze(1)  # (B, 1, D)
            means_expanded = class_means.to(device, dtype=torch.double).unsqueeze(
                0
            )  # (1, C, D)
            diff_classwise = feats_expanded - means_expanded  # (B, C, D)
            temp = torch.matmul(diff_classwise, prec_t)  # (B, C, D)
            maha_dists = (temp * diff_classwise).sum(dim=-1)  # (B, C)

            min_maha = maha_dists.min(dim=1).values  # (B,)
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

    feats_np = train_feats.detach().cpu().numpy().astype(np.float64)
    labels_np = train_labels.detach().cpu().numpy().astype(np.int64)
    feats_np /= np.linalg.norm(feats_np, axis=1, keepdims=True)

    d = feats_np.shape[1]

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

    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(centered_np)
    precision_np = ec.precision_  # (d,d)  float64

    precision = torch.from_numpy(precision_np).to(device, torch.float64)
    class_means = torch.from_numpy(class_means_np).to(device, torch.float64)  # (C,d)

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

    return np.asarray(scores, dtype=np.float64)


@torch.inference_mode()
def neco_score(
    train_feats: torch.Tensor,  # (N_train, d)
    test_loader: torch.utils.data.DataLoader,  # yields (features, logits)
    path: str,  # folder to cache scaler / pca
    use_logit: bool = True,
    device: str = "cuda",
) -> np.ndarray:
    os.makedirs(path, exist_ok=True)

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

    pca_path = os.path.join(path, "pca_neco_deitpre.pkl")
    if os.path.exists(pca_path):
        with open(pca_path, "rb") as f:
            pca: PCA = pickle.load(f)
    else:
        pca = PCA(svd_solver="randomized", random_state=0)
        pca.fit(complete_train)
        with open(pca_path, "wb") as f:
            pickle.dump(pca, f)

    cum_var = np.cumsum(pca.explained_variance_ratio_)
    neco_dim = np.where(cum_var >= 0.90)[0][0] + 1

    all_scores = []

    for feats_b, logits_b in test_loader:
        feats_np = feats_b.cpu().numpy()
        logits_np = logits_b.cpu().numpy()

        # standardize
        complete_vecs = ss.transform(feats_np)  # (B, d)

        reduced = pca.transform(complete_vecs)[:, :neco_dim]  # (B, neco_dim)

        full_norm = np.linalg.norm(complete_vecs, axis=1)
        red_norm = np.linalg.norm(reduced, axis=1)
        ratio = red_norm / full_norm  # (B,)

        if use_logit:
            max_logits = logits_np.max(axis=1)  # (B,)
            ratio *= max_logits

        all_scores.append(ratio)

    return np.concatenate(all_scores, axis=0)
