import os
import time
from util.args_loader import get_args
from util import metrics
import torch
import faiss
import numpy as np
import torchvision.models as models
from typing import Optional
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn


class LinearProbe(nn.Module):
    def __init__(self, in_features, num_classes):
        super(LinearProbe, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)


def plot_helper(
    score_in: np.ndarray,
    scores_out_test: np.ndarray,
    in_dataset: str,
    ood_dataset_name: str,
    function_name: str = "LAFO",
) -> None:
    plt.hist(score_in, bins=70, alpha=0.5, label="in", density=True)
    plt.hist(scores_out_test, bins=70, alpha=0.5, label="out", density=True)
    plt.legend(loc="upper right", fontsize=9)
    if ood_dataset_name == "dtd":
        ood_dataset_name = "Texture"
    print(ood_dataset_name)
    plt.xlabel("LAFO(x)", fontsize=16)
    # plt.xlabel(r'$\sin(\theta) / \sin(\alpha)$', fontsize=16)
    # plt.xlabel(r'$\sin(\theta)$', fontsize=16)
    # plt.xlabel(r'$\sin(\alpha)$', fontsize=16)
    # plt.xlabel(r'$\theta$', fontsize=16)
    plt.ylabel("Density", fontsize=16)
    plt.title(f"{in_dataset} vs {ood_dataset_name}", fontsize=16)
    plt.savefig(f"./{in_dataset}_{ood_dataset_name}_LAFO.png", dpi=600)
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
            logits_batch_initial = model.fc(feats_batch_initial.float())

            # logits_batch_initial = logits_batch_initial.to(device)
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


def lafo_score3(
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
            for stat in range(3):
                for class_id in class_idx:
                    logit_diff = max_logits - logits_batch_initial[:, class_id]
                    weight_diff = (
                        model.fc.weight[preds_initial] - model.fc.weight[class_id]
                    )
                    # weight_diff = model.classifier.weight[preds_initial] - model.classifier.weight[class_id]
                    weight_diff_norm = torch.linalg.norm(weight_diff, dim=1)

                    feats_batch_db = (
                        feats_batch_initial
                        - torch.divide(logit_diff, weight_diff_norm**2).view(-1, 1)
                        * weight_diff
                    )
                    if stat == 0:
                        centered_feats = (
                            feats_batch_initial - torch.max(class_means, dim=0).values
                        )
                        centered_feats_db = (
                            feats_batch_db - torch.max(class_means, dim=0).values
                        )

                    elif stat == 1:
                        centered_feats = (
                            feats_batch_initial - torch.min(class_means, dim=0).values
                        )
                        centered_feats_db = (
                            feats_batch_db - torch.min(class_means, dim=0).values
                        )
                    elif stat == 2:
                        centered_feats = (
                            feats_batch_initial
                            - torch.median(class_means, dim=0).values
                        )
                        centered_feats_db = (
                            feats_batch_db - torch.median(class_means, dim=0).values
                        )

                    # centered_feats = feats_batch_initial - torch.mean(class_means, dim=0)
                    # centered_feats_db = feats_batch_db - torch.mean(class_means, dim=0)
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
                    trajectory_list[:, class_id] += angles_origin

            trajectory_list[torch.isnan(trajectory_list)] = 0
            ood_score = torch.max(trajectory_list, dim=1).values
            # ood_score = torch.topk(trajectory_list, 2, largest=False, dim=1).values[:, 1]
            # ood_score = torch.mean(trajectory_list, dim=1)
            all_scores.append(ood_score)
    scores = np.asarray(torch.cat(all_scores).detach().cpu().numpy(), dtype=np.float32)
    return scores


def lafo_score(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    num_classes: int,
    class_means: torch.Tensor,
    device: str,
    thold: Optional[torch.Tensor] = None,
    text_features: Optional[torch.Tensor] = None,
    dataset_name: Optional[str] = None,
) -> np.ndarray:
    model = model.to(device)
    model.eval()

    text_features = text_features.to(device)
    class_means = F.normalize(class_means, dim=-1).float()

    all_scores = []
    total_size = 0

    class_idx = np.arange(num_classes)

    with torch.inference_mode():
        for feats_batch_initial, logits_batch_initial in test_loader:
            feats_batch_initial = feats_batch_initial.to(device)

            image_features = F.normalize(feats_batch_initial, dim=-1).float()
            text_features = F.normalize(text_features, dim=-1).float()
            probs = torch.sigmoid(image_features @ text_features.T)
            preds_initial = probs.argmax(1)

            total_size += feats_batch_initial.size(0)
            trajectory_list = torch.zeros(
                feats_batch_initial.size(0), num_classes, device=device
            )
            for class_id in class_idx:
                pred_text_feats = text_features[preds_initial]
                target_text_feats = text_features[class_id]

                diff = pred_text_feats - target_text_feats
                diff_norm = F.normalize(diff, p=2, dim=1)

                # image_features: BxD, diff_norm: BxD
                # image_feats_db = (
                #     image_features
                #     - torch.sum(torch.multiply(image_features, diff_norm), dim=1).view(
                #         -1, 1
                #     )
                #     * diff_norm
                # )
                image_feats_db = (
                    image_features
                    - torch.einsum("ij,ij->i", image_features, diff_norm).view(-1, 1)
                    * diff_norm
                )

                feats_batch_db = F.normalize(image_feats_db, p=2, dim=1)

                centered_feats = image_features - torch.mean(class_means, dim=0)
                centered_feats_db = feats_batch_db - torch.mean(class_means, dim=0)

                norm_centered_feats = F.normalize(centered_feats, p=2, dim=1)
                norm_centered_feats_db = F.normalize(centered_feats_db, p=2, dim=1)

                cos_sim_origin_perspective = torch.sum(
                    norm_centered_feats * norm_centered_feats_db, dim=1
                )
                angles_origin = torch.arccos(cos_sim_origin_perspective) / torch.pi

                # # fdbd original
                # distance_to_db = torch.linalg.norm(image_features - feats_batch_db, dim=1)
                # fdbd_score = distance_to_db / torch.linalg.norm(image_features - torch.mean(class_means, dim=0), dim=1)

                trajectory_list[:, class_id] = angles_origin

            trajectory_list[torch.isnan(trajectory_list)] = 0
            ood_score = torch.max(trajectory_list, dim=1).values
            # ood_score = torch.topk(trajectory_list, 2, largest=False, dim=1).values[:, 1]
            # ood_score = torch.mean(trajectory_list, dim=1)
            all_scores.append(ood_score)
    scores = np.asarray(torch.cat(all_scores).detach().cpu().numpy(), dtype=np.float32)
    return scores


def lafo_score2(
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
            if thold is not None:
                feats_batch_initial = react_filter(feats_batch_initial, thold).float()
                logits_batch_initial = model.fc(feats_batch_initial)
            preds_initial = logits_batch_initial.argmax(1)
            max_logits = logits_batch_initial.max(dim=1).values
            total_size += feats_batch_initial.size(0)
            trajectory_list = torch.zeros(
                feats_batch_initial.size(0), num_classes, device=device
            )
            for class_id2 in [1, 250, 500, 750, 999]:
                for class_id in class_idx:
                    logit_diff = max_logits - logits_batch_initial[:, class_id]
                    weight_diff = (
                        model.fc.weight[preds_initial] - model.fc.weight[class_id]
                    )
                    weight_diff_norm = torch.linalg.norm(weight_diff, dim=1)

                    feats_batch_db = (
                        feats_batch_initial
                        - torch.divide(logit_diff, weight_diff_norm**2).view(-1, 1)
                        * weight_diff
                    )

                    # centered_feats = feats_batch_initial - torch.mean(class_means, dim=0)
                    # centered_feats_db = feats_batch_db - torch.mean(class_means, dim=0)
                    centered_feats = feats_batch_initial - class_means[class_id2]
                    centered_feats_db = feats_batch_db - class_means[class_id2]

                    norm_centered_feats = F.normalize(centered_feats, p=2, dim=1)
                    norm_centered_feats_db = F.normalize(centered_feats_db, p=2, dim=1)

                    cos_sim_origin_perspective = torch.sum(
                        norm_centered_feats * norm_centered_feats_db, dim=1
                    )
                    angles_origin = torch.arccos(cos_sim_origin_perspective) / torch.pi
                    trajectory_list[:, class_id] += angles_origin

            trajectory_list[torch.isnan(trajectory_list)] = 0
            ood_score = torch.max(trajectory_list, dim=1).values
            # ood_score = torch.topk(trajectory_list, 2, largest=False, dim=1).values[:, 1]
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
            # logits_batch_initial = logits_batch_initial.to(device)
            logits_batch_initial = model.fc(feats_batch_initial.float())
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
    dataset_name: Optional[str] = None,
) -> np.ndarray:
    model = model.to(device)
    model.eval()
    all_scores = []
    with torch.inference_mode():
        for feats_batch_initial, logits_batch_initial in test_loader:
            feats_batch_initial = feats_batch_initial.to(device)
            logits_batch_initial = model.fc(feats_batch_initial.float())
            # logits_batch_initial = logits_batch_initial.to(device)
            probs = F.softmax(logits_batch_initial, dim=1)
            max_probs = torch.max(probs, dim=1).values
            all_scores.append(max_probs)
            # if dataset_name:
            # breakpoint()
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


def MCM_score(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    num_classes: int,
    class_means: torch.Tensor,
    device: str,
    thold: Optional[torch.Tensor] = None,
    text_features: Optional[torch.Tensor] = None,
    dataset_name: Optional[str] = None,
) -> np.ndarray:
    model = model.to(device)
    model.eval()

    text_features = text_features.to(device)
    class_means = F.normalize(class_means, dim=-1).float()

    all_scores = []
    total_size = 0

    class_idx = np.arange(num_classes)

    with torch.inference_mode():
        for feats_batch_initial, logits_batch_initial in test_loader:
            feats_batch_initial = feats_batch_initial.to(device)

            image_features = F.normalize(feats_batch_initial, dim=-1).float()
            text_features = F.normalize(text_features, dim=-1).float()
            # probs = torch.sigmoid(image_features @ text_features.T)
            # preds_initial = probs.argmax(1)
            # calculate softmax probabilities too
            probs = F.softmax(image_features @ text_features.T, dim=1)
            ood_score = torch.max(probs, dim=1).values
            all_scores.append(ood_score)
    scores = np.asarray(torch.cat(all_scores).detach().cpu().numpy(), dtype=np.float32)
    return scores


def plain_cos_score(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    num_classes: int,
    class_means: torch.Tensor,
    device: str,
    thold: Optional[torch.Tensor] = None,
    text_features: Optional[torch.Tensor] = None,
    dataset_name: Optional[str] = None,
) -> np.ndarray:
    model = model.to(device)
    model.eval()

    text_features = text_features.to(device)
    class_means = F.normalize(class_means, dim=-1).float()

    all_scores = []
    total_size = 0

    class_idx = np.arange(num_classes)

    with torch.inference_mode():
        for feats_batch_initial, logits_batch_initial in test_loader:
            feats_batch_initial = feats_batch_initial.to(device)

            image_features = F.normalize(feats_batch_initial, dim=-1).float()
            text_features = F.normalize(text_features, dim=-1).float()
            probs = image_features @ text_features.T
            # preds_initial = probs.argmax(1)
            # calculate softmax probabilities too
            # probs = F.softmax(image_features @ text_features.T, dim=1)
            ood_score = torch.max(probs, dim=1).values
            all_scores.append(ood_score)
    scores = np.asarray(torch.cat(all_scores).detach().cpu().numpy(), dtype=np.float32)
    return scores


args = get_args()

seed = args.seed
print(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

class_num = 1000
id_train_size = 1281167
id_val_size = 50000

cache_dir = f"cache/{args.in_dataset}_train_{args.name}_in"
feat_log = torch.from_numpy(
    np.memmap(
        f"{cache_dir}/feat.mmap", dtype=float, mode="r", shape=(id_train_size, 1024)
    )
).to(device)
score_log = torch.from_numpy(
    np.memmap(
        f"{cache_dir}/score.mmap",
        dtype=float,
        mode="r",
        shape=(id_train_size, class_num),
    )
).to(device)
label_log = torch.from_numpy(
    np.memmap(f"{cache_dir}/label.mmap", dtype=float, mode="r", shape=(id_train_size,))
).to(device)


cache_dir = f"cache/{args.in_dataset}_val_{args.name}_in"
feat_log_val = torch.from_numpy(
    np.memmap(
        f"{cache_dir}/feat.mmap", dtype=float, mode="r", shape=(id_val_size, 1024)
    )
).to(device)
score_log_val = torch.from_numpy(
    np.memmap(
        f"{cache_dir}/score.mmap", dtype=float, mode="r", shape=(id_val_size, class_num)
    )
).to(device)
label_log_val = torch.from_numpy(
    np.memmap(f"{cache_dir}/label.mmap", dtype=float, mode="r", shape=(id_val_size,))
).to(device)


from transformers import CLIPVisionModel

# model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
# linear_probe = LinearProbe(featdim, num_classes)

from open_clip import create_model_from_pretrained, get_tokenizer

model, preprocess = create_model_from_pretrained("hf-hub:apple/DFN5B-CLIP-ViT-H-14-384")
tokenizer = get_tokenizer("ViT-H-14")

with open("imagenet_classes.txt") as f:
    labels_list = f.read().splitlines()

text = tokenizer(labels_list, context_length=model.context_length)
text_features = model.encode_text(text)


model = model.to(device)


ood_feat_score_log = {}
ood_dataset_size = {"inat": 10000, "sun50": 10000, "places50": 10000, "dtd": 5640}

for ood_dataset in args.out_datasets:
    ood_feat_log = torch.from_numpy(
        np.memmap(
            f"cache/{ood_dataset}vs{args.in_dataset}_{args.name}_out/feat.mmap",
            dtype=float,
            mode="r",
            shape=(ood_dataset_size[ood_dataset], 1024),
        )
    ).to(device)
    ood_score_log = torch.from_numpy(
        np.memmap(
            f"cache/{ood_dataset}vs{args.in_dataset}_{args.name}_out/score.mmap",
            dtype=float,
            mode="r",
            shape=(ood_dataset_size[ood_dataset], class_num),
        )
    ).to(device)
    ood_feat_score_log[ood_dataset] = ood_feat_log, ood_score_log


######## get w, b; precompute demoninator matrix, training feature mean  #################

if args.name == "resnet50":
    net = models.resnet50(pretrained=True)
    for i, param in enumerate(net.fc.parameters()):
        if i == 0:
            w = param.data
        else:
            b = param.data

elif args.name == "resnet50-supcon":
    checkpoint = torch.load("ckpt/ImageNet_resnet50_supcon_linear.pth")
    w = checkpoint["model"]["fc.weight"]
    b = checkpoint["model"]["fc.bias"]

train_mean = torch.mean(feat_log, dim=0).to(device)

denominator_matrix = torch.zeros((1000, 1000)).to(device)
# for p in range(1000):
#   w_p = w - w[p,:]
#   denominator = torch.norm(w_p, dim=1)
#   denominator[p] = 1
#   denominator_matrix[p, :] = denominator

#################### fDBD score OOD detection #################

all_results = []
all_score_out = []

values, nn_idx = score_log_val.max(1)
logits_sub = torch.abs(score_log_val - values.repeat(1000, 1).T)
# pdb.set_trace()
# score_in = torch.sum(logits_sub/denominator_matrix[nn_idx], axis=1)/torch.norm(feat_log_val - train_mean , dim = 1)
# score_in = score_in.float().cpu().numpy()

model = LinearProbe(1024, 1000)
model.load_state_dict(torch.load("./linear_probe.pth"))
model = model.to(device)

in_dataset = torch.utils.data.TensorDataset(feat_log_val.cpu(), score_log_val.cpu())
in_loader = torch.utils.data.DataLoader(
    in_dataset, batch_size=128, shuffle=False, num_workers=2
)

class_means = []
for i in range(1000):
    class_means.append(feat_log_val[label_log_val == i].mean(0))
class_means = torch.stack(class_means).to(device)

# tholds = react_thold(feat_log, percentile=75)
tholds = None
# tholds = torch.from_numpy(tholds).to(device)
score_in = lafo_score(
    model,
    in_loader,
    1000,
    class_means,
    device,
    thold=tholds,
    text_features=text_features,
    dataset_name=None,
)
# score_in = MCM_score(
#     model,
#     in_loader,
#     1000,
#     class_means,
#     device,
#     thold=tholds,
#     text_features=text_features,
#     dataset_name=None,
# )
# score_in = plain_cos_score(
#     model,
#     in_loader,
#     1000,
#     class_means,
#     device,
#     thold=tholds,
#     text_features=text_features,
#     dataset_name=None,
# )
# score_in = energy_score(model, in_loader, 1000, class_means, device, tholds)
# score_in = msp_score(model, in_loader, 1000, class_means, device, tholds)
# score_in = fdbd_score(model, in_loader, 1000, class_means, device)
# score_in = React_energy(model, in_loader, 1000, class_means, device, tholds)
# breakpoint()
train_feats = feat_log[:]
# score_in = knn_score(model, in_loader, 1000, class_means, device, train_feats)


for ood_dataset, (feat_log, score_log) in ood_feat_score_log.items():
    print(ood_dataset)
    values, nn_idx = score_log.max(1)
    logits_sub = torch.abs(score_log - values.repeat(1000, 1).T)
    # scores_out_test = torch.sum(logits_sub/denominator_matrix[nn_idx], axis=1)/torch.norm(feat_log - train_mean , dim = 1)
    # scores_out_test = scores_out_test.float().cpu().numpy()
    out_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(feat_log.cpu(), score_log.cpu()),
        batch_size=128,
        shuffle=False,
        num_workers=2,
    )
    scores_out_test = lafo_score(
        model,
        out_loader,
        1000,
        class_means,
        device,
        thold=tholds,
        text_features=text_features,
        dataset_name=ood_dataset,
    )
    # scores_out_test = MCM_score(
    #     model,
    #     out_loader,
    #     1000,
    #     class_means,
    #     device,
    #     thold=tholds,
    #     text_features=text_features,
    #     dataset_name=ood_dataset,
    # )
    # scores_out_test = plain_cos_score(
    #     model,
    #     out_loader,
    #     1000,
    #     class_means,
    #     device,
    #     thold=tholds,
    #     text_features=text_features,
    #     dataset_name=ood_dataset,
    # )
    # scores_out_test = energy_score(model, out_loader, 1000, class_means, device, tholds)
    # scores_out_test = msp_score(model, out_loader, 1000, class_means, device, tholds, dataset_name=ood_dataset)
    # scores_out_test = fdbd_score(model, out_loader, 1000, class_means, device)
    # scores_out_test = React_energy(model, out_loader, 1000, class_means, device, tholds)
    # scores_out_test = knn_score(model, out_loader, 1000, class_means, device, train_feats)
    # plot histograms
    import matplotlib.pyplot as plt

    plot_helper(
        score_in, scores_out_test, args.in_dataset, ood_dataset, function_name="LAFO"
    )

    # scores_out_test = torch.sum(logits_sub/denominator_matrix[nn_idx], axis=1)/torch.norm(feat_log - train_mean , dim = 1)
    # scores_out_test = scores_out_test.float().cpu().numpy()
    all_score_out.extend(scores_out_test)
    results = metrics.cal_metric(score_in, scores_out_test)
    all_results.append(results)

metrics.print_all_results(all_results, args.out_datasets, "fDBD")
print()
