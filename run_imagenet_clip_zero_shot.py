import os
import time
from typing import Optional

import faiss
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from omegaconf import DictConfig
from util import metrics

# --- Constants ---
CLASS_NUM = 1000
ID_TRAIN_SIZE = 1281167
ID_VAL_SIZE = 50000
OOD_DATASET_SIZE = {"inat": 10000, "sun50": 10000, "places50": 10000, "dtd": 5640}
FEAT_DIM = 1024


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
    plt.ylabel("Density", fontsize=16)
    plt.title(f"{in_dataset} vs {ood_dataset_name}", fontsize=16)
    plt.savefig(f"./{in_dataset}_{ood_dataset_name}_ORA.png", dpi=600)
    plt.close()


def ora_score_clip(
    test_loader: torch.utils.data.DataLoader,
    num_classes: int,
    class_means: torch.Tensor,
    device: str,
    thold: Optional[torch.Tensor] = None,
    text_features: Optional[torch.Tensor] = None,
    dataset_name: Optional[str] = None,
) -> np.ndarray:

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

                trajectory_list[:, class_id] = angles_origin

            trajectory_list[torch.isnan(trajectory_list)] = 0
            ood_score = torch.max(trajectory_list, dim=1).values
            # ood_score = torch.topk(trajectory_list, 2, largest=False, dim=1).values[:, 1]
            # ood_score = torch.mean(trajectory_list, dim=1)
            all_scores.append(ood_score)
    scores = np.asarray(torch.cat(all_scores).detach().cpu().numpy(), dtype=np.float32)
    return scores


@hydra.main(config_path="configs", config_name="config_clip", version_base=None)
def main(cfg: DictConfig):
    seed = cfg.seed
    print(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu

    cache_dir_train = f"cache/{cfg.in_dataset}_train_{cfg.name}_in"
    feat_log = torch.from_numpy(
        np.memmap(
            f"{cache_dir_train}/feat.mmap",
            dtype=float,
            mode="r",
            shape=(ID_TRAIN_SIZE, FEAT_DIM),
        )
    ).to(device)
    score_log = torch.from_numpy(
        np.memmap(
            f"{cache_dir_train}/score.mmap",
            dtype=float,
            mode="r",
            shape=(ID_TRAIN_SIZE, CLASS_NUM),
        )
    ).to(device)
    label_log = torch.from_numpy(
        np.memmap(
            f"{cache_dir_train}/label.mmap",
            dtype=float,
            mode="r",
            shape=(ID_TRAIN_SIZE,),
        )
    ).to(device)

    cache_dir_val = f"cache/{cfg.in_dataset}_val_{cfg.name}_in"
    feat_log_val = torch.from_numpy(
        np.memmap(
            f"{cache_dir_val}/feat.mmap",
            dtype=float,
            mode="r",
            shape=(ID_VAL_SIZE, FEAT_DIM),
        )
    ).to(device)
    score_log_val = torch.from_numpy(
        np.memmap(
            f"{cache_dir_val}/score.mmap",
            dtype=float,
            mode="r",
            shape=(ID_VAL_SIZE, CLASS_NUM),
        )
    ).to(device)
    label_log_val = torch.from_numpy(
        np.memmap(
            f"{cache_dir_val}/label.mmap", dtype=float, mode="r", shape=(ID_VAL_SIZE,)
        )
    ).to(device)

    from open_clip import create_model_from_pretrained, get_tokenizer

    model, preprocess = create_model_from_pretrained(
        "hf-hub:apple/DFN5B-CLIP-ViT-H-14-384"
    )
    tokenizer = get_tokenizer("ViT-H-14")

    with open("imagenet_classes.txt") as f:
        labels_list = f.read().splitlines()

    text = tokenizer(labels_list, context_length=model.context_length)
    text_features = model.encode_text(text)

    ood_feat_score_log = {}

    for ood_dataset in cfg.out_datasets:
        ood_feat_log = torch.from_numpy(
            np.memmap(
                f"cache/{ood_dataset}vs{cfg.in_dataset}_{cfg.name}_out/feat.mmap",
                dtype=float,
                mode="r",
                shape=(OOD_DATASET_SIZE[ood_dataset], FEAT_DIM),
            )
        ).to(device)
        ood_score_log = torch.from_numpy(
            np.memmap(
                f"cache/{ood_dataset}vs{cfg.in_dataset}_{cfg.name}_out/score.mmap",
                dtype=float,
                mode="r",
                shape=(OOD_DATASET_SIZE[ood_dataset], CLASS_NUM),
            )
        ).to(device)
        ood_feat_score_log[ood_dataset] = ood_feat_log, ood_score_log

    all_results = []
    all_score_out = []

    in_dataset = torch.utils.data.TensorDataset(feat_log_val.cpu(), score_log_val.cpu())
    in_loader = torch.utils.data.DataLoader(
        in_dataset, batch_size=128, shuffle=False, num_workers=2
    )

    class_means = []
    for i in range(CLASS_NUM):
        class_means.append(feat_log_val[label_log_val == i].mean(0))
    class_means = torch.stack(class_means).to(device)

    tholds = None
    score_in = ora_score_clip(
        in_loader,
        CLASS_NUM,
        class_means,
        device,
        thold=tholds,
        text_features=text_features,
        dataset_name=None,
    )

    train_feats = feat_log[:]

    for ood_dataset, (feat_log, score_log) in ood_feat_score_log.items():
        print(ood_dataset)
        out_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(feat_log.cpu(), score_log.cpu()),
            batch_size=128,
            shuffle=False,
            num_workers=2,
        )
        scores_out_test = ora_score_clip(
            out_loader,
            CLASS_NUM,
            class_means,
            device,
            thold=tholds,
            text_features=text_features,
            dataset_name=ood_dataset,
        )

        plot_helper(
            score_in, scores_out_test, cfg.in_dataset, ood_dataset, function_name="ORA"
        )

        all_score_out.extend(scores_out_test)
        results = metrics.cal_metric(score_in, scores_out_test)
        all_results.append(results)

    metrics.print_all_results(all_results, cfg.out_datasets, "ORA")
    print()


if __name__ == "__main__":
    main()
