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
from sklearn.covariance import EmpiricalCovariance
from util import metrics
from util.ood_methods import (
    React_energy,
    calculate_acc_val,
    energy_score,
    fdbd_score,
    knn_score,
    mahalanobisplus_score,
    maxlogit_score,
    mds_score,
    msp_score,
    nci_score,
    neco_score,
    nnguide_score,
    ora_score,
    plaincos_score,
    plot_helper,
    rcos_score,
    react_filter,
    react_thold,
    rel_mahalonobis_distance_score,
    rel_mahalonobis_distance_score_2,
)

# --- Constants ---
CLASS_NUM = 1000
ID_TRAIN_SIZE = 1281167
ID_VAL_SIZE = 50000
OOD_DATASET_SIZE = {"inat": 10000, "sun50": 10000, "places50": 10000, "dtd": 5640}
FEAT_DIM = 1024


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

    model = LinearProbe(FEAT_DIM, CLASS_NUM)
    model.load_state_dict(torch.load("./linear_probe.pth"))
    model = model.to(device)

    in_dataset = torch.utils.data.TensorDataset(feat_log_val.cpu(), score_log_val.cpu())
    in_loader = torch.utils.data.DataLoader(
        in_dataset, batch_size=128, shuffle=False, num_workers=2
    )

    class_means = []
    for i in range(CLASS_NUM):
        class_means.append(feat_log_val[label_log_val == i].mean(0))
    class_means = torch.stack(class_means).to(device)

    tholds = None
    score_in = ora_score(model, in_loader, 1000, class_means, device, tholds)

    for ood_dataset, (feat_log, score_log) in ood_feat_score_log.items():
        print(ood_dataset)
        out_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(feat_log.cpu(), score_log.cpu()),
            batch_size=128,
            shuffle=False,
            num_workers=2,
        )
        scores_out_test = ora_score(
            model, out_loader, 1000, class_means, device, tholds
        )
        plot_helper(
            score_in,
            scores_out_test,
            cfg.in_dataset,
            ood_dataset,
            function_name="ORA",
        )
        all_score_out.extend(scores_out_test)
        results = metrics.cal_metric(score_in, scores_out_test)
        all_results.append(results)

    metrics.print_all_results(all_results, cfg.out_datasets, "ORA")
    print()


if __name__ == "__main__":
    main()
