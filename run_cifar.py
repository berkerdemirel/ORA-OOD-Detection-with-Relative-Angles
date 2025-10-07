import os
import pdb
import time
from typing import Optional

import faiss
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from util import metrics
from util.model_loader import load_model
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

CLASS_NUM = 10
ID_TRAIN_SIZE = 50000
ID_VAL_SIZE = 10000
FEAT_DIM = 512
OOD_DATASET_SIZE = {
    "SVHN": 26032,
    "iSUN": 8925,
    "places365": 10000,
    "dtd": 5640,
    "cifar100": 50000,
}


def load_features(path, size, feat_dim, class_num, device, has_label=True):
    """Load features, scores, and optionally labels from memory-mapped files."""
    feat = torch.from_numpy(
        np.memmap(f"{path}/feat.mmap", dtype=float, mode="r", shape=(size, feat_dim))
    ).to(device)
    score = torch.from_numpy(
        np.memmap(f"{path}/score.mmap", dtype=float, mode="r", shape=(size, class_num))
    ).to(device)
    if has_label:
        label = torch.from_numpy(
            np.memmap(f"{path}/label.mmap", dtype=float, mode="r", shape=(size,))
        ).to(device)
        return feat, score, label
    return feat, score


@hydra.main(version_base=None, config_path="configs", config_name="config_cifar")
def main(args: DictConfig) -> None:
    seed = args.seed
    print(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    feat_log, score_log, label_log = load_features(
        args.cache_dir_train, ID_TRAIN_SIZE, FEAT_DIM, CLASS_NUM, device
    )
    feat_log_val, score_log_val, label_log_val = load_features(
        args.cache_dir_val, ID_VAL_SIZE, FEAT_DIM, CLASS_NUM, device
    )

    print("ID ACC:", calculate_acc_val(score_log_val, label_log_val))

    ood_feat_score_log = {}
    for ood_dataset in args.out_datasets:
        ood_feat_log, ood_score_log = load_features(
            f"cache/{ood_dataset}vs{args.in_dataset}_{args.name}_out",
            OOD_DATASET_SIZE[ood_dataset],
            FEAT_DIM,
            CLASS_NUM,
            device,
            has_label=False,
        )
        ood_feat_score_log[ood_dataset] = ood_feat_log, ood_score_log

    all_results = []
    all_score_out = []

    model = load_model(args, 10)

    in_dataset = torch.utils.data.TensorDataset(feat_log_val.cpu(), score_log_val.cpu())
    in_loader = torch.utils.data.DataLoader(
        in_dataset, batch_size=128, shuffle=False, num_workers=2
    )

    class_means = []
    for i in range(10):
        class_means.append(feat_log_val[label_log_val == i].mean(0))
    class_means = torch.stack(class_means).to(device)

    tholds = None
    score_in = ora_score(model, in_loader, 10, class_means, device, tholds)

    for ood_dataset, (feat_log, score_log) in ood_feat_score_log.items():
        print(ood_dataset)
        out_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(feat_log.cpu(), score_log.cpu()),
            batch_size=128,
            shuffle=False,
            num_workers=2,
        )

        scores_out_test = ora_score(model, out_loader, 10, class_means, device, tholds)

        all_score_out.extend(scores_out_test)
        results = metrics.cal_metric(score_in, scores_out_test)
        all_results.append(results)

    metrics.print_all_results(all_results, args.out_datasets, "LAFO")
    print()


if __name__ == "__main__":
    main()
