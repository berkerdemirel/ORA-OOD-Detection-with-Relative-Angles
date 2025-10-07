import os
import pickle
import time
from typing import Optional

import faiss
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from omegaconf import DictConfig
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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

CLASS_NUM = 1000
ID_TRAIN_SIZE = 1281167
ID_VAL_SIZE = 50000
FEAT_DIM_MAP = {
    "resnet50": 2048,
    "resnet50-supcon": 2048,
    "convnext": 1024,
    "convnextpre": 1024,
    "deit": 768,
    "deitpre": 768,
    "densenet": 1024,
    "eva": 1024,
    "mobile": 1280,
    "swin": 1024,
    "swinpre": 1024,
}
OOD_DATASET_SIZE = {
    "inat": 10000,
    "SUN": 10000,
    "sun50": 10000,
    "iSUN": 10000,
    "places50": 10000,
    "places365": 10000,
    "dtd": 5640,
    "ssbhard": 49000,
    "ninco": 5878,
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


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(args: DictConfig) -> None:
    seed = args.seed
    print(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    feat_dim = FEAT_DIM_MAP[args.model_arch]

    feat_log, score_log, label_log = load_features(
        args.cache_dir_train, ID_TRAIN_SIZE, feat_dim, CLASS_NUM, device
    )
    feat_log_val, score_log_val, label_log_val = load_features(
        args.cache_dir_val, ID_VAL_SIZE, feat_dim, CLASS_NUM, device
    )

    print("ID ACC:", calculate_acc_val(score_log_val, label_log_val))

    ood_feat_score_log = {}
    for ood_dataset in args.out_datasets:
        path = f"cache/{ood_dataset}vs{args.in_dataset}_{args.name}_out"
        ood_feat_log, ood_score_log = load_features(
            path,
            OOD_DATASET_SIZE[ood_dataset],
            feat_dim,
            CLASS_NUM,
            device,
            has_label=False,
        )
        ood_feat_score_log[ood_dataset] = ood_feat_log, ood_score_log

    # train_mean = torch.mean(feat_log, dim=0).to(device)

    all_results = []
    all_score_out = []

    from util.model_loader import load_model

    model = load_model(args, 1000)
    if args.model_arch in ["convnext", "convnextpre"]:
        model.fc = model.head.fc
    elif args.model_arch in ["deit", "deitpre", "eva"]:
        model.fc = model.head

    in_dataset = torch.utils.data.TensorDataset(feat_log_val.cpu(), score_log_val.cpu())
    in_loader = torch.utils.data.DataLoader(
        in_dataset, batch_size=2048, shuffle=False, num_workers=2
    )

    # train_dataset = torch.utils.data.TensorDataset(feat_log.cpu(), score_log.cpu())
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=2048, shuffle=False, num_workers=2
    # )

    class_means = []
    for i in range(1000):
        class_means.append(feat_log_val[label_log_val == i].mean(0))
    class_means = torch.stack(class_means).to(device)

    # tholds = react_thold(feat_log, percentile=85)
    tholds = None
    # tholds = torch.from_numpy(tholds).to(device)
    score_in = ora_score(model, in_loader, 1000, class_means, device, tholds)
    # score_in = mahalanobisplus_score(feat_log, label_log, in_loader, 1000, device)
    # score_in = neco_score(feat_log, in_loader, "cache/neco", use_logit=True, device=device)
    # score_in = nci_score(model, in_loader, train_mean, device, alpha=0.0)

    # score_in = rcos_score(model, in_loader, 1000, class_means, device, tholds)
    # score_in = plaincos_score(model, in_loader, 1000, class_means, device, tholds)
    # score_in = rel_mahalonobis_distance_score(
    #     model, feat_log, label_log, in_loader, 1000, class_means, device, tholds
    # )
    # score_in = neco_score(
    #     model, feat_log, label_log, in_loader, 1000, class_means, device, tholds
    # )
    # score_in = mds_score(
    #     model, feat_log, label_log, in_loader, 1000, class_means, device, tholds
    # )

    # score_in = energy_score(model, in_loader, 1000, class_means, device, tholds)
    # score_in = msp_score(model, in_loader, 1000, class_means, device, tholds)
    # score_in = maxlogit_score(model, in_loader, 1000, class_means, device, tholds)
    # score_in = fdbd_score(model, in_loader, 1000, class_means, device)
    # score_in = React_energy(model, in_loader, 1000, class_means, device, tholds)
    # train_feats = feat_log[:]
    # train_labels = label_log[:]
    # score_in = knn_score(model, in_loader, 1000, class_means, device, train_feats)
    # score_in = nnguide_score(
    #     model, train_loader, in_loader, 1000, class_means, device, train_feats
    # )

    for ood_dataset, (feat_log, score_log) in ood_feat_score_log.items():
        print(ood_dataset)
        # mean_id = torch.mean(class_means, dim=0)
        # values, nn_idx = score_log.max(1)
        # logits_sub = torch.abs(score_log - values.repeat(1000, 1).T)
        # scores_out_test = torch.sum(logits_sub/denominator_matrix[nn_idx], axis=1)/torch.norm(feat_log - train_mean , dim = 1)
        # scores_out_test = scores_out_test.float().cpu().numpy()
        out_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(feat_log.cpu(), score_log.cpu()),
            batch_size=512,
            shuffle=False,
            num_workers=2,
        )

        scores_out_test = ora_score(
            model, out_loader, 1000, class_means, device, tholds
        )
        # scores_out_test = neco_score(
        #     train_feats, out_loader, "cache/neco", use_logit=True, device=device
        # )
        # scores_out_test = mahalanobisplus_score(
        #     train_feats, train_labels, out_loader, 1000, device
        # )
        # scores_out_test = nci_score(model, out_loader, mean_id, device, alpha=0.0)
        # scores_out_test = rcos_score(model, out_loader, 1000, class_means, device, tholds)
        # scores_out_test = plaincos_score(
        #     model, out_loader, 1000, class_means, device, tholds
        # )
        # scores_out_test = rel_mahalonobis_distance_score(
        #     model, train_feats, train_labels, out_loader, 1000, class_means, device, tholds
        # )
        # scores_out_test = neco_score(
        #     model, train_feats, train_labels, out_loader, 1000, class_means, device, tholds
        # )
        # scores_out_test = mds_score(
        #     model, train_feats, train_labels, out_loader, 1000, class_means, device, tholds
        # )
        # scores_out_test = energy_score(model, out_loader, 1000, class_means, device, tholds)
        # scores_out_test = msp_score(model, out_loader, 1000, class_means, device, tholds)
        # scores_out_test = maxlogit_score(
        #     model, out_loader, 1000, class_means, device, tholds
        # )
        # scores_out_test = fdbd_score(model, out_loader, 1000, class_means, device)
        # scores_out_test = React_energy(model, out_loader, 1000, class_means, device, tholds)
        # scores_out_test = knn_score(
        #     model, out_loader, 1000, class_means, device, train_feats
        # )
        # scores_out_test = nnguide_score(
        #     model, train_loader, out_loader, 1000, class_means, device, train_feats
        # )

        # plot histograms
        # import matplotlib.pyplot as plt
        # plot_helper(score_in, scores_out_test, args.in_dataset, ood_dataset, function_name="ORA")

        # scores_out_test = torch.sum(logits_sub/denominator_matrix[nn_idx], axis=1)/torch.norm(feat_log - train_mean , dim = 1)
        # scores_out_test = scores_out_test.float().cpu().numpy()
        all_score_out.extend(scores_out_test)
        results = metrics.cal_metric(score_in, scores_out_test)
        all_results.append(results)

    metrics.print_all_results(all_results, args.out_datasets, "ORA")
    print()
    print()
    print()


if __name__ == "__main__":
    main()
