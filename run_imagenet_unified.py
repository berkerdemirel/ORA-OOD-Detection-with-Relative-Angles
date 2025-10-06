import os
import pickle
import time
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
from util import metrics
from util.args_loader import get_args
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

if __name__ == "__main__":

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

    # A dictionary to map model architecture to feature dimension
    feat_dim_map = {
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
    feat_dim = feat_dim_map[args.model_arch]

    cache_dir = f"cache/{args.in_dataset}_train_{args.name}_in"
    feat_log = torch.from_numpy(
        np.memmap(
            f"{cache_dir}/feat.mmap",
            dtype=float,
            mode="r",
            shape=(id_train_size, feat_dim),
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
        np.memmap(
            f"{cache_dir}/label.mmap", dtype=float, mode="r", shape=(id_train_size,)
        )
    ).to(device)

    cache_dir = f"cache/{args.in_dataset}_val_{args.name}_in"
    feat_log_val = torch.from_numpy(
        np.memmap(
            f"{cache_dir}/feat.mmap",
            dtype=float,
            mode="r",
            shape=(id_val_size, feat_dim),
        )
    ).to(device)
    score_log_val = torch.from_numpy(
        np.memmap(
            f"{cache_dir}/score.mmap",
            dtype=float,
            mode="r",
            shape=(id_val_size, class_num),
        )
    ).to(device)
    label_log_val = torch.from_numpy(
        np.memmap(
            f"{cache_dir}/label.mmap", dtype=float, mode="r", shape=(id_val_size,)
        )
    ).to(device)

    print("ID ACC:", calculate_acc_val(score_log_val, label_log_val))

    ood_feat_score_log = {}
    ood_dataset_size = {
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

    for ood_dataset in args.out_datasets:
        ood_feat_log = torch.from_numpy(
            np.memmap(
                f"cache/{ood_dataset}vs{args.in_dataset}_{args.name}_out/feat.mmap",
                dtype=float,
                mode="r",
                shape=(ood_dataset_size[ood_dataset], feat_dim),
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

    train_mean = torch.mean(feat_log, dim=0).to(device)

    all_results = []
    all_score_out = []

    from util.model_loader import get_model

    model = get_model(args, 1000, load_ckpt=True)
    if args.model_arch in ["convnext", "convnextpre"]:
        model.fc = model.head.fc
    elif args.model_arch in ["deit", "deitpre", "eva"]:
        model.fc = model.head

    in_dataset = torch.utils.data.TensorDataset(feat_log_val.cpu(), score_log_val.cpu())
    in_loader = torch.utils.data.DataLoader(
        in_dataset, batch_size=2048, shuffle=False, num_workers=2
    )

    train_dataset = torch.utils.data.TensorDataset(feat_log.cpu(), score_log.cpu())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2048, shuffle=False, num_workers=2
    )

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
    # breakpoint()
    train_feats = feat_log[:]
    train_labels = label_log[:]
    # score_in = knn_score(model, in_loader, 1000, class_means, device, train_feats)
    # score_in = nnguide_score(
    #     model, train_loader, in_loader, 1000, class_means, device, train_feats
    # )

    for ood_dataset, (feat_log, score_log) in ood_feat_score_log.items():
        print(ood_dataset)
        mean_id = torch.mean(class_means, dim=0)
        values, nn_idx = score_log.max(1)
        logits_sub = torch.abs(score_log - values.repeat(1000, 1).T)
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
