# ORA: Out-of-Distribution Detection with Relative Angles

This repository contains the official implementation for the paper [**"Out-of-Distribution Detection with Relative Angles"**](https://openreview.net/pdf?id=ul5mlXrLZb) (NeurIPS 2025).

Our work introduces ORA, a novel method for Out-of-Distribution (OOD) detection that leverages the concept of relative angles in the feature space. By analyzing the angular displacement of a test sample with respect to class prototypes, ORA provides a robust and effective signal for distinguishing in-distribution (ID) data from OOD data.


## Workflow

The project is structured into two main stages: feature extraction and OOD detection.

### 1. Feature Extraction

First, you need to extract and cache the features from the backbone models. This is done using `feat_extract_all.py` for ImageNet-based models and `feat_extract_cifar.py` for CIFAR-based models. These scripts will save the feature representations, logits, and labels to disk, which allows for faster experimentation in the next stage.

### 2. OOD Detection

Once the features are cached, you can run the OOD detection experiments. The main scripts for this are:
- `run_imagenet_unified.py`: For experiments on ImageNet and its variants.
- `run_cifar.py`: For experiments on CIFAR-10.

These scripts load the pre-computed features and apply various OOD detection methods to evaluate their performance.

## Supported Components

### Backbone Models

The following backbone architectures are supported:

- `resnet18`
- `resnet18-supcon`
- `resnet50`
- `resnet50-supcon`
- `vit`
- `mobile`
- `convnext`
- `convnextpre`
- `swin`
- `swinpre`
- `deit`
- `deitpre`
- `densenet`
- `eva`
- `clip`

### OOD Datasets

The following OOD datasets are supported for evaluation against ImageNet:

- iNaturalist (`inat`)
- SUN (`sun50`)
- Places (`places50`)
- Describable Textures Dataset (`dtd`)

### OOD Detection Methods

This repository includes implementations for several OOD detection methods, including our proposed method, ORA. Implementations can be seen in utils/ood_methods.py

## Setup

The `download.sh` script is provided to download the necessary pretrained models (e.g., ResNets with SupCon loss) and the OOD datasets required to run the experiments.


## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@inproceedings{
demirel2025ood,
title={{OOD} Detection with Relative Angles},
author={Berker Demirel and Marco Fumero and Francesco Locatello},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=ul5mlXrLZb}
}
```

## Acknowledgements

Feature extraction and metric computations are adapted from the official repository for the fDBD paper. We thank the authors for making their code public.

- **fDBD Repository**: [https://github.com/litianliu/fDBD-OOD/tree/main](https://github.com/litianliu/fDBD-OOD/tree/main)