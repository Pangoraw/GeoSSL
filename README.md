# GeoSSL

This repository contains the code for the paper "[*Self-Supervised Learning for Scene Classification in Remote Sensing: current State of the Art and Perspectives*](https://www.mdpi.com/2072-4292/14/16/3995)".

## Implemented Methods

 - [SimCLR](http://proceedings.mlr.press/v119/chen20j/chen20j.pdf)[1]
 - [Momentum Contrast (v2)](https://arxiv.org/pdf/2003.04297.pdf)[2] (MoCo v2)
 - [BarlowTwins](http://proceedings.mlr.press/v139/zbontar21a/zbontar21a.pdf)[3]
 - [Bootstrap Your Own Latent](https://proceedings.neurips.cc/paper/2020/file/f3ada80d5c4ee70142b17b8192b2958e-Paper.pdf)[4] (BYOL)

## Installing

`geossl` was tested with and has the following dependencies:

```
python>=3.7
pytorch=1.10.2+cu102
torchvision=0.11.3+cu102
torchgeo=0.3.0
```

Exact dependencies are specified in the [`environment.yml`](https://github.com/Pangoraw/geossl/blob/main/environment.yml) file and it can be used to create an associated conda environment:

```bash
conda env create --file ./environment.yml
```

## Usage

The [`train_evaluate.py`](https://github.com/Pangoraw/geossl/blob/main/train_evaluate.py) script perform a self-supervised pre-training and evaluate the pre-trained model on the linear-evaluation protocol and fine-tuning (with 1% and 10% respectively). An example usage is available in [`scripts/train_evaluate.sh`](https://github.com/Pangoraw/geossl/blob/main/scripts/train_evaluate.sh).

## Pre-trained weights

The following pre-trained weights are available to download:

|Backbone|Pre-training dataset|Method|Identifier|
|--------|--------------------|------|----------|
|ResNet18|EuroSAT|SimCLR|`"resnet18/eurosat/simclr"`|
|ResNet18|EuroSAT|MoCo v2|`"resnet18/eurosat/moco"`|
|ResNet18|EuroSAT|BYOL|`"resnet18/eurosat/byol"`|
|ResNet18|EuroSAT|Barlow Twins|`"resnet18/eurosat/barlow"`|

There is an helper on the backbone class to instantiate the model with the pre-trained weights:

```python
from geossl.backbones import ResNetBackbone

model = ResNetBackbone.from_pretrained("resnet18/eurosat/simclr")
```

> **Note**
> Pre-trained weights for the Resisc-45 dataset will be released soon.

## Citing

If this work is useful to your research, consider citing the paper associated with this code: "_Self-Supervised Learning for Scene Classification in Remote Sensing: current State of the Art and Perspectives_".

```bibtex

@Article{rs14163995,
	author = {Berg, Paul and Pham, Minh-Tan and Courty, Nicolas},
	title = {Self-Supervised Learning for Scene Classification in Remote Sensing: Current State of the Art and Perspectives},
	journal = {Remote Sensing},
	volume = {14},
	year = {2022},
	number = {16},
	article-number = {3995},
	url = {https://www.mdpi.com/2072-4292/14/16/3995},
	issn = {2072-4292},
	doi = {10.3390/rs14163995}
}
```

## References

> [1]Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020, November). A simple framework for contrastive learning of visual representations.
> In International conference on machine learning (pp. 1597-1607). PMLR.

> [2]Chen, X., Fan, H., Girshick, R., & He, K. (2020). Improved baselines with momentum contrastive learning.
> arXiv preprint arXiv:2003.04297.

> [3]Zbontar, J., Jing, L., Misra, I., LeCun, Y., & Deny, S. (2021, July). Barlow twins: Self-supervised learning via redundancy reduction.
> In International Conference on Machine Learning (pp. 12310-12320). PMLR.

> [4]Grill, J. B., Strub, F., Altché, F., Tallec, C., Richemond, P., Buchatskaya, E., ... & Valko, M. (2020). Bootstrap your own latent-a new approach to self-supervised learning.
> Advances in Neural Information Processing Systems, 33, 21271-21284.
