import os
from typing import Any, Callable, Tuple

import torch
from torchvision.datasets.utils import (
    download_and_extract_archive,
    check_integrity,
    download_url,
)
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def identity(x):
    return x


class EuroSAT(torch.utils.data.Dataset):
    """
    Pytorch utility to extract the RGB data out of the EuroSAT [1] dataset using the split proposed in [2].

    [1] Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019). Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification.
        IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 12(7), 2217-2226.

    [2] Neumann, M., Pinto, A. S., Zhai, X., & Houlsby, N. (2019). In-domain representation learning for remote sensing.
        arXiv preprint arXiv:1911.06721.
    """

    url = "http://madm.dfki.de/files/sentinel/EuroSAT.zip"  # 2.0 GB download
    filename = "EuroSAT.zip"
    md5 = "c8fa014336c82ac7804f0398fcb19387"

    base_dir = "2750"

    splits = ["train", "val", "test"]
    split_urls = {
        "train": "https://storage.googleapis.com/remote_sensing_representations/eurosat-train.txt",
        "val": "https://storage.googleapis.com/remote_sensing_representations/eurosat-val.txt",
        "test": "https://storage.googleapis.com/remote_sensing_representations/eurosat-test.txt",
    }
    split_filenames = {
        "train": "eurosat-train.txt",
        "val": "eurosat-val.txt",
        "test": "eurosat-test.txt",
    }
    split_md5s = {
        "train": "908f142e73d6acdf3f482c5e80d851b1",
        "val": "95de90f2aa998f70a3b2416bfe0687b4",
        "test": "7ae5ab94471417b6e315763121e67c5f",
    }
    classes = [
        "Industrial",
        "Residential",
        "AnnualCrop",
        "PermanentCrop",
        "River",
        "SeaLake",
        "HerbaceousVegetation",
        "Highway",
        "Pasture",
        "Forest",
    ]
    classes_dict = {c: i for i, c in enumerate(classes)}

    def __init__(
        self,
        root: str,
        split: str = "train",
        download: bool = False,
        checksum: bool = True,
        transform: Callable = identity,
        target_transform: Callable = identity,
    ):
        self.root = root
        self.split = split
        assert isinstance(self.split, str) and self.split in self.splits

        self.checksum = checksum
        self.download = download
        self.transform = transform
        self.target_transform = target_transform

        self._download()
        if not self._check_integrity():
            raise Exception("Could not validate the files for EuroSAT")

        with open(os.path.join(self.root, self.split_filenames[self.split]), "r") as f:
            filenames = f.readlines()
        classes = [path.split("_")[0] for path in filenames]

        self.objects = [
            (os.path.join(cls, path.rstrip("\n")), self.classes_dict[cls])
            for path, cls in zip(filenames, classes)
        ]

    def _download(self):
        if not self.download:
            return
        download_and_extract_archive(
            self.url,
            self.root,
            self.root,
            filename=self.filename,
            md5=self.md5 if self.checksum else None,
        )
        for split in self.splits:
            download_url(
                self.split_urls[split],
                self.root,
                filename=self.split_filenames[split],
                md5=self.split_md5s[split] if self.checksum else None,
            )

    def _check_integrity(self) -> bool:
        if not self.checksum:
            return True

        return check_integrity(
            os.path.join(self.root, self.filename), md5=self.md5
        ) and all(
            check_integrity(
                os.path.join(self.root, self.split_filenames[split]),
                md5=self.split_md5s[split],
            )
            for split in self.splits
        )

    def __len__(self) -> int:
        return len(self.objects)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, cls = self.objects[index]
        path = os.path.join(self.root, self.base_dir, path)
        assert os.path.exists(path)
        with open(path, "rb") as f:
            img = Image.open(f)
            img.load()
        return self.transform(img.convert("RGB")), self.target_transform(cls)
