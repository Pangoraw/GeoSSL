import os
from typing import Any, Callable, Tuple
from torch.utils.data import Dataset
from torchvision.datasets.utils import (
    download_url,
    download_file_from_google_drive,
    check_integrity,
)
from PIL import Image


class Resisc45(Dataset):
    """
    Pytorch utility to extract the RGB data out of the NWPU-RESISC45 [1] dataset.

    [1] Cheng, G., Han, J., & Lu, X. (2017). Remote sensing image scene classification: Benchmark and state of the art.
        Proceedings of the IEEE, 105(10), 1865-1883.
    """

    url = "https://drive.google.com/file/d/1DnPSU5nVSN7xv95bpZ3XQ0JhKXZOKgIv"
    md5 = "d824acb73957502b00efd559fc6cfbbb"
    filename = "NWPU-RESISC45.rar"
    directory = "NWPU-RESISC45"

    splits = ["train", "val", "test"]
    split_urls = {
        "train": "https://storage.googleapis.com/remote_sensing_representations/resisc45-train.txt",
        "val": "https://storage.googleapis.com/remote_sensing_representations/resisc45-val.txt",
        "test": "https://storage.googleapis.com/remote_sensing_representations/resisc45-test.txt",
    }
    split_filenames = {
        "train": "resisc45-train.txt",
        "val": "resisc45-val.txt",
        "test": "resisc45-test.txt",
    }
    split_md5s = {
        "train": "b5a4c05a37de15e4ca886696a85c403e",
        "val": "a0770cee4c5ca20b8c32bbd61e114805",
        "test": "3dda9e4988b47eb1de9f07993653eb08",
    }

    classes = [
        "airplane",
        "airport",
        "baseball_diamond",
        "basketball_court",
        "beach",
        "bridge",
        "chaparral",
        "church",
        "circular_farmland",
        "cloud",
        "commercial_area",
        "dense_residential",
        "desert",
        "forest",
        "freeway",
        "golf_course",
        "ground_track_field",
        "harbor",
        "industrial_area",
        "intersection",
        "island",
        "lake",
        "meadow",
        "medium_residential",
        "mobile_home_park",
        "mountain",
        "overpass",
        "palace",
        "parking_lot",
        "railway",
        "railway_station",
        "rectangular_farmland",
        "river",
        "roundabout",
        "runway",
        "sea_ice",
        "ship",
        "snowberg",
        "sparse_residential",
        "stadium",
        "storage_tank",
        "tennis_court",
        "terrace",
        "thermal_power_station",
        "wetland",
    ]
    classes_dict = {c: i for i, c in enumerate(classes)}

    def __init__(
        self,
        root: str,
        split: str = "train",
        download: bool = False,
        checksum: bool = True,
        transform: Callable = lambda x: x,
        target_transform: Callable = lambda x: x,
    ):
        self.root = root
        self.split = split
        self.download = download
        self.checksum = checksum
        self.transform = transform
        self.target_transform = target_transform

        self._download()
        if not self._check_integrity():
            raise Exception("Failed to verify integrity of files")

        with open(os.path.join(self.root, self.split_filenames[self.split]), "r") as f:
            filenames = f.readlines()
        classes = ["_".join(path.split("_")[:-1]) for path in filenames]

        self.objects = [
            (os.path.join(cls, path.rstrip("\n")), self.classes_dict[cls])
            for path, cls in zip(filenames, classes)
        ]

    def _download(self):
        if not self.download:
            return
        download_file_from_google_drive(
            "1DnPSU5nVSN7xv95bpZ3XQ0JhKXZOKgIv",
            self.root,
            filename=self.filename,
            md5=self.md5 if self.checksum else None,
        )

        if not os.path.exists(os.path.join(self.root, self.directory)):
            import rarfile

            with rarfile.RarFile(os.path.join(self.root, self.filename), "r") as f:
                f.extractall(self.root)

        for split in self.splits:
            download_url(
                self.split_urls[split],
                self.root,
                self.split_filenames[split],
                md5=self.split_md5s[split] if self.checksum else None,
            )

    def _check_integrity(self,) -> bool:
        if not self.checksum:
            return True
        return check_integrity(
            os.path.join(self.root, self.filename), md5=self.md5
        ) and all(
            [
                check_integrity(
                    os.path.join(self.root, self.split_filenames[split]),
                    md5=self.split_md5s[split],
                )
                for split in self.splits
            ]
        )

    def __len__(self) -> int:
        return len(self.objects)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, cls = self.objects[index]
        img = Image.open(os.path.join(self.root, self.directory, path))
        return self.transform(img), self.target_transform(cls)
