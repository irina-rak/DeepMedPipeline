from glob import glob
from os import path, listdir
from pathlib import Path
from typing import Literal

from monai.data import CacheDataset, Dataset
from monai.transforms import Compose

from dmp.console import console



class CTCacheDataset:
    def __init__(
        self,
        data_dir: str,
        cache_rate: float = 1.0,
        num_workers: int = 4,
        transforms: Compose = None,
        mode: Literal["inference", "validation"] = "inference",
    ):
        self.data_dir = Path(data_dir)
        self.cache_rate = cache_rate
        self.num_workers = num_workers
        self.transforms = transforms
        self.mode = mode

        # Prepare data list
        self.data = self.create_data_list()

        if cache_rate > 0.0:
            self.dataset = CacheDataset(
                data=self.data,
                transform=self.transforms,
                cache_rate=self.cache_rate,
                num_workers=self.num_workers,
            )
        else:
            self.dataset = Dataset(
                data=self.data,
                transform=self.transforms
            )

    def __len__(self):
        return len(self.dataset)

    def create_data_list(self):
        cases = sorted(case for case in listdir(self.data_dir) if not case.startswith('.'))
        data = []
        for case in cases:
            image_pattern = str(self.data_dir / case / "CT" / "*.nii.gz")
            image_path = glob(image_pattern)[0] # Take the first match
            label_pattern = str(self.data_dir / case / "Labels" / "*combined_labels.nii.gz")
            label_files = glob(label_pattern)[0] # Take the first match
            
            if path.exists(image_path) and path.exists(label_files):
                entry = {"image": image_path, "name": case}
                if label_files and self.mode == "validation":
                    entry["label"] = label_files
                data.append(entry)
        return data

    def get_dataset(self):
        return self.dataset
