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

        # Create CacheDataset
        if cache_rate > 0.0:
            self.dataset = CacheDataset(
                data=self.data,
                transform=self.transforms,
                cache_rate=self.cache_rate,
                num_workers=self.num_workers,
                # copy_cache=False,
                # runtime_cache="processes",
            )
        else:
            self.dataset = Dataset(
                data=self.data,
                transform=self.transforms
            )

    def __len__(self):
        return len(self.dataset)

    def create_data_list(self):
        cases = listdir(str(self.data_dir))
        data = []
        for case in cases:
            image_path = str(self.data_dir / case / "CT" / "image.nii.gz")
            label_path = str(self.data_dir / case / "Labels" / "combined_labels.nii.gz")
            if path.exists(image_path):
                entry = {"image": image_path, "name": case}
                if path.exists(label_path) and self.mode == "validation":
                    entry["label"] = label_path
                data.append(entry)
        return data
    
    # def create_data_list(self):
    #     cases = listdir(str(self.data_dir))

    #     images = []
    #     labels = []
    #     names = []
    #     for case in cases:
    #         image_path = str(self.data_dir / case / "CT" / "image.nii.gz")
    #         label_path = str(self.data_dir / case / "Labels" / "combined_labels.nii.gz")
    #         if path.exists(image_path):
    #             images.append(image_path)
    #             names.append(case)
    #         if path.exists(label_path):
    #             labels.append(label_path)
                
    #     if len(labels) == 0:
    #         data = [{"image": image_name, "name": name} for image_name, name in zip(images, names)]
    #     else:
    #         data = [{"image": image_name, "label": label_name, "name": name} for image_name, label_name, name in zip(images, labels, names)]
    #     return data

    def get_dataset(self):
        return self.dataset
