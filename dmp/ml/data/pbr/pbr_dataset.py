from glob import glob
from os import path, listdir
from pathlib import Path

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
    ):
        self.data_dir = Path(data_dir)
        self.cache_rate = cache_rate
        self.num_workers = num_workers
        self.transforms = transforms

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
        # image_files = sorted(glob(str(self.data_dir / "CT" / "image.nii.gz")))
        # label_files = sorted(glob(str(self.data_dir / "Labels" / "combined_labels.nii.gz")))
        cases = listdir(str(self.data_dir))
        # image_files = [path.join(case, "CT", "image.nii.gz") for case in cases]
        # label_files = [path.join(case, "Labels", "combined_labels.nii.gz") for case in cases]

        # image_dict = {}
        # label_dict = {}
        # for case in cases:
        #     image_path = str(self.data_dir / case / "CT" / "image.nii.gz")
        #     label_path = str(self.data_dir / case / "Labels" / "combined_labels.nii.gz")
        #     if path.exists(image_path):
        #         image_dict[case] = image_path
        #     if path.exists(label_path):
        #         label_dict[case] = label_path

        images = []
        labels = []
        names = []
        for case in cases:
            image_path = str(self.data_dir / case / "CT" / "image.nii.gz")
            label_path = str(self.data_dir / case / "Labels" / "combined_labels.nii.gz")
            if path.exists(image_path):
                images.append(image_path)
                names.append(case)
            if path.exists(label_path):
                labels.append(label_path)

        # data = []
        # for filename in image_dict:
        #     if filename in label_dict:
        #         data.append({"image": image_dict[filename], "label": label_dict[filename]})
        #     else:
        #         print(f"Warning: no label found for {filename}")

        # data = [{"image": image_name, "label": label_name} for image_name, label_name in zip(images, labels)]
        data = [{"image": image_name, "label": label_name, "name": name} for image_name, label_name, name in zip(images, labels, names)]
        return data

    def get_dataset(self):
        return self.dataset
