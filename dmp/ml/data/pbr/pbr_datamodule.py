from typing import Literal, List, Optional

import lightning.pytorch as pl

from monai.data import DataLoader, NibabelReader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    CropForegroundd,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
)
from torch.utils.data._utils.collate import default_collate
from pydantic import BaseModel, ConfigDict

from dmp.console import console
from dmp.ml.data.pbr.pbr_dataset import CTCacheDataset


def get_transforms(
        # shape: tuple[int, int, int] = (500, 500, 250),
        patch_size: tuple[int, int, int] = (96, 96, 96),
        pixdim: tuple[float, float, float] = (1.0, 1.0, 2.0),
        margin: int = 45,
        keys: List[str] = ["image", "label"],
):
    """Get the transforms for training and validation. The training transforms
    include intensity normalization, patches extraction using RandGridPatchd, and data augmentation.
    The validation transforms include intensity normalization.

    Args:
        patch_size (tuple[int, int, int], optional): The size of the patches to be extracted. Defaults to (96, 96, 96).
        pixdim (tuple[float, float, float], optional): The pixel dimensions. Defaults to (1.0, 1.0, 2.0).

    Returns:
        tuple[Compose, Compose]: The training and validation transforms.
    """
    transforms = Compose([
        LoadImaged(keys=keys, reader=NibabelReader()),
        EnsureChannelFirstd(keys=keys),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-250,
            a_max=600,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        # CropForegroundd(keys=keys, source_key="image", margin=margin),
        # Orientationd(keys=keys, axcodes="RAS"),
        # Spacingd(keys=keys, pixdim=pixdim, mode=("bilinear", "nearest")),
        # NormalizeIntensityd(keys="image", nonzero=True, channel_wise=False)
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=pixdim, mode=("bilinear", "nearest")),
        # CropForegroundd(keys=keys, source_key="image", margin=margin),
        CropForegroundd(keys=keys, source_key="label", allow_smaller=True, margin=margin),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=False),
    ])

    return transforms


class ConfigPBR(BaseModel):
    """A Pydantic Model to validate the MedicalLitDataModule config givent by the user.

    Attributes
    ----------
    dir_train: str
        path to the directory holding the training data
    dir_val: str
        path to the directory holding the validating data
    dir_test: str, optional
        path to the directory holding the testing data
    batch_size: int, optional
        the batch size (default to 1)
    shape_img: tuple[float, float, float, float], optional
        the shape of the image (default to (96, 96, 96))
    shape_label: tuple[float, float, float, float], optional
        the shape of the label (default to (96, 96, 96))
    augment: bool, optional
        whether to use augmentation of data (default to False)
    preprocessed: bool, optional
        whether the data have already been preprocessed or not (default to True)
    num_workers: int, optional
        the number of workers for the DataLoaders (default to 0)
    """

    dir_train: str = None
    dir_val: str = None
    dir_test: str = None
    batch_size: int = 1
    shape_img: List = [96, 96, 96]
    augment: bool = False
    cache_rate: float = 1.0
    num_workers: int = 0
    margin: int = 45
    mode: Literal["inference", "validation"] = "inference"

    model_config = ConfigDict(extra="forbid")


class ConfigData_PBR(BaseModel):
    name: Literal["pbr"]
    config: ConfigPBR

    model_config = ConfigDict(extra="forbid")


class LitPBRDataModule(pl.LightningDataModule):
    """_summary_

    _extended_summary_

    Parameters
    ----------
    pl : _type_
        _description_
    """

    def __init__(
        self,
        # root_dir,
        dir_train: str,
        dir_val: str,
        dir_test: str,
        batch_size: int = 1,
        shape_img: tuple[float, float, float, float] = (96, 96, 96),
        augment: bool = False,
        cache_rate: float = 1.0,
        num_workers: int = 0,
        margin: int = 45,
        mode: Literal["inference", "validation"] = "inference",
    ):
        super().__init__()
        self.data_dir_train = dir_train
        self.data_dir_val = dir_val
        self.data_dir_test = dir_test
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.augment = augment
        self.cache_rate = cache_rate
        self.mode = mode

        self.transforms = get_transforms(
            patch_size=shape_img,
            pixdim=(1.0, 1.0, 2.0),
            margin=margin,
            keys=["image", "label"] if mode == "validation" else ["image"],
        )

    def setup(self, stage: Optional[str] = None):
        if self.cache_rate == 0.0:
            console.print("[red]Cache rate is set to 0.0, no caching will be performed.[/red]")

        if stage == "test" or stage is None:
            self.data_test = CTCacheDataset(
                data_dir=self.data_dir_test,
                transforms=self.transforms,
                cache_rate=self.cache_rate,
                num_workers=self.num_workers,
                mode=self.mode,
            ).get_dataset()

        else:
            raise ValueError(f"Unsupported stage: {stage}. Supported stages are 'test' or None.")

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
