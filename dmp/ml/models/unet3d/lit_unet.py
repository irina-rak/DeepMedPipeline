from typing import Literal, TypedDict

import lightning.pytorch as pl
import torch

from monai.data import NibabelWriter
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureType,
    KeepLargestConnectedComponent,
)
from monai.inferers import sliding_window_inference
from monai.metrics import DiceHelper
from monai.networks.nets import UNet
from pydantic import BaseModel, ConfigDict

from dmp.console import console



class ConfigUnet(BaseModel):
    """A Pydantic Model to validate the LitPBR config given by the user.

    Attributes
    ----------
    in_channels: int
        number of channels of the input
    out_channels: int
        number of channels of the output
    lr: float
        the learning rate
    """

    spatial_dims: int = 3
    in_channels: int
    out_channels: int
    channels: list[int]
    strides: list[int]
    num_res_units: int
    norm: str = "BATCH"
    patch_size: list[int] = [96, 96, 96]
    save_dir: str = "outputs"

    model_config = ConfigDict(extra="forbid")


class ConfigModel_UNet(BaseModel):
    """Pydantic BaseModel to validate Configuration for "U-Net" Model.

    Attributes
    ----------
    name:
        designation "unet" to choose
    config:
        configuration for the model LitUnet
    """

    name: Literal["unet"]
    config: ConfigUnet

    model_config = ConfigDict(extra="forbid")


class UnetSignature(TypedDict, total=False):
    """A TypedDict to represent the signature of both training and validation steps of U-Net model.

    Used in particular in train and test loops, to gather information on how many metrics are returned by the model.

    Attributes
    ----------
    dice_avg: torch.Tensor (optional)
        Average Dice score, calculated if labels are provided.
    message: str (optional)
        Message indicating inference completion, used when no labels are provided.
    """

    dice_avg: torch.Tensor
    message: str


def get_postprocessing(label_dims: int = 4) -> tuple[Compose, Compose]:
    post_pred = Compose(
        [
            EnsureType(),
            AsDiscrete(argmax=True, to_onehot=label_dims),
            KeepLargestConnectedComponent(is_onehot=True),
        ]
    )
    post_label = Compose([EnsureType(),  AsDiscrete(to_onehot=label_dims)])
    return post_pred, post_label


class LitUnet(pl.LightningModule):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: list[int],
        strides: list[int],
        num_res_units: int,
        norm: str,
        patch_size: list[int],
        save_dir: str,
        _logging: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            norm=norm,
        )
        self.dice_score = DiceHelper(
            include_background=False,
            softmax=True,
            reduction="mean",
            get_not_nans=False,
            num_classes=out_channels,
        )
        self.patch_size = patch_size
        self.save_dir = save_dir
        self._logging = _logging
        self._signature = UnetSignature

        self._post_pred, self._post_label = get_postprocessing(label_dims=out_channels)

    @property
    def signature(self):
        return self._signature
    
    def perform_inference(self, images: torch.Tensor, img_name: str) -> str:
        """Perform inference on the given images and save the output.
    
        Parameters
        ----------
        images : torch.Tensor
            The input images for inference.
        img_name : str
            The name of the image to use for saving the output.
    
        Returns
        -------
        str
            The path to the saved output file.
        """
        with torch.no_grad():
            outputs = sliding_window_inference(
                images, roi_size=self.patch_size, sw_batch_size=4, predictor=self.model
            )
            outputs = torch.argmax(outputs, dim=1)

        output_path = f"{self.save_dir}/output_{img_name[0]}"
        
        writer = NibabelWriter()
        writer.set_data_array(outputs, channel_dim=0)
        writer.set_metadata({"affine": images[0].affine, "original_affine": images[0].affine})
        writer.write(f"{output_path}.nii.gz", verbose=True)
        console.log(f"Output saved to {output_path}.nii.gz")

        return output_path
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> UnetSignature:
        """Validation step to compute evaluation metrics.
    
        Parameters
        ----------
        batch : torch.Tensor
            The input batch containing images and labels.
        batch_idx : int
            The index of the batch.
    
        Returns
        -------
        UnetSignature
            A dictionary containing evaluation metrics.
        """
        images = batch["image"]
        labels = batch["label"]  # Assume labels are always provided by the DataLoader
    
        # Perform inference
        outputs = sliding_window_inference(
            images, roi_size=self.patch_size, sw_batch_size=4, predictor=self.model
        )
    
        # Compute evaluation metrics
        dice = self.dice_score(outputs, labels)
        return {"dice_avg": dice.item()}