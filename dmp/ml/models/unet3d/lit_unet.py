from typing import Literal, TypedDict

import lightning.pytorch as pl
import torch

from monai.inferers import sliding_window_inference
from monai.metrics import DiceHelper
from monai.networks.nets import UNet
from pydantic import BaseModel, ConfigDict

from src.console import console



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
    lr: float

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

    @property
    def signature(self):
        return self._signature
        
    def validation_step(self, batch: torch.Tensor, img_name: str) -> UnetSignature:
        images = batch["image"]
        labels = batch.get("label", None)  # Check if "label" is provided in the batch
    
        outputs = sliding_window_inference(
            images, roi_size=self.patch_size, sw_batch_size=4, predictor=self.model
        )
    
        # Save the output image with monai
        output_path = f"{self.save_dir}/output_{img_name}.nii.gz"
    
        if labels is not None:
            dice = self.dice_score(outputs, labels)
            results = {
                "dice_avg": dice.item(),
            }
            return results
        else:
            # If no labels are provided, return an empty dictionary or a placeholder
            return {"message": f"Inference completed and saved to {output_path}"}