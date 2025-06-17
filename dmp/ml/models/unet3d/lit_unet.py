from typing import Literal, TypedDict

import lightning.pytorch as pl
import torch

from monai.data import decollate_batch, NibabelWriter
from monai.transforms import (
    AsDiscrete,
    Compose,
    EnsureType,
    KeepLargestConnectedComponent,
)
from monai.inferers import sliding_window_inference
from monai.metrics import DiceHelper, HausdorffDistanceMetric, SurfaceDistanceMetric
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
    hd_avg: torch.Tensor (optional)
        Hausdorff distance metric, calculated if labels are provided.
    sd_avg: torch.Tensor (optional)
        Surface distance metric, calculated if labels are provided.
    message: str (optional)
        Message indicating inference completion, used when no labels are provided.
    """

    dice_avg: torch.Tensor
    
    message: str


def get_postprocessing(label_dims: int = 4) -> tuple[Compose, Compose]:
    post_pred = Compose(
        [
            EnsureType(),
            AsDiscrete(argmax=True),
            KeepLargestConnectedComponent([0, 1, 2, 3], is_onehot=False),
        ]
    )
    post_label = Compose([EnsureType(),  AsDiscrete(argmax=True, to_onehot=label_dims)])
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
            reduction="none",  # Use None to return per-class scores
            get_not_nans=False,
            num_classes=out_channels,
        )
        self.hausdorff_distance = HausdorffDistanceMetric(
            include_background=False,
            reduction="none",  # Use None to return per-class scores
            percentile=95.0, # 95th percentile for Hausdorff distance
        )
        self.surface_distance = SurfaceDistanceMetric(
            include_background=False,
            reduction="none",
        )

        self.patch_size = patch_size
        self.save_dir = save_dir
        self._logging = _logging
        self._signature = UnetSignature

        self._post_pred, self._post_label = get_postprocessing(label_dims=out_channels)

    @property
    def signature(self):
        return self._signature
    
    # def perform_inference(self, images: torch.Tensor, img_name: str) -> str:
    #     """Perform inference on the given images and save the output.
    
    #     Parameters
    #     ----------
    #     images : torch.Tensor
    #         The input images for inference.
    #     img_name : str
    #         The name of the image to use for saving the output.
    
    #     Returns
    #     -------
    #     str
    #         The path to the saved output file.
    #     """
    #     with torch.no_grad():
    #         outputs = sliding_window_inference(
    #             images, roi_size=self.patch_size, sw_batch_size=4, predictor=self.model
    #         )
    #         outputs = [self._post_pred(output) for output in decollate_batch(outputs)][0]

    #     output_path = f"{self.save_dir}/output_{img_name[0]}"
        
    #     writer = NibabelWriter()
    #     writer.set_data_array(outputs, channel_dim=0)
    #     writer.set_metadata({"affine": images[0].affine, "original_affine": images[0].affine})
    #     writer.write(f"{output_path}.nii.gz", verbose=False)

    #     return output_path

    def perform_inference(self, images: torch.Tensor, img_names: list[str]) -> list[str]:
        """Perform inference on the given images and save the output.
    
        Parameters
        ----------
        images : torch.Tensor
            The input images for inference (batch).
        img_names : list[str]
            The names of the images to use for saving the outputs.
    
        Returns
        -------
        list[str]
            The paths to the saved output files.
        """
        output_paths = []
        
        with torch.no_grad():
            batch_outputs = sliding_window_inference(
                images,
                roi_size=self.patch_size,
                sw_batch_size=4,
                predictor=self.model
            )
            
            processed_outputs = [self._post_pred(output) for output in decollate_batch(batch_outputs)] # 4, H, W, D
            processed_outputs = torch.stack(processed_outputs, dim=0)  # Stack outputs into a batch: B, 1, H, W, D
            
            writer = NibabelWriter()
            
            for i, (output, img_name) in enumerate(zip(processed_outputs, img_names)):
                output_path = f"{self.save_dir}/output_{img_name}"
                
                writer.set_data_array(output, channel_dim=0)
                
                if hasattr(images[i], 'affine') and images[i].affine is not None:
                    affine_data = images[i].affine
                    writer.set_metadata({"affine": affine_data, "original_affine": affine_data})
                
                writer.write(f"{output_path}.nii.gz", verbose=False)
                output_paths.append(output_path)
        
        return output_paths, processed_outputs
    
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
        hd = self.hausdorff_distance(outputs, labels)
        sd = self.surface_distance(outputs, labels)
        
        return {"dice_avg": dice.item(), "hausdorff_distance": hd.item(), "surface_distance": sd.item()}
    
    def compute_metrics(self, outputs: torch.Tensor, labels: torch.Tensor) -> UnetSignature:
        """Compute evaluation metrics from the model outputs and labels.
    
        Parameters
        ----------
        outputs : torch.Tensor
            The model outputs.
        labels : torch.Tensor
            The ground truth labels.
    
        Returns
        -------
        UnetSignature
            A dictionary containing evaluation metrics.
        """
        label_names = {
            0: "Prostate",
            1: "Bladder",
            2: "Rectum",
        }

        dice = self.dice_score(outputs, labels).tolist()[0]
        dice_avg = torch.tensor(dice).mean().item()
        hd = self.hausdorff_distance(outputs, labels).tolist()[0]
        sd = self.surface_distance(outputs, labels).tolist()[0]

        # Match the output to the label_names
        dice = {label_names[i]: dice[i] for i in range(len(dice))}
        dice["dice_avg"] = dice_avg
        
        return {
            "dice": dice,
            "hd_avg": hd,
            "sd_avg": sd,
        }