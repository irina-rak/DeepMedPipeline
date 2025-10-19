from pathlib import Path
from typing import List, Literal, TypedDict, Union

import lightning.pytorch as pl
import torch
import torch.nn.functional as F

from monai.inferers import LatentDiffusionInferer
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler
from monai.utils import set_determinism
from pydantic import BaseModel, ConfigDict
from torch.amp import autocast

from pybiscus.core.pybiscus_logger import console
from klae.lit_klae import ConfigKLAutoEncoder, ConfigAlexDiscriminator, LitKLAutoEncoder


class ConfigDiffusionUnet(BaseModel):
    """A Pydantic Model to validate the generator config given by the user.

    Attributes
    ----------
    in_channels: int
        number of channels of the input
    out_channels: int
        number of channels of the output
    lr: float
        the learning rate
    """

    spatial_dims: int
    in_channels: int
    out_channels: int
    num_res_blocks: int
    channels: List[int]
    attention_levels: List[int]
    norm_num_groups: int
    num_head_channels: List[int]

    model_config = ConfigDict(extra="forbid")


class ConfigScheduler(BaseModel):
    """A Pydantic Model to validate the scheduler config given by the user.

    Attributes
    ----------
    num_train_timesteps: int
        number of training timesteps
    beta_start: float
        starting beta value for the scheduler
    beta_end: float
        ending beta value for the scheduler
    beta_schedule: Literal["linear", "cosine"]
        type of beta schedule to use
    """

    num_train_timesteps: int = 1000
    schedule: Literal["linear_beta", "cosine"] = "linear_beta"
    beta_start: float = 0.0015
    beta_end: float = 0.0195

    model_config = ConfigDict(extra="forbid")


class ConfigUnet(BaseModel):
    """A Pydantic Model to validate the LitPBR config given by the user.

    Attributes
    ----------
    unet_config: ConfigDiffusionUnet
        configuration for the UNet model
    scheduler_config: ConfigScheduler
        configuration for the scheduler
    lr: float
        learning rate for the optimizer
    autoencoder_warm_up_n_epochs: int
        number of epochs to warm up the autoencoder
    seed: Union[int, None]
        random seed for reproducibility
    _logging: bool
        whether to log training information
    """

    unet_config: ConfigDiffusionUnet
    scheduler_config: ConfigScheduler
    generator_config: ConfigKLAutoEncoder
    discriminator_config: ConfigAlexDiscriminator
    lr: float = 1e-4
    autoencoder_weights: Union[str, Path]
    seed: Union[int, None] = None
    _logging: bool = True

    model_config = ConfigDict(extra="forbid")


class ConfigModel_DiffusionUnet(BaseModel):
    """Pydantic BaseModel to validate Configuration for DiffusionUnet model.

    Attributes
    ----------
    name: Literal["dunet"]
        designation "dunet" to choose
    config:
        configuration for the model LitUnet
    """

    name: Literal["dunet"]
    config: ConfigUnet

    model_config = ConfigDict(extra="forbid")


class DUSignature(TypedDict):
    """A TypedDict to represent the signature of both training and validation steps of LDM model.

    Used in particular in train and test loops, to gather information on how many metrics are returned by the model.

    Attributes
    ----------
    loss: torch.Tensor
        The main loss
    """

    loss: torch.Tensor


class LitDiffusionUnet(pl.LightningModule):
    def __init__(
        self,
        unet_config: ConfigDiffusionUnet,
        scheduler_config: ConfigScheduler,
        generator_config: ConfigKLAutoEncoder,
        discriminator_config: ConfigAlexDiscriminator,
        lr: float = 1e-4,
        autoencoder_weights: Union[str, Path] = None,
        seed: Union[int, None] = None,
        _logging: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        if seed is not None and type(seed) is int:
            set_determinism(seed=seed)
            console.log(f"[bold][yellow]Setting seed to {seed}.[/yellow][/bold]")

        # Memory optimization settings
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.cuda.empty_cache()

        # Disable automatic optimization for manual control
        self.automatic_optimization = False

        self.autoencoderkl = LitKLAutoEncoder(
            # **(generator_config.model_dump() if hasattr(generator_config, "model_dump") else generator_config)
            generator_config=generator_config,
            discriminator_config=discriminator_config
        )
        try:
            self.autoencoderkl.load_state_dict(torch.load(autoencoder_weights, map_location=self.device)["state_dict"])
        except Exception as e:
            console.log(f"[bold][red]Error loading autoencoder weights: {e}[/red][/bold]")
            raise e
        self.autoencoder = self.autoencoderkl.autoencoderkl
        self.autoencoder.eval()  # That way, there's no gradient running around when the diffusion model is trained

        self.model = DiffusionModelUNet(
            **(unet_config.model_dump() if hasattr(unet_config, "model_dump") else unet_config)
        )

        self.scaling_factor = 1.0

        self.scheduler = DDPMScheduler(
            **(scheduler_config.model_dump() if hasattr(scheduler_config, "model_dump") else scheduler_config)
        )

        self.inferer = LatentDiffusionInferer(
            scheduler=self.scheduler,
        )

        self.lr = lr
        self._logging = _logging
        self._signature = DUSignature

    @property
    def signature(self):
        return self._signature

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)

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