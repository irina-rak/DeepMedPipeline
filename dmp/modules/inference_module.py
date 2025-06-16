from collections import OrderedDict
from typing import Union, Optional

from lightning.fabric import Fabric
from lightning.pytorch import LightningDataModule, LightningModule
from pydantic import BaseModel, ConfigDict, Field
from rich.progress import Progress

from dmp.console import console
from dmp.ml.loops_fabric import test_loop
from dmp.ml.models.unet3d.lit_unet import ConfigModel_UNet
from dmp.ml.data.pbr.pbr_datamodule import ConfigData_PBR



class ConfigFabric(BaseModel):
    """A Pydantic Model to validate the Client configuration given by the user.

    This is a (partial) reproduction of the Fabric API found here:
    https://lightning.ai/docs/fabric/stable/api/generated/lightning.fabric.fabric.Fabric.html#lightning.fabric.fabric.Fabric

    Attributes
    ----------
    accelerator:
        the type of accelerator to use: gpu, cpu, auto... See the Fabric documentation for more details.
    devices: optional
        either an integer (the number of devices needed); a list of integers (the id of the devices); or
        the string "auto" to let Fabric choose the best option available.
    """

    accelerator: str
    devices: Union[int, list[int], str] = Field(default="auto")


class ConfigInference(BaseModel):
    """A Pydantic Model to validate the configuration of the InferenceModule.

    Attributes
    ----------
    root_dir: str
        the path to a "root" directory, relatively to which can be found Data, Experiments and other useful directories
    model: dict
        a dictionnary holding all necessary keywords for the LightningModule used
    data: dict
        a dictionnary holding all necessary keywords for the LightningDataModule used.
    """

    root_dir: str
    mode: Optional[str] = "inference"
    paths: Optional[dict[str, str]] = Field(
        default_factory=lambda: OrderedDict(
            {
                "output_dir": "./outputs",
                "model_path": "./model.pth",
            }
        )
    )
    fabric: ConfigFabric
    model: Union[ConfigModel_UNet] = Field(discriminator="name")
    data: Union[ConfigData_PBR] = Field(discriminator="name")

    model_config = ConfigDict(extra="forbid")


class InferenceModule:
    """A Fabric-based, LightningModule-based module for inference and validation."""

    def __init__(
        self,
        fabric: Fabric,
        model: LightningModule,
        data: LightningDataModule,
        num_examples: dict[str, int],
    ) -> None:
        """Initialize the InferenceModule.

        Parameters
        ----------
        model: LightningModule
            The LightningModule to use for inference or validation.
        data: LightningDataModule
            The LightningDataModule to use for inference or validation.
        num_examples: dict[str, int]
            A dictionary containing the number of examples in the datasets.
        conf_fabric: ConfigFabric
            The configuration for the Fabric module.
        """
        self.model = model
        self.data = data
        self.num_examples = num_examples
        self.fabric = fabric

    def initialize(self):
        """Initialize the Fabric module and set up the model and dataloaders."""
        self.fabric.launch()
        self.model = self.fabric.setup(self.model)
        self._test_dataloader = self.fabric.setup_dataloaders(self.data.test_dataloader())

    def run(self, mode: str = "validation") -> dict:
        """Run the model in either validation or inference mode.

        Parameters
        ----------
        mode : str
            The mode of operation. Either "validation" or "inference".

        Returns
        -------
        dict
            A dictionary containing evaluation metrics or inference results.
        """
        metrics = {}
        console.log(f"Running in {mode} mode...")

        if mode not in ["validation", "inference"]:
            raise ValueError(f"Unsupported mode: {mode}. Use 'validation' or 'inference'.")

        with Progress() as progress:
            task = progress.add_task("Running inference...", total=len(self._test_dataloader))
            outputs_list = []
            
            for batch_idx, batch in enumerate(self._test_dataloader):
                images = batch["image"]
                # If labels are present, they can be used for validation metrics
                labels = batch.get("label", None)
                names = batch["name"]
                
                paths, outputs = self.model.perform_inference(images, names)
                outputs_list.append({
                    "batch_idx": batch_idx,
                    "names": names,
                    "outputs": paths
                })

                if mode == "validation":
                    results = self.model.compute_metrics(outputs, labels)
                    outputs_list[-1]["metrics"] = OrderedDict(results)

                progress.advance(task)
            progress.update(task, completed=len(self._test_dataloader))
            
            metrics["inference_outputs"] = outputs_list

        return metrics