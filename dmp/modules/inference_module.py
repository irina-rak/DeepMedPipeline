from collections import OrderedDict
from typing import Union, Optional

import torch
from lightning.fabric import Fabric
from lightning.pytorch import LightningDataModule, LightningModule
from pydantic import BaseModel, ConfigDict, Field

from dmp.console import console
from dmp.ml.loops_fabric import test_loop



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
    data_dir: str
        the path to the data directory, relatively to which can be found Data, Experiments and other useful directories
    model: dict
        a dictionnary holding all necessary keywords for the LightningModule used
    data: dict
        a dictionnary holding all necessary keywords for the LightningDataModule used.
    """

    root_dir: str
    data_dir: str
    model: Union[ConfigModel_Cifar10, ConfigModel_Noop,  ConfigModel_Paroma, ConfigModel_ChestXray, ConfigModel_BreastCancer] = Field(discriminator="name")
    data: Union[ConfigData_Cifar10, ConfigData_Paroma, ConfigData_ChestXray, ConfigData_BreastCancer] = Field(discriminator="name")

    model_config = ConfigDict(extra="forbid")


class InferenceModule:
    """A Fabric-based, LightningModule-based module for inference.
    This module is used to perform inference on a given model and data.
    It is a subclass of Fabric and uses the LightningModule and LightningDataModule
    classes from PyTorch Lightning.
    """

    def __init__(
        self,
        model: LightningModule,
        data: LightningDataModule,
        num_examples: dict[str, int],
        conf_fabric: ConfigFabric,
    ) -> None:
        """Initialize the InferenceModule.

        Parameters
        ----------
        model: LightningModule
            The LightningModule to use for inference.
        data: LightningDataModule
            The LightningDataModule to use for inference.
        num_examples: dict[str, int]
            A dictionary containing the number of examples in the training and validation datasets.
        conf_fabric: ConfigFabric
            The configuration for the Fabric module.
        """
        self.model = model
        self.data = data
        self.num_examples = num_examples

        self.num_examples = num_examples

        self.fabric = Fabric(**dict(conf_fabric))

    def initialize(self):
        self.fabric.launch()
        self.model = self.fabric.setup(self.model)
        self._test_dataloader = self.fabric.setup_dataloaders(self.data.test_dataloader())

    def evaluate(self, config):
        """Evaluate the model on the test dataset.
        
        Parameters
        ----------
        config: dict
            The configuration for the evaluation. This should include the server round number.

        Returns:
        --------
        metrics: dict
            A dictionary containing the evaluation metrics.
        """

        metrics = {}
        console.log(f"Round {config['server_round']}, evaluation Started...")
        results_evaluate = test_loop(
            self.fabric, self.model, self._test_dataloader
        )
        for key, val in results_evaluate.items():
            metrics[key] = val
        return metrics