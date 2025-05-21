from src.ml.data.pbr import LitPBRDataModule
from dmp.ml.models.unet3d.lit_unet import LitUnet

model_registry = {
    "pbr": LitUnet,
}

datamodule_registry = {
    "unet": LitPBRDataModule,
}
