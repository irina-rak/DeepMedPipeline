from dmp.ml.data.pbr.pbr_datamodule import LitPBRDataModule
from dmp.ml.models.unet3d.lit_unet import LitUnet

model_registry = {
    "unet": LitUnet,
}

datamodule_registry = {
    "pbr": LitPBRDataModule,
}
