from pathlib import Path
from typing import Annotated, OrderedDict

import torch
import typer
from lightning.fabric import Fabric
from omegaconf import OmegaConf
from pydantic import ValidationError

from dmp.console import console
from dmp.modules.inference_module import ConfigInference, InferenceModule
from dmp.modules.utils import compute_average_scores
from dmp.ml.registry import datamodule_registry, model_registry



def check_and_build_client_config(config: dict) -> tuple[dict, dict, dict]:
    _conf = ConfigInference(**config)
    console.log(_conf)

    conf       = dict(_conf)
    conf_paths = conf["paths"]
    conf_fabric = conf["fabric"]
    conf_data  = dict(conf["data"].config)
    conf_model = dict(conf["model"].config)
    return conf, conf_paths, conf_fabric, conf_data, conf_model

app = typer.Typer(pretty_exceptions_show_locals=False, rich_markup_mode="rich")

@app.callback()
def client():
    """The client part of Pybiscus for Paroma.

    It is made of two commands:

    * The command launch launches a client with a specified config file, to take part to a Federated Learning.
    * The command check checks if the provided configuration file satisfies the Pydantic constraints.
    """

# defines the "check" command in Typer app
@app.command(name="check")
def check_client_config(
    config:        Annotated[ Path, typer.Argument()],
    root_dir:      Annotated[ str,  typer.Option(rich_help_panel="Overriding some parameters") ] = None,
) -> None:
    """Check the provided client configuration file.

    The command loads the configuration file and checks the validity of the configuration using Pydantic.
    If the configuration is alright with respect to ConfigClient Pydantic BaseModel, nothing happens.
    Otherwise, raises the ValidationError by Pydantic -- which is quite verbose and should be useful understanding the issue with the configuration provided.

    You may pass optional parameters (in addition to the configuration file itself) to override the parameters given in the configuration.

    Parameters
    ----------
    config : Path
        the Path to the configuration file.
    num_rounds : int, optional
        the number of round of Federated Learnconf_fabricing, by default None
    server_adress : str, optional
        the server adress and port, by default None
    to_onnx : bool, optional
        if true, saves the final model into ONNX format. Only available now for Unet3D model! by default False

    Raises
    ------
    typer.Abort
        _description_
    typer.Abort
        _description_
    typer.Abort
        _description_
    ValidationError
        _description_
    """
    if config is None:
        print("No config file")
        raise typer.Abort()
    if config.is_file():
        conf_loaded = OmegaConf.load(config)
    elif config.is_dir():
        print("Config is a directory, will use all its config files")
        raise typer.Abort()
    elif not config.exists():
        print("The config doesn't exist")
        raise typer.Abort()

    if root_dir is not None:
        conf_loaded["root_dir"] = root_dir
    try:
        _ = check_and_build_client_config(conf_loaded)
        console.log("This is a valid config!")
    except ValidationError as e:
        console.log("This is not a valid config!")
        raise e


@app.command(name="launch")
def run_inference(
    config: Annotated[Path, typer.Argument()],
    root_dir: Annotated[str, typer.Option(rich_help_panel="Overriding some parameters")] = None,
    output_dir: Annotated[str, typer.Option(rich_help_panel="Directory to save inference results")] = None,
) -> None:
    """Run inference or validation using the provided configuration."""
    if config is None:
        print("No config file")
        raise typer.Abort()
    if config.is_file():
        conf_loaded = OmegaConf.load(config)
    elif config.is_dir():
        print("Config is a directory, will use all its config files")
        raise typer.Abort()
    elif not config.exists():
        print("The config doesn't exist")
        raise typer.Abort()

    if root_dir is not None:
        conf_loaded["root_dir"] = root_dir

    # Parse configuration
    conf, conf_paths, conf_fabric, conf_data, conf_model = check_and_build_client_config(config=conf_loaded)

    # Determine the output directory
    config_output_dir = conf_paths.get("output_dir", "./outputs")
    final_output_dir = output_dir if output_dir else config_output_dir

    # Ensure the output directory exists
    output_path = Path(final_output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load the model from the specified path
    model_path = conf_paths.get("model_path")
    if not model_path:
        console.log("Model path is not specified in the configuration!")
        raise typer.Abort()
    model_path = Path(model_path)
    if not model_path.exists():
        console.log(f"Model file does not exist at {model_path}")
        raise typer.Abort()
    
    # Initialize Fabric
    fabric = Fabric(**dict(conf_fabric))
    fabric.launch()

    mode = conf.get("mode", "inference").lower()

    if mode not in ["inference", "validation"]:
        console.log(f"Invalid mode: {mode}. Choose either 'inference' or 'validation'.")
        raise typer.Abort()

    # Initialize and run inference or validation
    console.log(f"Starting {mode}...")
    console.log("Initializing inference module...")

    # Initialize data module and model
    data = datamodule_registry[conf["data"].name](**conf_data, mode=mode)
    data.setup(stage="test")
    net = model_registry[conf["model"].name](**conf_model)

    # Load model weights
    console.log(f"Loading model weights from {model_path}...")
    try:
        state = fabric.load(model_path)["state_dict"]
    except KeyError:
        console.log(f"Model file {model_path} does not contain 'state_dict'. Trying to load 'model'.")
        state = fabric.load(model_path)["model"]
    net.load_state_dict(state, strict=True)

    # Prepare the InferenceModule
    num_examples = {"testset": len(data.test_dataloader())}
    conf_fabric = conf.get("fabric", {"accelerator": "auto", "devices": "auto"})
    inference_module = InferenceModule(
        fabric=fabric,
        model=net,
        data=data,
        num_examples=num_examples,
        # conf_fabric=ConfigInference(**conf_fabric),
    )
    inference_module.initialize()
    
    # TO DO: Update results calculation to include per-class metrics
    results = inference_module.run(mode=mode)
    if mode == "validation":
        results["average_scores"] = OrderedDict(compute_average_scores(results))

    # Log results
    console.log(f"{mode.capitalize()} results:")
    for key, value in results.items():
        console.log(f"{key}: {value}")

    # Save results to a file
    results_file = output_path / f"{mode}_results.json"
    with open(results_file, "w") as f:
        import json
        json.dump(results, f, indent=4)
    console.log(f"Results saved to {results_file}")
    


if __name__ == "__main__":
    app()
