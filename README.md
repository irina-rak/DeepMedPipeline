# DeepMedPipeline

DeepMedPipeline is a flexible repository for model inference and results visualization in medical imaging deep learning workflows. It is designed to streamline the process of running inference on medical imaging models, managing configurations, and visualizing results.

## Features
- **Flexible Inference Pipeline**: Easily run inference using pre-trained PyTorch deep learning models.
- **Configurable Workflows**: YAML-based configuration for data, model, and hardware settings.
- **Results Visualization**: Output results in standard formats for further analysis and visualization.
- **Command-Line Interface**: Simple CLI powered by [Typer](https://typer.tiangolo.com/) for running and validating inference jobs.
- **Extensible Design**: Modular codebase for easy extension to new models and datasets.

## Installation

### Requirements
- Python >= 3.11
- [Poetry](https://python-poetry.org/) (recommended for dependency management)

### Install with Poetry
```bash
poetry install
```

Or, using pip (not recommended):
```bash
pip install -r requirements.txt
```

## Usage

### Command-Line Interface
The main entry point is the CLI app:

```bash
poetry run dmp_app infer check CONFIG_PATH
```

- `infer check CONFIG_PATH`: Validates the provided YAML configuration file for inference.
- Additional commands for running inference and evaluation are available. See `--help` for details.

### Example Inference Command
```bash
poetry run dmp_app infer launch configs/pbr/infer.yml
```

### Using as a Python Package

If you have installed DeepMedPipeline as a package (e.g., via `pip install .` or `pip install deepmedpipeline`), you can use its functionality programmatically in your Python scripts:

```python
from dmp.main import app
# Use Typer app programmatically or import modules for custom workflows
```

You can also invoke the CLI directly if installed globally:

```bash
dmp_app infer launch configs/pbr/infer.yml
```

## Configuration

All pipeline settings are managed via YAML config files (see `configs/`). Example: `configs/pbr/infer.yml`:

```yaml
root_dir: /path/to/project
mode: validation
paths:
  model_path: /path/to/model
  output_dir: /path/to/output
# ...
```

- **root_dir**: Base directory for data and results
- **paths.model_path**: Path to the pre-trained model
- **paths.output_dir**: Directory to save inference results
- **data**: Data loading and preprocessing options
- **model**: Model architecture and parameters
- **fabric**: Hardware accelerator settings (e.g., GPU/CPU)

## Directory Structure

```
DeepMedPipeline/
├── configs/         # YAML configuration files
├── dmp/             # Main Python package
│   ├── commands/    # CLI commands
│   ├── ml/          # ML models and data modules
│   └── modules/     # Inference and utility modules
├── models/          # Pre-trained model weights
├── results/         # Inference outputs and results
├── main.py          # Entry point (if running as script)
├── pyproject.toml   # Project metadata and dependencies
└── README.md        # Project documentation
```

## Contributing

Contributions are welcome! Please open issues or submit pull requests for bug fixes, new features, or improvements.

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
