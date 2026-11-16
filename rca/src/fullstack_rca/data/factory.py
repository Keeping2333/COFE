"""
Dataset factory: config-driven dataset creation.

Usage:
    from fullstack_rca.data.factory import create_dataset, load_config
    config = load_config()                      # reads default config/dataset.yaml
    dataset = create_dataset(config["dataset"])  # returns the appropriate Dataset
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from torch.utils.data import Dataset

# Default config path: rca/config/dataset.yaml
_DEFAULT_CONFIG_PATH = str(
    Path(__file__).resolve().parents[3] / "config" / "dataset.yaml"
)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load dataset configuration from a YAML file.

    Args:
        config_path: Path to YAML config file. If None, uses the default
                     ``rca/config/dataset.yaml``.

    Returns:
        Parsed configuration dict.
    """
    path = config_path or _DEFAULT_CONFIG_PATH
    with open(path) as f:
        config = yaml.safe_load(f)
    return config


def create_dataset(dataset_config: Dict[str, Any]) -> Dataset:
    """Create a Dataset instance based on the config dict.

    Args:
        dataset_config: Dict with keys ``type``, ``path``, and optional ``params``.

    Returns:
        An instance of MicroServiceDataset, ExpRunsDataset, or SNDataset.

    Raises:
        ValueError: If the dataset type is unknown.
    """
    ds_type = dataset_config["type"]
    ds_path = dataset_config["path"]
    params = dict(dataset_config.get("params", {}))

    # Normalize param names across dataset types
    # anomod/sn use max_traces_per_scenario, exp_runs uses max_traces_per_experiment
    max_traces = params.pop("max_traces_per_experiment", None)
    max_traces_scenario = params.pop("max_traces_per_scenario", None)

    if ds_type == "anomod":
        from .dataset import MicroServiceDataset
        return MicroServiceDataset(
            data_dir=ds_path,
            max_traces_per_scenario=max_traces_scenario or max_traces or 200,
            **params,
        )
    elif ds_type in ("locust", "exp_runs"):
        from .exp_runs_dataset import ExpRunsDataset
        return ExpRunsDataset(
            data_dir=ds_path,
            max_traces_per_experiment=max_traces or max_traces_scenario or 200,
            **params,
        )
    elif ds_type == "sn":
        from .sn_dataset import SNDataset
        return SNDataset(
            data_dir=ds_path,
            max_traces_per_scenario=max_traces_scenario or max_traces or 50,
            **params,
        )
    else:
        raise ValueError(
            f"Unknown dataset type: {ds_type!r}. "
            f"Supported types: 'anomod', 'locust', 'sn'."
        )
