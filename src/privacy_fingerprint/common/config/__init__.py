import os
from typing import Dict, Optional

import yaml

from privacy_fingerprint.common.config.experiment_config import (
    ExperimentConfig,
)
from privacy_fingerprint.common.config.global_config import GlobalConfig

global_config = None
experiment_config = None

ENV_PREFIX = "PRIVACY_FINGERPRINT"


def load_global_config(config: Optional[Dict] = None) -> GlobalConfig:
    """Return the available global config, loading from the config parameter if supplied

    :param config: Dictionary of config values
    :returns: Global config object
    """
    global global_config
    if config is not None:
        global_config = GlobalConfig(**config)
    return global_config


def load_experiment_config(config: Optional[Dict] = None) -> ExperimentConfig:
    """Return the available experiment config, loading from the config parameter if supplied

    :param config: Dictionary of config values
    :returns: Experiment config object
    """
    global experiment_config
    if config is not None:
        experiment_config = ExperimentConfig(**config)
    return experiment_config


def load_global_config_from_file(
    filename: str, environ: Optional[Dict] = None
) -> GlobalConfig:
    """Load the global config for the privacy fingerprint tool

    Config values are first loaded from the file and then if matching values are found in
    the environment or the passed environ parameter they are overwritten. The format is
    PRIVACY_FINGERPRINT__LEVEL1__LEVEL2 with two underscores separating the prefix and each level.

    :param filename: Full path to the config file location
    :param environ: Dictionary of values used to override values in the file
    :returns: Config object"""
    with open(filename) as fp:
        config = yaml.safe_load(fp)
    if environ is None:
        environ = os.environ
    for k, v in environ.items():
        if not k.startswith(ENV_PREFIX):
            continue
        key_parts = k.lower().split("__")
        current_level = config
        for key_part in key_parts[1:]:
            if key_part in current_level:
                if isinstance(current_level[key_part], dict):
                    current_level = current_level[key_part]
                    continue
            current_level[key_part] = v
    return load_global_config(config)


def load_experiment_config_from_file(filename: str) -> ExperimentConfig:
    """Load the config for an experiment from a file

    :param filename: Full path to the config file location
    :returns: Config object"""
    with open(filename) as fp:
        config = yaml.safe_load(fp)
    return load_experiment_config(config)
