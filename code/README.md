# Learning Trajectories of Figurative Language for Pre-Trained Language Models — Code

This directory contains the official implementation for the paper *Learning Trajectories of Figurative Language for Pre-Trained Language Models* (EMNLP 2025).

## Contents

- [mdl_configs](mdl_configs): example experiment configurations, one for each figure of speech and experimental setting.
- [source](source): core source code for training and evaluation.  
- [environment.yml](environment.yml): specification file for creating the conda environment.  

## Installation

The codebase requires **Python 3.10**. We recommend creating a dedicated conda environment using the provided specification:

```bash
conda env create -f environment.yml
```

### Usage

Experiments are launched via the run_experiment entry point, passing the path to the desired configuration file.
Configuration files specify the model, dataset splits, training hyperparameters, and evaluation metrics.
All outputs (logs, checkpoints, metrics) are stored in the directory defined in the configuration.

For example:
```bash
python run_experiment --experiment_config mdl_configs/metaphor_config.yml
```
