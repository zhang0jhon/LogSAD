"""Sample evaluation script for track 2."""

import argparse
import importlib
import importlib.util

import torch
import logging
from torch import nn

# NOTE: The following MVTecLoco import is not available in anomalib v1.0.1.
# It will be available in v1.1.0 which will be released on April 29th, 2024.
# If you are using an earlier version of anomalib, you could install anomalib
# from the anomalib source code from the following branch:
# https://github.com/openvinotoolkit/anomalib/tree/feature/mvtec-loco
from anomalib.data import MVTecLoco
from anomalib.metrics.f1_max import F1Max
from anomalib.metrics.auroc import AUROC
from tabulate import tabulate
import numpy as np

# FEW_SHOT_SAMPLES = [0, 1, 2, 3]

def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--module_path", type=str, required=True)
    parser.add_argument("--class_name", default='MyModel', type=str, required=False)
    parser.add_argument("--weights_path", type=str, required=False)
    parser.add_argument("--dataset_path", default='/home/bhu/Project/datasets/mvtec_loco_anomaly_detection/', type=str, required=False)
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument("--viz", action='store_true', default=False)
    return parser.parse_args()


def load_model(module_path: str, class_name: str, weights_path: str) -> nn.Module:
    """Load model.

    Args:
        module_path (str): Path to the module containing the model class.
        class_name (str): Name of the model class.
        weights_path (str): Path to the model weights.

    Returns:
        nn.Module: Loaded model.
    """
    # get model class
    model_class = getattr(importlib.import_module(module_path), class_name)
    # instantiate model
    model = model_class()
    # load weights
    if weights_path:
        model.load_state_dict(torch.load(weights_path))
    return model


def run(module_path: str, class_name: str, weights_path: str, dataset_path: str, category: str, viz: bool) -> None:
    """Run the evaluation script.

    Args:
        module_path (str): Path to the module containing the model class.
        class_name (str): Name of the model class.
        weights_path (str): Path to the model weights.
        dataset_path (str): Path to the dataset.
        category (str): Category of the dataset.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Instantiate model class here
    # Load the model here from checkpoint.
    model = load_model(module_path, class_name, weights_path)
    model.to(device)

    # Create the dataset
    datamodule = MVTecLoco(root=dataset_path, eval_batch_size=1, image_size=(448, 448), category=category)
    datamodule.setup()

    model.set_viz(viz)
    model.set_save_coreset_features(True)
    

    FEW_SHOT_SAMPLES = range(len(datamodule.train_data)) # traverse all dataset to build coreset
    
    # pass few-shot images and dataset category to model
    setup_data = {
        "few_shot_samples": torch.stack([datamodule.train_data[idx]["image"] for idx in FEW_SHOT_SAMPLES]).to(device),
        "few_shot_samples_path": [datamodule.train_data[idx]["image_path"] for idx in FEW_SHOT_SAMPLES],
        "dataset_category": category,
    }
    model.setup(setup_data)
    


if __name__ == "__main__":
    args = parse_args()
    run(args.module_path, args.class_name, args.weights_path, args.dataset_path, args.category, args.viz)
