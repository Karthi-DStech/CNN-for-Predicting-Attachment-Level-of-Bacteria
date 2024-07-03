import argparse
import os
import sys
from typing import Dict, Union
import ast

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BaseOptions:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self) -> None:
        self.parser.add_argument(
            "--experiment_name",
            type=str,
            default="test",
            help="Name of the experiment",
        ),

        self.parser.add_argument(
            "--images_folder",
            type=str,
            default="../Datasets/Topographies/raw/FiguresStacked Same Size 4X4",
            help="Path to the folder containing the images",
        ),

        self.parser.add_argument(
            "--label_path",
            type=str,
            default="../Datasets/biology_data/TopoChip/AeruginosaWithClass.csv",
            help="Path to the file containing the labels",
        ),
        self.parser.add_argument(
            "--log_dir",
            type=str,
            default="../logs",
            help="Path to the folder containing the logs",
        ),

        self.parser.add_argument(
            "--img_size",
            type=int,
            default=224,
            help="Size of the images",
        ),

        self.parser.add_argument(
            "--batch_size",
            type=int,
            default=32,
            help="Batch size",
        ),

        self.parser.add_argument(
            "--num_workers",
            type=int,
            default=4,
            help="Number of workers",
        ),

        self.parser.add_argument(
            "--n_channels",
            type=int,
            default=1,
            help="Number of channels",
        ),

        self.parser.add_argument(
            "--n_classes",
            type=int,
            default=1,
            help="Number of classes",
        ),
        
        self.parser.add_argument(
            "--dataset_params",
            type=lambda x: ast.literal_eval(x),
            default={"mean": 0.5, "std": 0.5},
            help="Data normalisation parameters",
        ),
        self.parser.add_argument(
            "--img_type",
            type=str,
            default="png",
            help="Type of image",
        ),
        self.parser.add_argument(
            "--is_train",
            type=bool,
            default=False,
            help="Flag for training",
        ),
        self.parser.add_argument(
            "--network_name",
            type=str,
            default="cnn",
            choices=["cnn"],
            help="Name of the network",
        ),
        self.parser.add_argument(
            "--problem_type",
            type=str,
            default="regression",
            choices=["classification", "regression"],
            help="Type of problem",
        ),
        
        
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self._opt = self.parser.parse_args()

        return self._opt