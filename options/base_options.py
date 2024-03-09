import argparse
import ast
from typing import Dict, Union

class BaseOptions:

    def __init__(self) -> None:

        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self) -> None:

        self.parser.add_argument(
            "--experiment_name",
            type=str,
            default="text_experiment",
            help="Name of the experiment",
        )

        self.parser.add_argument(
            "--image_folder",
            type=str,
            default="../Datasets/Topographies/4-by-4/",
            help="Path to the folder containing images",
        )

        self.parser.add_argument(
            "--label_path",
            type=str,
            default="../Datasets/Topographies/4-by-4/labels.csv",
            help="Path to the csv file containing labels",
        )

        self.parser.add_argument(
            "--tb_logs",
            type=str,
            default="../Logs/",
            help="Path to the tensorboard logs",
        )

        self.parser.add_argument(
            "--image_size",
            type=int,
            default=224,
            choices=[128, 224, 256],
            help="Size of the image",
        )

        self.parser.add_argument(
            "--batch_size",
            type=int,
            default=32,
            choices=[32, 64],
            help="Size of the batch",
        )

        self.parser.add_argument(
            "--num_workers",
            type=int,
            default=4,
            help="Number of workers for dataloader",
        )

        self.parser.add_argument(
            "n_channels",
            type=int,
            default=1,
            choices=[1, 3],
            help="Number of channels in the image")
        
        self.parser.add_argument(
            "--dataset_normalisation",
            type=lambda x: ast.literal_eval(x),
            default={"mean": 0.5, "std": 0.5},
            help="Data normalisation parameters",
        ),

        self.parser.add_argument(
            "--network_name",
            type=str,
            default="cnn",
            choices=["cnn", "ann"],
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

    
    def parser(self) -> argparse.Namespace:
        if not self.initialized:
            self.initialize()

        self._opt = self.parser.parse_args()
        self._opt.is_train = self._is_train

        args = vars(self._opt)
        self._print(args)

        return self._opt
    
    def _print(self, args: Dict) -> None:
        """
        Prints the arguments passed to the script

        Parameters
        ----------
        args: dict
            The arguments to print

        Returns
        -------
        None
        """

        print("------------ Options -------------")
        for k, v in args.items():
            print(f"{str(k)}: {str(v)}")
        print("-------------- End ---------------")


    def float_or_none(self, value: str) -> Union[float, None]:
        """
        Converts a string to float or None

        Parameters
        ----------
        value: str
            The value to convert

        Returns
        -------
        float
            The converted value
        """

        if value.lower() == "none":
            return None
        try:
            return float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Invalid float or 'none' value: {}".format(value)
            )

    def list_or_none(self, value: str) -> Union[list, None]:
        """
        Converts a string to list or None

        Parameters
        ----------
        value: str
            The value to convert

        Returns
        -------
        list
            The converted value
        """

        if value.lower() == "none":
            return None
        try:
            return ast.literal_eval(value)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Invalid list or 'none' value: {}".format(value)
            )
