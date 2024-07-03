import os
import sys

from options.base_options import BaseOptions

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TrainOptions(BaseOptions):
    """Train options"""

    def __init__(self) -> None:
        super().__init__()


    def initialize(self) -> None:
        """Initialize train options"""
        BaseOptions.initialize(self)

        self.parser.add_argument(
            "--optimizer_type",
            type=str,
            default="adam",
            help="Type of optimizer",
        ),

        self.parser.add_argument(
            "--lr",
            type=float,
            default=0.001,
            help="Learning rate",
        ),

        self.parser.add_argument(
            "--n_epochs",
            type=int,
            default=2,
            help="Number of epochs",
        ),
        self.parser.add_argument(
            "--split_size",
            type=float,
            default=0.2,
            help="Size of the Test set",
        ),
        self.parser.add_argument(
            "--shuffle_data",
            type=bool,
            default=True,   
            help="Shuffle the dataset",
        ),
        self.parser.add_argument(
            "--is_eval",
            type=bool,
            default=True,
            help="Whether to evaluate the best model",
        ),


        # Update --is_train to True
        self.parser.set_defaults(is_train=True),



