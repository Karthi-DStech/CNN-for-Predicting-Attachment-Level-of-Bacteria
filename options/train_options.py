from options.base_options import BaseOptions


class TrainOptions(BaseOptions):
    """
    TrainOptions class inherits from BaseOptions and defines the options used during training.
    """

    def __init__(self) -> None:

        super().__init__()

    def initialize(self) -> None:
        """Initializes the TrainOptions class"""

        BaseOptions.initialize(self)

        self.parser.add_argument(
            "--optimizer",
            type=str,
            default="adam",
            choices=["adam", "sgd"],
            help="Optimizer to be used",
        )

        self.parser.add_argument(
            "--lr",
            type=float,
            default=0.0003,
            choices=[0.0001, 0.0002, 0.0003],
            help="Learning rate",
        )

        self.parser.add_argument(
            "--n_epochs",
            type=int,
            default=10,
            choices = [10, 20, 30],
            help="Number of epochs",
        )

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

        self._parser.add_argument(
            "--init_type",
            type=str,
            required=False,
            default="xavier_normal",
            help="initialization type",
            choices=["normal", "xavier_normal", "kaiming_normal"],
        )

        self._parser.add_argument(
            "--loss_function",
            type=str,
            required=False,
            default="MSELoss",
            choices = ["MSELoss", "CrossEntropyLoss", "Categorical Cross-Entropy Loss"],
            help="loss function"
        )

        self._parser.add_argument(
            "--weight_decay",
            type=int,
            required=False,
            default=0.0003,
            choices = [0.0001, 0.0002, 0.0003],
            help="number of epochs",
        )

        self._is_train = True