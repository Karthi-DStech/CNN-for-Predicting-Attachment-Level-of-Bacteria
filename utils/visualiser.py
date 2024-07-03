import argparse
from datetime import datetime
from tensorboardX import SummaryWriter
import os


class Visualizer:    
    """
    A class for logging and visualizing the training and test performance of the model
    """

    def __init__(self, opt: argparse.Namespace) -> None:
        """
        Initialize the visualizer

        Parameters
        ----------
        opt: argparse.Namespace
            The training options

        Returns
        -------
        None
        """
        self.opt = opt
        self.log_dir = os.path.join(self.opt.log_dir, self.opt.experiment_name)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.opt_path = os.path.join(self.log_dir, "opt.json")
        self.log_path = os.path.join(self.log_dir, "log.txt")
        
        with open(self.opt_path, "w") as f:
            for key, value in vars(opt).items():
                f.write(f"{key}: {value}\n")

        message = f'{"="*20} Experiment Log ({datetime.now()}) {"="*20}\n'
        print(f"{message}")
        with open(self.log_path, "w") as f:
            f.write(message + "\n")

    def log_performance(
        self,
        loss: float,
        epoch: int,
        is_train: bool = True,
        print_freq: int = 1
    ) -> None:
        """
        Log the performance of the model

        Parameters
        ----------
        loss: float
            The loss of the model
        epoch: int
            The current epoch
        is_train: bool
            Whether the model is in training mode
        print_freq: int
            The frequency of printing the performance

        Returns
        -------
        None
        """

        
        sum_name = "{}/{}".format("Train" if is_train else "Test", "Loss")
        self.writer.add_scalar(sum_name, loss, self.opt.n_epochs)            
        self._print_performance(loss, epoch,is_train,print_freq)

    def log_best_performance(
        self,
        loss: float,
        epoch: int,
        metrics: dict,
        is_train: bool = True,
        print_freq: int = 1
    ) -> None:
        """
        Log and print best performance of the model

        Parameters
        ----------
        loss: float
            The loss of the model
        epoch: int
            The current epoch
        metrics: dict
            The evaluation metrics of the model
        is_train: bool
            Whether the model is in training mode
        print_freq: int
            The frequency of printing the performance

        Returns
        -------
        None
        """
        if is_train:
            message = "Train "
        else:
            message = "Test "

        message += f'Best Model Metrics'
        print(f"{message}")
        with open(self.log_path, "a") as f:
            f.write(message + "\n")

        self._print_performance(loss, epoch=epoch, is_train=is_train, print_freq=1)

        message = ""
        for key, value in metrics.items():
            message += f"{key}: {value:.4f}\n"
            print(message)
            with open(self.log_path, "a") as f:
                f.write(message + "\n")



    def _print_performance(
        self,
        loss: float,
        epoch: int,
        is_train: bool = True,
        print_freq: int =1
    ) -> None:
        """
        Print the performance of the model

        Parameters
        ----------
        losses: dict
            The losses of the model
        epoch: int
            The current epoch
        is_train: bool
            Whether the model is in training mode
        print_freq: int
            The frequency of printing the performance

        Returns
        -------
        None
        """
        if epoch % print_freq == 0:
            if is_train:
                message = "Train "
            else:
                message = "Test "
            message += f"[Epoch {epoch}/{self.opt.n_epochs}] \n"
            message += f"Loss: {loss:.4f} \n"
            print(message)
            with open(self.log_path, "a") as f:
                f.write(message + "\n")
        else:
            pass
    