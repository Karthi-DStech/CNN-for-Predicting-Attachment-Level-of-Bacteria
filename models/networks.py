import torch
import torch.nn as nn
from typing import Tuple,  Union



class BaseNetwork(nn.module):
    """
    Base class for all networks
    """
    def __init__(self, opt):
        """
        Initializes the BaseNetwork class
        
        Parameters
        ----------
        opt: argparse.Namespace
            The options for the network
        
        Returns
        -------
        None
        """

        super().__init__()
        self._name("BaseNetwork")
        self._opt = opt
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    @property
    def name(self) -> str:
        """
        Returns the name of the network
        """

        return self._name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network
        
        Parameters
        ----------
        x: torch.Tensor
            The input tensor
            
        Returns
        -------
        torch.Tensor
            The output tensor
        """

        raise NotImplementedError("The forward method must be implemented")
    
    
    def __str__(self) -> str:
        """
        Returns the string representation of the network
        """
        return self._name
    
    def init_weights(self, init_type: str = "normal") -> None:
        """
        Initializes the weights of the network
        
        Parameters
        ----------
        init_type: str
            The type of initialization to be used
            
        Returns
        -------
        Not implemented error if the init_type is not implemented
        """
        
        if init_type == "normal":
            self._init_normal()

        elif init_type == "xavier_normal":
            self._init_xavier_normal()

        elif init_type == "kaiming_normal":
            self._init_kaiming_normal()

        else:
            raise NotImplementedError(f"Invalid initialization type {init_type}")
        

    def _make_loss(self, problem_type) -> None:
        """
        Initializes the loss function based on the problem type
        
        Parameters
        ----------
        problem_type: str
            The type of problem
            
        Returns
        -------
        Not implemented error if the loss is not implemented
        """
        if problem_type == "classification":
            self.loss = nn.CrossEntropyLoss()
        elif problem_type == "regression":
            self.loss = nn.MSELoss()
        else:
            raise NotImplementedError(f"Problem type {problem_type} not implemented")
        
        
    def _set_optimiser(self, optimizer: str, lr: float) -> None:
        """
        Sets the optimizer for the network
        
        Parameters
        ----------
        optimizer: str
            The optimizer to be used
            
        lr: float
            The learning rate
            
        Returns
        -------
        Not implemented error if the optimizer is not implemented
        """
        if optimizer == "adam":
            self._optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif optimizer == "sgd":
            self._optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        elif optimizer == "rmsprop":
            self._optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)
        else:
            raise NotImplementedError(f"Optimizer {optimizer} not implemented")
        
    def set_input(self, data: torch.Tensor):
        """
        Sets the input of the model

        Parameters
        ----------
        data: torch.Tensor
            The input data

        Returns
        -------
        None
        """
        # Extract images and labels
        self.inputs, self.labels = data
        self.inputs, self.labels = self.inputs.to(self.device), self.labels.to(self.device)
        
    def get_num_params(self) -> Tuple[int, int]:
        """
        Returns the number of parameters in the network

        Returns
        -------
        all_params: int
            The total number of parameters in the network
        trainable_params: int
            The total number of trainable parameters in the network
        """
        all_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return all_params, trainable_params

        

    def save_model(self):
        pass
    
    def save_performance(self, epoch, train_loss, test_loss):
        pass
    
    def train_step(self):
        pass

    def test_step(self):
        pass


"""
TODO:
- save_model
- save_performance
- train_step
- test_step
"""

