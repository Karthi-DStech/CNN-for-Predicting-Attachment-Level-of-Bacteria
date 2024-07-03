import os
import sys
import torch
from torcheval.metrics import R2Score

import torch.nn as nn
from utils.weight_init import *

class BaseNetwork(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self._name = "BaseNetwork"
        self._opt = opt
        # set cpu or gpu
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        

    def forward(self, x):
        raise NotImplementedError
    
    @property
    def name(self) -> str:
        """
        Returns the name of the network
        """
        return self._name
    
    def init_weights(self, init_type: str = "normal") -> None:
        """
        Initializes the weights of the network

        Parameters
        ----------
        init_type: str
            The type of initialization

        Raises
        ------
        NotImplementedError
            if the method is not implemented
        """
        if init_type == "normal":
            self.apply(normal_init)
        elif init_type == "xavier_normal":
            self.apply(xavier_init)
        elif init_type == "kaiming_normal":
            self.apply(kaiming_init)
        else:
            raise NotImplementedError(f"Invalid init type: {init_type}")
        
    def _make_loss(self, problem_type):
        if problem_type == "classification":
            self.loss = nn.CrossEntropyLoss()
        elif problem_type == "regression":
            self.loss = nn.MSELoss()
        else:
            raise NotImplementedError(f"Problem type {problem_type} not implemented")
        
    def _make_optimizer(self, optimizer_type, lr):
        if optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        elif optimizer_type == "rmsprop":
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)
        else:
            raise NotImplementedError(f"Optimizer type {optimizer_type} not implemented")
        
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
    
    def save_model(self):
        """
        Saves the model

        Returns
        -------
        None
        """
        #torch.save(self.state_dict(), os.path.join(self._opt.checkpoints_dir, f"{self._name}.pth"))
        raise NotImplementedError("Method not implemented")
    
    def save_performance(self, epoch, train_loss, test_loss):
        """
        Saves the performance of the model

        Parameters
        ----------
        epoch: int
            The epoch number
        train_loss: float
            The training loss
        test_loss: float
            The test loss

        Returns
        -------
        None
        """
        raise NotImplementedError("Method not implemented")
    
    def train_step(self):
        # Set the model to training mode
        self.train()
        # Predict the output
        preds = self(self.inputs)
        # Evaluate error between original image and reconstructed image
        loss = self.loss(self.labels.float(), preds.squeeze())
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def test_step(self):
        # Set the model to evaluation mode
        self.eval()
        # Predict the output
        preds = self(self.inputs)
        # Evaluate error between original image and reconstructed image
        loss = self.loss(self.labels.float(), preds.squeeze())

        return loss.item()
    

    def eval_step(self):
        # Set the model to evaluation
        self.eval()
        # Dictionary to store evaluation metrics
        metrics = {}
        if self._opt.problem_type == "regression":
            with torch.no_grad():
                preds = self(self.inputs)

                # Compute R-squared
                metric = R2Score()
                metric.update(self.labels.float(), preds.squeeze()) 
                metrics['r_squared'] = metric.compute().item()

                return metrics
        else:
            raise NotImplementedError(f"Problem type {self._opt.problem_type} not implemented")
       
    def save_model(self):
        """
        Saves the model

        Returns
        -------
        None
        """
        torch.save(self.state_dict(), os.path.join(self._opt.log_dir, 
                                                   self._opt.experiment_name,
                                                   f"{self._name}.pth"))
    
    def load_model(self):
        """
        Loads the model

        Returns
        -------
        None
        """
        self.load_state_dict(torch.load(os.path.join(self._opt.log_dir, 
                                                   self._opt.experiment_name,
                                                   f"{self._name}.pth")))
        self.eval()


    


    