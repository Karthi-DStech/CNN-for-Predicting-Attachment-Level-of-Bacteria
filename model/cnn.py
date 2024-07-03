import torch
import os
import sys
import torch.nn as nn
from model.networks import BaseNetwork

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class CNN(BaseNetwork):
    def __init__(self, opt):
        super().__init__(opt)
        self._name = "CNN"
        self._n_channels = opt.n_channels
        self._n_classes = opt.n_classes

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(self._n_channels, 32, 3),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((2,2),padding=1),
            nn.Conv2d(32, 64, 3),
            nn.Tanh(),
            nn.MaxPool2d((2,2),padding=1),
            nn.Conv2d(64, 128, 3),
            nn.Tanh(),
            nn.MaxPool2d((2,2),padding=1),
            nn.Conv2d(128, 128, 3),
            nn.Tanh(),
            nn.MaxPool2d((2,2),padding=1)
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.latent_feat = nn.Sequential(
            nn.Linear(25088, 64),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, self._n_classes)            
        )       

        if opt.is_train:
            self._make_loss(opt.problem_type)
            self._make_optimizer(opt.optimizer_type, opt.lr)

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.latent_feat(x)
        return x
    
    

    
    