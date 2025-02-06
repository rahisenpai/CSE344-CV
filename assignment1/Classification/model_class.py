import torchvision 
from torchvision import models
import torch
import torch.nn as nn


class ConvNet(nn.Module):
    """
    Implements a custom CNN architecture with 3 convolution layers using PyTorch
    to be used on Russian Wildlife Dataset for Image Classification Task.
    """

    def __init__(self, num_classes: int):
        """
        Initializes the CNN model.
        """
        super(ConvNet, self).__init__()

        #initially, image size is 256x256
        self.conv = nn.Sequential(
            #conv layer 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            #after conv layer 1, image size is 63x63

            #conv layer2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #after conv layer 2, image size is 31x31

            #conv layer 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #after conv layer 3, image size is 16x16
        )
        self.fc = nn.Linear(128 * 16*16, num_classes) #fully connected layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.
        """
        out = self.conv(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

""" Refernces
- https://github.com/rahisenpai/CSE343-ML/blob/main/asgn4/code_c.ipynb
- https://github.com/rahisenpai/CSE556-NLP/blob/main/assignment1/task3.py
"""


class ResNet18(nn.Module):
    def __init__(self, num_classes: int=10):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(weights='IMAGENET1K_V1') #Use resnet18 from torchvision with Pretrained Weights.
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes) #Replace None with resnet18's number of input features for FC.
    def forward(self, x):
        return self.model(x)