import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Setting up the Sequential of CNN Layers
        # self.cnn1 = nn.Sequential(
        #     nn.Conv2d(1, 96, kernel_size=11, stride=1),
        #     nn.ReLU(inplace=True),
        #     nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
        #     nn.MaxPool2d(3, stride=2),
        #     nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=2),
        #     nn.MaxPool2d(3, stride=2),
        #     nn.Dropout2d(p=0.3),
        #     nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(3, stride=2),
        #     nn.Dropout2d(p=0.3),
        # )

        # # Defining the fully connected layers
        # self.fc1 = nn.Sequential(
        #     nn.Linear(30976, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout2d(p=0.5),
        #     nn.Linear(1024, 128),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(128, 2),
        # )
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=12 * 4 * 4, out_features=120),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Linear(in_features=120, out_features=60),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Linear(in_features=60, out_features=10),
        )

    def forward_once(self, x):
        # Forward pass
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)
        return output1, output2


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2)
            + (label)
            * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )

        return loss_contrastive
