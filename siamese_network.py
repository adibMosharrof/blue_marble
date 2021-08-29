import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding
from torch.nn.modules.linear import Linear


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(6),
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(24),
            nn.Conv2d(in_channels=24, out_channels=36, kernel_size=3),
        )

        # in_dims = 3
        # out_dims = None
        # self.layers = nn.ModuleList()
        # for i in range(self.layers):
        #     out_dims = in_dims * 2
        #     self.layers.append(
        #         nn.Conv2d(in_channels=in_dims, out_dims=out_dims, kernel_size=3)
        #     )
        #     self.layers.append(nn.ReLU())
        #     self.layers.append(nn.BatchNorm2d(out_dims))

    def forward_once(self, x):
        # Forward pass
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        # output = self.fc1(output)
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
