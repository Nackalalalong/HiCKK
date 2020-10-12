import torch.nn as nn
import torch.nn.functional as F
import torch

class OurNet(nn.Module):
    def __init__(self):
        super(OurNet, self).__init__()
        self.filter_numbers = [8,3,5,8,1]
        self.filter_sizes = [7,5,3,5,7]
        self.padding = 6
        self.output_dim = 40 - self.padding

        self.conv3 = nn.Conv2d(1, 8, 5, padding=2)
        self.conv4 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv5 = nn.Conv2d(1, 8, 1)

        self.conv = nn.Conv2d(8, 8, 3, padding=1)
        self.dropout = nn.Dropout2d(0.3)
        self.batchnorm = nn.BatchNorm2d(8)

        self.seqout = nn.Sequential(
            nn.Conv2d(8, 1, 7),
            nn.ReLU(),
        )

    def forward(self, x):
        x3 = self.dropout(self.batchnorm(F.relu(self.conv3(x))))
        x4 = self.dropout(self.batchnorm(F.relu(self.conv4(x))))
        x5 = self.dropout(self.batchnorm(F.relu(self.conv5(x))))

        x10 = self.dropout(self.batchnorm(F.relu(self.conv(x3))))
        x11 = x10 + x4

        x12 = self.dropout(self.batchnorm(F.relu(self.conv(x11))))
        x13 = x12 + x5

        out = self.seqout(x13)        

        return out