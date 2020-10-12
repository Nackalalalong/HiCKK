import torch.nn as nn
import torch.nn.functional as F
import torch

class OurNet(nn.Module):
    def __init__(self):
        super(OurNet, self).__init__()
        self.filter_numbers = [8,3,5,8,1]
        self.filter_sizes = [7,5,3,5,7]
        self.padding = sum(self.filter_sizes)-1-5-3-5-1

        self.seq = nn.Sequential(
            nn.Conv2d(1, self.filter_numbers[0], self.filter_sizes[0]),
            nn.ReLU(),
            nn.BatchNorm2d(self.filter_numbers[0]),
            nn.Dropout2d(0.3),
            nn.Conv2d(self.filter_numbers[0], self.filter_numbers[1], self.filter_sizes[1], padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(self.filter_numbers[1]),
            nn.Dropout2d(0.3),
            nn.Conv2d(self.filter_numbers[1], self.filter_numbers[2], self.filter_sizes[2], padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.filter_numbers[2]),
            nn.Dropout2d(0.3),
            nn.Conv2d(self.filter_numbers[2], self.filter_numbers[3], self.filter_sizes[3], padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(self.filter_numbers[3]),
            nn.Dropout2d(0.3),
            nn.Conv2d(self.filter_numbers[3], self.filter_numbers[4], self.filter_sizes[4]),
            nn.ReLU(),
        )

    def forward(self, x):

        return self.seq(x)