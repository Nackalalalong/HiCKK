import torch.nn as nn
import torch.nn.functional as F
import torch

class OurNet(nn.Module):
    def __init__(self):
        super(OurNet, self).__init__()
        self.filter_numbers = [8,3,5,8,1]
        self.filter_sizes = [7,5,3,5,7]
        self.padding = 12
        self.output_dim = 40 - self.padding

        self.seq1 = nn.Sequential(
            nn.Conv2d(1, 5, 3),
            nn.ReLU(),
            nn.BatchNorm2d(5),
            nn.Dropout2d(0.3),
            nn.Conv2d(5, 5, 3),
            nn.ReLU(),
            nn.BatchNorm2d(5),
            nn.Dropout2d(0.3)
        )

        self.seq2 = nn.Sequential(
            nn.Conv2d(1,5,5),
            nn.ReLU(),
            nn.BatchNorm2d(5),
            nn.Dropout2d(0.3),
        )

        self.conv = nn.Conv2d(5, 5, 3)

        self.seq3 = nn.Sequential(
            nn.Conv2d(1,5,7),
            nn.ReLU(),
            nn.BatchNorm2d(5),
            nn.Dropout2d(0.3),
        )

        self.seq4 = nn.Sequential(
            nn.Conv2d(1, 5, 9),
            nn.ReLU(),
            nn.BatchNorm2d(5),
            nn.Dropout2d(0.3),
        )
        
        self.seqout = nn.Sequential(
            nn.Conv2d(5, 1, 5),
            nn.ReLU(),
            # nn.BatchNorm2d(8),
            # nn.Dropout2d(0.3),
            # nn.Conv2d(8, 4, 3),
            # nn.ReLU(),
            # nn.BatchNorm2d(4),
            # nn.Dropout2d(0.3),
            # nn.Conv2d(4, 1, 5),
            # nn.ReLU(),
        )


    def forward(self, x):
        x1 = self.seq1(x)
        x2 = self.seq2(x)
        x3 = self.seq3(x)
        x4 = x1 + x2
        x5 = self.conv(x4)
        x6 = x5 + x3
        x7 = self.conv(x6)
        x9 = self.seq4(x)
        x8 = x7 + x9

        out = self.seqout(x8)        

        return out