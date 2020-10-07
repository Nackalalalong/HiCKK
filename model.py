import torch.nn as nn
import torch.nn.functional as F

class OurNetV2(nn.Module):
    def __init__(self):
        super(OurNetV2, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.filter_numbers = [8,3,5,8,1]
        self.filter_sizes = [7,5,3,5,7]
        self.padding = sum(self.filter_sizes)-1-5-3-5-1
        self.conv1 = nn.Conv2d(1, self.filter_numbers[0], self.filter_sizes[0])
        self.conv2 = nn.Conv2d(self.filter_numbers[0], self.filter_numbers[1], self.filter_sizes[1], padding=2)
        self.conv3 = nn.Conv2d(self.filter_numbers[1], self.filter_numbers[2], self.filter_sizes[2], padding=1)
        self.conv4 = nn.Conv2d(self.filter_numbers[2], self.filter_numbers[3], self.filter_sizes[3], padding=2)
        self.conv5 = nn.Conv2d(self.filter_numbers[3], self.filter_numbers[4], self.filter_sizes[4])

    def forward(self, x):
        #print("start forwardingf")
        x1 = self.conv1(x)
        x = F.relu(x1)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = x + x1
        x = self.conv5(x)
        x = F.relu(x)

        # x1 = self.conv1(x)
        # x = self.swish(x1)
        # x = self.conv2(x)
        # x = self.swish(x)
        # x = self.conv3(x)
        # x = self.swish(x)
        # x = self.conv4(x)
        # x = self.swish(x)
        # x = x + x1
        # x = self.conv5(x)
        # x = self.swish(x)

        return x

    def swish(self,x):
        return x * F.sigmoid(x)