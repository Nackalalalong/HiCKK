import torch.nn as nn
import torch.nn.functional as F
import torch

class OurNetV4(nn.Module):
    def __init__(self, batch_size):
        super(OurNetV4, self).__init__()
        self.batch_size = batch_size

        self.filter_numbers = [8,3,5,8,1]
        self.filter_sizes = [7,5,3,5,7]
        self.padding = sum(self.filter_sizes)-1-5-3-5-1
        self.output_dim = 40 - self.padding
        
        self.conv1 = nn.Conv2d(1, self.filter_numbers[0], self.filter_sizes[0])
        self.conv2 = nn.Conv2d(self.filter_numbers[0], self.filter_numbers[1], self.filter_sizes[1], padding=2)
        self.conv3 = nn.Conv2d(self.filter_numbers[1], self.filter_numbers[2], self.filter_sizes[2], padding=1)
        self.conv4 = nn.Conv2d(self.filter_numbers[2], self.filter_numbers[3], self.filter_sizes[3], padding=2)
        self.conv5 = nn.Conv2d(self.filter_numbers[3], self.filter_numbers[4], self.filter_sizes[4])

        self.lstm_input_size = self.output_dim ** 2
        self.lstm_hidden_size = self.output_dim ** 2
        self.lstm_num_layer = 1
        self.lstm_seq_len = 1

        self.lstm = nn.LSTM(
            self.lstm_input_size,
            self.lstm_hidden_size,
            self.lstm_num_layer,
            batch_first=True
        )

        # self.linear = nn.Sequential(
        #     nn.Linear(self.lstm_hidden_size, self.output_dim**2),
        # )

        self.dropout = nn.Dropout2d(0.3)

    def forward(self, x):
        x1 = self.conv1(x)
        x = F.relu(x1)
        x = self.dropout(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x + x1
        x = self.conv5(x)
        x = F.relu(x)

        x, (h, c) = self.lstm(x.view(x.size(0), self.lstm_seq_len, self.lstm_input_size), (self.lstm_h, self.lstm_c))
        h = h.detach()
        c = c.detach()
        self.lstm_h = h
        self.lstm_c = c
        # x = self.linear(x.view(x.size(0), -1))

        return x.view(x.size(0), 1, self.output_dim, self.output_dim)

    def swish(self,x):
        return x * F.sigmoid(x)

    def init_lstm_state(self, device):
        lstm_h = torch.zeros(self.lstm_num_layer, self.batch_size, self.lstm_hidden_size)
        lstm_c = torch.zeros(self.lstm_num_layer, self.batch_size, self.lstm_hidden_size)

        if device == 'cuda':
            lstm_h = lstm_h.cuda()
            lstm_c = lstm_c.cuda()

        self.lstm_h = lstm_h
        self.lstm_c = lstm_c

    def set_state(self, h,c):
        self.lstm_h = h
        self.lstm_c = c