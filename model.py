import torch.nn as nn
import torch.nn.functional as F
import torch

class OurNet(nn.Module):
    def __init__(self, batch_size):
        super(OurNet, self).__init__()
        self.batch_size = batch_size
        
        self.filter_numbers = [8,3,5,8,1]
        self.filter_sizes = [7,5,3,5,7]
        self.padding = sum(self.filter_sizes)-1-5-3-5-1
        self.output_dim = 40 - self.padding

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

        self.lstm_input_size = self.output_dim
        self.lstm_hidden_size = self.output_dim
        self.lstm_num_layer = 1
        self.lstm_seq_len = self.output_dim

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
        x = self.seq(x)

        x, (h, c) = self.lstm(x.view(x.size(0), self.lstm_seq_len, self.lstm_input_size), (self.lstm_h, self.lstm_c))
        h = h.detach()
        c = c.detach()
        self.lstm_h = h
        self.lstm_c = c
        
        x = x.view(x.size(0), 1, self.output_dim, self.output_dim)
        x = F.relu(x)

        return x
  
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
