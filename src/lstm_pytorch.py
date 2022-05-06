import torch
import torch.nn as nn

"""
nn.lstm needs two arguments: size and number of neurons that are fed recursively
For regression output of linear layer is 1.
LSTM layers have 3 outputs Outputs: output, (h_n, c_n)
We have to initialize a hidden cell state so that we can can use it 
as an input for the next time_stamp
LSTM algorithm accepts three inputs: previous hidden state,
previous cell state and current input.
"""


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer=1026):
        super().__init__()
        self.hidden_layer = hidden_layer
        self.lstm = nn.LSTM(1, hidden_layer)
        self.linear = nn.Linear(hidden_layer, 1)
        self.hidden_cell = (
            torch.zeros(1, 1, self.hidden_layer),
            torch.zeros(1, 1, self.hidden_layer),
        )

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(
            input_seq.view(len(input_seq), 1, -1), self.hidden_cell
        )
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]
