import torch
import torch.nn as nn
import pandas as pd
import torchvision

class LSTM(nn.Module):
    def __init__(self, input_size , hidden_layer = 1026):
        super().__init__()
        self.input_size = input_size
        self.hidden_layer = hidden_layer
        #nn.lstm needs two arguments: size and number of neurons that are fed
    #recursively
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_layer  
                            ,batch_first = True)
    #for regression output of linear layer is 1.
        self.linear = nn.Linear(hidden_layer , 1)
    ##LSTM layers have 3 outputs Outputs: output, (h_n, c_n)
    #We have to initialize a hidden cell state so that we can can use it 
    #as an input for the next time_stamp 
    #LSTM algorithm accepts three inputs: previous hidden state, 
    #previous cell state and current input.
    def forward(self,x):
        self.lstm.flatten_parameters()
        output, (hn, cn) = self.lstm(x)
        out = self.linear(hn[-1])
        return out  
