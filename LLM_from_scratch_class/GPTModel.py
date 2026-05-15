'''
The Model class
'''
import torch
import torch.nn as nn

class GPTModel(nn.Module):
    def __init__(self):

        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x.float() * self.w + self.b