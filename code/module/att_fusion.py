import torch
import torch.nn as nn
import torch.nn.functional as F


class AttFusion(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(AttFusion, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1)
