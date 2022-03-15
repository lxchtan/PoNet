import torch
import torch.nn as nn



class FFTAttention(nn.Module):

  def __init__(self, config):
    super().__init__()

  def forward(self, X, mask):
    X = torch.fft.fft(torch.fft.fft(X, dim=-1), dim=-2).real

    return X
