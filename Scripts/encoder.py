import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    super().__init__(
        nn.Conv2d(in_channels=3,out_channels=128,kernel_size=3,padding=1),

    )