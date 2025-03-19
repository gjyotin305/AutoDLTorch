import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F

## Intuition: Write a Good AutoEncoder, Try to use convolutions, attention, etc.

class SimpleGenerator(nn.Module):
    def __init__(self, noise_dim, hidden_dim):
        super().__init__()
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.model = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 784)
        )
    
    def forward(self, x):
        x = rearrange(x, "b c h w -> b (c h w)")
        out = self.model(x)
        out = F.tanh(out)
        out = rearrange(out, "b (c h w) -> b c h w")
        return out


## Write a good image classifier for the same, we can also use convolutions, attentions, etc.

class SimpleDiscriminator(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.hid_dim = hidden_dim
        self.in_dim = in_dim
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        x = rearrange(x, "b c h w -> b (c h w)")
        out = self.model(x)
        out = F.sigmoid(out)
        return out