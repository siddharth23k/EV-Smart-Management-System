import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceAutoencoder(nn.Module):

    def __init__(self, input_dim=3, latent_dim=4):
        super().__init__()

        # Applied independently at each timestep: 3 → 8 → latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, latent_dim)
        )

        # latent_dim → 8 → 3
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim)
        )

    def forward(self, x):
        # x: (batch, time_steps, input_dim) → flatten time for per-step encoding
        batch_size, time_steps, input_dim = x.shape
        x_flat = x.view(batch_size * time_steps, input_dim)

        latent = self.encoder(x_flat)
        reconstructed = self.decoder(latent)

        return reconstructed.view(batch_size, time_steps, input_dim)

    def encode(self, x):
        batch_size, time_steps, input_dim = x.shape
        x_flat = x.view(batch_size * time_steps, input_dim)
        latent = self.encoder(x_flat)
        return latent.view(batch_size, time_steps, -1)