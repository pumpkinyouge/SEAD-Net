# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

__all__ = ['Autoencoder']

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_dim_2, crop_size, num_classes=2):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(crop_size, input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim_2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim_2, hidden_dim),
            nn.Linear(hidden_dim, input_dim),
            nn.Linear(input_dim, crop_size),
        )

        self.fc = nn.Linear(hidden_dim_2, num_classes)
        self.fc1 = nn.Linear(crop_size + hidden_dim_2 + 1, 128)
        self.fc2 = nn.Linear(128 + 1, 32)
        self.fc3 = nn.Linear(32 + 1, 1)

    def forward(self, x):
        x = x.float()
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        output = self.fc(encoded)

        # Calculate the reconstruction error
        sub_result = decoded - x
        mse = decoded - x
        # Calculate the L2 norm of the reconstruction residual vector
        sub_norm2 = torch.norm(sub_result, p=2, dim=1, keepdim=True)
        # Calculate the result of dividing the reconstruction residual vector by the L2 norm
        sub_result = sub_result / sub_norm2
        # Connection tensor
        conca_tensor = torch.cat([sub_result, encoded], dim=1)
        conca_tensor = torch.cat([conca_tensor, sub_norm2], dim=1)

        conca_tensor = self.fc1(conca_tensor)
        conca_tensor = torch.cat([conca_tensor, sub_norm2], dim=1)

        conca_tensor = self.fc2(conca_tensor)
        conca_tensor = torch.cat([conca_tensor, sub_norm2], dim=1)

        score = self.fc3(conca_tensor)

        return output, score, sub_result, mse

def autoencoder(args, mode, **kwargs):
    hidden_dim_2 = int(args.hidden_dim / 2)
    return Autoencoder(args.input_dim, args.hidden_dim, hidden_dim_2, args.crop_size)
