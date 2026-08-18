import random
import torch
import argparse
import torch.nn as nn

from NoiseTransformer import NoiseTransformer
from SVDNoiseUnet import SVDNoiseUnet


class NPNet(nn.Module):
    def __init__(self, pretrained_path=True, device='cuda') -> None:
        super(NPNet, self).__init__()

        self.device = device
        self.pretrained_path = pretrained_path

        self.unet_svd = SVDNoiseUnet(resolution=256).to(self.device).to(torch.float32)
        self.unet_embedding = NoiseTransformer(resolution=256).to(self.device).to(torch.float32)
        self._beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, Init_X_T, X_T_prime, mask, mask_synthetic):
        golden_embedding = self.unet_embedding(X_T_prime.float(), mask, mask_synthetic)
        golden_noise = self.unet_svd(Init_X_T.float(), mask,
                                     mask_synthetic) + self._beta * golden_embedding

        return golden_noise

    def reference(self, Init_X_T, mask, mask_synthetic):
        X_T_prime_bar = self.unet_svd(Init_X_T.float(), mask, mask_synthetic)

        golden_embedding = self.unet_embedding(X_T_prime_bar, mask, mask_synthetic)
        golden_noise = X_T_prime_bar + self._beta * golden_embedding

        return golden_noise


if __name__ == '__main__':
    dtype = torch.float16
    device = torch.device('cuda')
    model = NPNet(device=device)
    Init_X_T = torch.randn(1, 3, 256, 256, dtype=dtype).to(device)
    X_T_prime = torch.randn(1, 3, 256, 256, dtype=dtype).to(device)
    mask = torch.randn(1, 1, 256, 256, dtype=dtype).to(device)
    mask_synthetic = torch.randn(1, 3, 256, 256, dtype=dtype).to(device)
    # create NPNet to get the target noise
    golden_noise = model(Init_X_T, X_T_prime, mask, mask_synthetic)  #
    print(golden_noise.shape)
