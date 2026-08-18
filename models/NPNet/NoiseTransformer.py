import torch.nn as nn
from dominate.tags import output

from torch.nn import functional as F
from timm import create_model
import torch

__all__ = ['NoiseTransformer']


class NoiseTransformer(nn.Module):
    def __init__(self, resolution=256):
        super().__init__()
        self.downsample = lambda x: F.interpolate(x, [resolution, resolution])
        self.upconv = nn.Conv2d(8, 3, (1, 1), (1, 1), (0, 0))
        self.downconv = nn.Conv2d(7, 3, (1, 1), (1, 1), (0, 0))
        self.swin = create_model("swinv2_tiny_window8_256", pretrained=True)

    def forward(self, X_T_prime, mask, mask_synthetic):
        input = torch.cat([X_T_prime, mask, mask_synthetic], dim=1)

        x = self.downconv(input)
        x = self.swin.forward_features(x)
        x = self.downsample(x)
        noise = self.upconv(x)
        # x = self.upconv(self.downsample(self.swin.forward_features(self.downconv(self.upsample(x))))) + x
        return mask_synthetic * (1. - mask) + mask * noise


if __name__ == '__main__':
    import torch
    from utils.showImage import Show

    device = torch.device('cuda')
    model = NoiseTransformer().to(device)

    Init_X_T = torch.randn(1, 3, 256, 256).to(device)
    X_T_prime = torch.randn(1, 3, 256, 256).to(device)
    mask = torch.randn(1, 1, 256, 256).to(device)
    mask_synthetic = torch.randn(1, 3, 256, 256).to(device)

    output = model(Init_X_T, mask, mask_synthetic)
    print(output.size)
