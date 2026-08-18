import torch
from torch import nn
from models.mapping_modules.unet import UNet as Mnet
# from models.decod_b_model.unet import UNet as B
# from models.VAE.autoencoder.autoencoder import AautoencoderKL as B
# from models.BOPL_VAE.unet import GlobalGenerator_DCDCv2 as A
# from models.VAE.encode_unet.unet import UNet as A
# from models.VAE.autoencoder.autoencoder import AautoencoderKL as A
# from models.VAE.encode_a_model.unet import UNet as A
from models.BOPL_VAE.unet_org import GlobalGenerator_DCDCv2 as A


class Map(nn.Module):
    def __init__(self, module_name, unet_A, mapping):
        super().__init__()
        self.module_name = module_name
        self.A = A(**unet_A)
        # self.B = B(**unet_B)
        self.Map = Mnet(**mapping)

    def encodeing(self, x, inst):
        hidden = self.A.encodeing(x)
        return hidden

    def decodeing(self, h, gammas, mask):
        out = self.Map(x=h, gammas=gammas, mask=mask)
        return out

    def forward(self, x, gammas, mask=None, inst=None):
        hidden, hs = self.A.encodeing(x)
        out = self.Map(x=hidden, hs_up=hs, gammas=gammas, mask=mask)
        return out
