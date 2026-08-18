import torch.nn as nn
from nn import Encoder, Decoder, SiLU
import torch
from models.VAE.nn import DiagonalGaussianDistribution
from torch.nn.utils import spectral_norm


class AautoencoderKL(nn.Module):
    def __init__(self,
                 double_z=True,
                 z_channels=4,
                 resolution=256,
                 in_channels=3,
                 out_ch=3,
                 ch=128,
                 ch_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_resolutions=[],
                 dropout=0.0,
                 embed_dim=4,
                 name=None,
                 using_class=False,
                 ):
        super().__init__()
        self.double_z = double_z
        cond_embed_dim = ch * len(ch_mult)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = Encoder(double_z=double_z, ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
                               resolution=resolution, z_channels=z_channels, cond_embed_dim=cond_embed_dim)
        self.decoder = Decoder(ch=ch, out_ch=out_ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                               attn_resolutions=attn_resolutions, dropout=dropout, in_channels=in_channels,
                               resolution=resolution, z_channels=z_channels, cond_embed_dim=cond_embed_dim)
        self.quant_conv = torch.nn.Conv2d(2 * z_channels if double_z else z_channels,
                                          2 * embed_dim if double_z else embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, z_channels, 1)
        self.embed_dim = embed_dim  # T
        self.logvar = nn.Parameter(torch.ones(size=()) * 0.0)
        self.name = name
        self.using_class = using_class
        if self.using_class:
            self.embedding = nn.Sequential(
                spectral_norm(nn.Embedding(num_embeddings=3, embedding_dim=ch), eps=1e-6),
                nn.Linear(ch, cond_embed_dim),
                SiLU(),
                nn.Linear(cond_embed_dim, cond_embed_dim),
            )

    def encodeing(self, x, imgc=None, *args, **kwargs):
        # if imgc is not None:
        #     assert self.using_class is True, "There is no image class"
        #     embedded = self.embedding(imgc.long())
        # else:
        #     assert self.using_class is False, "image class haven't entered"
        #     embedded = None
        embedded = self.get_embedding(imgc)
        h, hs = self.encoder(x, embc=embedded, *args, **kwargs)
        h = self.quant_conv(h)
        return h

    def decodeing(self, h, imgc=None, *args, **kwargs):
        # if imgc is not None:
        #     assert self.using_class is True, "There is no image class"
        #     embedded = self.embedding(imgc.long())
        # else:
        #     assert self.using_class is False, "image class haven't entered"
        #     embedded = None
        embedded = self.get_embedding(imgc)
        h = self.post_quant_conv(h)  #
        dec = self.decoder(h, embc=embedded, *args, **kwargs)  #
        return dec

    def forward(self, x, imgc=None, *args, **kwargs):
        hidden = self.encodeing(x, imgc=imgc, *args, **kwargs)
        out = self.decodeing(hidden, imgc=imgc, *args, **kwargs)
        return out

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def get_embedding(self, imgc=None, *args, **kwargs):
        if imgc is not None:
            assert self.using_class is True, "There is no image class"
            embedded = self.embedding(imgc.long())
        else:
            assert self.using_class is False, "image class haven't entered"
            embedded = None
        return embedded


if __name__ == '__main__':
    import torch

    VAE = AautoencoderKL(double_z=False,
                         z_channels=256,
                         resolution=256,
                         in_channels=3,
                         out_ch=3,
                         ch=128,
                         ch_mult=[1, 2, 4],
                         num_res_blocks=2,
                         attn_resolutions=[],
                         dropout=0.0,
                         embed_dim=128,
                         using_class=True, )
    torch.save(VAE.state_dict(), "VAE.pt")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.manual_seed(0)
    # VAE.to(device)
    #
    # image = torch.randn(4, 3, 256, 256).to(device)
    # label = torch.tensor([0, 1, 2, 2]).long().to(device)
    # posterior = VAE.encodeing(x=image, imgc=label)
    # posterior.kl(index=label != 2)
    # emb = posterior.sample(inference=True)
    # print(emb)
    # with torch.no_grad():
    #     posterior = VAE.encodeing(x=image, imgc=label)
    #     emb = posterior.sample(inference=True)
    #     kl = posterior.kl(index=[label != 2])
    #     out = VAE.decodeing(h=emb, imgc=label)
    #     emb = VAE.get_embedding(imgc=label)
    #     print(emb.shape)
    # out = VAE(image, torch.tensor([0, 1, 2, 2]).long().to(device))

    # out = VAE(x=image, imgc=torch.tensor([0, 1, 2]).long().to(device))
    # torch.save(VAE.state_dict(), 'autoencoder.pt')
    # print(out.shape)
