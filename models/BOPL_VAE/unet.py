import torch
import torch.nn as nn
import functools
from torch.nn.utils import spectral_norm


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(
            self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False, dilation=1,
            emb_channels=64, use_scale_shift_norm=True,
    ):
        super(ResnetBlock, self).__init__()
        self.use_scale_shift_norm = use_scale_shift_norm
        self.emb_layers = nn.Sequential(
            SiLU(),
            nn.Linear(
                emb_channels,
                2 * dim if use_scale_shift_norm else dim,
            ),
        )
        self.dilation = dilation
        self.in_layers = self.build_in_block(dim, padding_type, norm_layer, activation, use_scale_shift_norm)

        self.out_layers = self.build_out_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_in_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(self.dilation)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(self.dilation)]
        elif padding_type == "zero":
            p = self.dilation
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, dilation=self.dilation),
            norm_layer(dim),
            activation,
        ]
        return nn.Sequential(*conv_block)

    def build_out_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        conv_block += [norm_layer(dim)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "replicate":
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, dilation=1), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        out = h + x
        return out


class GlobalGenerator_DCDCv2(nn.Module):
    def __init__(
            self,
            input_nc,
            output_nc,
            ngf=64,
            k_size=3,
            n_downsampling=8,
            norm_layer=functools.partial(nn.InstanceNorm2d, affine=False),
            padding_type="reflect",
            mc=64,
            start_r=1,
            spatio_size=64,
            feat_dim=-1,
            use_segmentation_model=False,
    ):
        super(GlobalGenerator_DCDCv2, self).__init__()
        activation = nn.ReLU(True)

        cond_embed_dim = ngf
        self.cond_embed = nn.Sequential(
            nn.Linear(ngf, cond_embed_dim),
            activation,
            nn.Linear(cond_embed_dim, cond_embed_dim),
            nn.InstanceNorm1d(cond_embed_dim),
        )
        self.embedding = spectral_norm(nn.Embedding(num_embeddings=3, embedding_dim=ngf))

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, min(ngf, mc), kernel_size=7, padding=0),  #
            norm_layer(ngf),
            activation,
        ]
        ### downsample
        for i in range(start_r):
            mult = 2 ** i
            model += [
                nn.Conv2d(
                    min(ngf * mult, mc),
                    min(ngf * mult * 2, mc),
                    kernel_size=k_size,
                    stride=2,
                    padding=1,
                ),
                norm_layer(min(ngf * mult * 2, mc)),
                activation,
            ]
            model += [
                ResnetBlock(
                    min(ngf * mult * 2, mc),
                    padding_type=padding_type,
                    activation=activation,
                    norm_layer=norm_layer,
                )
            ]
            model += [
                ResnetBlock(
                    min(ngf * mult * 2, mc),
                    padding_type=padding_type,
                    activation=activation,
                    norm_layer=norm_layer,
                )
            ]
        for i in range(start_r, n_downsampling - 1):
            mult = 2 ** i
            model += [
                nn.Conv2d(
                    min(ngf * mult, mc),
                    min(ngf * mult * 2, mc),
                    kernel_size=k_size,
                    stride=2,
                    padding=1,
                ),
                norm_layer(min(ngf * mult * 2, mc)),
                activation,
            ]
            model += [
                ResnetBlock(
                    min(ngf * mult * 2, mc),
                    padding_type=padding_type,
                    activation=activation,
                    norm_layer=norm_layer,
                )
            ]
            model += [
                ResnetBlock(
                    min(ngf * mult * 2, mc),
                    padding_type=padding_type,
                    activation=activation,
                    norm_layer=norm_layer,
                )
            ]
        mult = 2 ** (n_downsampling - 1)

        if spatio_size == 32:
            model += [
                nn.Conv2d(
                    min(ngf * mult, mc),
                    min(ngf * mult * 2, mc),
                    kernel_size=k_size,
                    stride=2,
                    padding=1,
                ),
                norm_layer(min(ngf * mult * 2, mc)),
                activation,
            ]
        if spatio_size == 64:
            model += [
                ResnetBlock(
                    min(ngf * mult * 2, mc),
                    padding_type=padding_type,
                    activation=activation,
                    norm_layer=norm_layer,
                )
            ]
        model += [
            ResnetBlock(
                min(ngf * mult * 2, mc),
                padding_type=padding_type,
                activation=activation,
                norm_layer=norm_layer,
            )
        ]
        # model += [nn.Conv2d(min(ngf * mult * 2, opt.mc), min(ngf, opt.mc), 1, 1)]
        if feat_dim > 0:
            model += [nn.Conv2d(min(ngf * mult * 2, mc), feat_dim, 1, 1)]
        self.encoder = nn.Sequential(*model)

        # decode
        model = []
        if feat_dim > 0:  #
            model += [nn.Conv2d(feat_dim, min(ngf * mult * 2, mc), 1, 1)]
        # model += [nn.Conv2d(min(ngf, opt.mc), min(ngf * mult * 2, opt.mc), 1, 1)]
        o_pad = 0 if k_size == 4 else 1  #
        mult = 2 ** n_downsampling  #
        model += [
            ResnetBlock(
                min(ngf * mult, mc),
                padding_type=padding_type,
                activation=activation,
                norm_layer=norm_layer,
            )
        ]

        if spatio_size == 32:
            model += [
                nn.ConvTranspose2d(
                    min(ngf * mult, mc),
                    min(int(ngf * mult / 2), mc),
                    kernel_size=k_size,
                    stride=2,
                    padding=1,
                    output_padding=o_pad,
                ),
                norm_layer(min(int(ngf * mult / 2), mc)),
                activation,
            ]
        if spatio_size == 64:
            model += [
                ResnetBlock(
                    min(ngf * mult, mc),
                    padding_type=padding_type,
                    activation=activation,
                    norm_layer=norm_layer,
                )
            ]

        for i in range(1, n_downsampling - start_r):
            mult = 2 ** (n_downsampling - i)
            model += [
                ResnetBlock(
                    min(ngf * mult, mc),
                    padding_type=padding_type,
                    activation=activation,
                    norm_layer=norm_layer,
                )
            ]
            model += [
                ResnetBlock(
                    min(ngf * mult, mc),
                    padding_type=padding_type,
                    activation=activation,
                    norm_layer=norm_layer,
                )
            ]
            model += [
                nn.ConvTranspose2d(
                    min(ngf * mult, mc),
                    min(int(ngf * mult / 2), mc),
                    kernel_size=k_size,
                    stride=2,
                    padding=1,
                    output_padding=o_pad,
                ),
                norm_layer(min(int(ngf * mult / 2), mc)),
                activation,
            ]
        for i in range(n_downsampling - start_r, n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    min(ngf * mult, mc),
                    min(int(ngf * mult / 2), mc),
                    kernel_size=k_size,
                    stride=2,
                    padding=1,
                    output_padding=o_pad,
                ),
                norm_layer(min(int(ngf * mult / 2), mc)),
                activation,
            ]

        if use_segmentation_model:
            model += [nn.ReflectionPad2d(3), nn.Conv2d(min(ngf, mc), output_nc, kernel_size=7, padding=0)]
        else:
            model += [
                nn.ReflectionPad2d(3),
                nn.Conv2d(min(ngf, mc), output_nc, kernel_size=7, padding=0),
                nn.Tanh(),
            ]
        self.decoder = nn.Sequential(*model)

    def encodeing(self, x, imgc=None):
        emb = self.get_embedding(imgc=imgc)
        h = x.type(torch.float32)
        for model in self.encoder:
            if isinstance(model, ResnetBlock):
                h = model(h, emb)
            else:
                h = model(h)
        return h
    def decodeing(self, h, imgc=None):
        emb = self.get_embedding(imgc=imgc)
        for model in self.decoder:
            if isinstance(model, ResnetBlock):
                h = model(h, emb)
            else:
                h = model(h)
        return h

    def forward(self, x, imgc):
        hidden = self.encodeing(x, imgc)
        out = self.decodeing(hidden, imgc)
        return out

    def get_embedding(self, imgc=None):
        if imgc is not None:
            imgc = imgc.view(-1, )
            embedded = self.cond_embed(self.embedding(imgc))
        else:
            embedded = None
        return embedded


if __name__ == '__main__':
    from data.dataset import InpaintDataset
    from torch.utils.data import DataLoader
    import tqdm
    from utils.showImage import Show
    import torch


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = GlobalGenerator_DCDCv2(input_nc=3, output_nc=3, ngf=64, k_size=4, n_downsampling=3).to(device)
    torch.save(model,'model.pth')
    # model_path = 'checkpoint/A_model/latest_net_G.pth'  # 替换为模型文件的实际路径
    # model.load_state_dict(torch.load(model_path, map_location=device))


    dataset = InpaintDataset(data_root='datasets')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


    for data in tqdm.tqdm(dataloader):
        syn = data['synthetic'].to(device)
        mask = data['mask'].to(device)
        inst = data['inst'].to(device)
        noise = torch.randn(syn[0].shape).to(device)
        input = mask * syn + (1 - mask) * noise
        input = syn
        Show(input, 'input')
        output = model(input,imgc=inst)
        Show(output, 'output')
