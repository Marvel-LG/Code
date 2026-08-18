import math
import torch
import torch.nn as nn
import numpy as np
import functools
from abc import abstractmethod


# class ResnetBlock(nn.Module):
#     def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
#                  dropout,ch):
#         super().__init__()
#         self.in_channels = in_channels
#         self.ch = ch
#         out_channels = in_channels if out_channels is None else out_channels
#         self.out_channels = out_channels
#         self.use_conv_shortcut = conv_shortcut
#
#         self.norm1 = Normalize(in_channels)  # 组正则化
#         self.conv1 = torch.nn.Conv2d(in_channels,
#                                      out_channels,
#                                      kernel_size=3,
#                                      stride=1,
#                                      padding=1)  # 正常巻积
#         self.norm2 = Normalize(out_channels)
#         self.dropout = torch.nn.Dropout(dropout)
#         self.conv2 = torch.nn.Conv2d(out_channels,
#                                      out_channels,
#                                      kernel_size=3,
#                                      stride=1,
#                                      padding=1)
#         if self.in_channels != self.out_channels:
#             if self.use_conv_shortcut:  # 是否使用巻积跳连
#                 self.conv_shortcut = torch.nn.Conv2d(in_channels,
#                                                      out_channels,
#                                                      kernel_size=3,
#                                                      stride=1,
#                                                      padding=1)
#             else:
#                 self.nin_shortcut = torch.nn.Conv2d(in_channels,
#                                                     out_channels,
#                                                     kernel_size=1,
#                                                     stride=1,
#                                                     padding=0)
#
#     def forward(self, x, embc):
#         h = x
#         h = self.norm1(h)
#         h = nonlinearity(h)
#         h = self.conv1(h)
#
#         if embc is not None:
#             h = h + nonlinearity(embc).repeat(1, h.shape[1]//self.ch)[:, :, None, None]
#
#         h = self.norm2(h)
#         h = nonlinearity(h)
#         h = self.dropout(h)
#         h = self.conv2(h)
#
#         if self.in_channels != self.out_channels:
#             if self.use_conv_shortcut:
#                 x = self.conv_shortcut(x)
#             else:
#                 x = self.nin_shortcut(x)
#
#         return x + h

def Instancenormalization(in_channels):
    return torch.nn.InstanceNorm2d(in_channels, affine=True)


def nonlinearity(x):
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none"], f'attn_type {attn_type} unknown'
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    else:
        return nn.Identity(in_channels)


class EmbedBlock(nn.Module):
    """
    Any module where forward() takes embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` embeddings.
        """


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class ResnetBlock(EmbedBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of embedding channels.
    :param dropout: the rate of dropout.
    :param out_channel: if specified, the number of out channels.
    :param use_conv: if True and out_channel is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
            self,
            channels,
            emb_channels,
            dropout,
            out_channel=None,
            use_conv=False,
            use_scale_shift_norm=False,
            use_checkpoint=False,
            up=False,
            down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channel = out_channel or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            nn.Conv2d(channels, self.out_channel, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False)
            self.x_upd = Upsample(channels, False)
        elif down:
            self.h_upd = Downsample(channels, False)
            self.x_upd = Downsample(channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channel if use_scale_shift_norm else self.out_channel,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channel),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv2d(self.out_channel, self.out_channel, 3, padding=1)
            ),
        )

        if self.out_channel == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(
                channels, self.out_channel, 3, padding=1
            )
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channel, 1)

    def forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        if emb is not None:
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
        else:
            if self.use_scale_shift_norm:
                out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
                h = out_norm(h)
                h = out_rest(h)
            else:
                h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,  # 展平得到q
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,  #
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class Encoder(nn.Module):  # 编码器
    def __init__(self, *,
                 ch,  # 内部通道数的基础
                 out_ch,  # 输出通道数
                 ch_mult=(1, 2, 4, 8),  # 通道数的倍数
                 num_res_blocks,  # 残差块的数量
                 attn_resolutions,  # 注意力机制的添加位置
                 dropout=0.0,
                 resamp_with_conv=True,  # 是否下采样
                 in_channels,  # 输入通道数
                 resolution,
                 z_channels,  # z通道数
                 cond_embed_dim,
                 double_z=True,  # 是否两倍z
                 use_linear_attn=False,  # 是否使用线性注意力
                 attn_type="vanilla",  # 默认注意力类型
                 use_checkpoint=False,
                 use_scale_shift_norm=True
                 ):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.num_resolutions = len(ch_mult)  # 下采样次数
        self.num_res_blocks = num_res_blocks  # 连续残差块个数
        self.resolution = resolution  #
        self.in_channels = in_channels
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_checkpoint = use_checkpoint
        self.cond_embed_dim = cond_embed_dim
        self.chs = [self.ch]
        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,  # 扩大通道数
                                       self.chs[-1],
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution  # 分辨率大小
        in_ch_mult = (1,) + tuple(ch_mult)  # 通道数扩大的倍数
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()  # 下采样列表
        for i_level in range(self.num_resolutions):  # 下采样次数
            block = nn.ModuleList()
            attn = nn.ModuleList()

            block_in = ch * in_ch_mult[i_level]  # 输入通道数
            block_out = ch * ch_mult[i_level]  # 输出通道数

            for i_block in range(self.num_res_blocks):  # 残差块
                block.append(ResnetBlock(block_in,
                                         cond_embed_dim,
                                         dropout,
                                         out_channel=block_out,
                                         use_checkpoint=use_checkpoint,
                                         use_scale_shift_norm=use_scale_shift_norm
                                         ))
                self.chs.append(block_out)
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()  # 下采样模块
            down.block = block
            down.attn = attn  # 注意力模块
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                self.chs.append(block_in)
                curr_res = curr_res // 2  # 当前大小
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in,
                                       cond_embed_dim,
                                       dropout,
                                       out_channel=block_in,
                                       use_checkpoint=use_checkpoint,
                                       use_scale_shift_norm=use_scale_shift_norm
                                       )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(block_in,
                                       cond_embed_dim,
                                       dropout,
                                       out_channel=block_in,
                                       use_checkpoint=use_checkpoint,
                                       use_scale_shift_norm=use_scale_shift_norm)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        2 * z_channels if double_z else z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)  # 最后编码8通道

    def forward(self, x, embc, *args, **kwargs):
        # downsampling
        hs = [self.conv_in(x)]  # 输入
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], embc)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, embc)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, embc)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h, hs


class Decoder(nn.Module):
    def __init__(self, *,
                 ch,
                 out_ch,
                 ch_mult=(1, 2, 4, 8),
                 num_res_blocks,
                 attn_resolutions,
                 dropout=0.0,
                 resamp_with_conv=True,
                 in_channels,
                 resolution,
                 z_channels,
                 cond_embed_dim,
                 give_pre_end=False,
                 tanh_out=False,
                 use_linear_attn=False,
                 attn_type="vanilla",
                 use_checkpoint=False,
                 use_scale_shift_norm=True,
                 **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.cond_embed_dim = cond_embed_dim

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in,
                                       cond_embed_dim,
                                       dropout,
                                       out_channel=block_in,
                                       use_checkpoint=use_checkpoint,
                                       use_scale_shift_norm=use_scale_shift_norm)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)  # 注意力模块
        self.mid.block_2 = ResnetBlock(block_in,
                                       cond_embed_dim,
                                       dropout,
                                       out_channel=block_in,
                                       use_checkpoint=use_checkpoint,
                                       use_scale_shift_norm=use_scale_shift_norm)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):  # 转置通道倍数
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(block_in,
                                         cond_embed_dim,
                                         dropout,
                                         out_channel=block_out,
                                         use_checkpoint=use_checkpoint,
                                         use_scale_shift_norm=use_scale_shift_norm))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z, embc):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        # temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, embc)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, embc)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, embc)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h
