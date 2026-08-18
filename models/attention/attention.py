import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from einops import rearrange
from .HighDctFrequencyExtractor import HighDctFrequencyExtractor
from .LowDctFrequencyExtractor import LowDctFrequencyExtractor

Conv2d = nn.Conv2d

##########################################################################
## Layer Norm
def to_2d(x):
    return rearrange(x, 'b c h w -> b (h w c)')


def to_3d(x):
    #    return rearrange(x, 'b c h w -> b c (h w)')
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    #    return rearrange(x, 'b c (h w) -> b c h w',h=h,w=w)
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

##########################################################################
## Dynamic-range Histogram Self-Attention (DHSA)

class Attention_histogram(nn.Module):
    def __init__(self, dim, num_heads, bias, ifBox=True):
        super(Attention_histogram, self).__init__()
        self.low_dct = LowDctFrequencyExtractor()
        self.high_dct = HighDctFrequencyExtractor()
        self.factor = num_heads
        self.ifBox = ifBox
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 使用卷积层将输入维度转换为dim*2
        self.qk1 = nn.Conv2d(2 * dim, 2 * dim, kernel_size=1, bias=bias)
        # 深度可分离卷积，不改变通道数但进行局部特征提取
        self.qk1_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)

        # 使用卷积层将输入维度转换为dim*2
        self.v = nn.Conv2d(3 * dim, dim, kernel_size=1, bias=bias)
        # 深度可分离卷积，不改变通道数但进行局部特征提取
        self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        # 使用卷积层将输入维度转换为dim*2
        self.qk2 = nn.Conv2d(2 * dim, 2 * dim, kernel_size=1, bias=bias)
        # 深度可分离卷积，不改变通道数但进行局部特征提取
        self.qk2_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)

        # 输出投影层，将结果映射回原始维度
        self.project_out = nn.Conv2d(2 * dim, dim, kernel_size=1, bias=bias)

    def pad(self, x, factor):
        hw = x.shape[-1]
        t_pad = [0, 0] if hw % factor == 0 else [0, (hw // factor + 1) * factor - hw]
        x = F.pad(x, t_pad, 'constant', 0)
        return x, t_pad

    def unpad(self, x, t_pad):
        _, _, hw = x.shape
        return x[:, :, t_pad[0]:hw - t_pad[1]]

    def softmax_1(self, x, dim=-1):
        logit = x.exp()
        logit = logit / (logit.sum(dim, keepdim=True) + 1)
        return logit

    def normalize(self, x):
        mu = x.mean(-2, keepdim=True)
        sigma = x.var(-2, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5)  # * self.weight + self.bias

    def reshape_attn(self, q, k, v, ifBox):
        b, c = q.shape[:2]
        q, t_pad = self.pad(q, self.factor)
        k, t_pad = self.pad(k, self.factor)
        v, t_pad = self.pad(v, self.factor)
        hw = q.shape[-1] // self.factor
        shape_ori = "b (head c) (factor hw)" if ifBox else "b (head c) (hw factor)"
        shape_tar = "b head (c factor) hw"
        q = rearrange(q, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        k = rearrange(k, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        v = rearrange(v, '{} -> {}'.format(shape_ori, shape_tar), factor=self.factor, hw=hw, head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = self.softmax_1(attn, dim=-1)
        out = (attn @ v)
        out = rearrange(out, '{} -> {}'.format(shape_tar, shape_ori), factor=self.factor, hw=hw, b=b,
                        head=self.num_heads)
        out = self.unpad(out, t_pad)
        return out

    def forward(self, x):
        b, c, h, w = x.shape  # 获取输入张量的形状
        high_freq = self.high_dct(x)
        low_freq = self.low_dct(x)
        input_high = torch.concat([x, high_freq], dim=1)
        input_low = torch.concat([x, low_freq], dim=1)
        input_all = torch.concat([x, high_freq, low_freq], dim=1)

        # 通过卷积层获取qkv
        qk1 = self.qk1_dwconv(self.qk1(input_high))  # 1C 变5C 然后 5C变5C
        q1, k1 = qk1.chunk(2, dim=1)  # 将qkv分割成五个部分 5C变1C

        v = self.v_dwconv(self.v(input_all))

        qk2 = self.qk2_dwconv(self.qk2(input_low))
        q2, k2 = qk2.chunk(2, dim=1)
        # 对v进行排序，并根据相同的索引更新q1, k1, q2, k2   B C (H W)
        v, idx = v.view(b, c, -1).sort(dim=-1)

        # torch.gather 函数用于根据索引 idx 从 q1 中收集特定位置的元素。    B C (H W)
        q1 = torch.gather(q1.view(b, c, -1), dim=2, index=idx)
        k1 = torch.gather(k1.view(b, c, -1), dim=2, index=idx)

        q2 = torch.gather(q2.view(b, c, -1), dim=2, index=idx)
        k2 = torch.gather(k2.view(b, c, -1), dim=2, index=idx)

        # 计算两次注意力机制
        out1 = self.reshape_attn(q1, k1, v, False)  # 长距离依赖关系
        out2 = self.reshape_attn(q2, k2, v, True)  # 局部信息

        out1 = torch.scatter(out1, 2, idx, out1).view(b, c, h, w)
        out2 = torch.scatter(out2, 2, idx, out2).view(b, c, h, w)

        out = out1 * out2  # 合并两个输出
        out = torch.cat((x, out), dim=1)

        out = self.project_out(out)  # 投影输出
        return out

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        #        self.weight = nn.Parameter(torch.ones(normalized_shape))
        #        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5)  # * self.weight + self.bias
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        #        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5)  # * self.weight

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type="WithBias"):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## Dual-scale Gated Feed-Forward Network (DGFF)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv_5 = Conv2d(hidden_features // 4, hidden_features // 4, kernel_size=5, stride=1, padding=2,
                               groups=hidden_features // 4, bias=bias)
        self.dwconv_dilated2_1 = Conv2d(hidden_features // 4, hidden_features // 4, kernel_size=3, stride=1, padding=2,
                                        groups=hidden_features // 4, bias=bias, dilation=2)
        self.p_unshuffle = nn.PixelUnshuffle(2)
        self.p_shuffle = nn.PixelShuffle(2)

        self.project_out = Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.p_shuffle(x)

        x1, x2 = x.chunk(2, dim=1)
        #        x2_1, x2_2 = x2.chunk(2, dim=1)
        x1 = self.dwconv_5(x1)
        x2 = self.dwconv_dilated2_1(x2)
        #        x2_2 = self.dwconv_dilated3_1( x2_2 )
        #        x2 = torch.cat([x2_1, x2_2], dim=1)
        x = F.mish(x2) * x1
        x = self.p_unshuffle(x)
        x = self.project_out(x)

        #        x1 = self.dwconv_5(x)
        #        x2 = self.dwconv_dilated_2(x)
        #        x = F.mish(x2) * x1 + x

        return x

class TransformerBlock(nn.Module):  # 主
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):#36,1,2.667,False.WithBais
        super(TransformerBlock, self).__init__()

        self.attn_g = Attention_histogram(dim, num_heads, bias, True)#注意力
        self.norm_g = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

        self.norm_ff1 = LayerNorm(dim, LayerNorm_type)

    def forward(self, x):
        x = x + self.attn_g(self.norm_g(x))
        x_out = x + self.ffn(self.norm_ff1(x))

        return x_out