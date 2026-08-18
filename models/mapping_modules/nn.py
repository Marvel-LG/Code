"""
Various utilities for neural networks.
"""

import math
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import functools

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)



def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])


def gamma_embedding(gammas, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param gammas: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=gammas.device)
    args = gammas[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def get_norm_layer(norm_type="instance"):
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer
class ResnetBlock(nn.Module):
    def __init__(
            self, dim, padding_type, norm_layer, opt, activation=nn.ReLU(True), use_dropout=False, dilation=1
    ):
        super(ResnetBlock, self).__init__()
        self.opt = opt
        self.dilation = dilation
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
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

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
class Patch_Attention_4(nn.Module):  ## While combine the feature map, use conv and mask
    def __init__(self, in_channels, inter_channels, patch_size):
        super(Patch_Attention_4, self).__init__()
        self.patch_size = patch_size
        self.F_Combine = nn.Conv2d(in_channels=in_channels*2+1, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=True)
        norm_layer = get_norm_layer(norm_type="instance")
        activation = nn.ReLU(True)
        model = []
        for i in range(1):
            model += [
                ResnetBlock(
                    inter_channels,
                    padding_type="reflect",
                    activation=activation,
                    norm_layer=norm_layer,
                    opt=None,
                )
            ]
        self.res_block = nn.Sequential(*model)
    def Hard_Compose(self, input, dim, index):
        # batch index select
        # input: [B,C,HW]
        # dim: scalar > 0
        # index: [B, HW]
        views = [input.size(0)] + [1 if i != dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def forward(self, z, mask):  ## Reduce the extra memory cost

        x = self.res_block(z)

        b, c, h, w = x.shape

        mask = F.interpolate(mask, (x.size(2), x.size(3)), mode="bilinear")#将mask与输入大小对齐

        ## 1: mask position 0: non-mask

        mask_unfold = F.unfold(mask, kernel_size=(self.patch_size, self.patch_size), padding=0, stride=self.patch_size)
        non_mask_region = (torch.mean(mask_unfold, dim=1, keepdim=True) == 1).float()[0, 0, :]  # 1*1*all_patch_num
        #>0.6是将其变成true或false，选择出有mask的地方
        # all_patch_num = h * w / self.patch_size / self.patch_size
        mask_index = torch.nonzero(non_mask_region, as_tuple=True)[0]#因为返回的是一个张量，所以要用[0]取出需要的结果
        unmask_index = torch.nonzero(non_mask_region != 1, as_tuple=True)[0]
        # print(mask_index.shape[0]/mask_unfold.shape[2])
        if len(mask_index) == 0 or len(unmask_index) == 0 or True:  ## No mask patch is selected, no attention is needed
            composed_fold = x
        else:
            #找到第一个不为1的索引
            x_unfold = F.unfold(x, kernel_size=(self.patch_size, self.patch_size), padding=0, stride=self.patch_size)
            #将x按照patch_size划分块
            Query_Patch = torch.index_select(x_unfold, 2, mask_index)#取出x中非0的patch
            Key_Patch = torch.index_select(x_unfold, 2, unmask_index)#取出x中为0的patch

            Query_Patch = Query_Patch.permute(0, 2, 1)#相当于转制
            Query_Patch_normalized = F.normalize(Query_Patch, dim=2)
            Key_Patch_normalized = F.normalize(Key_Patch, dim=1)
            #Q*K
            correlation_matrix = torch.bmm(Query_Patch_normalized, Key_Patch_normalized)
            correlation_matrix = F.softmax(correlation_matrix, dim=2)

            R, max_arg = torch.max(correlation_matrix, dim=2)#沿着第二维度取找到最大值，并且返回在该维度上的索引

            composed_unfold = self.Hard_Compose(Key_Patch, 2, max_arg)
            x_unfold[:, :, mask_index] = composed_unfold
            composed_fold = F.fold(x_unfold, output_size=(h, w), kernel_size=(self.patch_size, self.patch_size),
                                   padding=0, stride=self.patch_size)

        concat_1 = torch.cat((z, composed_fold, mask), dim=1)
        concat_1 = self.F_Combine(concat_1)

        return concat_1