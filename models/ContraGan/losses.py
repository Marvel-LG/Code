from torch import autograd
import torch
import torch.distributed as dist
import numpy as np

class GatherLayer(torch.autograd.Function):
    """
    This file is copied from
    https://github.com/open-mmlab/OpenSelfSup/blob/master/openselfsup/models/utils/gather_layer.py
    Gather tensors from all process, supporting backward propagation
    """
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out

class ConditionalContrastiveLoss(torch.nn.Module):#L_2c损失
    def __init__(self, num_classes, temperature, master_rank='cuda'):
        super(ConditionalContrastiveLoss, self).__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.master_rank = master_rank#设备名
        self.calculate_similarity_matrix = self._calculate_similarity_matrix()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def _make_neg_removal_mask(self, labels):
        labels = labels.detach().cpu().numpy()
        n_samples = labels.shape[0]
        mask_multi, target = np.zeros([self.num_classes, n_samples]), 1.0
        for c in range(self.num_classes):#统计同类的向量
            c_indices = np.where(labels == c)
            mask_multi[c, c_indices] = target
        return torch.tensor(mask_multi).type(torch.long).to(self.master_rank)

    def _calculate_similarity_matrix(self):
        return self._cosine_simililarity_matrix

    def _remove_diag(self, M):
        h, w = M.shape
        assert h == w, "h and w should be same"
        mask = np.ones((h, w)) - np.eye(h)
        mask = torch.from_numpy(mask)
        mask = (mask).type(torch.bool).to(self.master_rank)
        return M[mask].view(h, -1)

    def _cosine_simililarity_matrix(self, x, y):
        v = self.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, embed, proxy, label, **_):

        sim_matrix = self.calculate_similarity_matrix(embed, embed)#分子上的右边
        sim_matrix = torch.exp(self._remove_diag(sim_matrix) / self.temperature)#去除无关样本
        neg_removal_mask = self._remove_diag(self._make_neg_removal_mask(label)[label])#负样本移除
        sim_pos_only = neg_removal_mask * sim_matrix#计算mask的分子右边

        emb2proxy = torch.exp(self.cosine_similarity(embed, proxy) / self.temperature)#图像和类别的对比损失，分子分母的左边

        numerator = emb2proxy + sim_pos_only.sum(dim=1)#分子部分
        denomerator = torch.cat([torch.unsqueeze(emb2proxy, dim=1), sim_matrix], dim=1).sum(dim=1)#分子部分，没有移除负样本
        return -torch.log(numerator / denomerator).mean()
