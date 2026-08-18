"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np


class SupConLoss(nn.Module):
    def __init__(self, num_classes, temperature, master_rank='cuda'):
        super(SupConLoss, self).__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.master_rank = master_rank  # 设备名
        self.calculate_similarity_matrix = self._calculate_similarity_matrix()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def _make_neg_removal_mask(self, labels):
        labels = labels.detach().cpu().numpy()
        n_samples = labels.shape[0]
        mask_multi, target = np.zeros([self.num_classes, n_samples]), 1.0
        for c in range(self.num_classes):  # 统计同类的向量
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
        # print(embed.size(), proxy.size())
        sim_matrix = self.calculate_similarity_matrix(embed, embed)  # 分母的右边
        sim_matrix = torch.exp(self._remove_diag(sim_matrix) / self.temperature)  # 去除本身相关度
        neg_removal_mask = self._remove_diag(self._make_neg_removal_mask(label)[label])  # 去除负样的mask
        sim_pos_only = neg_removal_mask * sim_matrix  # 计算mask的分子右边

        emb2proxy = torch.exp(self.cosine_similarity(embed, proxy) / self.temperature)  # 图像和类别的对比损失，分子分母的左边

        numerator = emb2proxy + sim_pos_only.sum(dim=1)  # 分子部分
        denomerator = torch.cat([torch.unsqueeze(emb2proxy, dim=1), sim_matrix], dim=1).sum(dim=1)  # 分子部分，没有移除负样本
        return -torch.log(numerator / denomerator).mean()

if __name__ == '__main__':
    device = torch.device('cpu')
    features = torch.randn((4, 128)).to(device)
    proxy = torch.randn((4, 128)).to(device)
    labels = torch.tensor([0, 2, 1, 2]).to(device)
    ContraGanLoss = SupConLoss(num_classes=2, temperature=1.0, master_rank='cpu')
    loss = ContraGanLoss(features, proxy, (labels == 2).long())
    print(loss)
