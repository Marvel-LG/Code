import torch
from torch.autograd import Variable


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, double_z=False):
        self.double_z = double_z
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self, inference):
        if inference:
            if self.double_z:
                x = self.mean + self.std
            else:
                x = self.parameters
        else:
            if self.double_z:
                x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
            else:
                x = self.parameters + torch.randn(self.parameters.shape).to(device=self.parameters.device)
                # Variable(torch.randn(self.parameters.size()).cuda(self.parameters.data.get_device()))#
                #torch.randn(self.parameters.shape).to(device=self.parameters.device)
        return x

    def kl(self, index=None):
        if self.double_z:
            kl = 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3]) * 1e-6
        else:
            kl = torch.sum(torch.pow(self.parameters, 2), dim=[1, 2, 3])
        if index is not None:
            return torch.mean(kl[index])
        else:
            return torch.mean(kl)
