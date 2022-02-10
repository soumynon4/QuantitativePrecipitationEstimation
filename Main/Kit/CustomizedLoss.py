# @Time     : 12/27/2021 2:11 PM
# @Author   : ZhangJianchang
# @Email    : zz19970227@gmail.com
# @File     : CustomizedLoss.py
# @Project  : QuantitativePrecipitationEstimation
import torch


class Loss(torch.nn.Module):
    def __init__(self, delta: float = 1.0):
        super(Loss, self).__init__()
        self.delta = delta

    def forward(self, estimation: torch.Tensor, target: torch.Tensor):
        assert estimation.size() == target.size()

        # Huber Loss
        residual = torch.abs(estimation - target)
        largeLoss = 0.5 * torch.square(residual)
        smallLoss = self.delta * residual - 0.5 * (self.delta ** 2)
        cond = torch.less_equal(residual, self.delta)
        HuberLoss = torch.where(cond, largeLoss, smallLoss)

        # weight
        weightCond = torch.less_equal(target, 20.0)
        weight = torch.where(weightCond, torch.ones_like(target), torch.sqrt(target))

        assert HuberLoss.size() == weight.size()

        weightedLoss = torch.mean(HuberLoss * weight)
        return weightedLoss
