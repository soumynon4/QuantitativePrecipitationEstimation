# @Time     : 12/20/2021 11:36 AM
# @Author   : ZhangJianchang
# @Email    : zz19970227@gmail.com
# @File     : MLP.py
# @Project  : QuantitativePrecipitationEstimation
import torch


class MultilayerPerceptron(torch.nn.Module):
    def __init__(self, inputFeatureSize):
        super(MultilayerPerceptron, self).__init__()

        self.fc1 = torch.nn.Linear(inputFeatureSize, 64)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.relu1 = torch.nn.GELU()

        self.fc2 = torch.nn.Linear(64, 256)
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.relu2 = torch.nn.GELU()

        self.fc3 = torch.nn.Linear(256, 512)
        self.bn3 = torch.nn.BatchNorm1d(512)
        self.relu3 = torch.nn.GELU()

        self.fc4 = torch.nn.Linear(512, 512)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.relu4 = torch.nn.GELU()

        self.fc5 = torch.nn.Linear(512, 256)
        self.bn5 = torch.nn.BatchNorm1d(256)
        self.relu5 = torch.nn.GELU()
        self.fc6 = torch.nn.Linear(256, 1)

    def forward(self, inputFeature):
        outcome = self.relu1(self.bn1(self.fc1(inputFeature)))
        outcome = self.relu2(self.bn2(self.fc2(outcome)))
        outcome = self.relu3(self.bn3(self.fc3(outcome)))
        outcome = self.relu4(self.bn4(self.fc4(outcome)))
        outcome = self.relu5(self.bn5(self.fc5(outcome)))
        outcome = self.fc6(outcome)
        return outcome
