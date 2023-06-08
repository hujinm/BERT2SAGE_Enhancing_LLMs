import torch
import torch.nn as nn


class ConvNet_v1(nn.Module):

    def __init__(self):
        super(ConvNet_v1, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(2, 60, kernel_size=433, stride=1, padding=1),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv1d(60, 120, kernel_size=300, stride=1, padding=1),
            nn.ReLU())


        self.layer3 = nn.Sequential(
        nn.Conv1d(120, 240, kernel_size=200, stride=1, padding=1),
        nn.ReLU())


        self.layer4 = nn.Sequential(
            nn.Conv1d(240, 1, kernel_size=100, stride=1))

    def forward(self, x):
         out = self.layer1(x)
         out = self.layer2(out)
         out = self.layer3(out)
         out = self.layer4(out)
         out = out.view(-1)
         return out  