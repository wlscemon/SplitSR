import torch
from torch import nn

from modules.blocks import MeanShift, ResidualBlock, Upsample, SplitSRBlock


class SplitSR(nn.Module):
    def __init__(self):
        super(SplitSR, self).__init__()

        self.ResidualGroup1 = nn.Sequential(
            ResidualBlock(channels=64),
            SplitSRBlock(channels=64, kernel=3, alpha=0.25),
            SplitSRBlock(channels=64, kernel=3, alpha=0.25),
            ResidualBlock(channels=64),
            SplitSRBlock(channels=64, kernel=3, alpha=0.25),
            SplitSRBlock(channels=64, kernel=3, alpha=0.25),
        )
        self.ResidualGroup2 = nn.Sequential(
            ResidualBlock(channels=64),
            SplitSRBlock(channels=64, kernel=3, alpha=0.25),
            SplitSRBlock(channels=64, kernel=3, alpha=0.25),
            ResidualBlock(channels=64),
            SplitSRBlock(channels=64, kernel=3, alpha=0.25),
            SplitSRBlock(channels=64, kernel=3, alpha=0.25),
        )
        self.ResidualGroup3 = nn.Sequential(
            ResidualBlock(channels=64),
            SplitSRBlock(channels=64, kernel=3, alpha=0.25),
            SplitSRBlock(channels=64, kernel=3, alpha=0.25),
            ResidualBlock(channels=64),
            SplitSRBlock(channels=64, kernel=3, alpha=0.25),
            SplitSRBlock(channels=64, kernel=3, alpha=0.25),
        )
        self.ResidualGroup4 = nn.Sequential(
            ResidualBlock(channels=64),
            SplitSRBlock(channels=64, kernel=3, alpha=0.25),
            SplitSRBlock(channels=64, kernel=3, alpha=0.25),
            ResidualBlock(channels=64),
            SplitSRBlock(channels=64, kernel=3, alpha=0.25),
            SplitSRBlock(channels=64, kernel=3, alpha=0.25),
        )
        self.ResidualGroup5 = nn.Sequential(
            ResidualBlock(channels=64),
            SplitSRBlock(channels=64, kernel=3, alpha=0.25),
            SplitSRBlock(channels=64, kernel=3, alpha=0.25),
            ResidualBlock(channels=64),
            SplitSRBlock(channels=64, kernel=3, alpha=0.25),
            SplitSRBlock(channels=64, kernel=3, alpha=0.25),
        )
        self.ResidualGroup6 = nn.Sequential(
            ResidualBlock(channels=64),
            SplitSRBlock(channels=64, kernel=3, alpha=0.25),
            SplitSRBlock(channels=64, kernel=3, alpha=0.25),
            ResidualBlock(channels=64),
            SplitSRBlock(channels=64, kernel=3, alpha=0.25),
            SplitSRBlock(channels=64, kernel=3, alpha=0.25),
        )
        self.conv_head = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=3 // 2, bias=True)
        self.conv_back = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=3 // 2, bias=True)
        self.upsample = Upsample(64)
        self.MeanSubstract = MeanShift(-1)
        self.MeanAdd = MeanShift(1)
        self.relu = nn.ReLU()

    def forward(self, x):
        #x = self.MeanSubstract(x)
        x = self.conv_head(x)

        x = self.ResidualGroup1(x)
        x = self.ResidualGroup2(x)
        x = self.ResidualGroup3(x)
        x = self.ResidualGroup4(x)
        x = self.ResidualGroup5(x)
        x = self.ResidualGroup6(x)

        x = self.upsample(x)
        #print(f"up:{x[0][:5]}")
        x = self.conv_back(x)
        #print(f"cov:{x[0][:5]}")
        #x = self.MeanAdd(x)
        #print(f"mean:{x[0][:5]}")
        #x = self.relu(x)

        return x


if __name__ == '__main__':
    model = SplitSR()

    x = torch.randn(1, 3, 96, 96)
    y = model(x)

    assert y.shape[-1] == x.shape[-1] * 4 & y.shape[-2] == x.shape[-2] * 4

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params: {params}')
# print(model.paremeters.keys)
# model = SplitSR()
# print(model.state_dict().keys())