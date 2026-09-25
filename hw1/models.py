import torch
import torch.nn as nn


class SmallCNN(nn.Module):

  def __init__(self):
    super().__init__()

    # Conv 7x7 s2 3->32, MaxPool 3x3 s2 p1 -> S/4
    self.block1 = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    )

    # Conv 5x5 32->64 -> S/4
    self.block2 = nn.Sequential(
        nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, bias=False),
        nn.ReLU(inplace=True),
    )

    # Conv 3x3 s2 64->128 -> S/8
    self.block3 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
        nn.ReLU(inplace=True),
    )

    # Conv 1x1 128->256 -> S/8
    self.block4 = nn.Sequential(
        nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False),
        nn.ReLU(inplace=True),
    )

    # Conv 3x3 s2 256->256 -> S/16
    self.block5 = nn.Sequential(
        nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
        nn.ReLU(inplace=True),
    )

    # Conv 1x1 256->512 -> S/16
    self.block6 = nn.Sequential(
        nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False),
        nn.ReLU(inplace=True),
    )

    # Head: GlobalAvgPool, Linear 512->256, ReLU, Linear 256->100
    self.head = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, 100),
    )

  def forward(self, x):
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    x = self.block5(x)
    x = self.block6(x)
    x = self.head(x)
    return x
