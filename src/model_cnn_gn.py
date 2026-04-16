import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConvGN(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()

        # make sure num_groups divides out_channels
        if out_channels % num_groups != 0:
            for g in [8, 4, 2, 1]:
                if out_channels % g == 0:
                    num_groups = g
                    break

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetDownscaleGN(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, num_groups=8):
        super().__init__()

        self.inc = DoubleConvGN(in_channels, 32, num_groups=num_groups)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvGN(32, 64, num_groups=num_groups)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvGN(64, 128, num_groups=num_groups)
        )

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConvGN(128, 64, num_groups=num_groups)

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConvGN(64, 32, num_groups=num_groups)

        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x = self.up1(x3)

        diffY = x2.size(2) - x.size(2)
        diffX = x2.size(3) - x.size(3)
        x = F.pad(
            x,
            [diffX // 2, diffX - diffX // 2,
             diffY // 2, diffY - diffY // 2]
        )

        x = torch.cat([x, x2], dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)

        diffY = x1.size(2) - x.size(2)
        diffX = x1.size(3) - x.size(3)
        x = F.pad(
            x,
            [diffX // 2, diffX - diffX // 2,
             diffY // 2, diffY - diffY // 2]
        )

        x = torch.cat([x, x1], dim=1)
        x = self.conv_up2(x)

        return self.outc(x)