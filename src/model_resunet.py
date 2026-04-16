import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Match channels for residual connection if needed
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        out = out + identity
        out = self.relu(out)
        return out


class ResUNetDownscale(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super().__init__()

        # Encoder
        self.inc = ResidualBlock(in_channels, 32)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualBlock(32, 64)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualBlock(64, 128)
        )

        # Decoder
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up1 = ResidualBlock(128, 64)   # 64 up + 64 skip

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up2 = ResidualBlock(64, 32)    # 32 up + 32 skip

        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)      # (B, 32, H, W)
        x2 = self.down1(x1)   # (B, 64, H/2, W/2)
        x3 = self.down2(x2)   # (B, 128, H/4, W/4)

        # Decoder 1
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

        # Decoder 2
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