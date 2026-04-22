import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class PositionalEncoding2D(nn.Module):
    """
    Learnable 2D positional embedding for transformer bottleneck.
    """
    def __init__(self, embed_dim, max_h=64, max_w=64):
        super().__init__()
        self.row_embed = nn.Parameter(torch.randn(1, embed_dim // 2, max_h, 1))
        self.col_embed = nn.Parameter(torch.randn(1, embed_dim - embed_dim // 2, 1, max_w))

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        b, c, h, w = x.shape
        row = self.row_embed[:, :, :h, :].expand(b, -1, h, w)
        col = self.col_embed[:, :, :, :w].expand(b, -1, h, w)
        pos = torch.cat([row, col], dim=1)
        return x + pos


class TransformerBottleneck(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_layers=3,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoding = PositionalEncoding2D(embed_dim)

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        x = self.pos_encoding(x)

        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)   # [B, H*W, C]
        x = self.transformer(x)
        x = x.transpose(1, 2).reshape(b, c, h, w)
        return x


class UNetTransformerDownscale(nn.Module):
    def __init__(
        self,
        in_channels=2,       # T + elevation
        out_channels=1,
        base_channels=32,
        bottleneck_channels=128,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()

        # Encoder: 32 -> 64 -> 128
        self.inc = DoubleConv(in_channels, base_channels)                       # 2 -> 32
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_channels, base_channels * 2)                        # 32 -> 64
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(base_channels * 2, bottleneck_channels)                  # 64 -> 128
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(bottleneck_channels, bottleneck_channels)                # 128 -> 128
        )

        # 1x1 projection into transformer embedding space
        self.pre_transform = nn.Conv2d(bottleneck_channels, embed_dim, kernel_size=1)

        # Transformer bottleneck
        self.bottleneck = TransformerBottleneck(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=4.0,
            dropout=dropout,
        )

        # 1x1 projection back to decoder channels
        self.post_transform = nn.Conv2d(embed_dim, bottleneck_channels, kernel_size=1)

        # Decoder
        self.up1 = nn.ConvTranspose2d(bottleneck_channels, bottleneck_channels, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(bottleneck_channels * 2, bottleneck_channels)   # 128 + 128

        self.up2 = nn.ConvTranspose2d(bottleneck_channels, base_channels * 2, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(base_channels * 4, base_channels * 2)            # 64 + 64

        self.up3 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(base_channels * 2, base_channels)                # 32 + 32

        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)        # [B, 32, H, W]
        x2 = self.down1(x1)     # [B, 64, H/2, W/2]
        x3 = self.down2(x2)     # [B, 128, H/4, W/4]
        x4 = self.down3(x3)     # [B, 128, H/8, W/8]

        # Transformer bottleneck
        x4 = self.pre_transform(x4)     # [B, embed_dim, H/8, W/8]
        x4 = self.bottleneck(x4)
        x4 = self.post_transform(x4)    # [B, 128, H/8, W/8]

        # Decoder 1
        x = self.up1(x4)         # [B, 128, ~H/4, ~W/4]
        diffY = x3.size(2) - x.size(2)
        diffX = x3.size(3) - x.size(3)
        x = F.pad(
            x,
            [diffX // 2, diffX - diffX // 2,
             diffY // 2, diffY - diffY // 2]
        )
        x = torch.cat([x, x3], dim=1)   # [B, 256, H/4, W/4]
        x = self.conv_up1(x)            # [B, 128, H/4, W/4]

        # Decoder 2
        x = self.up2(x)         # [B, 64, ~H/2, ~W/2]
        diffY = x2.size(2) - x.size(2)
        diffX = x2.size(3) - x.size(3)
        x = F.pad(
            x,
            [diffX // 2, diffX - diffX // 2,
             diffY // 2, diffY - diffY // 2]
        )
        x = torch.cat([x, x2], dim=1)   # [B, 128, H/2, W/2]
        x = self.conv_up2(x)            # [B, 64, H/2, W/2]

        # Decoder 3
        x = self.up3(x)         # [B, 32, ~H, ~W]
        diffY = x1.size(2) - x.size(2)
        diffX = x1.size(3) - x.size(3)
        x = F.pad(
            x,
            [diffX // 2, diffX - diffX // 2,
             diffY // 2, diffY - diffY // 2]
        )
        x = torch.cat([x, x1], dim=1)   # [B, 64, H, W]
        x = self.conv_up3(x)            # [B, 32, H, W]

        return self.outc(x)             # [B, 1, H, W]