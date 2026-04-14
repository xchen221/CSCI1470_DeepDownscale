# import torch.nn as nn


# class SimpleCNN(nn.Module):
#     def __init__(self, in_channels=2, hidden_channels=32, out_channels=1):
#         super().__init__()

#         self.net = nn.Sequential(
#             nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(hidden_channels, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(64, hidden_channels, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
#         )

#     def forward(self, x):
#         return self.net(x)
    
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNetDownscale(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super().__init__()
        # Encoder (Downsampling)
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))

        # Decoder (Upsampling)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(128, 64) # 64 from up + 64 from skip
        
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(64, 32)  # 32 from up + 32 from skip
        
        self.outc = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # 1. Encoder Path
        x1 = self.inc(x)           # Size: (batch, 32, H, W)
        x2 = self.down1(x1)        # Size: (batch, 64, H/2, W/2)
        x3 = self.down2(x2)        # Size: (batch, 128, H/4, W/4)
        
        # 2. First Upsample + Skip Connection
        x = self.up1(x3)
        
        # Padding to fix dimension mismatches (e.g., 154 vs 155)
        diffY = x2.size()[2] - x.size()[2]
        diffX = x2.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x, x2], dim=1) 
        x = self.conv_up1(x)
        
        # 3. Second Upsample + Skip Connection
        x = self.up2(x)
        
        diffY = x1.size()[2] - x.size()[2]
        diffX = x1.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x, x1], dim=1) 
        x = self.conv_up2(x)
        
        # 4. Final Output
        return self.outc(x)