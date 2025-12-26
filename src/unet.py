import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tf

class DoubleConvolutional(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    
class Encoder(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.conv1 = DoubleConvolutional(in_channels, 64) 
        self.conv2 = DoubleConvolutional(64, 128)
        self.conv3 = DoubleConvolutional(128, 256)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x1 = self.conv1(x)              
        x2 = self.conv2(self.pool(x1))  
        x3 = self.conv3(self.pool(x2)) 
        x4 = self.pool(x3)              
        return x1, x2, x3, x4
    

class Bottleneck(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = DoubleConvolutional(256, 512)

    def forward(self, x):
        return self.conv(x)



class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConvolutional(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConvolutional(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConvolutional(128, 64)

    def forward(self, x, x3, x2, x1):
        x = self.up1(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv3(x)

        return x

class UNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.encoder = Encoder()
        self.bottleneck = Bottleneck()
        self.decoder = Decoder()
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        x = self.bottleneck(x4)
        x = self.decoder(x, x3, x2, x1)
        return self.final(x)

    
class RoverLanding(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.encoder = Encoder()
        self.bottleneck = Bottleneck()
        self.decoder = Decoder()


        self.segmentation_head = nn.Conv2d(64, n_classes, kernel_size=1)


        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.safety_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(32, 1) 
        )

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        x_bottleneck = self.bottleneck(x4)
        feature_map = self.decoder(x_bottleneck, x3, x2, x1) 


        surface_map = self.segmentation_head(feature_map)


        pooled = self.global_pool(feature_map) 
        pooled = pooled.view(pooled.size(0), -1)
        safety_score = self.safety_head(pooled)

        return surface_map, safety_score