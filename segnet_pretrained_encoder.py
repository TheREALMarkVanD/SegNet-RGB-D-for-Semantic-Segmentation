import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, vgg_layers):
        super(EncoderBlock, self).__init__()
        self.vgg_layers = vgg_layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x):
        for layer in self.vgg_layers:
            x = layer(x)
        x, indices = self.pool(x)
        return x, indices

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super(DecoderBlock, self).__init__()
        layers = []
        for _ in range(num_layers - 1):
            layers.append(ConvBNReLU(in_channels, in_channels))
        # Adjust the last layer to have the appropriate number of input and output channels
        layers.append(ConvBNReLU(in_channels, out_channels))
        self.decoder = nn.Sequential(*layers)
        self.last_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, indices, is_last_block=False):
        x = F.max_unpool2d(x, indices, kernel_size=2, stride=2)
        if is_last_block:
            for layer in self.decoder[:-1]:
                x = layer(x)
            x = self.last_conv(x)
        else:
            x = self.decoder(x)
        return x

class SegNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SegNet, self).__init__()
        # Load pretrained VGG16 model
        vgg16 = models.vgg16(pretrained=True).features

        # Encoder
        self.enc1 = EncoderBlock(vgg16[:5])
        self.enc2 = EncoderBlock(vgg16[5:10])
        self.enc3 = EncoderBlock(vgg16[10:17])
        self.enc4 = EncoderBlock(vgg16[17:24])
        self.enc5 = EncoderBlock(vgg16[24:31])

        # Decoder
        self.dec5 = DecoderBlock(512, 256, 3)
        self.dec4 = DecoderBlock(256, 128, 3)
        self.dec3 = DecoderBlock(128, 64, 3)
        self.dec2 = DecoderBlock(64, 64, 2)
        self.dec1 = DecoderBlock(64, num_classes, 2)

    def forward(self, x):
        # Encoder
        x, indices1 = self.enc1(x)
        x, indices2 = self.enc2(x)
        x, indices3 = self.enc3(x)
        x, indices4 = self.enc4(x)
        x, indices5 = self.enc5(x)

        # Decoder
        x = self.dec5(x, indices5)
        x = self.dec4(x, indices4)
        x = self.dec3(x, indices3)
        x = self.dec2(x, indices2)
        x = self.dec1(x, indices1, is_last_block=True)

        return x
