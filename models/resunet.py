import torch
import torch.nn as nn


class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)


class ResUNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResUNet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(64, 128, stride=2, padding=1)
        self.residual_conv_2 = ResidualConv(128, 256, stride=2, padding=1)

        self.bridge = ResidualConv(256, 512, stride=2, padding=1)

        self.upsample_1 = Upsample(512, 512, kernel=2, stride=2)
        self.up_residual_conv1 = ResidualConv(512 + 256, 256, stride=1, padding=1)

        self.upsample_2 = Upsample(256, 256, kernel=2, stride=2)
        self.up_residual_conv2 = ResidualConv(256 + 128, 128, stride=1, padding=1)

        self.upsample_3 = Upsample(128, 128, kernel=2, stride=2)
        self.up_residual_conv3 = ResidualConv(128 + 64, 64, stride=1, padding=1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(64, out_ch, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        # Bridge
        x4 = self.bridge(x3)
        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        output = self.output_layer(x10)
        output[output<0.5] = 0
        output[output>=0.5] = 1

        return output