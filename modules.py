# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class DoubleConv(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""
#
#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super(DoubleConv, self).__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#
#         self.double_conv = nn.Sequential(
#             nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
#             nn.BatchNorm3d(mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm3d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         return self.double_conv(x)
#
#
# class Down(nn.Module):
#     """Downscaling with maxpool then double conv"""
#
#     def __init__(self, in_channels, out_channels):
#         super(Down, self).__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool3d(2),
#             DoubleConv(in_channels, out_channels)
#         )
#
#     def forward(self, x):
#         return self.maxpool_conv(x)
#
#
# class Up(nn.Module):
#     """Upscaling then double conv"""
#
#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super(Up, self).__init__()
#
#         # if bilinear, use the normal convolutions to reduce the number of channels
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#             self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
#         else:
#             self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
#             self.conv = DoubleConv(in_channels, out_channels)
#
#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         # input is CHW
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]
#
#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)
#
#
# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()
#         self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
#
#     def forward(self, x):
#         return self.conv(x)
#
#
# """ Full assembly of the parts to form the complete network """
#
#
# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=True):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
#
#         self.inconv = DoubleConv(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         factor = 2 if bilinear else 1
#         self.down4 = Down(512, 1024 // factor)
#         self.up1 = Up(1024, 512 // factor, bilinear)
#         self.up2 = Up(512, 256 // factor, bilinear)
#         self.up3 = Up(256, 128 // factor, bilinear)
#         self.up4 = Up(128, 64, bilinear)
#         self.outconv = OutConv(64, n_classes)
#
#     def forward(self, x):
#         x1 = self.inconv(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         x = self.outconv(x)
#         return x
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=2, training=True):
        super(UNet, self).__init__()
        self.training = training
        self.encoder1 = nn.Conv3d(in_channel, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2 = nn.Conv3d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3 = nn.Conv3d(64, 128, 3, stride=1, padding=1)
        self.encoder4 = nn.Conv3d(128, 256, 3, stride=1, padding=1)
        self.encoder5 = nn.Conv3d(256, 512, 3, stride=1, padding=1)

        self.decoder1 = nn.Conv3d(512, 256, 3, stride=1, padding=1)  # b, 16, 5, 5
        self.decoder2 = nn.Conv3d(256, 128, 3, stride=1, padding=1)  # b, 8, 15, 1
        self.decoder3 = nn.Conv3d(128, 64, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder4 = nn.Conv3d(64, 32, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv3d(32, 2, 3, stride=1, padding=1)

        self.map4 = nn.Sequential(
            nn.Conv3d(2, out_channel, 1, 1),
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear'),
            nn.Softmax(dim=1)
        )

        # 128*128 尺度下的映射
        self.map3 = nn.Sequential(
            nn.Conv3d(64, out_channel, 1, 1),
            nn.Upsample(scale_factor=(4, 8, 8), mode='trilinear'),
            nn.Softmax(dim=1)
        )

        # 64*64 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Conv3d(128, out_channel, 1, 1),
            nn.Upsample(scale_factor=(8, 16, 16), mode='trilinear'),
            nn.Softmax(dim=1)
        )

        # 32*32 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Conv3d(256, out_channel, 1, 1),
            nn.Upsample(scale_factor=(16, 32, 32), mode='trilinear'),
            nn.Softmax(dim=1)
        )

    def forward(self, x):

        out = F.relu(F.max_pool3d(self.encoder1(x), 2, 2))
        t1 = out
        out = F.relu(F.max_pool3d(self.encoder2(out), 2, 2))
        t2 = out
        out = F.relu(F.max_pool3d(self.encoder3(out), 2, 2))
        t3 = out
        out = F.relu(F.max_pool3d(self.encoder4(out), 2, 2))
        t4 = out
        out = F.relu(F.max_pool3d(self.encoder5(out), 2, 2))

        # t2 = out
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=(2, 2, 2), mode='trilinear'))
        # print(out.shape,t4.shape)
        out = torch.add(F.pad(out, [0, 0, 0, 0, 0, 1]), t4)
        output1 = self.map1(out)
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2, 2), mode='trilinear'))
        out = torch.add(out, t3)
        output2 = self.map2(out)
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2, 2), mode='trilinear'))
        out = torch.add(out, t2)
        output3 = self.map3(out)
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2, 2), mode='trilinear'))
        out = torch.add(out, t1)

        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2, 2), mode='trilinear'))
        output4 = self.map4(out)
        # print(out.shape)
        # print(output1.shape,output2.shape,output3.shape,output4.shape)
        if self.training is True:
            return output1, output2, output3, output4
        else:
            return output4