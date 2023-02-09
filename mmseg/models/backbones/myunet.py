import torch
import torch.nn as nn
from ..builder import BACKBONES
from mmcv.runner import BaseModule


class DoubleConv(nn.Module):
    """两次卷积"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.first = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU()
        self.second = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU()

    def forward(self, x):
        x = self.first(x)
        x=self.bn1(x)
        x = self.act1(x)
        x = self.second(x)
        x=self.bn2(x)
        x = self.act2(x)
        return x


class DownSample(nn.Module):
    def __init__(self):
        super(DownSample, self).__init__()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(x)


class UpSample(nn.Module):
    """用转置卷积上采样"""

    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        # upsample让长和宽*2
        self.up=nn.Upsample(scale_factor=2)
        # conv让通道减半，方便后续与前面合并
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x=self.up(x)

        x=self.conv(x)
        x=self.bn(x)
        x=self.act(x)
        return x


@BACKBONES.register_module()
class MyUnet(BaseModule):
    def __init__(self, in_channels):
        super(MyUnet, self).__init__()

        self.down_conv = nn.ModuleList([DoubleConv(i, o) for i, o in [(in_channels, 64),
                                                                      (64, 128),
                                                                      (128, 256),
                                                                      (256, 512)]])
        # 只是为了好看，实际上maxpooling一个就行
        self.down_sample = nn.ModuleList([DownSample() for _ in range(4)])
        self.mid_conv = DoubleConv(512, 1024)
        self.up_sample = nn.ModuleList([UpSample(i, o) for i, o in [(1024, 512),
                                                                    (512, 256),
                                                                    (256, 128),
                                                                    (128, 64)]])
        self.up_conv = nn.ModuleList([DoubleConv(i, o) for i, o in [(1024, 512),
                                                                    (512, 256),
                                                                    (256, 128),
                                                                    (128, 64)]])

    def forward(self, x: torch.Tensor):
        # 方便跟decode合并
        encode_out = []
        # 4次下采样
        for i in range(4):
            x = self.down_conv[i](x)
            encode_out.append(x)
            x = self.down_sample[i](x)
        # channels=512
        x = self.mid_conv(x)
        decode_out = []
        # reverse是为了channels匹配
        for i, encode in enumerate(reversed(encode_out)):
            x = self.up_sample[i](x)
            # print(encode.shape,x.shape)
            x = torch.cat((encode, x),dim=1)
            x = self.up_conv[i](x)
            decode_out.append(x)
        return decode_out


if __name__ == '__main__':
    model = MyUnet(in_channels=3)
    x = torch.randn((5, 3, 224, 224))
    y = model(x)
    print(y[-1].shape)
