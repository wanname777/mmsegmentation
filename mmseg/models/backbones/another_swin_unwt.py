from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import einops
from .swin import SwinTransformer, SwinBlock, SwinBlockSequence

from ..builder import BACKBONES
from mmcv.runner import BaseModule


class PatchExpand(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.expand = nn.Linear(in_features=channels, out_features=2 * channels)
        self.norm = nn.LayerNorm(channels // 2)

    def forward(self, x, hw_shape):
        B, L, C = x.shape
        H, W = hw_shape

        x = x.view(B, H, W, C)
        x = self.expand(x)
        # B,H,W,2C->B,H,W,C//2
        x = einops.rearrange(x, 'b h w (p1 p2 c)->b (h p1) (w p2) c',
                             p1=2, p2=2, c=C // 2)
        x = x.view(B, -1, C // 2)
        x = self.norm(x)
        return x, (H * 2, H * 2)


class FinalExpand(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.expand = nn.Linear(in_features=channels,
                                out_features=16 * channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, L, C = x.shape
        # todo:这里直接开根不太合适？
        H = int(np.sqrt(L))
        W = H

        x = x.view(B, H, W, C)
        x = self.expand(x)
        # B,H,W,2C->B,H,W,C//2
        x = einops.rearrange(x, 'b h w (p1 p2 c)->b (h p1) (w p2) c',
                             p1=4, p2=4, c=C)
        x = x.view(B, -1, C)
        x = self.norm(x)
        return x


@BACKBONES.register_module()
class AnotherSwinUnet(BaseModule):
    def __init__(self, img_size=(224, 224)):
        super().__init__()
        self.down = SwinTransformer(depths=(2, 2, 2, 2))
        self.expand0 = PatchExpand(channels=768)
        self.swin_up = nn.ModuleList()
        self.swin_up.append(SwinBlockSequence(embed_dims=384, num_heads=12,
                                              feedforward_channels=12, depth=2,
                                              downsample=PatchExpand(
                                                  channels=384)))
        self.swin_up.append(SwinBlockSequence(embed_dims=192,
                                              num_heads=6,
                                              feedforward_channels=12,
                                              depth=2,
                                              downsample=PatchExpand(
                                                  channels=192)))
        self.swin_up.append(SwinBlockSequence(
            embed_dims=96, num_heads=3, feedforward_channels=12, depth=2,
            downsample=None))

        self.conv = nn.ModuleList([
            nn.Linear(768, 384),
            nn.Linear(384, 192),
            nn.Linear(192, 96)])
        self.final = FinalExpand(channels=96)

    def forward(self, x):
        B,H,W,C=x.shape
        encode_outputs = self.down(x)

        for i, eo in enumerate(encode_outputs):
            # print(eo.shape)
            B, C, H, W = eo.shape
            encode_outputs[i] = eo.view(B, H * W, C)
        out = encode_outputs.pop(-1)
        # print(out.shape)
        hw_shape = (7, 7)
        out, hw_shape = self.expand0(out, hw_shape)
        for i, encode_output in enumerate(reversed(encode_outputs)):
            out = torch.cat((out, encode_output), dim=-1)
            out = self.conv[i](out)
            out, hw_shape, temp1, temp2 = self.swin_up[i](out, hw_shape)

        out = self.final(out)
        # print(out.shape)
        return [out.reshape(B, 96, 224, 224)]


if __name__ == '__main__':
    x = torch.randn((4, 3, 224, 224))
    net = AnotherSwinUnet()
    out = net(x)
    print(out.shape)
