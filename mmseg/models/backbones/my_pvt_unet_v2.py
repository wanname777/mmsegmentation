from mmseg.models.backbones.pvt_v2 import pvt_v2_b2,Block
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger
from mmcv.runner import load_checkpoint
import math
def to_img_shape(x):
    B, L, C = x.shape
    H = int(np.sqrt(L))
    W = H
    return x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()


def to_swin_shape(x):
    B, C, H, W = x.shape
    return x.view(B, C, H * W).permute(0, 2, 1).contiguous()
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1,act=nn.GELU):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = act()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class UpBlock(nn.Module):
    def __init__(self,in_channels,out_channels,mode='bilinear',scale_factor=2):
        super(UpBlock, self).__init__()
        self.up=nn.Upsample(scale_factor=scale_factor,mode=mode)
        self.conv=ConvBlock(in_channels=in_channels,out_channels=out_channels)
    def forward(self,input):
        output=self.up(input)
        output=self.conv(output)
        return output
class BasicLayerUp(nn.Module):
    def __init__(self,in_channels,out_channels,
                 depth,embed_dim,num_head,mlp_ratio,
                 qkv_bias,qk_scale,
                 attn_drop_rate,drop_rate,dpr,
                 norm_layer,sr_ratio,linear):
        super(BasicLayerUp, self).__init__()

        self.up=UpBlock(in_channels=in_channels,out_channels=out_channels)
        self.conv=ConvBlock(in_channels=2*out_channels,out_channels=out_channels)

        self.block = nn.ModuleList([Block(
                dim=embed_dim, num_heads=num_head,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer,
                sr_ratio=sr_ratio, linear=linear)
               for i in range(depth)])
    def forward(self,x,encoder_input):
        x=self.up(x)
        x=torch.cat([x,encoder_input],dim=1)
        x=self.conv(x)
        B,C,H,W=x.shape
        x=to_swin_shape(x)
        for blk in self.block:
            x=blk(x,H,W)

        out=to_img_shape(x)
        return out
class Head4x(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.up1 = nn.Upsample(scale_factor=2)
        self.conv1 = ConvBlock(in_channels=in_channels,
                               out_channels=mid_channels)
        self.up2 = nn.Upsample(scale_factor=2)
        self.conv2 = ConvBlock(in_channels=mid_channels,
                               out_channels=out_channels)

    def forward(self, x):
        x = self.up1(x)
        x = self.conv1(x)
        x = self.up2(x)
        x = self.conv2(x)
        return x
@BACKBONES.register_module(force=True)
class MyPvtUnetV2(nn.Module):
    def __init__(self,patch_size=4, embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True,qk_scale=None, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            attn_drop_rate=0.,drop_rate=0.0, drop_path_rate=0.1,linear=False,
                 pretrained=None):
        super(MyPvtUnetV2, self).__init__()
        self.encoder=pvt_v2_b2(pretrained=pretrained)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule
        cur=0
        basic_layer_up=[]
        for i in range(len(embed_dims)-1):
            basic_layer_up.append(BasicLayerUp(in_channels=embed_dims[i+1],out_channels=embed_dims[i],
                                               depth=depths[i],embed_dim=embed_dims[i],
                                               num_head=num_heads[i],mlp_ratio=mlp_ratios[i],
                                               qkv_bias=qkv_bias,qk_scale=qk_scale,
                                               attn_drop_rate=attn_drop_rate,drop_rate=drop_rate,
                                               dpr=dpr[cur:cur+depths[i]],norm_layer=norm_layer,
                                               sr_ratio=sr_ratios[i],linear=linear,
                                               ))
        self.decoder=nn.ModuleList(basic_layer_up)
        print(self.decoder)
        # out为96,就是一个特殊值，为了跟swin对齐，没什么特殊意义
        self.final4x=Head4x(in_channels=embed_dims[0],mid_channels=2*embed_dims[0],out_channels=96)
    def forward(self,x):
        outputs=[]
        encoder_outputs=self.encoder(x)
        x=encoder_outputs.pop(-1)
        for i in range(len(encoder_outputs)-1,-1,-1):
            x=self.decoder[i](x,encoder_outputs[i])
            outputs.append(x)
        out=self.final4x(x)
        outputs.append(out)
        return outputs


if __name__ == '__main__':
    model=MyPvtUnetV2(pretrained=None)
    input=torch.randn((4,3,256,256))
    outputs=model(input)
    for o in outputs:
        print(o.shape)
