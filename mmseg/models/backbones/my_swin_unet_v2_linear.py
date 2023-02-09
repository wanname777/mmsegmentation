import einops

from .swin_transformer_v2 import SwinTransformerV2, BasicLayer
import torch.nn as nn
import torch
import numpy as np
from collections import OrderedDict
from einops import rearrange
import copy
from ..builder import BACKBONES


def to_img_shape(x):
    B, L, C = x.shape
    H = int(np.sqrt(L))
    W = H
    return x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()


def to_swin_shape(x):
    B, C, H, W = x.shape
    return x.view(B, C, H * W).permute(0, 2, 1).contiguous()


def change_shape(x):
    # 输入B L C 输出 B 4*L C//2，即长宽加倍，通道减半
    B, L, C = x.shape
    H = int(np.sqrt(L))
    W = H
    x = x.view(B, H, W, C)
    # 这里也是硬变的，从batch*3136*96变成了batch*224*224*96
    x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c',
                  p1=2, p2=2,
                  c=C // 4)
    x = x.view(B, -1, C // 4)
    return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


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


class PatchExpand(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.expand = nn.Linear(dim, 2 * dim)
        self.norm = norm_layer(dim // 2)

    def forward(self, x):
        """
        x: B, L, C
        """
        x = self.expand(x)
        x = change_shape(x)
        x = self.norm(x)
        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.expand1 = PatchExpand(dim=dim)
        self.expand2 = PatchExpand(dim=dim // 2)

    def forward(self, x):
        """
        x: B, L, C
        """
        x = self.expand1(x)
        x = self.expand2(x)
        return x


class PatchUp(nn.Module):
    def __init__(self, dim):
        super(PatchUp, self).__init__()
        # dim表示768
        self.dim = dim
        # seq1
        self.up1 = nn.Upsample(scale_factor=2)
        self.conv1 = ConvBlock(in_channels=dim, out_channels=dim // 2)
        self.conv2 = ConvBlock(in_channels=dim, out_channels=dim // 2)
        # 空洞卷积的替换
        # self.up1 = nn.ConvTranspose2d(in_channels=dim, out_channels=dim // 2,
        #                               kernel_size=4, stride=2, padding=1)
        # self.conv2 = ConvBlock(in_channels=dim, out_channels=dim // 2,
        #                        kernel_size=1, padding=0)

        # seq2
        self.linear1 = nn.Linear(in_features=dim, out_features=dim * 2)
        self.norm1 = nn.LayerNorm(dim // 2)
        self.linear2 = nn.Linear(in_features=dim, out_features=dim // 2)
        self.norm2 = nn.LayerNorm(dim // 2)

        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x, encoder_output):
        # x和encoder_output都是B L C

        x1 = x
        x1 = to_img_shape(x1)
        x1 = self.up1(x1)
        x1 = self.conv1(x1)
        temp_encoder = to_img_shape(encoder_output)
        x1 = torch.cat([temp_encoder, x1], dim=1)
        x1 = self.conv2(x1)
        x1 = to_swin_shape(x1)

        x2 = x
        x2 = self.linear1(x2)
        x2 = change_shape(x2)
        x2 = torch.cat([encoder_output, x2], dim=-1)
        x2 = self.linear2(x2)
        x2 = self.norm2(x2)

        return self.norm(x1 + x2)


class BasicLayerUp(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None,
                 use_checkpoint=False,
                 pretrained_window_size=0):
        super().__init__()

        # swin-unet作者的放大
        self.up = PatchExpand(dim=2 * dim)
        self.linear = nn.Linear(in_features=2 * dim, out_features=dim)
        self.norm = norm_layer(dim)

        self.layers = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=False,
            pretrained_window_size=0)

    def forward(self, input, encoder_output):
        # 作者的流程
        output = self.up(input)
        output = torch.cat([encoder_output, output], dim=-1)
        output = self.norm(self.linear(output))

        # 注释要改
        output_shortcut, output = self.layers(output)

        return output


class MultiDecoder(nn.Module):
    def __init__(self, max_dim, min_dim):
        super(MultiDecoder, self).__init__()
        self.depths = int(np.log2(max_dim / min_dim))
        self.decoders = nn.ModuleList([nn.Sequential() for i in range(self.depths)])

        temp_dim = max_dim
        for i in range(self.depths):
            for j in range(self.depths):
                if j > i:
                    break
                self.decoders[j].add_module(str(i), PatchExpand(dim=temp_dim))
            temp_dim //= 2

    def forward(self, intputs):
        outputs = []
        for i in range(self.depths):
            output = self.decoders[i](intputs[i])
            outputs.append(output)
        outputs.append(intputs[-1])

        output = torch.cat(outputs, dim=-1)
        return output


@BACKBONES.register_module()
class MySwinUnetV2Linear(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 pretrained_window_sizes=[0, 0, 0, 0],
                 pretrain_path=None,
                 **kwargs):
        super().__init__()
        # encoders和decoders采用不同的加载预训练权重方式
        self.encoders = SwinTransformerV2(img_size=img_size,
                                          window_size=window_size,
                                          drop_path_rate=drop_path_rate,
                                          ape=ape)

        temp_size = img_size // patch_size
        temp_dim = embed_dim

        self.decoders = nn.ModuleList()
        for i in range(len(depths) - 1):
            # print(temp_size, temp_dim)
            layer_up = BasicLayerUp(
                dim=int(temp_dim),
                input_resolution=(int(temp_size), int(temp_size)),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate,
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=False,
                pretrained_window_size=0
            )
            self.decoders.append(layer_up)

            temp_size /= 2
            temp_dim *= 2

        # 多尺度信息
        self.multi_decoder = MultiDecoder(max_dim=temp_dim, min_dim=embed_dim)

        # 作者的
        self.final4x = FinalPatchExpand_X4(dim=4 * embed_dim)
        if pretrain_path is not None:
            self.load_pretrain_weight(pretrain_path)

    def forward(self, inputs):
        encoder_outputs = self.encoders(inputs)

        outputs = []
        outputs.append(encoder_outputs[-1])
        output = encoder_outputs.pop(-1)

        # range=2,1,0
        for i in range(len(encoder_outputs) - 1, -1, -1):
            output = self.decoders[i](output, encoder_outputs[i])
            outputs.append(output)

        output = self.multi_decoder(outputs)
        # 作者的
        output = self.final4x(output)
        output = to_img_shape(output)
        outputs.append(output)
        return outputs

    def load_pretrain_weight(self, pretrain_path):
        encoders_weight = torch.load(pretrain_path)['model']
        decoders_weight = OrderedDict()

        for k, v in encoders_weight.items():
            # print(k)
            if 'layers' in k:
                ks = k.split('.')
                ks[0], ks[1] = ks[1], ks[0]
                new_k = '.'.join(ks)
                # print(k,new_k)
                decoders_weight[new_k] = v
        self.encoders.load_state_dict(encoders_weight, strict=False)
        msg = self.decoders.load_state_dict(decoders_weight, strict=False)
        print(msg)


if __name__ == '__main__':
    x = torch.randn((4, 3, 224, 224))
    net = MySwinUnetV2(depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24])
    outputs = net(x)
    for output in outputs:
        print(output.shape)
