from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import einops
from mmseg.models.backbones.swin import SwinTransformer,SwinBlock
from mmseg.models.backbones.another_swin_unwt import AnotherSwinUnet
from mmseg.models.backbones.my_swin_unet_v2 import MySwinUnetV2
from mmseg.models.backbones.my_swin_unet_v2_linear import MySwinUnetV2Linear
# from ..builder import BACKBONES
# from mmcv.runner import BaseModule
if __name__ == '__main__':
    # x=torch.randn((4,3,224,224))
    # net=SwinTransformer()
    # x_forward=torch.randn((4,56*56,96))
    # # block=SwinBlock(embed_dims=96,num_heads=3,feedforward_channels=int(4 * 3))
    # # out=block(x_forward,(56,56))
    # # print(out.shape)
    #
    # # net.init_weights()
    # outputs=net(x)
    # for output in outputs:
    #     print(output.shape)

    x = torch.randn((4, 3, 256, 256))
    net = MySwinUnetV2(img_size=256,window_size=8,pretrain_path='../checkpoint/swinv2_tiny_patch4_window8_256.pth')
    out = net(x)
    for o in out:
        print(o.shape)

    # x=torch.randn((1,3,16,16))
    # net=nn.ConvTranspose2d(in_channels=3,out_channels=10,kernel_size=4,stride=2,padding=1)
    # out=net(x)
    # print(out.shape)