import torch
import torch.nn as nn

from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class MyFCNHead(BaseDecodeHead):
    def __init__(self, **kargs):
        super().__init__(**kargs)
        # self.linear = nn.Conv2d(in_channels=self.in_channels,
        #                         out_channels=self.channels,
        #                         kernel_size=kernel_size,
        #                         padding=kernel_size // 2)

    def forward(self, inputs):
        # out = self.linear(inputs[self.in_index])
        # BaseDecodeHead实现好的分类函数，例如可以用来将channels:64->2
        out = self.cls_seg(inputs[self.in_index])
        return out
