from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import einops
from mmseg.models.builder import BACKBONES
from mmcv.runner import BaseModule


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=0, dropout=0.5):
        super().__init__()
        # 常规信息
        self.in_features = in_features
        if hidden_features == 0:
            self.hidden_features = in_features
        else:
            self.hidden_features = hidden_features
        self.out_features = in_features

        # todo:考虑这里的sequential对于mmseg网络初始化是否有影响
        self.seq = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_features, self.out_features),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.seq(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, head_nums, channels, window_size, dropout=0.5):
        super().__init__()
        # 常规信息
        self.head_nums = head_nums
        self.channels = channels
        self.window_size = window_size

        # attention相关
        # 3指的是qkv一共3个
        self.qkv = nn.Linear(channels, 3 * channels)
        # 每个头的根下d
        self.scale = (channels // head_nums) ** -0.5
        # todo:randn是默认不需要梯度的，但是放在这里是否合适？同时，这里没有用论文作者的相对位置bias
        self.bias = nn.Parameter(
            torch.randn((head_nums, (window_size ** 2), (window_size ** 2))))
        self.attn_drop = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

        # 奇怪的映射层
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # print(f"msa_input:{x.shape}")
        B, L, C = x.shape
        # 把x最后的3*C分成3*head_nums*(C/head_nums)
        # 然后总变量再变成3,B,head_nums,L,C/head_nums
        qkv = self.qkv(x) \
            .reshape(B, L, 3, self.head_nums, C // self.head_nums) \
            .permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # print(f"q.shape:{q.shape},k.shape:{k.shape},v.shape:{v.shape}")
        # QK^T/根下d
        q = q * self.scale
        # bias利用了广播机制
        attn = q @ k.transpose(-2, -1) + self.bias
        # print(attn.shape)
        # print(f"attn.shape:{attn.shape}")
        # 基于mask做softmax
        if mask is not None:
            # print(f"mask.shape:{mask.shape}")
            W = mask.shape[0]
            attn = attn.view(B // W, W, self.head_nums, L, L) + \
                   mask.unsqueeze(1)
            attn = attn.view(B, self.head_nums, L, L)
        attn = self.softmax(attn)

        # 剩下的与V相乘即可获得attention
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, L, C)

        # 奇怪的线性映射
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinBlock(nn.Module):
    def __init__(self, channels, shift_size=0, window_size=7, head_nums=3,
                 dropout=0.5):
        super().__init__()
        # B L C等常规信息
        self.channels = channels
        self.shift_size = shift_size
        self.window_size = window_size
        # head_nums最大能取多少取决于channels // head_nums，只要小于channels都能行，但一般不取太大
        self.head_nums = head_nums
        # 网络信息
        self.attention = MultiHeadAttention(head_nums=self.head_nums,
                                            channels=self.channels,
                                            window_size=self.window_size,
                                            dropout=dropout)
        self.norm1 = nn.LayerNorm(self.channels)
        self.norm2 = nn.LayerNorm(self.channels)
        self.mlp = MLP(self.channels, dropout=dropout)

    def forward(self, x):
        B, L, C = x.shape
        # todo:这里直接开根不太合适？
        H = int(np.sqrt(L))
        W = H

        # 注意这句是指针
        x_shortcut = x
        # 下面这个x变成了正则化之后的数值
        x = self.norm1(x)

        x = x.view(B, H, W, C)
        # todo:循环位移，为什么默认为3位？
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size),
                           dims=(1, 2))
        # x窗口化作为输入
        # 将B,H,W,C转成-1,window_size,window_size,C的形式做注意力机制
        # 而这里强行用了多次view+reshape是因为我们希望将window_size*window_size的元素视为“一句话”
        x = x.view(B,
                   H // self.window_size, self.window_size,
                   W // self.window_size, self.window_size,
                   C)
        # 这里要注意view和reshape的区别，reshape=view.contiguous.view
        # 所以reshape从内存空间上改变了tensor的元素，而view只是改变了访问顺序，而并不改变元素本身的存储顺序
        # view size is not compatible with input tensor's size and stride
        # (at least one dimension spans across two contiguous subspaces).
        # Use .reshape(...) instead.
        x = x.permute(0, 1, 3, 2, 4, 5) \
            .reshape(-1, self.window_size, self.window_size, C)
        # 窗口化之后在换成能被attention接受的维度，即B,L,C
        x = x.view(-1, self.window_size * self.window_size, C)

        # 获取mask
        if self.shift_size > 0:
            # # B, L, C = self.input_shape
            # # L==H*W
            # H = int(np.sqrt(self.input_shape[1]))
            # W = H
            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H, W, 1), device=x.device)  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = img_mask.view(1,
                                         H // self.window_size,
                                         self.window_size,
                                         W // self.window_size,
                                         self.window_size,
                                         1)
            # 下面两步可以合并但是似乎没必要
            mask_windows = mask_windows.permute(0, 1, 3, 2, 4, 5) \
                .reshape(-1, self.window_size, self.window_size, 1)
            # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1,
                                             self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)) \
                .masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
            # 基于mask做attention
        x = self.attention(x, mask=attn_mask)

        # 反attention
        x = x.view(-1, self.window_size, self.window_size, C)
        # x反窗口化
        x = x.view(B,
                   H // self.window_size, W // self.window_size,
                   self.window_size, self.window_size,
                   C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)
        # 循环位移归位
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size),
                           dims=(1, 2))
        x = x.view(B, H * W, C)

        # 跳连接1
        x = x + x_shortcut
        # 用shortcut暂时存一下，方便下次跳连接
        x_shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        # 跳连接2
        x = x + x_shortcut

        return x


class PatchMerging(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.LayerNorm(4 * channels)
        self.reduction = nn.Linear(in_features=4 * channels,
                                   out_features=2 * channels)

    def forward(self, x):
        B, L, C = x.shape
        # todo:这里直接开根不太合适？
        H = int(np.sqrt(L))
        W = H

        x = x.view(B, H, W, C)
        # 这个其实是以2*2=4个小块为基准，分别取每个小块的左上、右上、左下、右下的元素，
        # 与其他小块对应位置的元素进行拼接，形成x0123，这样每个x都是batch*112*112*96
        # 96*4=384
        # 拼接形成batch*112*112*384
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        # print(x0.shape,x1.shape,x2.shape,x3.shape)
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchExpand(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.expand = nn.Linear(in_features=channels, out_features=2 * channels)
        self.norm = nn.LayerNorm(channels // 2)

    def forward(self, x,hw_shape):
        B, L, C = x.shape
        # todo:这里直接开根不太合适？
        H = int(np.sqrt(L))
        W = H

        x = x.view(B, H, W, C)
        x = self.expand(x)
        # B,H,W,2C->B,H,W,C//2
        x = einops.rearrange(x, 'b h w (p1 p2 c)->b (h p1) (w p2) c',
                             p1=2, p2=2, c=C // 2)
        x = x.view(B, -1, C // 2)
        x = self.norm(x)
        return x


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


class PatchEmbed(nn.Module):
    def __init__(self, embedding_dim=96):
        super().__init__()
        self.embedding_dim = embedding_dim
        # todo:kernel_size不能改变
        self.embedding = nn.Conv2d(in_channels=3, out_channels=embedding_dim,
                                   kernel_size=4, stride=4)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = self.embedding(x)
        B, C, H, W = x.shape
        x = x.view(B, -1, C)
        x = self.norm(x)

        return x


class DownSample(nn.Module):
    def __init__(self, channels, shift_size, window_size=7, head_nums=3,
                 is_merge=True, dropout=0.5):
        super().__init__()
        self.channels = channels
        self.swin1 = SwinBlock(channels=channels,
                               shift_size=0,
                               window_size=window_size,
                               head_nums=head_nums,
                               dropout=dropout)
        self.swin2 = SwinBlock(channels=channels,
                               shift_size=shift_size,
                               window_size=window_size,
                               head_nums=head_nums,
                               dropout=dropout)
        self.is_merge = is_merge
        if is_merge:
            self.merge = PatchMerging(channels)

    def forward(self, x):
        x_shortcut = x
        x = self.swin1(x)
        x = self.swin2(x)
        if self.is_merge:
            x = self.merge(x)
        return x_shortcut, x


class UpSample(nn.Module):
    def __init__(self, channels, shift_size, window_size=7, head_nums=3,
                 is_expand=True, dropout=0.5):
        super().__init__()

        self.channels = channels
        self.swin1 = SwinBlock(channels=channels,
                               shift_size=0,
                               window_size=window_size,
                               head_nums=head_nums,
                               dropout=dropout)
        self.swin2 = SwinBlock(channels=channels,
                               shift_size=shift_size,
                               window_size=window_size,
                               head_nums=head_nums,
                               dropout=dropout)
        self.is_expand = is_expand
        if is_expand:
            self.expand = PatchExpand(channels)

    def forward(self, x):
        x = self.swin1(x)
        x = self.swin2(x)
        if self.is_expand:
            x = self.expand(x)
        return x


@BACKBONES.register_module(force=True)
class MySwinUnet(BaseModule):
    def __init__(self, img_size, depth=3, embedding_dim=96, shift_size=3,
                 window_size=7, dropout=0., ape=False):
        super().__init__()
        self.img_size = img_size
        self.embedding_dim = embedding_dim
        self.depth = depth
        self.shift_size = shift_size
        self.window_size = window_size
        self.dropout = dropout
        # embedding只支持RGB通道
        self.patch_embedding = PatchEmbed(embedding_dim=embedding_dim)
        self.ape = ape

        # todo:加入绝对位置信息
        if ape:
            pass

        self.encoders = nn.ModuleList()
        H, W = img_size
        H, W, C = H // 4, W // 4, embedding_dim
        L = H * W
        temp_head_nums = 3

        for i in range(depth):
            self.encoders.append(DownSample(channels=C,
                                            shift_size=shift_size,
                                            window_size=window_size,
                                            head_nums=temp_head_nums,
                                            is_merge=True,
                                            dropout=dropout))
            L = L // 4
            C = C * 2
            temp_head_nums = temp_head_nums * 2

        self.norm1 = nn.LayerNorm(C)

        decoders = []
        linears = []
        for i in range(depth + 1):
            decoders.append(UpSample(channels=C,
                                     shift_size=shift_size,
                                     window_size=window_size,
                                     head_nums=temp_head_nums,
                                     is_expand=True if i != self.depth else False,
                                     dropout=dropout))
            linears.append(nn.Linear(in_features=C, out_features=C // 2))
            L = L * 4
            C = C // 2
            temp_head_nums = temp_head_nums // 2
        self.neck = decoders.pop(0)
        linears.pop()
        self.decoders = nn.ModuleList(decoders)
        self.linears = nn.ModuleList(linears)

        L = L // 4
        C = C * 2

        self.norm2 = nn.LayerNorm(C)

        self.final_expand = FinalExpand(channels=C)



    def forward(self, x):
        # print(f"input_shape:{x.shape}")
        B, C, H, W = x.shape
        x = self.patch_embedding(x)
        encoder_output = []
        for encoder in self.encoders:
            x_shortcut, x = encoder(x)
            encoder_output.append(x_shortcut)
        x = self.neck(x)

        for i in range(self.depth):
            x = torch.cat([encoder_output[self.depth - 1 - i], x], dim=-1)
            x = self.linears[i](x)
            x = self.decoders[i](x)

        # x=self.norm2(x)
        x = self.final_expand(x)
        x = x.view(B, H, W, -1)
        x = x.permute(0, 3, 1, 2)
        return [x]


if __name__ == '__main__':
    # H和W应该能被window_size整除
    x = torch.randn((2, 3, 224, 224))
    swin_unet = MySwinUnet(img_size=(224, 224), embedding_dim=96, window_size=7,
                           dropout=0.)
    # print(swin_unet)
    x = swin_unet(x)
    print(x[-1].shape)
