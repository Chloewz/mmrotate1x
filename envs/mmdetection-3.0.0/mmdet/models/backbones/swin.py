# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmengine.logging import MMLogger
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import constant_init, trunc_normal_, trunc_normal_init
from mmengine.runner.checkpoint import CheckpointLoader
from mmengine.utils import to_2tuple

from mmdet.registry import MODELS
from ..layers import PatchEmbed, PatchMerging


class WindowMSA(BaseModule):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.

    WindowMSA模块通过基于窗口的多头自注意力机制实现高效的图像特征提取，具有以下特点：
        局部性：只在每个窗口内计算注意力，减少计算量
        相对位置偏移：显式建模窗口内的相对位置关系
        灵活性：支持掩膜矩阵用于处理不规则窗口或边界条件

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
        init_cfg (dict | None, optional): The Config for initialization.
            Default: None.
    """

    def __init__(
        self,
        embed_dims,
        num_heads,
        window_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop_rate=0.0,
        proj_drop_rate=0.0,
        init_cfg=None,
    ):

        super().__init__()
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads   # 每个注意力头的维度
        self.scale = qk_scale or head_embed_dims**-0.5  # 用于归一化q向量
        self.init_cfg = init_cfg

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        # 每个窗口元素的相对位置都有一个偏置项

        # About 2x faster than original impl
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)  # 生成窗口内坐标序列
        # shape: (1, Wh*Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T  # 计算每个相对位置的索引
        # 两者相加时，触发广播机制，结果shape: (Wh*Ww, Wh*Ww)
        rel_position_index = rel_position_index.flip(1).contiguous()    # 翻转顺序，调整索引的访问模式
        # * flip(1): 翻转矩阵的第二维(列)；对于每一行，将列的顺序颠倒
        # 为后续访问relative_position_bias_table提供一个一致的顺序
        self.register_buffer("relative_position_index", rel_position_index)
        # * register_buffer存储为不可训练的缓冲区。即模型训练时不会更新，但是保存模型时，该组参数作为模型不可或缺的一部分被保存

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias) # 将输入变换为q, k, v
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)   # 对注意力权重归一化

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        # * trunc_normal_截断正态分布初始化函数

    def forward(self, x, mask=None):
        """
        Args:

            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        B, N, C = x.shape   # N=Wh*Ww, 窗口内的像素数
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)  # (B, N, 3, #heads, head_dim)
            .permute(2, 0, 3, 1, 4) # (3, B, #heads, N, head_dim)
        )
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale  # 缩放q
        attn = q @ k.transpose(-2, -1)  # shape: (B, #heads, N, N)
        # * @操作是矩阵乘法

        # 将预定义的相对位置偏置标映射到当前窗口的相对位置索引矩阵上
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)   # shape: (Wh*Ww * Wh*Ww, nH), 其中每一行是一个相对位置的偏置值，针对所有注意力头
        ].view( # 恢复到二维形式
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH 
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)   # 调整注意力分布以考虑相对位置关系
        # relative_position_bias shape: (1, nH, Wh*Ww, Wh*Ww)

        if mask is not None:    # 如果提供了掩膜矩阵mask，将其加到attn上，用于调整窗口内的注意力权重
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)   # 归一化

        attn = self.attn_drop(attn) # Dropout

        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # 计算加权结果
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod   # * 转换为静态方法。静态方法不需要类实例作为第一个参数(self)，不需要类本身作为第一个参数(cls)
    # * 静态方法可以在不创建实例的情况下直接通过类名调用
    def double_step_seq(step1, len1, step2, len2):
        """
        用于生成一个二维序列，并最终展平为一维
        """
        seq1 = torch.arange(0, step1 * len1, step1)
        # * torch.arange(start, end, step)生成一个从start到end(不包括)的序列，步长为step
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)
        # * seq1[:, None]将seq1转换为列向量(len1,1); seq2[None,:]将seq2转换为行向量(1,len2)
        # * 相加操作利用了boradcasting机制，得到形状为(len1,len2)的二维矩阵,
        # * 每一行都是seq1中的一个元素分别加上seq2的对应元素
        # * 每一列都是seq2中的一个元素分别加上seq1的对应元素


class ShiftWindowMSA(BaseModule):
    """Shifted Window Multihead Self-Attention Module.

    ShiftWindowMSA是一种增强版的多头自注意力机制，通过滑动窗口实现跨窗口的上下文信息交互
    核心思想：
        窗口划分：在局部范围内计算自注意力
        滑动窗口：通过平移窗口实现跨窗口交互
        掩码机制：避免无效的跨窗口计算

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window.
        shift_size (int, optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Defaults: 0.
        proj_drop_rate (float, optional): Dropout ratio of output.
            Defaults: 0.
        dropout_layer (dict, optional): The dropout_layer used before output.
            Defaults: dict(type='DropPath', drop_prob=0.).
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(
        self,
        embed_dims,
        num_heads,
        window_size,
        shift_size=0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop_rate=0,
        proj_drop_rate=0,
        dropout_layer=dict(type="DropPath", drop_prob=0.0),
        init_cfg=None,
    ):
        super().__init__(init_cfg)

        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            init_cfg=None,
        )

        self.drop = build_dropout(dropout_layer)

    def forward(self, query, hw_shape):
        B, L, C = query.shape   # L: 序列长度
        H, W = hw_shape
        assert L == H * W, "input feature has wrong size"
        query = query.view(B, H, W, C)  # 使其能够表示二维图像

        # pad feature maps to multiples of window size
        # 计算横向和纵向需要填充的像素数，使得图像宽高能够整除窗口大小
        pad_r = (self.window_size - W % self.window_size) % self.window_size    # 横向填充
        # 对结果再取模，确保当无需填充时(余数为0)填充量为0，而不是窗口大小本身
        pad_b = (self.window_size - H % self.window_size) % self.window_size    # 纵向填充
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        # (0, 0, 0, pad_r, 0, pad_b) 前二：通道方向不进行填充；中二：宽度方向右侧填充，左侧不填充；后二：高度方向底部填充，顶部不填充
        H_pad, W_pad = query.shape[1], query.shape[2]

        # cyclic shift 循环平移
        if self.shift_size > 0:
            shifted_query = torch.roll(
                query, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
            # * torch.roll(input, shifts, dims=None)用于对张量元素移位，沿给定维数滚动张量，移动到最后一个位置以外的元素将在第一个位置重新引入
            # * shifts 张量元素移位的位数。如果该参数是一个元组(shifts=(x,y))，则dims必须是一个相同大小的元组(如dims=(a,b))
            # *     相当于再第a维度移x位，在b维度移y位
            # * dims为确定的维度
            # * shifts为正数相当于向下挤牙膏，挤出的牙膏又从顶部塞回牙膏里面

            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            # * 使用slice将图像划分为多个子区域，每个子区域标记一个唯一的编号
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt  # img_mask的值为编号，表示每个区域对应的窗口编号
                    cnt += 1

            mask_windows = self.window_partition(img_mask)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)   # 每个窗口展开为1维，向量化
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            # shape: (#W, W_size^2, W_size^2)，得到窗口内所有位置两两之间的编号差值
            # 如果两个位置的编号相等，表示在同一窗口，掩码值为0，否则非0
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0))
            # 窗口编号不相等，填充-100，在softmax操作中趋近于0，屏蔽跨窗口的注意力
            # 窗口编号相等的位置，填充为0，不影响窗口内注意力权重的计算
            # ! 每个窗口块的掩码用于限定注意力计算的范围，仅保留同一窗口内的注意力
        else:
            shifted_query = query
            attn_mask = None

        # 窗口划分并变形
        query_windows = self.window_partition(shifted_query)    # (nW*B, window_size, window_size, C)
        query_windows = query_windows.view(-1, self.window_size**2, C)  # (nW*B, window_size*window_size, C)

        # W-MSA/SW-MSA, 计算多头注意力
        attn_windows = self.w_msa(query_windows, mask=attn_mask)    # (nW*B, window_size*window_size, C)

        # merge windows, 将窗口恢复为二维
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # 重新拼接成完整的特征图
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad) # B, H_pad, W_pad, C
        # reverse cyclic shift, 反向循环位移
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x

        if pad_r > 0 or pad_b:  # 去除填充区域
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C) # 重塑为输出格式, L=H*W

        x = self.drop(x)
        return x

    def window_reverse(self, windows, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size)) # 计算batch
        x = windows.view(
            B, H // window_size, W // window_size, window_size, window_size, -1
        )
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows


class SwinBlock(BaseModule):
    """ "
    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        window_size (int, optional): The local window scale. Default: 7.
        shift (bool, optional): whether to shift window or not. Default False.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(
        self,
        embed_dims,
        num_heads,
        feedforward_channels,
        window_size=7,
        shift=False,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        act_cfg=dict(type="GELU"),
        norm_cfg=dict(type="LN"),
        with_cp=False,
        init_cfg=None,
    ):

        super(SwinBlock, self).__init__()

        self.init_cfg = init_cfg
        self.with_cp = with_cp

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = ShiftWindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            dropout_layer=dict(type="DropPath", drop_prob=drop_path_rate),
            init_cfg=None,
        )

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=2,
            ffn_drop=drop_rate,
            dropout_layer=dict(type="DropPath", drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=True,
            init_cfg=None,
        )

    def forward(self, x, hw_shape):

        def _inner_forward(x):
            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape)

            x = x + identity

            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)

            return x

        # 选择性地应用gradient checkpointing(梯度检查点)机制的逻辑，目的是在训练过程中节省内存
        if self.with_cp and x.requires_grad:    # 是否启用，启用的话可以通过checkpointing节省内存
            x = cp.checkpoint(_inner_forward, x)
            # * cp.checkpoint(): cp是torch.utils.checkpoint模块的缩写，提供了一种延迟计算某些中间结果的机制
            # * checkpoint()函数会在前向传播时不保存中间激活值，从而节省内存；在反向传播时重新计算中间激活值，因此会增加计算时间
        else:
            x = _inner_forward(x)

        return x


class SwinBlockSequence(BaseModule):
    """Implements one stage in Swin Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        depth (int): The number of blocks in this stage.
        window_size (int, optional): The local window scale. Default: 7.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float | list[float], optional): Stochastic depth
            rate. Default: 0.
        downsample (BaseModule | None, optional): The downsample operation
            module. Default: None.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(
        self,
        embed_dims,
        num_heads,
        feedforward_channels,
        depth,
        window_size=7,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        downsample=None,
        act_cfg=dict(type="GELU"),
        norm_cfg=dict(type="LN"),
        with_cp=False,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]

        self.blocks = ModuleList()
        for i in range(depth):
            block = SwinBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                window_size=window_size,
                shift=False if i % 2 == 0 else True,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None,
            )
            self.blocks.append(block)

        self.downsample = downsample

    def forward(self, x, hw_shape):
        for block in self.blocks:
            x = block(x, hw_shape)

        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape


@MODELS.register_module()
class SwinTransformer(BaseModule):
    """Swin Transformer
    A PyTorch implement of : `Swin Transformer:
    Hierarchical Vision Transformer using Shifted Windows`  -
        https://arxiv.org/abs/2103.14030

    Inspiration from
    https://github.com/microsoft/Swin-Transformer

    Args:
        pretrain_img_size (int | tuple[int]): The size of input image when
            pretrain. Defaults: 224.
        in_channels (int): The num of input channels.
            Defaults: 3.
        embed_dims (int): The feature dimension. Default: 96.
        patch_size (int | tuple[int]): Patch size. Default: 4.
        window_size (int): Window size. Default: 7.
        mlp_ratio (int): Ratio of mlp hidden dim to embedding dim.
            Default: 4.
        depths (tuple[int]): Depths of each Swin Transformer stage.
            Default: (2, 2, 6, 2).
        num_heads (tuple[int]): Parallel attention heads of each Swin
            Transformer stage. Default: (3, 6, 12, 24).
        strides (tuple[int]): The patch merging or patch embedding stride of
            each Swin Transformer stage. (In swin, we set kernel size equal to
            stride.) Default: (4, 2, 2, 2).
        out_indices (tuple[int]): Output from which stages.
            Default: (0, 1, 2, 3).
        qkv_bias (bool, optional): If True, add a learnable bias to query, key,
            value. Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        patch_norm (bool): If add a norm layer for patch embed and patch
            merging. Default: True.
        drop_rate (float): Dropout rate. Defaults: 0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Defaults: 0.1.
        use_abs_pos_embed (bool): If True, add absolute position embedding to
            the patch embedding. Defaults: False.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer at
            output of backone. Defaults: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        pretrained (str, optional): model pretrained path. Default: None.
        convert_weights (bool): The flag indicates whether the
            pre-trained model is from the original repo. We may need
            to convert some keys to make it compatible.
            Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            Default: -1 (-1 means not freezing any parameters).
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(
        self,
        pretrain_img_size=224,
        in_channels=3,
        embed_dims=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        use_abs_pos_embed=False,
        act_cfg=dict(type="GELU"),
        norm_cfg=dict(type="LN"),
        with_cp=False,
        pretrained=None,
        convert_weights=False,
        frozen_stages=-1,
        init_cfg=None,
    ):
        self.convert_weights = convert_weights
        self.frozen_stages = frozen_stages
        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)    # * to_2tuple将输入参数转换为长度为2的元组
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])
            assert len(pretrain_img_size) == 2, (
                f"The size of image should have length 1 or 2, "
                f"but got {len(pretrain_img_size)}"
            )

        assert not (
            init_cfg and pretrained
        ), "init_cfg and pretrained cannot be specified at the same time"
        if isinstance(pretrained, str):
            warnings.warn(
                "DeprecationWarning: pretrained is deprecated, "
                'please use "init_cfg" instead'
            )
            self.init_cfg = dict(type="Pretrained", checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = init_cfg
        else:
            raise TypeError("pretrained must be a str or None")

        super(SwinTransformer, self).__init__(init_cfg=init_cfg)

        num_layers = len(depths)
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed

        assert strides[0] == patch_size, "Use non-overlapping patch embed."

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type="Conv2d",
            kernel_size=patch_size,
            stride=strides[0],
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None,
        )

        if self.use_abs_pos_embed:
            patch_row = pretrain_img_size[0] // patch_size
            patch_col = pretrain_img_size[1] // patch_size
            num_patches = patch_row * patch_col
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros((1, num_patches, embed_dims))
            )

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # set stochastic depth decay rule
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        self.stages = ModuleList()
        in_channels = embed_dims
        for i in range(num_layers):
            if i < num_layers - 1:
                downsample = PatchMerging(
                    in_channels=in_channels,
                    out_channels=2 * in_channels,
                    stride=strides[i + 1],
                    norm_cfg=norm_cfg if patch_norm else None,
                    init_cfg=None,
                )
            else:
                downsample = None

            stage = SwinBlockSequence(
                embed_dims=in_channels,
                num_heads=num_heads[i],
                feedforward_channels=mlp_ratio * in_channels,
                depth=depths[i],
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                # 为每个阶段分配对应的drop概率列表，确保每层的路径drop概率与其全局索引一致
                downsample=downsample,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None,
            )
            self.stages.append(stage)
            if downsample:
                in_channels = downsample.out_channels

        self.num_features = [int(embed_dims * 2**i) for i in range(num_layers)]
        # Add a norm layer for each output
        for i in out_indices:
            layer = build_norm_layer(norm_cfg, self.num_features[i])[1]
            layer_name = f"norm{i}"
            self.add_module(layer_name, layer)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            if self.use_abs_pos_embed:
                self.absolute_pos_embed.requires_grad = False
            self.drop_after_pos.eval()

        for i in range(1, self.frozen_stages + 1):

            if (i - 1) in self.out_indices:
                norm_layer = getattr(self, f"norm{i-1}")
                # * getattr是一个内置函数，用于动态获取对象的属性。具体作用是从一个对象中获取指定属性的值
                # * 即使这个属性的名字是在运行时以字符串形式确定的
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

            m = self.stages[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        logger = MMLogger.get_current_instance()
        if self.init_cfg is None:
            logger.warn(
                f"No pre-trained weights for "
                f"{self.__class__.__name__}, "
                f"training start from scratch"
            )
            if self.use_abs_pos_embed:
                trunc_normal_(self.absolute_pos_embed, std=0.02)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=0.02, bias=0.0)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)
        else:
            assert "checkpoint" in self.init_cfg, (
                f"Only support "
                f"specify `Pretrained` in "
                f"`init_cfg` in "
                f"{self.__class__.__name__} "
            )
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location="cpu"
            )
            if "state_dict" in ckpt:
                _state_dict = ckpt["state_dict"]
            elif "model" in ckpt:
                _state_dict = ckpt["model"]
            else:
                _state_dict = ckpt
            if self.convert_weights:
                # supported loading weight from original repo,
                _state_dict = swin_converter(_state_dict)   # 如果需要转换权重格式，则调用转换

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith("backbone."):   # 去掉backbone.前缀，使权重字典的键名与模型一致
                # * startswith()方法用于检查字符串是否以指定子字符串开头，如果是返回True，如果不是返回False
                    state_dict[k[9:]] = v

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith("module."):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # reshape absolute position embedding
            # 如果预训练权重和当前模型的形状不匹配，记录警告并跳过
            if state_dict.get("absolute_pos_embed") is not None:
                absolute_pos_embed = state_dict["absolute_pos_embed"]
                N1, L, C1 = absolute_pos_embed.size()
                N2, C2, H, W = self.absolute_pos_embed.size()
                if N1 != N2 or C1 != C2 or L != H * W:
                    logger.warning("Error in loading absolute_pos_embed, pass")
                else:
                    state_dict["absolute_pos_embed"] = (
                        absolute_pos_embed.view(N2, H, W, C2)
                        .permute(0, 3, 1, 2)
                        .contiguous()
                    )

            # interpolate position bias table if needed
            # 如果相对位置偏置表形状不匹配，通过双三次插值调整大小，使预训练权重与当前模型兼容
            relative_position_bias_table_keys = [
                k for k in state_dict.keys() if "relative_position_bias_table" in k
            ]
            for table_key in relative_position_bias_table_keys:
                table_pretrained = state_dict[table_key]
                table_current = self.state_dict()[table_key]
                L1, nH1 = table_pretrained.size()
                L2, nH2 = table_current.size()
                if nH1 != nH2:
                    logger.warning(f"Error in loading {table_key}, pass")
                elif L1 != L2:
                    S1 = int(L1**0.5)
                    S2 = int(L2**0.5)
                    table_pretrained_resized = F.interpolate(
                        table_pretrained.permute(1, 0).reshape(1, nH1, S1, S1),
                        size=(S2, S2),
                        mode="bicubic",
                    )
                    state_dict[table_key] = (
                        table_pretrained_resized.view(nH2, L2)
                        .permute(1, 0)
                        .contiguous()
                    )

            # load state_dict
            self.load_state_dict(state_dict, False)

    def forward(self, x):
        x, hw_shape = self.patch_embed(x)

        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                out = norm_layer(out)
                out = (
                    out.view(-1, *out_hw_shape, self.num_features[i])
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                outs.append(out)

        return outs


def swin_converter(ckpt):

    new_ckpt = OrderedDict()

    def correct_unfold_reduction_order(x):
        out_channel, in_channel = x.shape
        x = x.reshape(out_channel, 4, in_channel // 4)
        x = x[:, [0, 2, 1, 3], :].transpose(1, 2).reshape(out_channel, in_channel)
        return x

    def correct_unfold_norm_order(x):
        in_channel = x.shape[0]
        x = x.reshape(4, in_channel // 4)
        x = x[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)
        return x

    for k, v in ckpt.items():
        if k.startswith("head"):
            continue
        elif k.startswith("layers"):
            new_v = v
            if "attn." in k:
                new_k = k.replace("attn.", "attn.w_msa.")
            elif "mlp." in k:
                if "mlp.fc1." in k:
                    new_k = k.replace("mlp.fc1.", "ffn.layers.0.0.")
                elif "mlp.fc2." in k:
                    new_k = k.replace("mlp.fc2.", "ffn.layers.1.")
                else:
                    new_k = k.replace("mlp.", "ffn.")
            elif "downsample" in k:
                new_k = k
                if "reduction." in k:
                    new_v = correct_unfold_reduction_order(v)
                elif "norm." in k:
                    new_v = correct_unfold_norm_order(v)
            else:
                new_k = k
            new_k = new_k.replace("layers", "stages", 1)
        elif k.startswith("patch_embed"):
            new_v = v
            if "proj" in k:
                new_k = k.replace("proj", "projection")
            else:
                new_k = k
        else:
            new_v = v
            new_k = k

        new_ckpt["backbone." + new_k] = new_v

    return new_ckpt
