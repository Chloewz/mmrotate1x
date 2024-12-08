# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList
from ..layers import (
    DetrTransformerDecoder,
    DetrTransformerEncoder,
    SinePositionalEncoding,
)
from .base_detr import DetectionTransformer


@MODELS.register_module()
class DETR(DetectionTransformer):
    r"""Implementation of `DETR: End-to-End Object Detection with Transformers.

    <https://arxiv.org/pdf/2005.12872>`_.

    Code is modified from the `official github repo
    <https://github.com/facebookresearch/detr>`_.
    """

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(**self.positional_encoding)
        self.encoder = DetrTransformerEncoder(**self.encoder)
        self.decoder = DetrTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        # NOTE The embed_dims is typically passed from the inside out.
        # For example in DETR, The embed_dims is passed as
        # self_attn -> the first encoder layer -> encoder -> detector.
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        # * nn.Embedding字典映射表，存储固定字典和大小的嵌入。通常用于存储词嵌入并使用索引检索它们。
        # * 模块的输入是索引列表，输出是相应的词嵌入。

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, (
            "embed_dims should be exactly 2 times of num_feats. "
            f"Found {self.embed_dims} and {num_feats}."
        )

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        for coder in self.encoder, self.decoder:
            for p in coder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def pre_transformer(
        self, img_feats: Tuple[Tensor], batch_data_samples: OptSampleList = None
    ) -> Tuple[Dict, Dict]:
        """Prepare the inputs of the Transformer.

        对输入特征进行预处理，将其转换为适合encoder和decoder输入的字典格式
        关键步骤：1.构造特征图的有效区域掩码（mask）
                2.基于掩码生成位置编码
                3.调整特征图、位置编码、掩码的形状
                4.构造encoder和decoder的输入字典

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            img_feats (Tuple[Tensor]): Tuple of features output from the neck,
                has shape (bs, c, h, w).
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such as
                `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None. 用于生成特征图对应的掩码（mask）

        Returns:
            tuple[dict, dict]: The first dict contains the inputs of encoder
            and the second dict contains the inputs of decoder.

            - encoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_encoder()`, which includes 'feat', 'feat_mask',
              and 'feat_pos'.
            - decoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_decoder()`, which includes 'memory_mask',
              and 'memory_pos'.
        """

        feat = img_feats[-1]  # NOTE img_feats contains only one feature. （只是用了最后一层特征图）
        batch_size, feat_dim, _, _ = feat.shape
        # construct binary masks which for the transformer. 构造二值掩码
        assert batch_data_samples is not None
        batch_input_shape = batch_data_samples[0].batch_input_shape
        img_shape_list = [sample.img_shape for sample in batch_data_samples]

        input_img_h, input_img_w = batch_input_shape
        masks = feat.new_ones((batch_size, input_img_h, input_img_w))   # 初始全1
        for img_id in range(batch_size):
            img_h, img_w = img_shape_list[img_id]
            masks[img_id, :img_h, :img_w] = 0   # 根据真实尺寸img_h, img_w将对应的区域设置为0(0是有效区域)
        # NOTE following the official DETR repo, non-zero values represent
        # ignored positions, while zero values mean valid positions.

        masks = (
            F.interpolate(masks.unsqueeze(1), size=feat.shape[-2:])
            .to(torch.bool)
            .squeeze(1)
        )   # 将掩码masks的尺寸调整为与特征图相匹配
        # * F.interpolate()插值，将掩码从输入图片的尺寸缩小到特征图的分辨率
        # [batch_size, embed_dim, h, w]
        pos_embed = self.positional_encoding(masks)

        # 将特征图、位置编码、掩码调整为Transformer所需的格式
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        # [bs, c, h, w] -> [bs, h*w, c] [batch_size, h*w, feat_dim]
        feat = feat.view(batch_size, feat_dim, -1).permute(0, 2, 1)
        pos_embed = pos_embed.view(batch_size, feat_dim, -1).permute(0, 2, 1)
        # [bs, h, w] -> [bs, h*w]
        masks = masks.view(batch_size, -1)

        # prepare transformer_inputs_dict
        encoder_inputs_dict = dict(feat=feat, feat_mask=masks, feat_pos=pos_embed)
        decoder_inputs_dict = dict(memory_mask=masks, memory_pos=pos_embed)
        return encoder_inputs_dict, decoder_inputs_dict

    def forward_encoder(
        self, feat: Tensor, feat_mask: Tensor, feat_pos: Tensor
    ) -> Dict:
        """Forward with Transformer encoder.

        通过Transformer encoder对输入特征进行处理

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            feat (Tensor): Sequential features, has shape (bs, num_feat_points,
                dim).
            feat_mask (Tensor): ByteTensor, the padding mask of the features,
                has shape (bs, num_feat_points).
            feat_pos (Tensor): The positional embeddings of the features, has
                shape (bs, num_feat_points, dim).

        Returns:
            dict: The dictionary of encoder outputs, which includes the
            `memory` of the encoder output.
        """
        memory = self.encoder(
            query=feat, query_pos=feat_pos, key_padding_mask=feat_mask
        )  # for self_attn
        encoder_outputs_dict = dict(memory=memory)
        return encoder_outputs_dict

    def pre_decoder(self, memory: Tensor) -> Tuple[Dict, Dict]:
        """Prepare intermediate variables before entering Transformer decoder,
        such as `query`, `query_pos`.

        1.准备查询位置嵌入（query_pos）
        2.初始化decoder的查询向量（query）
        3.组装decoder输入字典和头部输入字典

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).

        Returns:
            tuple[dict, dict]: The first dict contains the inputs of decoder
            and the second dict contains the inputs of the bbox_head function.

            - decoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_decoder()`, which includes 'query', 'query_pos',
              'memory'. 
              query_pos引入查询的位置信息；query作为decoder的初始输入用于后续预测；memory提供全局上下文信息帮助decoder生成最终输出
            - head_inputs_dict (dict): The keyword args dictionary of the
              bbox_head functions, which is usually empty, or includes
              `enc_outputs_class` and `enc_outputs_class` when the detector
              support 'two stage' or 'query selection' strategies.
                为bbox_head函数准备的输入字典，通常为空，除非支持两阶段或查询选择策略
        """

        batch_size = memory.size(0)  # (bs, num_feat_points, dim)
        query_pos = self.query_embedding.weight
        # query_embedding是一个可训练的嵌入矩阵，形状为(num_queries, dim) -- (查询向量的数量，每个查询向量的维度)
        # (num_queries, dim) -> (bs, num_queries, dim)
        query_pos = query_pos.unsqueeze(0).repeat(batch_size, 1, 1)
        # unsqueeze(0)在第0维增加一个批量维度，.repeat(bs, 1, 1)将位置嵌入扩展到整个批量，形状变为(bs, #queries, dim)
        query = torch.zeros_like(query_pos) # 初始化查询全零

        decoder_inputs_dict = dict(query_pos=query_pos, query=query, memory=memory)
        head_inputs_dict = dict()
        return decoder_inputs_dict, head_inputs_dict

    def forward_decoder(
        self,
        query: Tensor,
        query_pos: Tensor,
        memory: Tensor,
        memory_mask: Tensor,
        memory_pos: Tensor,
    ) -> Dict:
        """Forward with Transformer decoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            query (Tensor): The queries of decoder inputs, has shape
                (bs, num_queries, dim).
            query_pos (Tensor): The positional queries of decoder inputs,
                has shape (bs, num_queries, dim).
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points).
            memory_pos (Tensor): The positional embeddings of memory, has
                shape (bs, num_feat_points, dim).

        Returns:
            dict: The dictionary of decoder outputs, which includes the
            `hidden_states` of the decoder output.

            - hidden_states (Tensor): Has shape
              (num_decoder_layers, bs, num_queries, dim)
        """

        hidden_states = self.decoder(
            query=query,
            key=memory,
            value=memory,
            query_pos=query_pos,
            key_pos=memory_pos,
            key_padding_mask=memory_mask,
        )  # for cross_attn

        head_inputs_dict = dict(hidden_states=hidden_states)
        return head_inputs_dict
