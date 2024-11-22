# Copyright (c) OpenMMLab. All rights reserved.
from unicodedata import category

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmrotate.registry import MODELS
from mmdet.models.dense_heads.anchor_head import AnchorHead
from torch import Tensor

from mmdet.structures.bbox import cat_boxes, get_box_tensor
from mmdet.models.utils import images_to_levels, multi_apply
from typing import List
from mmdet.utils import InstanceList, OptInstanceList, ConfigType

from scipy.stats import linregress
import numpy as np


@MODELS.register_module()
class RetinaAngleHead(AnchorHead):
    r"""An anchor-based head used in `RetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Example:
        # >>> import torch
        # >>> self = RetinaHead(11, 7)
        # >>> x = torch.rand(1, 7, 32, 32)
        # >>> cls_score, bbox_pred = self.forward_single(x)
        # >>> # Each anchor predicts a score for each class except background
        # >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        # >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        # >>> assert cls_per_anchor == (self.num_classes)
        # >>> assert box_per_anchor == 4
    """

    def __init__(
        self,
        num_classes,
        in_channels,
        stacked_convs=4,
        conv_cfg=None,
        norm_cfg=None,
        anchor_generator=dict(
            type="AnchorGenerator",
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128],
        ),
        init_cfg=dict(
            type="Normal",
            layer="Conv2d",
            std=0.01,
            override=dict(type="Normal", name="retina_cls", std=0.01, bias_prob=0.01),
        ),
        loss_angle: ConfigType=dict(type='SCALoss'),
        **kwargs,
    ):
        assert stacked_convs >= 0, (
            "`stacked_convs` must be non-negative integers, "
            f"but got {stacked_convs} instead."
        )
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        super(RetinaAngleHead, self).__init__(
            num_classes,
            in_channels,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs,
        )
        self.loss_angle = MODELS.build(loss_angle)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        in_channels = self.in_channels
        for i in range(self.stacked_convs):
            self.cls_convs.append(
                ConvModule(
                    in_channels,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                )
            )
            self.reg_convs.append(
                ConvModule(
                    in_channels,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                )
            )
            in_channels = self.feat_channels
        self.retina_cls = nn.Conv2d(
            in_channels, self.num_base_priors * self.cls_out_channels, 3, padding=1
        )
        reg_dim = self.bbox_coder.encode_size
        self.retina_reg = nn.Conv2d(
            in_channels, self.num_base_priors * reg_dim, 3, padding=1
        )
        self.sca_regress = SimilarCategoryAngleRegression(self.num_classes)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)

        if cls_score.size(2) == 128:
            sca = self.sca_regress(cls_score)
        else:
            sca = None

        return cls_score, bbox_pred, sca

    def loss_by_feat_single(
        self,
        cls_score: Tensor,
        bbox_pred: Tensor,
        sca: float,
        anchors: Tensor,
        labels: Tensor,
        label_weights: Tensor,
        bbox_targets: Tensor,
        bbox_weights: Tensor,
        avg_factor: int,
    ) -> tuple:
        """Calculate the loss of a single scale level based on the features
        extracted by the detection head.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            avg_factor (int): Average factor that is used to average the loss.

        Returns:
            tuple: loss components.
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=avg_factor
        )
        # regression loss
        target_dim = bbox_targets.size(-1)
        bbox_targets = bbox_targets.reshape(-1, target_dim)
        bbox_weights = bbox_weights.reshape(-1, target_dim)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(
            -1, self.bbox_coder.encode_size
        )
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, anchors.size(-1))
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
            bbox_pred = get_box_tensor(bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred, bbox_targets, bbox_weights, avg_factor=avg_factor
        )
        # angle loss
        loss_angle = self.loss_angle(sca)
        return loss_cls, loss_bbox, loss_angle

    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        sca: float,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None,
    ) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                has shape (N, num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict: A dictionary of loss components.
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device
        )
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
        )
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            avg_factor,
        ) = cls_reg_targets

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(cat_boxes(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list, num_level_anchors)

        losses_cls, losses_bbox, loss_angle= multi_apply(
            self.loss_by_feat_single,
            cls_scores,
            bbox_preds,
            sca,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            avg_factor=avg_factor,
        )
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_angle=loss_angle)


class SimilarCategoryAngleRegression(nn.Module):
    def __init__(self, num_classes):
        super(SimilarCategoryAngleRegression, self).__init__()
        self.num_classes = num_classes

    def forward(self, cls_score):
        cls_score_angle = cls_score.permute(0, 2, 3, 1).contiguous()
        cls_score_angle = cls_score_angle.view(
            cls_score_angle.size(0), -1, self.num_classes
        )
        scores = cls_score_angle.sigmoid()
        scores_mean = scores.mean(dim=0)

        keep_idxs = self.filte_scores(scores_mean, 0.05, 20000)
        if keep_idxs.size(0) != 0:
            keep_idxs = torch.unique(keep_idxs)
            scores_mean = scores_mean[keep_idxs]

            category_x = "large-vehicle"
            category_y = "contianer"

            similar_category = scores_mean[:, [3, 5]].clone()
            similar_category_np = similar_category.numpy()
            labels = np.wher(
                similar_category_np[:, 0] > similar_category_np[:, 1],
                category_x,
                category_y,
            )
            similar_x = similar_category_np[labels == category_x]
            similar_y = similar_category_np[labels == category_y]

            slope_x, _, _, _, _ = linregress(similar_x[:, 0], similar_x[:, 1])
            slope_y, _, _, _, _ = linregress(similar_y[:, 0], similar_y[:, 1])

            angle = np.arctan(np.abs((slope_y - slope_x) / (1 + slope_y * slope_x)))
            sca = np.degrees(angle)
        else:
            sca = None
        return sca

    def filte_scores(self, scores, scores_threshold, topk):
        valid_mask = scores > scores_threshold
        scores = scores[valid_mask]
        valid_idxs = torch.nonzero(valid_mask)

        num_topk = min(topk, valid_idxs.size(0))
        scores, idxs = scores.sort(descending=True)
        topk_idx = valid_idxs[idxs[:num_topk]]
        keep_idxs, labels = topk_idx.unbind(dim=1)

        return keep_idxs
