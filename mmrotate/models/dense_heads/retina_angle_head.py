# Copyright (c) OpenMMLab. All rights reserved.
from unicodedata import category

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmrotate.registry import MODELS
from mmdet.models.dense_heads.anchor_head import AnchorHead
from torch import Tensor
from mmengine.config import ConfigDict

from mmdet.structures.bbox import cat_boxes, get_box_tensor
from mmdet.models.utils import (
    images_to_levels,
    multi_apply,
    filter_scores_and_topk,
    select_single_mlvl,
)
from typing import List, Optional
from mmdet.utils import InstanceList, OptInstanceList, ConfigType
import copy
from mmengine.structures import InstanceData

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
        loss_angle: ConfigType = dict(type="SCALoss"),
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

        # TODO: 这里只对最底层的特征图进行了sca的计算与损失，但是其他层级的特征也会对分类产生影响，是否全用有待商榷
        if cls_score.size(2) == 128:    # 只对最底层的特征图做sca的计算与损失计算
            sca = self.sca_regress(cls_score)
        else:
            sca = 0.0

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

        losses_cls, losses_bbox, loss_angle = multi_apply(
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

    def predict_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        sca: float,
        score_factors: Optional[List[Tensor]] = None,
        batch_img_metas: Optional[List[dict]] = None,
        cfg: Optional[ConfigDict] = None,
        rescale: bool = False,
        with_nms: bool = True,
    ) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Note: When score_factors is not None, the cls_scores are
        usually multiplied by it then obtain the real score used in NMS,
        such as CenterNess in FCOS, IoU branch in ATSS.

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            score_factors (list[Tensor], optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, num_priors * 1, H, W). Defaults to None.
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_scores) == len(bbox_preds)

        if score_factors is None:
            # e.g. Retina, FreeAnchor, Foveabox, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, AutoAssign, etc.
            with_score_factors = True
            assert len(cls_scores) == len(score_factors)

        num_levels = len(cls_scores)

        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes, dtype=cls_scores[0].dtype, device=cls_scores[0].device
        )

        result_list = []

        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            cls_score_list = select_single_mlvl(cls_scores, img_id, detach=True)
            bbox_pred_list = select_single_mlvl(bbox_preds, img_id, detach=True)
            # sca_list = select_single_mlvl(sca, img_id, detach=True)
            if with_score_factors:
                score_factor_list = select_single_mlvl(
                    score_factors, img_id, detach=True
                )
            else:
                score_factor_list = [None for _ in range(num_levels)]

            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                sca_list=sca,
                score_factor_list=score_factor_list,
                mlvl_priors=mlvl_priors,
                img_meta=img_meta,
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms,
            )
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(
        self,
        cls_score_list: List[Tensor],
        bbox_pred_list: List[Tensor],
        sca_list: List[Tensor],
        score_factor_list: List[Tensor],
        mlvl_priors: List[Tensor],
        img_meta: dict,
        cfg: ConfigDict,
        rescale: bool = False,
        with_nms: bool = True,
    ) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image, each item has shape
                (num_priors * 1, H, W).
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (mmengine.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        if score_factor_list[0] is None:
            # e.g. Retina, FreeAnchor, etc.
            with_score_factors = False
        else:
            # e.g. FCOS, PAA, ATSS, etc.
            with_score_factors = True

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta["img_shape"]
        nms_pre = cfg.get("nms_pre", -1)

        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        mlvl_labels = []
        mlvl_scas = []
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None
        for level_idx, (cls_score, bbox_pred, sca, score_factor, priors) in enumerate(
            zip(
                cls_score_list, bbox_pred_list, sca_list, score_factor_list, mlvl_priors
            )
        ):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            dim = self.bbox_coder.encode_size
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, dim)
            if with_score_factors:
                score_factor = score_factor.permute(1, 2, 0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # remind that we set FG labels to [0, num_class-1]
                # since mmdet v2.0
                # BG cat_id: num_class
                scores = cls_score.softmax(-1)[:, :-1]

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            score_thr = cfg.get("score_thr", 0)

            results = filter_scores_and_topk(
                scores, score_thr, nms_pre, dict(bbox_pred=bbox_pred, priors=priors)
            )
            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results["bbox_pred"]
            priors = filtered_results["priors"]

            if with_score_factors:
                score_factor = score_factor[keep_idxs]

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)
            mlvl_scas.append(sca)

            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        bbox_pred = torch.cat(mlvl_bbox_preds)
        # print(bbox_pred.shape)
        priors = cat_boxes(mlvl_valid_priors)
        bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)

        results = InstanceData()
        results.bboxes = bboxes
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)
        # results.sca = torch.tensor(mlvl_scas[0])
        if with_score_factors:
            results.score_factors = torch.cat(mlvl_score_factors)

        # print(results.bboxes.shape)

        return self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta,
        )


class SimilarCategoryAngleRegression(nn.Module):
    def __init__(self, num_classes):
        super(SimilarCategoryAngleRegression, self).__init__()
        self.num_classes = num_classes

    def forward(self, cls_score):
        # cls_score shape: (1,81,128,128)
        cls_score_angle = cls_score.permute(
            0, 2, 3, 1
        ).contiguous()  # cls_score_angle shape: (1,128,128,81)
        cls_score_angle = cls_score_angle.view(
            cls_score_angle.size(0), -1, self.num_classes
        )  # cls_score_angle shape: (1,147456,9) 1 is batchsize
        scores = cls_score_angle.sigmoid()  # scores shape: (1,147456,9)
        # TODO: 这里的取mean操作，把整个batch的所有特征图都参与进了计算中，可能会导致所有类别的dets都有变化，是否做修改再思考
        scores_mean = scores.mean(dim=0)  # scores_mean shape: (147456,9)

        keep_idxs = self.filte_scores(
            scores_mean, 0.05, 2000
        )  # keep_idxs shape: (2000,)
        if keep_idxs.size(0) != 0:
            keep_idxs = torch.unique(keep_idxs)
            scores_mean = scores_mean[keep_idxs]

            category_x = "large-vehicle"
            category_y = "contianer"

            similar_category = scores_mean[:, [3, 5]].clone()
            similar_category_np = similar_category.detach().cpu().numpy()
            labels = np.where(
                similar_category_np[:, 0] > similar_category_np[:, 1],
                category_x,
                category_y,
            )

            similar_x = similar_category_np[labels == category_x]
            similar_y = similar_category_np[labels == category_y]

            if len(similar_x) != 0 and len(similar_y.shape) != 0:
                slope_x, _, _, _, _ = linregress(similar_x[:, 0], similar_x[:, 1])
                slope_y, _, _, _, _ = linregress(similar_y[:, 0], similar_y[:, 1])

                TINY = 1e-5
                angle = np.arctan(
                    np.abs((slope_y - slope_x) / (1 + slope_y * slope_x + TINY))
                )
                sca = np.degrees(angle)
            else:
                sca = 0.0
        else:
            sca = 0.0
        return sca

    def filte_scores(self, scores, scores_threshold, topk):
        # scores shape:(147456,9)
        valid_mask = scores > scores_threshold  # valid_mask shape:(147456,9)
        scores = scores[valid_mask]
        valid_idxs = torch.nonzero(valid_mask)

        num_topk = min(topk, valid_idxs.size(0))
        scores, idxs = scores.sort(descending=True)
        topk_idx = valid_idxs[idxs[:num_topk]]
        keep_idxs, labels = topk_idx.unbind(dim=1)  # keep_idxs shape (2000,)

        return keep_idxs
