# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union
from torch import Tensor
from mmrotate.registry import MODELS
from mmdet.models.utils import multi_apply
from .s2a_head import S2AHead

# import numpy as np


@MODELS.register_module()
class S2AHeadFeatureMap(S2AHead):
    r"""An anchor-based head used in `S2A-Net
    <https://ieeexplore.ieee.org/document/9377550>`_.
    """  # noqa: W605

    def forward(self, x: Tuple[Tensor]) -> Tuple[List[Tensor]]:
        cls_scores, bbox_preds = multi_apply(self.forward_single, x)
        self.output = cls_scores
        # self.output = bbox_preds
        return cls_scores, bbox_preds
