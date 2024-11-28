import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch import dtype

from mmrotate.registry import MODELS
from mmdet.models import weight_reduce_loss


@MODELS.register_module()
class SCALoss(nn.Module):

    def __init__(
        self):
        super(SCALoss, self).__init__()

    def forward(
        self,
        sca
    ):
        if sca != 0:
            sca = torch.tensor(sca, dtype=torch.float32)
            sca = torch.deg2rad(sca)
            loss_angle = torch.cos(sca)
        else:
            sca = torch.tensor(sca, dtype=torch.float32)
            loss_angle = torch.tensor(0.0)
        return loss_angle
