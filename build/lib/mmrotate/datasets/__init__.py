# Copyright (c) OpenMMLab. All rights reserved.
from .dior import DIORDataset  # noqa: F401, F403
from .dota import DOTAv2Dataset  # noqa: F401, F403
from .dota import DOTADataset, DOTAv15Dataset
from .hrsc import HRSCDataset  # noqa: F401, F403
from .transforms import *  # noqa: F401, F403
from .soda_dota_fix import SODAA_DOTADataset

__all__ = [
    'DOTADataset', 'DOTAv15Dataset', 'DOTAv2Dataset', 'HRSCDataset',
    'DIORDataset', 'SODAA_DOTADataset'
]