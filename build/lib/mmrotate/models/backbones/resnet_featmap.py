# Copyright (c) OpenMMLab. All rights reserved.
from mmrotate.registry import MODELS
from mmdet.models.backbones.resnet import ResNet


@MODELS.register_module()
class ResNetFeaturemap(ResNet):

    def forward(self, x):
        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        self.output = outs
        return tuple(outs)
