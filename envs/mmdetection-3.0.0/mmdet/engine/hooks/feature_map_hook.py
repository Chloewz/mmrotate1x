from mmengine.hooks import Hook
from mmdet.registry import HOOKS
import os
import numpy as np

@HOOKS.register_module()
class FeatureMapHook(Hook):
    def __init__(self, save_dir, layer_names=None):
        """
        Args:
            save_dir: 保存特征图的位置
            layer_names: 要提取的层名，默认为None
        """
        self.save_dir = save_dir
        self.layer_names = layer_names

    
    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model

        if hasattr(model.backbone, 'output'):
            feature_maps = model.backbone.output
        elif hasattr(model.neck, 'output'):
            feature_maps = model.neck.output
        elif hasattr(model.bbox_head_init, 'output'):
            feature_maps = model.bbox_head_init.output
        elif hasattr(model.bbox_head_refine[0], 'output'):
            feature_maps = model.bbox_head_refine[0].output
        else:
            raise AttributeError("No 'output' attribute, please fix related code")
        
        for idx, feature_map in enumerate(feature_maps):
            save_path = os.path.join(self.save_dir, f'batch_{batch_idx}_layer_{idx}.npy')
            np.save(save_path, feature_map.detach().cpu().numpy())

        print(f"Saved feature maps for batch {batch_idx} to {self.save_dir}")

    # def after_forward(self, runner, batch_idx, data_batch, outputs):
    #     """
    #     在forward过程结束之后，提取并保存特征图
    #     """
    #     model = runner.model
    #     # 获取特定层输出
    #     if hasattr(model, 'module'):
    #         model = model.module

    #     feature_maps = {}
    #     for name, module in model.name_modules():
    #         if self.layer_names is None or name in self.layer_names:
    #             if hasattr(module, 'output'):
    #                 feature_maps[name] = module.output.detach().cpu().numpy()

    #     # 保存特征图
    #     for name, fmap in feature_maps.items():
    #         save_path = os.path.join(self.save_dir, f"batch_{name}.npy")
    #         os.makedirs(os.path.dirname(save_path, exist_ok=True))
    #         np.save(save_path, fmap)