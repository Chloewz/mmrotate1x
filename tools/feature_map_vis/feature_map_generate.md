# 1 Feature Map的生成
## 1.1 通过自定义Hook，实现在模型测试过程中，Feature Map的生成

Feature Map生成自定义Hook的位置：envs/mmdetection-3.0.0/mmdet/engine/hooks/feature_map_hook.py


在该代码中，具体实现功能的函数是*after_test_iter*，确定在模型的哪个位置输出特征图，以S2ANet为例：

```
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
```

S2ANet的配置文件中显示该网络有四个模块组成，backbone、neck、bbox_head_init、bbox_head_refine，因此判断了这四个模块中是否有output，如果有的话，存入feature_maps中并进行后续的操作


其中，需要注意的是，**bbox_head_refine的类型是ModuleList**，这一点很重要，S2ANet网络只有一个refine头，因此采用*model.bbox_head_refine[0].output*

## 1.2 修改对应的模型文件

在自定义Hook中修改设置完成在模型的那一部分输出Feature Map后，需要在模型的对应部分修改forward函数，这一部分相对简单，**将forward的输出赋给self.output即可**，以bbox_head_redine为例：
```
class S2ARefineHeadFeatureMap(S2ARefineHead):
    
    def forward(self, x: Tuple[Tensor]) -> Tuple[List[Tensor]]:
        cls_scores, bbox_preds = multi_apply(self.forward_single, x)
        # self.output = cls_scores
        self.output = bbox_preds
        return cls_scores, bbox_preds
```
其中，需要注意一点的是，省事的办法是在源代码中直接加一行如*self.output=out*类似的语句，但保险起见，**目前采用的都是方式是新建一个模块文件，修改forward函数后，再注册到mmrotate或mmdetection中**

## 1.3 修改网络的配置文件
包括两个步骤：

1. 随后，修改configs下对应的配置文件，将对应的模块代替原有的网络模块，这里为了保险起见，也是新建一个配置文件，在新的配置文件中修改

2. 修改default_runtime文件，加入自定义hook部分，**其中参数save_dir的作用是修改生成Feature Map的保存路径**：
```
custom_hooks = [dict(
    type='mmdet.FeatureMapHook',
    save_dir='/mnt/d/exp/sodaa_sob/a6000result/0924_baseline/test/featuremap/',
)]
```

## 1.4 运行test文件
运行
```shell
python tools/test.py
```
其中，要将对应的参数根据test文件的要求配置好

## 1.5 Feature Map生成小结
1. 修改*envs/mmdetection-3.0.0/mmdet/engine/hooks/feature_map_hook.py*中的自定义hook，查看网络是否有对应的模块

2. 修改对应模块下的forward函数，在输出部分（或中间部分）添加*self.output=out*代码

3. 修改对应的模型配置文件，网络相关的模型配置文件，用修改后的模块代替原有模块

4. 修改对应的模型训练配置，在*default_runtime_featmap.py*文件中根据自己的需求，调整输出Feature Map的保存路径*save_dir*

5. 修改测试设置*test.py*，完成在测试的同时输出Feature Map

# 2 Feature Map可视化