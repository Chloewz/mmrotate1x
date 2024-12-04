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
## 2.1 FeatureMap由npy文件转为png格式可视化
使用*tools/feature_map_vis/feature_map_vis.py*代码能够完成Feature Map的可视化。具体的操作流程如下：

1. 把从1步骤中生成的Feature Map汇总到一个文件夹下面，可以由不同的子文件夹组成，如：
```
s2anet_feature_map
|
|___backbone.resnet
|   |___batch_0_layer_0.npy
|   |___batch_0_layer_1.npy
|   |___batch_0_layer_2.npy
|   |___batch_0_layer_3.npy
|   |___batch_0_layer_4.npy
|___bbox_head_init.s2anet_init.bbox_preds
|   | ... 
|___bbox_head_init.s2anet_init.cls_scores
|   | ... 
|___bbox_head_refine.s2anet_refine.bbox_preds
|   | ... 
|___bbox_head_refine.s2anet_refine.cls_scores
|   | ... 
|___neck.fpn
|   | ... 
```

2. 随后将*feature_map_vis.py*中的*folder_path*参数修改为该文件夹路径"../s2anet_feature_map/"，执行该函数，即可完成该函数的功能，遍历该文件夹下面的所有子文件夹，并完成这些文件夹下面的npy格式Feature Map可视化

# 3 Heat Map可视化
## 3.1 由Feature Map计算Heat Map并可视化
使用*tools/feature_map_vis/heat_map_from_featmap.py*完成Feature Map到Heat Map的计算。具体的实现细节就是归一化后可视化。运行该代码也很简单，和2一样，将*Heat_map_from_featmap.py*中的*folder_path*参数修改为同样的文件夹路径"../s2anet_feature_map/"即可。执行该函数，即可完成该函数的功能，遍历该文件夹下面的所有子文件夹，并完成由Feature Map到Heat Map的计算与可视化。