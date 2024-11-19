# SODAA S2ANET
## Train
python tools/train.py --config configs/sodaa-benchmarks/s2anet-le90_r50_fpn_1x_sodaa.py --work-dir /mnt/d/exp/sodaa_sob/4060/0924
## Confusion Matrix
python tools/analysis_tools/confusion_matrix_better.py configs/sodaa-benchmarks/s2anet-le90_r50_fpn_1x_sodaa.py /mnt/d/exp/sodaa_sob/a6000result/0924_baseline/test/s2anet_sodaa.pkl /mnt/d/exp/sodaa_sob/a6000result/0924_baseline/test/ --show

# EpisonHotExp
## SODAA S2ANET LSFocalLoss
python tools/train.py --config configs/sodaa-benchmarks/s2anet-le90_r50_fpn_1x_sodaa_smooth.py --work-dir /mnt/d/exp/sodaa_sob/4060/0924_smooth
## SODAA S2ANet EpisonHotLoss
python tools/train.py --config configs/sodaa-benchmarks/s2anet-le90_r50_fpn_1x_sodaa_epison.py --work-dir /mnt/d/exp/sodaa_sob/4060/1105_epison --cfg-options randomness.seed=42

# Analysis Tools
## ANALYZE LOG
python tools/analysis_tools/analyze_logs.py plot_curve /mnt/d/exp/sodaa_sob/a6000result/1105_cbam/20241105_212549/vis_data/20241105_212549.json --keys loss_cls_refine_0 loss_bbox_refine_0 --legend loss_cls loss_bbox

# SODAA R3Det
## Train
python tools/train.py --config configs/sodaa-benchmarks/r3det-refine-oc_r50_fpn_1x_dota.py --work-dir /mnt/d/exp/sodaa_sob/4060/r3det_sodaa
## Confusion Matrix
python tools/analysis_tools/confusion_matrix_better.py configs/sodaa-benchmarks/r3det-refine-oc_r50_fpn_1x_dota.py /mnt/d/exp/sodaa_sob/a6000result/1107_r3det/test/r3det_sodaa.pkl /mnt/d/exp/sodaa_sob/a6000result/1107_r3det/test/ --show

# SODAA RetinaNet
## Test
python tools/test.py configs/sodaa-benchmarks/rotated-retinanet-rbox-le90_r50_fpn_1x_dota.py /mnt/d/exp/sodaa_sob/a6000result/1107_retinanet/epoch_12.pth --work-dir /mnt/d/exp/sodaa_sob/a6000result/1107_retinanet/work_dir --out /mnt/d/exp/sodaa_sob/a6000result/1107_retinanet/test/test_all.pkl
## Confusion Matrix
python tools/analysis_tools/confusion_matrix_better.py configs/sodaa-benchmarks/rotated-retinanet-rbox-le90_r50_fpn_1x_dota.py /mnt/d/exp/sodaa_sob/a6000result/1107_retinanet/test/retinanet_sodaa.pkl /mnt/d/exp/sodaa_sob/a6000result/1107_retinanet/test --show

# SODAA RoiTransformer
## Train
python tools/train.py --config configs/sodaa-benchmarks/sodaa_comparison/roi-trans-le90_r50_fpn_1x_sodaa.py --work-dir /mnt/d/exp/sodaa_sob/4060/roitrans_sodaa
## Test
python tools/test.py configs/sodaa-benchmarks/sodaa_comparison/roi-trans-le90_r50_fpn_1x_sodaa.py /mnt/d/exp/sodaa_sob/4060/roitrans_sodaa/epoch_1.pth --work-dir /mnt/d/exp/sodaa_sob/4060/roitrans_sodaa/test/
