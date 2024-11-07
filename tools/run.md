# SODAA S2ANET
python tools/train.py --config configs/sodaa-benchmarks/s2anet-le90_r50_fpn_1x_sodaa.py --work-dir /mnt/d/exp/sodaa_sob/4060/0924

# SODAA S2ANET LSFocalLoss
python tools/train.py --config configs/sodaa-benchmarks/s2anet-le90_r50_fpn_1x_sodaa_smooth.py --work-dir /mnt/d/exp/sodaa_sob/4060/0924_smooth

python tools/train.py --config configs/sodaa-benchmarks/s2anet-le90_r50_fpn_1x_sodaa_epison.py --work-dir /mnt/d/exp/sodaa_sob/4060/1105_epison --cfg-options randomness.seed=42

# ANALYZE LOG
python tools/analysis_tools/analyze_logs.py plot_curve /mnt/d/exp/sodaa_sob/a6000result/1105_cbam/20241105_212549/vis_data/20241105_212549.json --keys loss_cls_refine_0 loss_bbox_refine_0 --legend loss_cls loss_bbox

# SODAA R3Det
python tools/train.py --config configs/sodaa-benchmarks/r3det-refine-oc_r50_fpn_1x_dota.py --work-dir /mnt/d/exp/sodaa_sob/4060/r3det_sodaa