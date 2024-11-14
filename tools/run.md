# SODAA S2ANET
python tools/train.py --config configs/sodaa-benchmarks/s2anet-le90_r50_fpn_1x_sodaa.py --work-dir /mnt/d/exp/sodaa_sob/4060/0924

# SODAA S2ANET LSFocalLoss
python tools/train.py --config configs/sodaa-benchmarks/s2anet-le90_r50_fpn_1x_sodaa_smooth.py --work-dir /mnt/d/exp/sodaa_sob/4060/0924_smooth

python tools/train.py --config configs/sodaa-benchmarks/s2anet-le90_r50_fpn_1x_sodaa_epison.py --work-dir /mnt/d/exp/sodaa_sob/4060/1105_epison --cfg-options randomness.seed=42

# ANALYZE LOG
python tools/analysis_tools/analyze_logs.py plot_curve /mnt/d/exp/sodaa_sob/a6000result/1105_cbam/20241105_212549/vis_data/20241105_212549.json --keys loss_cls_refine_0 loss_bbox_refine_0 --legend loss_cls loss_bbox

# SODAA R3Det
python tools/train.py --config configs/sodaa-benchmarks/r3det-refine-oc_r50_fpn_1x_dota.py --work-dir /mnt/d/exp/sodaa_sob/4060/r3det_sodaa

# SODAA featmap_vis
python tools/featmap_vis_demo.py /mnt/d/exp/sodaa_sob/featmap_vis_test/00105__800__1300___650.jpg configs/sodaa-benchmarks/rotated-retinanet-rbox-le90_r50_fpn_1x_dota.py /mnt/d/exp/sodaa_sob/a6000result/1107_retinanet/epoch_12.pth --target-layers neck --channel-reduction select_max --out-dir /mnt/d/exp/sodaa_sob/featmap_vis_test/neck/

# SODAA RetinaNet Test
python tools/test.py configs/sodaa-benchmarks/rotated-retinanet-rbox-le90_r50_fpn_1x_dota.py /mnt/d/exp/sodaa_sob/a6000result/1107_retinanet/epoch_12.pth --work-dir /mnt/d/exp/sodaa_sob/a6000result/1107_retinanet/work_dir --out /mnt/d/exp/sodaa_sob/a6000result/1107_retinanet/test/test_all.pkl

# SODAA RetinaNet Output Similiar Test
python tools/test.py configs/sodaa-benchmarks/rotated-retinanet-rbox-le90_r50_fpn_1x_dota.py /mnt/d/exp/sodaa_sob/a6000result/1107_retinanet/epoch_12.pth --work-dir /mnt/d/exp/sodaa_sob/a6000result/1107_retinanet/test/work_dir --out /mnt/d/exp/sodaa_sob/a6000result/1107_retinanet/test/similiar.pkl 

--show-dir /mnt/d/exp/sodaa_sob/a6000result/1107_retinanet/test/similiar_output_vis/

# SODAA featmap_vis
python tools/featmap_vis.py /mnt/d/exp/sodaa_sob/featmap_vis_test/00105__800__1300___650.jpg /mnt/d/exp/sodaa_sob/featmap_vis_test/neck/ configs/sodaa-benchmarks/rotated-retinanet-rbox-le90_r50_fpn_1x_dota.py /mnt/d/exp/sodaa_sob/a6000result/1107_retinanet/epoch_12.pth

# SODAA RetinaNet Output Similiar Confusion Matrix
python tools/analysis_tools/confusion_matrix.py configs/sodaa-benchmarks/rotated-retinanet-rbox-le90_r50_fpn_1x_dota.py /mnt/d/exp/sodaa_sob/a6000result/1107_retinanet/test/similiar.pkl /mnt/d/exp/sodaa_sob/a6000result/1107_retinanet/test/cm --show
