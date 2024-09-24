# SODAA S2ANET
python tools/train.py --config configs/sodaa-benchmarks/s2anet-le90_r50_fpn_1x_sodaa.py --work-dir /mnt/d/exp/sodaa_sob/4060/0924

# SODAA S2ANET LSFocalLoss
python tools/train.py --config configs/sodaa-benchmarks/s2anet-le90_r50_fpn_1x_sodaa_smooth.py --work-dir /mnt/d/exp/sodaa_sob/4060/0924_smooth --resume /mnt/d/exp/sodaa_sob/a6000result/0907/epoch_9.pth