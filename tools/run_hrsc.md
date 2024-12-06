# HRSC2016 S2ANET
## Train
python tools/train.py --config configs/hrsc2016-benchmarks/s2anet-le135_r50_fpn_1x_hrsc_dota.py --work-dir /mnt/d/try/
## Test
python tools/test.py 
configs/hrsc2016-benchmarks/s2anet-le135_r50_fpn_1x_hrsc_dota.py 
/mnt/d/exp/hrsc_gf2/hrsc_full/epoch_12.pth 
--work-dir /mnt/d/exp/hrsc_gf2/test/ 
--out /mnt/d/exp/hrsc_gf2/test/crop.pkl

# DOTA Split
python tools/data/dota/split/img_split.py 
--base-json tools/data/dota/split/split_configs/ss_test.json
