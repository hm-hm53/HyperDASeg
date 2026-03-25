
export PYTHONPATH=$PWD
CUDA_VISIBLE_DEVICES=0 python tools/init_prototypes_dg.py --config-path st.hyperdaseg.vaihingenpRgb_segformer

CUDA_VISIBLE_DEVICES=0 python tools/train_warmup_hyp.py --config-path st.hyperdaseg.vaihingenpRgb_segformer \
  --loss-kd HyperbolicPrototypeContrastiveLoss

CUDA_VISIBLE_DEVICES=0 python tools/train_ssl.py \
  --config-path st.hyperdaseg.vaihingenpRgb_segformer \
  --ckpt-model-tea log/hyperdaseg/SegFormer_MiT-B2/vaihingenpRgb/src_warmup_s/Potsdam_rgb_tea_curr.pth \
  --ckpt-model-stu log/hyperdaseg/SegFormer_MiT-B2/vaihingenpRgb/src_warmup_s/Potsdam_rgb_stu_curr.pth \
  --ckpt-proto log/hyperdaseg/SegFormer_MiT-B2/vaihingenpRgb/prototypes/warmup_prototypes.pth \
  --percent 0.9

