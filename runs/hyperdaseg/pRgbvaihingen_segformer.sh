
export PYTHONPATH=$PWD
CUDA_VISIBLE_DEVICES=0 python tools/init_prototypes_dg.py --config-path st.hyperdaseg.pRgbvaihingen_segformer

CUDA_VISIBLE_DEVICES=0 python tools/train_warmup_hyp.py --config-path st.hyperdaseg.pRgbvaihingen_segformer \
  --loss-kd HyperbolicPrototypeContrastiveLoss

CUDA_VISIBLE_DEVICES=0 python tools/train_ssl.py \
  --config-path st.hyperdaseg.pRgbvaihingen_segformer \
  --ckpt-model-tea log/hyperdaseg/SegFormer_MiT-B2/pRgbvaihingen/src_warmup_s/Vaihingen_tea_curr.pth \
  --ckpt-model-stu log/hyperdaseg/SegFormer_MiT-B2/pRgbvaihingen/src_warmup_s/Vaihingen_stu_curr.pth \
  --ckpt-proto log/hyperdaseg/SegFormer_MiT-B2/pRgbvaihingen/prototypes/warmup_prototypes.pth \
  --percent 0.9

