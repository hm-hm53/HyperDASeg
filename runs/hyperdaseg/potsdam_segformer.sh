export PYTHONPATH=$PWD
CUDA_VISIBLE_DEVICES=0 python tools/init_prototypes_dg.py --config-path st.hyperdaseg.potsdam_segformer

CUDA_VISIBLE_DEVICES=0 python tools/train_warmup_hyp.py --config-path st.hyperdaseg.potsdam_segformer \
  --loss-kd HyperbolicPrototypeContrastiveLoss

CUDA_VISIBLE_DEVICES=0 python tools/train_ssl.py \
  --config-path st.hyperdaseg.potsdam_segformer \
  --ckpt-model-tea log/hyperdaseg/SegFormer_MiT-B2/potsdam/src_warmup_s/Potsdam_tea_curr.pth \
  --ckpt-model-stu log/hyperdaseg/SegFormer_MiT-B2/potsdam/src_warmup_s/Potsdam_stu_curr.pth \
  --ckpt-proto log/hyperdaseg/SegFormer_MiT-B2/potsdam/prototypes/warmup_prototypes.pth \
  --percent 0.9

