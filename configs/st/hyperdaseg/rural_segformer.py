from configs.ToRURAL import SOURCE_DATA_CONFIG, EVAL_DATA_CONFIG, \
    PSEUDO_DATA_CONFIG, TEST_DATA_CONFIG, TARGET_SET, target_dir, DATASETS
import hyperdaseg.aug.augmentation as mag
from albumentations import Normalize


# teacher cfg
MODEL_TEACHER = 'VitSeg_2'
BACKBONE_TEACHER = 'dinov3_vitl16'
PRETRAINED_TEACHER = ('/home/yons/文档/论文/HyperDASeg/HyperDASeg/ckpts/backbones/dinov3-vitl16-pretrain-lvd1689m')
# student cfg
MODEL_STUDENT = 'SegFormer'
BACKBONE_STUDENT = 'MiT-B2'
# PRETRAINED_STUDENT = None
PRETRAINED_STUDENT = '/home/yons/文档/论文/HyperDASeg/HyperDASeg/ckpts/backbones/mit_b2.pth'


BATCH_SIZE = 8

IGNORE_LABEL = -1
MOMENTUM = 0.9

SNAPSHOT_DIR = f'./log/hyperdaseg/{MODEL_STUDENT}_{BACKBONE_STUDENT}/rural'

# Hyper Paramters
OPTIMIZER = 'AdamW'
WEIGHT_DECAY = 0.01
LEARNING_RATE = 6e-5
# WEIGHT_DECAY = 0.0005
# LEARNING_RATE = 1e-2

STAGE1_STEPS = 10000
STAGE2_STEPS = 6000
NUM_STEPS = None        # for learning rate poly
PREHEAT_STEPS = 500     # for warm-up
POWER = 0.9             # lr poly power
EVAL_EVERY = 1000
GENE_EVERY = 1000
KD_TEM = 0.2
CUTOFF_TOP = 0.8
CUTOFF_LOW = 0.6


HYP_C = 0.1
DIST_A = 0.5
CLIP_R = 1


HR_LATER = 0.9
TAU_LATER = 0.6

TARGET_DATA_CONFIG = dict(
    image_dir=target_dir['image_dir'],
    mask_dir=[None],
    transforms=mag.Compose([
        mag.RandomCrop((512, 512)),
        mag.RandomHorizontalFlip(0.5),
        mag.RandomVerticalFlip(0.5),
        mag.RandomRotate90(0.5),
        mag.ColorJitter(p=0.5),
        mag.Normalize(
            mean=(73.53223948, 80.01710095, 74.59297778),
            std=(41.5113661, 35.66528876, 33.75830885)
        ),
    ]),
    CV=dict(k=10, i=-1),
    training=True,
    batch_size=8,
    num_workers=8,
    pin_memory=True,
    label_type='prob',
    read_sup=False,
)