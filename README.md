# Hyperbolic Geometry-Guided Unsupervised Domain Adaptive Semantic Segmentation in Remote Sensing

## Overall

<div>
<img src="overall/overall.png"/>
</div>


## Install

step1:creative an environment for project
````shell
conda create -n hyperdaseg python=3.10
conda activate hyperdaseg
````

step2:Install pytorch and torchvision matching your CUDA version(This project runs under CUDA version 13.0):
````shell
pip install torch==2.8.0 torchvision==0.23.0
pip install -r HyperDASeg/requirements.txt
````

## Pretrained Weights of HyperDASeg

[model_weights](https://pan.baidu.com/s/1m34ypzfwna5p1xWoYme0tg?pwd=ygqq)

## Data Preprocessing
#### 1. you can download the processed data
- Download the processed datasets from <a href="https://pan.baidu.com/s/1m34ypzfwna5p1xWoYme0tg?pwd=ygqq">here</a>.
- reorganize the directory tree.
#### 2. The prepared data is formatted as follows:
"\
./data\
----&nbsp;IsprsDA\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;----&nbsp;Potsdam\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;----&nbsp;ann_dir\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;----&nbsp;img_dir\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;----&nbsp;Vaihingen\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;----&nbsp;ann_dir\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;----&nbsp;img_dir\
----&nbsp;LoveDA\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;----&nbsp;Test\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;----&nbsp;Rural\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;----&nbsp;Urban\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;----&nbsp;Train\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;----&nbsp;Rural\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;----&nbsp;Urban\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;----&nbsp;Val\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;----&nbsp;Rural\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;----&nbsp;Urban\
"
#### 3. you can also generate the raw data
- Run the preprocess script in ./convert_datasets/ to crop train, val, test sets:\
`python convert_datasets/convert_potsdam.py`\
`python convert_datasets/convert_vaihingen.py`
## Training
#### 1. On Vaihingen (IRRG) -> Potsdam (IRRG) task
````shell
bash runs/hyperdaseg/potsdam_segformer.sh
````
#### 2.On Potsdam (IRRG) task -> On Vaihingen (IRRG)
````shell
bash runs/hyperdaseg/vaihingen_segformer.sh
````
#### 3. On Rural -> Urban task
````shell
bash runs/hyperdaseg/urban_segformer.sh
````
#### 4. On Urban -> rural task
````shell
bash runs/hyperdaseg/rural_segformer.sh
````
#### 5. On Potsdam (RGB) -> Vaihingen (IRRG) task
````shell
bash runs/hyperdaseg/pRgbvaihingen_segformer.sh
````
#### 6. On Vaihingen (IRRG) -> Potsdam (RGB) task
````shell
bash runs/hyperdaseg/vaihingenpRgb_segformer.sh
````


## Testing


````shell
PYTHONPATH=. python tools/eval.py \
--config-path st.hyperdaseg.vaihingen_segformer \
--ckpt-path log/lfmda/SegFormer_MiT-B2/urban/ssl_proto_pseudo_class-wise_threshold_0.9/Vaihingen_stu_best.pth \
--test 1
````
