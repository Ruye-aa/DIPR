# DIPR: Efficient Point Cloud Registration via Dynamic Iteration

PyTorch implementation of the paper: [DIPR: Efficient Point Cloud Registration via Dynamic Iteration](https://arxiv.org/pdf/2312.02877)

## Introduction

Point cloud registration (PCR) is an essential task in 3D vision. Existing methods achieve increasingly higher accuracy. However, a large proportion of non-overlapping points in point cloud registration consume a lot of computational resources while negatively affecting registration accuracy. To overcome this challenge, we introduce a novel Efficient Point Cloud Registration via Dynamic Iteration framework, DIPR, that makes the neural network interactively focus on overlapping points based on sparser input points. We design global and local registration stages to achieve efficient course-to-fine processing. Beyond basic matching modules, we propose the Refined Nodes to narrow down the scope of overlapping points by using adopted density-based clustering to significantly reduce the computation amount. And our SC Classifier serves as an early-exit mechanism to terminate the registration process in time according to matching accuracy. Extensive experiments on multiple datasets show that our proposed approach achieves superior registration accuracy while significantly reducing computational time and GPU memory consumption compared to state-of-the-art methods. 


## Installation

Please use the following command for installation.

```bash
# It is recommended to create a new environment
conda create -n registration python==3.8
conda activate registration

pip install torch==1.7.1+cu110
# Install packages and other dependencies
pip install -r requirements.txt
python setup.py build develop
cd transformer/cpp_wrappers/pointops
python setup.py build
```
## 3DMatch

### Data preparation
The dataset can be downloaded from [PREDATOR]
(https://github.com/prs-eth/OverlapPredator). The data should be organized as follows:

```text
--data--3DMatch--metadata
              |--data--train--7-scenes-chess--cloud_bin_0.pth
                    |      |               |--...
                    |      |--...
                    |--test--7-scenes-redkitchen--cloud_bin_0.pth
                          |                    |--...
                          |--...
```

### Training

The code for 3DMatch is in `experiments/3dmatch`. Use the following command for training.

```bash
python experiments/3dmatch-KPConv/train.py
python experiments/3dmatch-GA/train.py
```


### Testing

Use the following command for testing.

```bash
# 3DMatch
python experiments/3dmatch-KPConv/test.py  --benchmark=3DMatch   --snapshot=output/3DMatch/snapshot/snapshot.pth.tar  --stage_nums=iter_num  --classifier_threshold=200
python experiments/3dmatch-KPConv/eval.py  --benchmark=3DMatch   --method=lgr

python experiments/3dmatch-GA/test.py  --benchmark=3DMatch   --snapshot=output/3DMatch/snapshot/snapshot.pth.tar  --stage_nums=iter_num  --classifier_threshold=200 --epsilon=0.125
python experiments/3dmatch-GA/eval.py  --benchmark=3DMatch   --method=lgr

# 3DLoMatch
python experiments/3dmatch-KPConv/test.py  --benchmark=3DLoMatch --snapshot=output/3DMatch/snapshot/snapshot.pth.tar  --stage_nums=iter_num  --classifier_threshold=200
python experiments/3dmatch-KPConv/eval.py  --benchmark=3DLoMatch --method=lgr

python experiments/3dmatch-KPConv/test.py  --benchmark=3DLoMatch --snapshot=output/3DMatch/snapshot/snapshot.pth.tar  --stage_nums=iter_num  --classifier_threshold=150 --epsilon=0.3
python experiments/3dmatch-KPConv/eval.py  --benchmark=3DLoMatch --method=lgr
```
`iter_num` is the iteration number. The default is 3.


## Kitti odometry

### Data preparation

Download the data from the [Kitti official website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) into `data/Kitti` and run `data/Kitti/downsample_pcd.py` to generate the data. The data should be organized as follows:


```text
--data--Kitti--metadata
            |--sequences--00--velodyne--000000.bin
            |              |         |--...
            |              |...
            |--downsampled--00--000000.npy
                         |   |--...
                         |--...
```

### Training

The code for Kitti is in `experiments/kitti`. Use the following command for training.

```bash
python experiments/kitti/train.py
```

### Testing

Use the following command for testing.

```bash
python experiments/kitti/test.py    --snapshot=output/kitti/snapshot/snapshot.pth.tar  --stage_nums=iter_num  --classifier_threshold=30
python experiments/kitti/eval.py    --method=lgr
```
`iter_num` is the iteration number. The default is 2.
