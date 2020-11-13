# PIE-NET: Parametric Inference of Point Cloud Edges

This is the code repository for "PIE-NET: Parametric Inference of Point Cloud Edges”

Created by Xiaogang Wang, Yuelang Xu, Kai Xu, Andrea Tagliasacchi, Bin Zhou, Ali Mahdavi-Amiri, Hao Zhang

## Prerequisites: 
    Numpy (ver. 1.13.3)
    TensorFlow (ver. 1.4.1)
    scipy (ver. 0.19.1)
    Matlab (ver. 2015a) 
    Python (ver. 2.7)
    
This repository is based on Tensorflow and the TF operators from PointNet++ and PointNet. Therefore, you need to compile PointNet++ ([here](https://github.com/charlesq34/pointnet2)） and PointNet ([here](https://github.com/charlesq34/pointnet)).
The code is tested under TensorFlow 1.4.1 and Python 2.7 on Ubuntu 16.04.


## Train：

    to train edge and corner points detection module

    python train_stage_12.py --stage=1



## Test：

    to test edge and corner points detection module

    python test_stage_12.py --stage=1

	
    
## Visualization:
    visualization.m: This file is used to visualize the detection results.
