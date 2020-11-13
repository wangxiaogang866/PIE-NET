# PIE-NET: Parametric Inference of Point Cloud Edges

This is the code repository for "PIE-NET: Parametric Inference of Point Cloud Edges”
Created by Xiaogang Wang, Yuelang Xu, Kai Xu, Andrea Tagliasacchi, Bin Zhou, Ali Mahdavi-Amiri, Hao Zhang

## Prerequisites: 
    Numpy (ver. 1.13.3)
    TensorFlow (ver. 1.4.1)
    scipy (ver. 0.19.1)
    Matlab (ver. 2015a) 
    Python (ver. 2.7)
    Cuda (ver. 8.0)
    
This repository is based on Tensorflow and the TF operators from PointNet++ and PointNet. Therefore, you need to compile PointNet++ ([here](https://github.com/charlesq34/pointnet2)） and PointNet ([here](https://github.com/charlesq34/pointnet)).
The code is tested under TensorFlow 1.4.1 and Python 2.7 on Ubuntu 16.04.

## Introduction
There are seven versions of our approach, including a main version (8096 points Input), and six versions of the stress test, include: Noise (+0.01, +0.02, +0.05) and Sparse (1024, 2048, 4096) .

In each version, we provided pre-trained models ('.\main\stage_1_log\...') and test data ('.\main\test_data\...').

In the main version, the test data we provided included two parts :
1, ABC dataset ('.\model\test_data\101.mat', '102.mat') with GT.
2, Novel Categories ('.\model\test_data\135.mat', '136.mat', '137.mat') without GT.

In other versions, we only provided the ABC Dataset as test data.
Also, in the main version, we provide training samples ('.\model\train_data\'), if you want to retrain the model ('Python train_stage_12.py --stage=1').

We provide the pre-trained model in folder ('.\main\stage_1_log\...'). 
To evaluate the model, you need to put the test point clouds ('.mat' format) in the folder '.\main\test_data\...'

## Train：
    to train edge and corner points detection    
    cd main
    python train_stage_12.py --stage=1

## Test：
    to test edge and corner points detection    
    cd main
    python test_stage_12.py --stage=1

	
    
## Visualization:
    visualization.m: This file is used to visualize the detection results.
    

