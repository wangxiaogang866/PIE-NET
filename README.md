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

## Stage 2 training data generation

The file '/stage2/main/Gen_stage2_train_data_all.m' that can be used to generate training data for Stage2. 
Taking the data we provided as an example, you only need to test all the training data './ train_data/xxx.mat' with the trained Stage1 network and get the corresponding output './test_result/test_pre_xxx.mat'. 
Then, the training data for Stage2 (./train_data_2_1/xxx.mat) can be generated using the file 'Gen_stage2_train_data_all.m' .

Note that the generation of the training data for Stage2 depends on the nine ground truth items (corner_points_label, open_gt_pair_idx, open_gt_type, open_gt_res) in './train_data/xxx.mat';
Therefore, if you are using your own data to generate Stage2 training samples, you will need to provide the following nine data items:

PC_8096_edge_points_label_bin: which points in the point cloud are edge points;
corner_points_label: which points in the point cloud are corner points;
open_gt_pair_idx: index of two endpoints of each open curve;
open_gt_type: Curve type of each open curve (arc/B-spline/line)
open_gt_res: Residuals between the two endpoints and GT endpoints.
open_gt_sample_points：64 points were sampled for each GT open curve. (Because there is a slight deviation between the endpoints in the input point cloud and the endpoints of the GT curve.)
closed_gt_type: Curve type of each closed curve (In this paper, we include only circles)
closed_gt_sample_points: 64 points were sampled for each GT closed curve.

## Train_stage2 (./Stage2/main/)
    python train_stage_2.py --stage=1
## Test_stage2 (./Stage2/main/)
    python test_stage_2.py --stage=1
    
## Visualization:
    visualization.m: This file is used to visualize the detection results.
    Vis_closed_final.m (./stage2/main/): this file is used for closed curve detection post-precessing
    Vis_open_final.m (./stage2/main/): this file is used for open curve detection post-precessing
    
## Comparison with EC-Net

The following is the complete process of generating training samples（EC-NET）using ABC data (see Figure 4 in EC-NET paper):

1, Extract the mesh information from the existing file（see Figure 4 (a)）

   1.1,  Extract all 'vertices', 'faces'  from '.obj' file

1.2, Extract all 'curves' from '.yml' file
      Fields:  
       sharp: true;   
       type: Bspline/line/Cycle;
       vert_indices: Contains all the vertex indexes that belong to the curve
   Note that in our work, we only consider sharp curve ('sharp: true'). A non-sharp curve ('sharp: false') provided in the '.yml' file, which is not in our consideration.

1.3, For EC-NET annotated edges （see Figure 4 (b)）
    'faces': N_f*9;  each facet is represented by three vertices
    'vertices'：N_v*3; （x,y,z）
    'curves' : N_c*6;  each 'curve' is made up of a large number of small line segments, each represented by two 'vertices'
 
1.4, point cloud sampling  （see Figure 4 (c)）

   1.4.1，'Dense_sample_points' :  According to 'Vertices' and  'Faces' (based on the area of the triangular face) , 100,000 points are sampled for each model;
   1.4.2,  'Sparse_sample_points' :  Using the Farthest Point Sampling (FPS) algorithm to sample 8096 points from 100,000 points.

1.5, generate EC-NET training data  （see Figure 4 (d) (e)）
     
   Finally, using the above entries ('faces', 'vertices', 'curves', 'Sparse_sample_points') as input, you can generate the training data using the '../code/prepare_data.py' file provided by EC-Net.

