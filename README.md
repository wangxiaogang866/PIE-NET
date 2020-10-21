# PIE-NET: Parametric Inference of Point Cloud Edges

This is the code repository for "PIE-NET: Parametric Inference of Point Cloud Edges”


## Prerequisites: 
    Numpy (ver. 1.13.3)
    TensorFlow (ver. 1.4.1)
    scipy (ver. 0.19.1)
    Matlab (ver. 2015a)   


## Train：

    to train edge and corner points detection module

    python train_stage_1 --stage=1

    to train curve proposal generation module

    python train_stage_2_3 --stage=1


## Test：

    to test edge and corner points detection module

    python test_stage_1 --stage=1

    to test curve proposal generation module

    python test_stage_2_3 --stage=1
    
## Visualization:
    visualization.m: This file is used to visualize the detection results.
