# Deep co-supervision and attention fusion strategy for automatic COVID-19 lung infection segmentation on CT images

> **Authors:** 
> [Haigen Hu](https://dpfan.net/), 
> [Leizhao Shen](https://taozh2017.github.io/), 
> [Qiu Guan](https://scholar.google.com/citations?user=oaxKYKUAAAAJ&hl=en), 
> [Xiaoxin Li](https://scholar.google.com/citations?hl=zh-CN&user=EnDCJKMAAAAJ), 
> [Qianwei Zhou](https://www.researchgate.net/profile/Geng_Chen13) and
> [Su Ruan](http://hzfu.github.io/)

## 0. Preface

- This repository provides code for "_**Deep co-supervision and attention fusion strategy for automatic COVID-19 lung infection segmentation on CT images**_". 
([arXiv Pre-print](https://arxiv.org/abs/2004.14133) & [medrXiv](https://www.medrxiv.org/content/10.1101/2020.04.22.20074948v1) & [中译版](http://dpfan.net/wp-content/uploads/TMI20_InfNet_Chinese_Finalv2.pdf) & [Patter Recognition](https://www.sciencedirect.com/science/article/pii/S0031320321006282))

- If you have any questions about our paper, feel free to contact us. And if you are using COVID-SemiSeg Dataset, 
Inf-Net or evaluation toolbox for your research, please cite this paper ([BibTeX](#8-citation)).
- We elaborately collect COVID-19 imaging-based AI research papers and datasets [awesome-list](https://github.com/HzFu/COVID19_imaging_AI_paper_list).

### 0.1. Table of Contents

- [Inf-Net: Automatic COVID-19 Lung Infection Segmentation from CT Images](#inf-net--automatic-covid-19-lung-infection-segmentation-from-ct-scans)
  * [0. Preface](#0-preface)
    + [0.1. :fire: NEWS :fire:](#01--fire--news--fire-)
    + [0.2. Table of Contents](#02-table-of-contents)
  * [1. Introduction](#1-introduction)
    + [1.1. Task Description](#11-task-description)
  * [2. Proposed Methods](#2-proposed-methods)
    + [2.1. Inf-Net](#21-inf-net)
      - [2.1.1 Overview](#211-overview)
      - [2.1.2. Usage](#212-usage)
    + [2.2. Semi-Inf-Net](#22-semi-inf-net)
      - [2.2.1. Overview](#221-overview)
      - [2.2.2. Usage](#222-usage)
    + [2.3. Semi-Inf-Net + Multi-class UNet](#23-semi-inf-net---multi-class-unet)
      - [2.3.1. Overview](#231-overview)
      - [2.3.2. Usage](#232-usage)
  * [3. Evaluation Toolbox](#3-evaluation-toolbox)
    + [3.1. Introduction](#31-introduction)
    + [3.2. Usage](#32-usage)
  * [4. COVID-SemiSeg Dataset](#4-covid-semiseg-dataset)
    + [3.1. Training set](#31-training-set)
    + [3.2. Testing set](#32-testing-set)
  * [4. Results](#4-results)
    + [4.1. Download link:](#41-download-link-)
  * [5. Visualization Results:](#5-visualization-results-)
  * [6. Paper list of COVID-19 related (Update continue)](#6-paper-list-of-covid-19-related--update-continue-)
  * [7. Manuscript](#7-manuscript)
  * [8. Citation](#8-citation)
  * [9. LICENSE](#9-license)
  * [10. Acknowledgements](#10-acknowledgements)
  * [11. TODO LIST](#11-todo-list)
  * [12. FAQ](#12-faq)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

## 1. Introduction

### 1.1. Task Descriptions

<p align="center">
    <img src="http://dpfan.net/wp-content/uploads/COVID19-Infection-1.png"/> <br />
    <em>
      Fig. 1. An illustration of challenging task for identification the infected lesions (contours in red) of COVID-19 on CT images. (a) The infections have various scales and shapes. (b) There is no obvious difference between normal and infected tissues. (For interpretation of the references to color in this figure legend, the reader is referred to the web version of this article.)
    </em>
</p>

## 2. Proposed Methods

- **Preview:**

    Our proposed methods consist of three individual components under three different settings: 

    - ESM (Edge supervised module)
    
    - ASSM (Auxiliary semantic supervised module)
    
    - AFM (Attention fusion module)
   
- **Dataset Preparation:**

    Firstly, you should download the training/testing set ([COVID-19 - Medical segmentation_Link1](https://medicalsegmentation.com/covid19/), [COVID-19 CT Lung and Infection Segmentation Dataset | Zenodo_Link2](https://zenodo.org/record/3757476#.Xpz8OcgzZPY))

    > [32] “COVID-19 CT segmentation dataset, 2020, https://medicalsegmentation.com/covid19/. 
>
    > [33] “COVID-19 CT segmentation dataset, 2020, https://gitee.com/junma11/COVID-19-CT-Seg-Benchmark.

    and put it into `./data/` repository.

- **Pretrained Model:**

    There is no pretrained model used in our paper. If you want to use pretrained model, put them into `./data/pretrained_models/` repository.
    
- **Configuring your environment (Prerequisites):**

    Note that Inf-Net series is only tested on Ubuntu OS 16.04 with the following environments (CUDA-10.2). 
    It may work on other operating systems as well but we do not guarantee that it will.
    
    + Creating a virtual environment in terminal: `conda create -n InfNet python=3.7`.
    + Installing the pytorch environment: `conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.2 -c pytorch` 
    + Installing necessary packages: `pip install -r requirements.txt`.
    

### 2.1. Inf-Net

#### 2.1.1 Overview

<p align="center">
    <img src="http://dpfan.net/wp-content/uploads/Inf-Net.png"/> <br />
    <em> 
    Figure 2. The architecture of our proposed Inf-Net model, which consists of three reverse attention 
    (RA) modules connected to the paralleled partial decoder (PPD).
    </em>
</p>

#### 2.1.2. Usage

1. Train

    - We provide multiple backbone versions (see this [line](https://github.com/DengPingFan/Inf-Net/blob/0df52ec8bb75ef468e9ceb7c43a41b0886eced3a/MyTrain_LungInf.py#L133)) in the training phase, i.e., ResNet, Res2Net, and VGGNet, but we only provide the Res2Net version in the Semi-Inf-Net. Also, you can try other backbones you prefer to, but the pseudo labels should be RE-GENERATED with corresponding backbone.
    
    - Turn off the semi-supervised mode (`--is_semi=False`) turn off the flag of whether use pseudo labels (`--is_pseudo=False`) in the parser of `MyTrain_LungInf.py` and just run it! (see this [line](https://github.com/DengPingFan/Inf-Net/blob/0df52ec8bb75ef468e9ceb7c43a41b0886eced3a/MyTrain_LungInf.py#L121))

1. Test
   
    - When training is completed, the weights will be saved in `./Snapshots/save_weights/Inf-Net/`. 
    You can also directly download the pre-trained weights from [Google Drive](https://drive.google.com/open?id=19p_G8NS4NwF4pZOEOX3w06ZLVh0rj3VW).
    
    - Assign the path `--pth_path` of trained weights and `--save_path` of results save and in `MyTest_LungInf.py`.
    
    - Just run it and results will be saved in `./Results/Lung infection segmentation/Inf-Net`

### 2.2. Semi-Inf-Net

#### 2.2.1. Overview

<p align="center">
    <img src="http://dpfan.net/wp-content/uploads/Semi-InfNet.png"/> <br />
    <em> 
    Figure 3. Overview of the proposed Semi-supervised Inf-Net framework.
    </em>
</p>

#### 2.2.2. Usage

1. Data Preparation for pseudo-label generation. (Optional)
   
    - Dividing the 1600 unlabeled image into 320 groups (1600/K groups, we set K=5 in our implementation), 
      in which images with `*.jpg` format can be found in `./Dataset/TrainingSet/LungInfection-Train/Pseudo-label/Imgs/`. 
      (I suppose you have downloaded all the train/test images following the instructions [above](#2-proposed-methods))
      Then you only just run the code stored in `./SrcCode/utils/split_1600.py` to split it into multiple sub-dataset, 
      which are used in the training process of pseudo-label generation. The 1600/K sub-datasets will be saved in 
      `./Dataset/TrainingSet/LungInfection-Train/Pseudo-label/DataPrepare/Imgs_split/`
      
    - **You can also skip this process and download them from [Google Drive](https://drive.google.com/open?id=1Rn_HhgTEhy-M7qmPVspdJiamWC_kueFJ) that is used in our implementation.**

1. Generating Pseudo Labels (Optional)

    - After preparing all the data, just run `PseudoGenerator.py`. It may take at least day and a half to finish the whole generation.
    
    - **You can also skip this process and download intermediate generated file from [Google Drive](https://drive.google.com/open?id=1Rn_HhgTEhy-M7qmPVspdJiamWC_kueFJ) that is used in our implementation.**
    
    - When training is completed, the images with pseudo labels will be saved in `./Dataset/TrainingSet/LungInfection-Train/Pseudo-label/`.

1. Train

    - Firstly, turn off the semi-supervised mode (`--is_semi=False`) and turn on the flag of whether using pseudo labels
     (`--is_pseudo=True`) in the parser of `MyTrain_LungInf.py` and modify the path of training data to the pseudo-label 
      repository (`--train_path='Dataset/TrainingSet/LungInfection-Train/Pseudo-label'`). Just run it!
    
    - When training is completed, the weights (trained on pseudo-label) will be saved in `./Snapshots/save_weights/Inf-Net_Pseduo/Inf-Net_pseudo_100.pth`. Also, you can directly download the pre-trained weights from [Google Drive](https://drive.google.com/open?id=1NMM0BoVJU9DS8u4yTG0L-V_mcHKajrmN). Now we have prepared the weights that is pre-trained on 1600 images with pseudo labels. Please note that these valuable images/labels can promote the performance and the stability of training process, because of ImageNet pre-trained models are just design for general object classification/detection/segmentation tasks initially.
    
    - Secondly, turn on the semi-supervised mode (`--is_semi=True`) and turn off the flag of whether using pseudo labels
     (`--is_pseudo=False`) in the parser of `MyTrain_LungInf.py` and modify the path of training data to the doctor-label (50 images)
      repository (`--train_path='Dataset/TrainingSet/LungInfection-Train/Doctor-label'`). Just run it.

1. Test
   
    - When training is completed, the weights will be saved in `./Snapshots/save_weights/Semi-Inf-Net/`. 
    You also can directly download the pre-trained weights from [Google Drive](https://drive.google.com/open?id=1iHMQ9bjm4-qZaZZFXigZ-iruZOSB9mty).
    
    - Assign the path `--pth_path` of trained weights and `--save_path` of results save and in `MyTest_LungInf.py`.
    
    - Just run it! And results will be saved in `./Results/Lung infection segmentation/Semi-Inf-Net`.

### 2.3. Semi-Inf-Net + Multi-class UNet

#### 2.3.1. Overview

Here, we provide a general and simple framework to address the multi-class segmentation problem. We modify the 
original design of UNet that is used for binary segmentation, and thus, we name it as _Multi-class UNet_. 
More details can be found in our paper.

<p align="center">
    <img src="http://dpfan.net/wp-content/uploads/MultiClassInfectionSeg.png"/> <br />
    <em> 
    Figure 3. Overview of the proposed Semi-supervised Inf-Net framework.
    </em>
</p>

#### 2.3.2. Usage

1. Train

    - Just run `MyTrain_MulClsLungInf_UNet.py`
    
    - Note that `./Dataset/TrainingSet/MultiClassInfection-Train/Prior` is just borrowed from `./Dataset/TestingSet/LungInfection-Test/GT/`, 
    and thus, two repositories are equally.

1. Test
   
    - When training is completed, the weights will be saved in `./Snapshots/save_weights/Semi-Inf-Net_UNet/`. Also, you can directly download the pre-trained weights from [Google Drive](https://drive.google.com/open?id=1V4VJl5X4kJVD6HHWYsGzbrbPh8Q_Vc0I).
    
    - Assigning the path of weights in parameters `snapshot_dir` and run `MyTest_MulClsLungInf_UNet.py`. All the predictions will be saved in `./Results/Multi-class lung infection segmentation/Consolidation` and `./Results/Multi-class lung infection segmentation/Ground-glass opacities`.

## 3. Evaluation Toolbox

### 3.1. Introduction

We provide one-key evaluation toolbox for LungInfection Segmentation tasks, including Lung-Infection and Multi-Class-Infection. 
Please download the evaluation toolbox [Google Drive](https://drive.google.com/open?id=1BGUUmrRPOWPxdxnawFnG9TVZd8rwLqCF).

### 3.2. Usage

- Prerequisites: MATLAB Software (Windows/Linux OS is both works, however, we suggest you test it in the Linux OS for convenience.)

- run `cd ./Evaluation/` and `matlab` open the Matlab software via terminal

- Just run `main.m` to get the overall evaluation results.

- Edit the parameters in the `main.m` to evaluate your custom methods. Please refer to the instructions in the `main.m`.

## 4. COVID-SemiSeg Dataset

We also build a semi-supervised COVID-19 infection segmentation (**COVID-SemiSeg**) dataset, with 100 labelled CT scans 
from the COVID-19 CT Segmentation dataset [1] and 1600 unlabeled images from the COVID-19 CT Collection dataset [2]. 
Our COVID-SemiSeg Dataset can be downloaded at [Google Drive](https://drive.google.com/open?id=1bbKAqUuk7Y1q3xsDSwP07oOXN_GL3SQM).

> [1]“COVID-19 CT segmentation dataset,” https://medicalsegmentation.com/covid19/, accessed: 2020-04-11.
> [2]J. P. Cohen, P. Morrison, and L. Dao, “COVID-19 image data collection,” arXiv, 2020.


### 3.1. Training set

1. Lung infection which consists of 50 labels by doctors (Doctor-label) and 1600 pseudo labels generated (Pseudo-label) 
by our Semi-Inf-Net model. [Download Link](http://dpfan.net/wp-content/uploads/LungInfection-Train.zip).

1. Multi-Class lung infection which also composed of 50 multi-class labels (GT) by doctors and 50 lung infection 
labels (Prior) generated by our Semi-Inf-Net model. [Download Link](http://dpfan.net/wp-content/uploads/MultiClassInfection-Train.zip).


### 3.2. Testing set

1. The Lung infection segmentation set contains 48 images associate with 48 GT. [Download Link](http://dpfan.net/wp-content/uploads/LungInfection-Test.zip).

1. The Multi-Class lung infection segmentation set has 48 images and 48 GT. [Download Link](http://dpfan.net/wp-content/uploads/MultiClassInfection-Test.zip).

1. The download link ([Google Drive](https://drive.google.com/file/d/1WHHIJ1R6DRV9maUD6qKT_-aiDktKLd0q/view?usp=sharing)) of our 638-dataset, which is used in Table.V of our paper.

== Note that ==: In our manuscript, we said that the total testing images are 50. However, we found there are two images with very small resolution and black ground-truth. Thus, we discard these two images in our testing set. The above link only contains 48 testing images. 


## 4. Results

To compare the infection regions segmentation performance, we consider the two state-of-the-art models U-Net and U-Net++. 
We also show the multi-class infection labelling results in Fig. 5. As can be observed, 
our model, Semi-Inf-Net & FCN8s, consistently performs the best among all methods. It is worth noting that both GGO and 
consolidation infections are accurately segmented by Semi-Inf-Net & FCN8s, which further demonstrates the advantage of 
our model. In contrast, the baseline methods, DeepLabV3+ with different strides and FCNs, all obtain unsatisfactory 
results, where neither GGO and consolidation infections can be accurately segmented.

Overall results can be downloaded from this [link](http://dpfan.net/wp-content/uploads/COVID-SemiSeg-Results.zip).

### 4.1. Download link:

Lung infection segmentation results can be downloaded from this [link](http://dpfan.net/wp-content/uploads/Lung-infection-segmentation.zip)

Multi-class lung infection segmentation can be downloaded from this [link](https://drive.google.com/file/d/1mIA9ggiftwhdzSAMl2sIAl3rkupt2_AY/view?usp=sharing)

## 5. Visualization Results:

<p align="center">
    <img src="http://dpfan.net/wp-content/uploads/InfectionSeg.png"/> <br />
    <em> 
    Figure 4. Visual comparison of lung infection segmentation results.
    </em>
</p>

<p align="center">
    <img src="http://dpfan.net/wp-content/uploads/MultiClassInfectionSeg.png"/> <br />
    <em> 
    Figure 5. Visual comparison of multi-class lung infection segmentation results, where the red and green labels 
    indicate the GGO and consolidation, respectively.
    </em>
</p>

## 6. Paper list of COVID-19 related (Update continue)

> Ori GitHub Link: https://github.com/HzFu/COVID19_imaging_AI_paper_list

<p align="center">
    <img src="http://dpfan.net/wp-content/uploads/paper-list-cover.png"/> <br />
    <em> 
    Figure 6. This is a collection of COVID-19 imaging-based AI research papers and datasets.
    </em>
</p>

## 7. Manuscript
https://arxiv.org/pdf/2004.14133.pdf

## 8. Citation

Please cite our paper if you find the work useful: 

	@article{hu2022deep,
		author = {Haigen Hu and Leizhao Shen and Qiu Guan and Xiaoxin Li and Qianwei Zhou and Su Ruan},
		journal = {Pattern Recognition},
		title = {Deep co-supervision and attention fusion strategy for automatic COVID-19 lung infection segmentation on CT images},
		year = {2022},
		volume = {124},
		pages = {108452},
		doi = {https://doi.org/10.1016/j.patcog.2021.108452},
	}

 

## 9. LICENSE

- The **COVID-SemiSeg Dataset** is made available for non-commercial purposes only. Any comercial use should get formal permission first.

- You will not, directly or indirectly, reproduce, use, or convey the **COVID-SemiSeg Dataset** 
or any Content, or any work product or data derived therefrom, for commercial purposes.


## 10. Acknowledgements

We would like to thank the whole organizing committee for considering the publication of our paper in this special issue (Special Issue on Imaging-Based Diagnosis of COVID-19) of IEEE Transactions on Medical Imaging. More papers refer to [Link](https://ieeexplore.ieee.org/xpl/tocresult.jsp?isnumber=9153182).



## 12. FAQ

1. If the image cannot be loaded in the page (mostly in the domestic network situations).

    [Solution Link](https://blog.csdn.net/weixin_42128813/article/details/102915578)

2. I tested the U-Net, however, the Dice score is different from the score in TABLE II (Page 8 on our manuscript)? <br>
   Note that, the our Dice score is the mean dice score rather than the max Dice score. You can use our evaluation tool box [Google Drive](https://drive.google.com/open?id=1BGUUmrRPOWPxdxnawFnG9TVZd8rwLqCF). 
   The training set of each compared model (e.g., U-Net, Attention-UNet, Gated-UNet, Dense-UNet, U-Net++, Inf-Net (ours)) is the 48 images rather than 48 image+1600 images.

**[⬆ back to top](#0-preface)**









# COVID-19-Lung-Infection-Segentation

Due to the irregular shapes,various sizes and indistinguishable boundaries between the normal and infected tissues, it is still a challenging task to accurately segment the infected lesions of COVID-19 on CT images. In this paper, a novel segmentation scheme is proposed for the infections of COVID-19 by enhancing supervised information and fusing multi-scale feature maps of different levels based on the encoder-decoder architecture. To this end, a deep collaborative supervision (Co-supervision) scheme is proposed to guide the network learning the features of edges and semantics. More specifically, an Edge Supervised Module (ESM) is firstly designed to highlight low-level boundary features by incorporating the edge supervised information into the initial stage of down-sampling. Meanwhile, an Auxiliary Semantic Supervised Module (ASSM) is proposed to strengthen high-level semantic information by integrating mask supervised information into the later stage. Then an Attention Fusion Module (AFM) is developed to fuse multiple scale feature maps of different levels by using an attention mechanism to reduce the semantic gaps between high-level and low-level feature maps. Finally, the effectiveness of the proposed scheme is demonstrated on four various COVID-19 CT datasets. The results show that the proposed three modules are all promising. Based on the baseline (ResUnet), using ESM, ASSM, or AFM alone can respectively increase Dice metric by 1.12%, 1.95%,1.63% in our dataset, while the integration by incorporating three models together can rise 3.97%. Compared with the existing approaches in various datasets, the proposed method can obtain better segmentation performance in some main metrics, and can achieve the best generalization and comprehensive performance.

# COVID19

you can train your own dataset

creat a new folder 'data' and 'checkpoint'，then the directory structure is as follows:

-checkpoint

-data

```
-train   (train dataset)
    -image    (CT images)
    -mask     (GT images)
    -mask_    (EdgeEGT images)
-val        (validation dataset)
    -image    (CT images)
    -mask     (GT images)
    -mask_    (EdgeEGT images)
```

You can modify the parameter settings in /resources/train_config.yaml

-batch_size

-learning_rate

-weight_decay

-checkpoint_save_dir

-loss_function
...

finally run train.py, the model will saved in checkpoint folder.

