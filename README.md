
# Deep co-supervision and attention fusion strategy for automatic COVID-19 lung infection segmentation on CT images

> **Authors:** [Haigen Hu](), [Leizhao Shen](), [Qiu Guan](), [Xiaoxin Li](), [Qianwei Zhou](), [Su Ruan]()

## Abstract
Due to the irregular shapes,various sizes and indistinguishable boundaries between the normal and infected tissues, it is still a challenging task to accurately segment the infected lesions of COVID-19 on CT images. In this paper, a novel segmentation scheme is proposed for the infections of COVID-19 by enhancing supervised information and fusing multi-scale feature maps of different levels based on the encoder-decoder architecture. To this end, a deep collaborative supervision (Co-supervision) scheme is proposed to guide the network learning the features of edges and semantics. More specifically, an Edge Supervised Module (ESM) is firstly designed to highlight low-level boundary features by incorporating the edge supervised information into the initial stage of down-sampling. Meanwhile, an Auxiliary Semantic Supervised Module (ASSM) is proposed to strengthen high-level semantic information by integrating mask supervised information into the later stage. Then an Attention Fusion Module (AFM) is developed to fuse multiple scale feature maps of different levels by using an attention mechanism to reduce the semantic gaps between high-level and low-level feature maps. Finally, the effectiveness of the proposed scheme is demonstrated on four various COVID-19 CT datasets. The results show that the proposed three modules are all promising. Based on the baseline (ResUnet), using ESM, ASSM, or AFM alone can respectively increase Dice metric by 1.12%, 1.95%,1.63% in our dataset, while the integration by incorporating three models together can rise 3.97%. Compared with the existing approaches in various datasets, the proposed method can obtain better segmentation performance in some main metrics, and can achieve the best generalization and comprehensive performance.

## Usage

You can train your own dataset.

Create a new folder `data` and `checkpoints`，then the directory structure is as follows:

```shell
./COVID-19-Lung-Infection-Segentation
├── checkpoints
├── data
│   ├── COVID19
│   │   ├── train       (1. Trainning Dataset)
│   │   │   ├── image           (CT images)
│   │   │   ├── mask            (GT images)
│   │   │   └── mask_           (EdgeEGT images)
│   │   └── val         (2. Validating Dataset)
│   │       ├── image           (CT images)
│   │       ├── mask            (GT images)
│   │       └── mask_           (EdgeEGT images)
│   └── pretrained_models
├── datasets
├── losses
├── models
├── resources
└── trainer
```
You can modify the parameter settings in `/resources/train_config.yaml`
```yaml
batch_size
learning_rate
weight_decay
checkpoint_save_dir
loss_function
...
```
finally run `train.py`, the model will saved in `./checkpoints` folder.

**[⬆ back to top](#0-preface)**

## 0. Preface

- This repository provides code for "_**Deep co-supervision and attention fusion strategy for automatic COVID-19 lung infection segmentation on CT images**_". ([Pattern Recognition](https://www.sciencedirect.com/science/article/pii/S0031320321006282))

- If you have any questions about our paper, feel free to contact us. And if you are using the methods proposed in this paper for your research, please cite this paper ([BibTeX](#8-citation)).

<!-- 
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

-->

- [Deep co-supervision and attention fusion strategy for automatic COVID-19 lung infection segmentation on CT images](#deep-co-supervision-and-attention-fusion-strategy-for-automatic-covid-19-lung-infection-segmentation-on-ct-images)
  * [Abstract](#abstract)
  * [0. Preface](#0-preface)
    + [0.1. Table of Contents](#01-table-of-contents)
  * [1. Introduction](#1-introduction)
    + [1.1. Task Descriptions](#11-task-descriptions)
  * [2. Proposed Methods](#2-proposed-methods)
    + [2.1. ESM (Edge supervised module)](#21-esm--edge-supervised-module-)
    + [2.2. ASSM (Auxiliary semantic supervised module)](#22-assm--auxiliary-semantic-supervised-module-)
    + [2.3. AFM (Attention fusion module)](#23-afm--attention-fusion-module-)
  * [3. Experiments](#3-experiments)
  * [4. Visualized Results:](#4-visualized-results-)
  * [5. Citation](#5-citation)
  * [6. Acknowledgements](#6-acknowledgements)
  * [7. FAQ](#7-faq)
  * [Usage](#usage)

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
    

### 2.1. ESM (Edge supervised module)

### 2.2. ASSM (Auxiliary semantic supervised module)

### 2.3. AFM (Attention fusion module)


## 3. Experiments

## 4. Visualized Results

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



## 5. Citation

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


## 6. Acknowledgements

The authors would like to express their appreciation to the referees for their helpful comments and suggestions. This work was supported in part by Zhejiang Provincial Natural Science Foundation of China (Grant nos. LGF20H180002 and GF22F037921), and in part by National Natural Science Foundation of China (Grant nos. 61802347, 61801428 and 61972354), the National Key Research and Development Program of China (Grant no.2018YFB1305202), and the Microsystems Technology Key Laboratory Foundation of China

## 7. FAQ

