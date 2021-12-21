

# Deep co-supervision and attention fusion strategy for automatic COVID-19 lung infection segmentation on CT images

> **The Paper Links**: [Pattern Recognition](https://www.sciencedirect.com/science/article/pii/S0031320321006282), [Arxiv](https://arxiv.org/abs/2112.10368).   
> **Authors:** [Haigen Hu](), [Leizhao Shen](), [Qiu Guan](), [Xiaoxin Li](), [Qianwei Zhou](), [Su Ruan]()

## Abstract
Due to the irregular shapes,various sizes and indistinguishable boundaries between the normal and infected tissues, it is still a challenging task to accurately segment the infected lesions of COVID-19 on CT images. In this paper, a novel segmentation scheme is proposed for the infections of COVID-19 by enhancing supervised information and fusing multi-scale feature maps of different levels based on the encoder-decoder architecture. To this end, a deep collaborative supervision (Co-supervision) scheme is proposed to guide the network learning the features of edges and semantics. More specifically, an Edge Supervised Module (ESM) is firstly designed to highlight low-level boundary features by incorporating the edge supervised information into the initial stage of down-sampling. Meanwhile, an Auxiliary Semantic Supervised Module (ASSM) is proposed to strengthen high-level semantic information by integrating mask supervised information into the later stage. Then an Attention Fusion Module (AFM) is developed to fuse multiple scale feature maps of different levels by using an attention mechanism to reduce the semantic gaps between high-level and low-level feature maps. Finally, the effectiveness of the proposed scheme is demonstrated on four various COVID-19 CT datasets. The results show that the proposed three modules are all promising. Based on the baseline (ResUnet), using ESM, ASSM, or AFM alone can respectively increase Dice metric by 1.12%, 1.95%,1.63% in our dataset, while the integration by incorporating three models together can rise 3.97%. Compared with the existing approaches in various datasets, the proposed method can obtain better segmentation performance in some main metrics, and can achieve the best generalization and comprehensive performance.

## Usage

You can build a runtime environment and prepare your dataset by following these steps：

- **Dataset Preparation:**

  Firstly, you should download the training/testing set ([COVID-19 - Medical segmentation_Link1](https://medicalsegmentation.com/covid19/), [COVID-19 CT Lung and Infection Segmentation Dataset | Zenodo_Link2](https://zenodo.org/record/3757476#.Xpz8OcgzZPY))

  > [1] COVID-19 CT segmentation dataset, 2020, https://medicalsegmentation.com/covid19.  
  > [2] COVID-19 CT segmentation dataset, 2020, https://gitee.com/junma11/COVID-19-CT-Seg-Benchmark.

  and put it into `./data/` repository.

- **Pretrained Model:**

  There is no pretrained model used in our paper. If you want to use pretrained model, put them into `./data/pretrained_models/` repository.

- **Configuring your environment (Prerequisites):**

  Note that our model is only tested on Ubuntu OS 16.04 with the following environments (CUDA-10.2). 
  It may work on other operating systems as well but we do not guarantee that it will.

  + Creating a virtual environment in terminal: `conda create -n ResUNet python=3.7`, and then run `conda activate ResUNet`.
  + Installing the pytorch environment: `conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.2 -c pytorch` 
  + Installing necessary packages: `pip install -r requirements.txt`.

  

Create a new folder `data` and `checkpoints`，then the directory structure is as follows:

```
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

```
batch_size
learning_rate
weight_decay
checkpoint_save_dir
loss_function
...
```

Finally, run `train.py`, the model will saved in `./checkpoints` folder.



## 0. Preface

- This repository provides code for "_**Deep co-supervision and attention fusion strategy for automatic COVID-19 lung infection segmentation on CT images**_". ([Pattern Recognition](https://www.sciencedirect.com/science/article/pii/S0031320321006282), [Arxiv](https://arxiv.org/abs/2112.10368))

- If you have any questions about our paper, feel free to contact us. And if you are using the methods proposed in this paper for your research, please cite this paper ([BibTeX](#5-citation)).

### 0.1. Table of Content

- [Deep co-supervision and attention fusion strategy for automatic COVID-19 lung infection segmentation on CT images](#deep-co-supervision-and-attention-fusion-strategy-for-automatic-covid-19-lung-infection-segmentation-on-ct-images)
  * [Abstract](#abstract)
  * [Usage](#usage)
  * [0. Preface](#0-preface)
    + [0.1. Table of Content](#01-table-of-content)
  * [1. Introduction](#1-introduction)
    + [1.1. Task Descriptions](#11-task-descriptions)
    + [1.2. Architecture](#12-architecture)
  * [2. Proposed Methods](#2-proposed-methods)
  * [3. Experiments](#3-experiments)
  * [4. Visualized Results](#4-visualized-results)
  * [5. Citation](#5-citation)
  * [6. Acknowledgements](#6-acknowledgements)
  * [7. FAQ](#7-faq)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

## 1. Introduction

### 1.1. Task Descriptions

<p align="center">
    <img src="https://raw.githubusercontent.com/HuHaigen/COVID-19-Lung-Infection-Segentation/main/resources/Fig1.png"/>
    <em>
      Fig. 1. An illustration of challenging task for identification the infected lesions (contours in red) of COVID-19 on CT images. (a) The infections have various scales and shapes. (b) There is no obvious difference between normal and infected tissues. (For interpretation of the references to color in this figure legend, the reader is referred to the web version of this article.)
    </em>
</p>



### 1.2. Architecture

<p align="center">
  <img src="https://raw.githubusercontent.com/HuHaigen/COVID-19-Lung-Infection-Segentation/main/resources/Fig2.png"></img>
	<em>
	Fig. 2. An Illustration of the overall network architecture. The proposed architecture comprises of ASSM, ESM and AFM based on encoder-decoder structure.
	</em>
</p>

where (1) **ESM** is used to further highlight the low-level features in the initial shallow layers of the encoder, and it can capture more detailed information like object boundaries. (2) While **ASSM** is employed to strengthen high-level semantic information by integrating object mask supervised information into the later stages of the encoder. (3) Finally, **AFM** is utilized to fuse multi-scale feature maps of different levels in the decoder.



## 2. Proposed Methods

Our proposed methods consist of three individual components under three different settings: 

- ESM (Edge supervised module)
- ASSM (Auxiliary semantic supervised module)
- AFM (Attention fusion module)

Please refer to the paper ([Link](https://arxiv.org/pdf/2112.10368.pdf)) for details.


## 3. Experiments

We have done a series of qualitative and quantitative experimental comparisons on our proposed method, please refer to the paper ([Link](https://arxiv.org/pdf/2112.10368.pdf)) for the specific experimental results.

## 4. Visualized Results

<p align="center">
    <img src="https://raw.githubusercontent.com/HuHaigen/COVID-19-Lung-Infection-Segentation/main/resources/Fig6.png"/> <br />
    <em> 
    Fig. 6. Visual qualitative comparison of lung infection segmentation results among the different methods.
    </em>
</p>

<p align="center">
    <img src="https://raw.githubusercontent.com/HuHaigen/COVID-19-Lung-Infection-Segentation/main/resources/Fig7.png"/> <br />
    <em> 
    Fig. 7. Visualization of each stage supervised by <strong>ESM</strong>.
    </em>
</p>

<p align="center">
    <img src="https://raw.githubusercontent.com/HuHaigen/COVID-19-Lung-Infection-Segentation/main/resources/Fig8.png"/> <br />
    <em> 
    Fig. 8. Visual results of the fusion process based on the proposed <strong>AFM</strong>.
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

The authors would like to express their appreciation to the referees for their helpful comments and suggestions. This work was supported in part by Zhejiang Provincial Natural Science Foundation of China (Grant nos. LGF20H180002 and GF22F037921), and in part by National Natural Science Foundation of China (Grant nos. 61802347, 61801428 and 61972354), the National Key Research and Development Program of China (Grant no.2018YFB1305202), and the Microsystems Technology Key Laboratory Foundation of China.

## 7. FAQ

**[⬆ back to top](#0-preface)**
