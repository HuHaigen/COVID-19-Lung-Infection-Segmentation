# Deep Co-supervision and Attention Fusion Strategy for Automatic COVID-19 Lung Infection Segmentation on CT Images
Authors: Haigen Hu, Leizhao Shen, Qiu Guan, Xiaoxin Li, Qianwei Zhou, Su Ruan

# Abstract
Due to the irregular shapes,various sizes and indistinguishable boundaries between the normal and infected tissues, it is still a challenging task to accurately segment the infected lesions of COVID-19 on CT images. In this paper, a novel segmentation scheme is proposed for the infections of COVID-19 by enhancing supervised information and fusing multi-scale feature maps of different levels based on the encoder-decoder architecture. To this end, a deep collaborative supervision (Co-supervision) scheme is proposed to guide the network learning the features of edges and semantics. More specifically, an Edge Supervised Module (ESM) is firstly designed to highlight low-level boundary features by incorporating the edge supervised information into the initial stage of down-sampling. Meanwhile, an Auxiliary Semantic Supervised Module (ASSM) is proposed to strengthen high-level semantic information by integrating mask supervised information into the later stage. Then an Attention Fusion Module (AFM) is developed to fuse multiple scale feature maps of different levels by using an attention mechanism to reduce the semantic gaps between high-level and low-level feature maps. Finally, the effectiveness of the proposed scheme is demonstrated on four various COVID-19 CT datasets. The results show that the proposed three modules are all promising. Based on the baseline (ResUnet), using ESM, ASSM, or AFM alone can respectively increase Dice metric by 1.12%, 1.95%,1.63% in our dataset, while the integration by incorporating three models together can rise 3.97%. Compared with the existing approaches in various datasets, the proposed method can obtain better segmentation performance in some main metrics, and can achieve the best generalization and comprehensive performance.


# Usage

you can train your own dataset <br>
creat a new folder 'data' and 'checkpoint'，then the directory structure is as follows:<br>
-checkpoint <br>
-data <br>
``` <br>
-train   (train dataset)<br>
    -image    (CT images)<br>
    -mask     (GT images)<br>
    -mask_    (EdgeEGT images)<br>
-val        (validation dataset)<br>
    -image    (CT images)<br>
    -mask     (GT images)<br>
    -mask_    (EdgeEGT images)<br>
```
You can modify the parameter settings in /resources/train_config.yaml<br>
-batch_size<br>
-learning_rate<br>
-weight_decay<br>
-checkpoint_save_dir<br>
-loss_function<br>
...<br>
finally run train.py, the model will saved in checkpoint folder.<br>

# Citation
Please cite our paper if you find the work useful:

@article{hu2022deep,<br>
    author = {Haigen Hu and Leizhao Shen and Qiu Guan and Xiaoxin Li and Qianwei Zhou and Su Ruan},<br>
    journal = {Pattern Recognition},<br>
    title = {Deep co-supervision and attention fusion strategy for automatic COVID-19 lung infection segmentation on CT images},<br>
    year = {2022},<br>
    volume = {124},<br>
    pages = {108452},<br>
    doi = {https://doi.org/10.1016/j.patcog.2021.108452},<br>
}<br>
