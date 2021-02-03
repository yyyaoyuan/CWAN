# CWAN

Multi-source Heterogeneous Domain Adaptation with Conditional Weighting Adversarial Network

# Running Environment

## Prerequisites
* python 3.6
* tensorflow-gpu 1.4
* CUDA 8.0
* cudnn 6.0
* numpy
* scipy
* matplotlib
* scikit_learn

## Step-by-step Installation

```
$ conda create -n cwan python=3.6
$ conda activate cwan

$ pip install tensorflow-gpu==1.4
$ conda install cudatoolkit=8.0
$ conda install cudnn=6.0
$ conda install scipy
$ conda install matplotlib
$ conda install scikit-learn
$ conda install matplotlib
```

# Datasets

You can download the example dataset from [here](), and put in the folder of datasets.

# Running

1. You can run this code by inputing: 
```
python -W ignore main.py
```
The results should be close to 59.67 (A (D_{4096}), D (R_{2048}) -> W (S_{800})). Note that different environmental outputs may be different.

2. You can use your datasets by replacing:
```
source_exp = [ad.SAD, ad.SDR]
target_exp = [ad.TWS]
```

3. You can tune the parameters, i.e., lr_1 (learning rate for g(\dot), f(\dot)), lr_2 (learning rate for d(\dot)), T, d, beta, tau, for different applications.

4. The default parameters are: lr_1 = 0.004, lr_2 = 0.001, T = 500, d = 256, beta = 0.03, tau = 0.004.

# Citation

If you find this helpful, please cite:
```
@article{DBLP:journals/corr/abs-2008-02714,
  author    = {Yuan Yao and
               Xutao Li and
               Yu Zhang and
               Yunming Ye},
  title     = {Multi-source Heterogeneous Domain Adaptation with Conditional Weighting
               Adversarial Network},
  journal   = {CoRR},
  volume    = {abs/2008.02714},
  year      = {2020},
  url       = {https://arxiv.org/abs/2008.02714},
  archivePrefix = {arXiv},
}
```
