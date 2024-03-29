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
conda create -n cwan python=3.6
conda activate cwan

pip install tensorflow-gpu==1.4
conda install cudatoolkit=8.0
conda install cudnn=6.0
conda install scipy
conda install matplotlib
conda install scikit-learn
```

# Datasets

You can download the example dataset from [here](https://pan.baidu.com/s/1-SuuuOFkC-sQ9z8_fq3zZg) (Password: 8i7c), and put in the folder of datasets.

All of the datasets can be downloaded from [here](https://pan.baidu.com/s/1lkSIsNRQJg6i5KffM1mxRg) (Password: 4y1q).

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
@ARTICLE{9530273,
  author={Yao, Yuan and Li, Xutao and Zhang, Yu and Ye, Yunming},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Multisource Heterogeneous Domain Adaptation With Conditional Weighting Adversarial Network}, 
  year={2021},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TNNLS.2021.3105868}}
```
