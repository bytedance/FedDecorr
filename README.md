# Introduction
This Repo contains the **official implementation** of the following paper:

|Venue|Method|Paper Title|
|----|-----|-----|
|ICLR'23|FedDecorr|[Towards Understanding and Mitigating Dimensional Collapse in Heterogeneous Federated Learning](https://arxiv.org/abs/2210.00226)|

and unofficial implementation of the following papers:

|Venue|Method|Paper Title|
|----|-----|-----|
|AISTATS'17|FedAvg|[Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)|
|ArXiv'19|FedAvgM|[Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification](https://arxiv.org/abs/1909.06335)|
|MLSys'20|FedProx|[Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127)|
|NeurIPS'20|FedNova|[Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization](https://arxiv.org/abs/2007.07481)|
|CVPR'21|MOON|[Model-Contrastive Federated Learning](https://arxiv.org/abs/2103.16257)|
|ICLR'21|FedAdagrad/Yogi/Adam|[Adaptive Federated Optimization](https://openreview.net/forum?id=LkFG3lB13U5)|
|KDD'21|FedRS|[FedRS: Federated Learning with Restricted Softmax for Label Distribution Non-IID Data](http://www.lamda.nju.edu.cn/lixc/papers/FedRS-KDD2021-Lixc.pdf)|
|ICML'22|FedLogitCal|[Federated Learning with Label Distribution Skew via Logits Calibration](https://arxiv.org/abs/2209.00189)|
|ICML'22/ECCV'22|FedSAM|[Generalized Federated Learning via Sharpness Aware Minimization](https://arxiv.org/pdf/2206.02618.pdf)/[Improving Generalization in Federated Learning by Seeking Flat Minima](https://arxiv.org/abs/2203.11834)|
|ICLR'23|FedExp|[FedExP: Speeding up Federated Averaging via Extrapolation](https://openreview.net/forum?id=IPrzNbddXV)|


# Dataset preprocessing
***TinyImageNet***:
1) Download the dataset to "data" directory from this link: http://cs231n.stanford.edu/tiny-imagenet-200.zip
2) Unzip the downloaded file under "data" directory.
3) Lastly, to reformat the validation set, under the folder "data/tiny-imagenet-200", run:
```
python3 preprocess_tiny_imagenet.py
```




# Running Instructions
Shell scripts to reproduce experimental results in our paper are under "run\_scripts" folder. Simply changing the "ALPHA" variable to run under different degree of heterogeneity.

Here are commands that replicate our results:

FedAvg on CIFAR10:
```
bash run_scripts/cifar10_fedavg.sh
```

FedAvg + FedDecorr on CIFAR10:
```
bash run_scripts/cifar10_fedavg_feddecorr.sh
```

Experiments on other methods (FedAvgM, FedProx, MOON) and other datasets (CIFAR100, TinyImageNet) follow the similar manner.


# Citation
If you find our repo/paper helpful, please consider citing our work :)
```
@article{shi2022towards,
  title={Towards Understanding and Mitigating Dimensional Collapse in Heterogeneous Federated Learning},
  author={Shi, Yujun and Liang, Jian and Zhang, Wenqing and Tan, Vincent YF and Bai, Song},
  journal={arXiv preprint arXiv:2210.00226},
  year={2022}
}
```


# Contact
Yujun Shi (shi.yujun@u.nus.edu)

# Acknowledgement
Some of our code is borrowed following projects: [MOON](https://github.com/QinbinLi/MOON), [NIID-Bench](https://github.com/Xtra-Computing/NIID-Bench), [SAM(Pytorch)](https://github.com/davda54/sam)

