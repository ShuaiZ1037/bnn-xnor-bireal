### BNN-XNOR-BiReal
This is a project contains Pytorch implementations of the BNN*, XNOR-Net and BiReal-Net.
(NOTE: the first and last layer of BNN* is floating point.)

---
#### Results 
| **Method**   | **Acc**  | 
|:--:|:--:|
|   BNN  | 71.70%  |
| XNOR  | 80.77%  |
| BiReal  | 83.17% |

#### Dependencies
- Python 3.5
- Pytorch 1.3
- Cuda

#### Train
- Train a binary model with bireal:
```
python train.py --binary_method bireal
```
- Train a binary model with xnor:
```
python train.py --binary_method xnor
```
- Train a binary model with bnn:
```
python train.py --binary_method bnn
```
#### Todo
- Acc on cifar10 with bnn and xnor
- Pretrain model on cifar10

#### Reference
- [XNOR-Net-Pytorch](https://github.com/jiecaoyu/XNOR-Net-PyTorch)
- [Bi-Real-net](https://github.com/liuzechun/Bi-Real-net)
- [BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or −1](https://arxiv.org/pdf/1602.02830v1.pdf)
- [XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks](https://arxiv.org/pdf/1603.05279.pdf)
- [Bi-Real Net: Enhancing the Performance of 1-bit CNNs With Improved Representational Capability and Advanced Training Algorithm](https://openaccess.thecvf.com/content_ECCV_2018/papers/zechun_liu_Bi-Real_Net_Enhancing_ECCV_2018_paper.pdf)
