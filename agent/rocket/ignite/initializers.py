import torch
import torch.nn as nn
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BacthNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)


def isin(x, list):
    for i in list:
        if x == i:
            return True


@torch.no_grad()
def init_conv(m):
    print(m)
    if isin(type(m), [nn.Conv1d, nn.Conv2d, nn.Conv3d]):
        nn.init.xavier_uniform_(m.weight.data)
        # nn.init.xavier_uniform_(m.bias.data)


@torch.no_grad()
def init_linear(m):
    print(m)
    if isin(type(m), [nn.Linear]):
        nn.init.xavier_uniform_(m.weight.data)
