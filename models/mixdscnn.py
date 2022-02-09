import math
import numpy as np
import torch.nn as nn

from . import quant_module as qm

# MR
__all__ = [
]

# MR
def conv1x1(conv_func, in_planes, out_planes, stride=1, **kwargs):
    "1x1 convolution with padding"
    return conv_func(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False, groups = 1, **kwargs)
# MR
def dw3x3(conv_func, in_planes, out_planes, stride=1, **kwargs):
    "3x3 convolution dw with padding"
    #return conv_func(in_planes, out_planes, kernel_size=3, stride=stride,
    #                 padding=1, bias=False, groups=1, **kwargs)
    # ^ Previous implementation... I think is wrong because it is not true depthwise
    # If I set groups=1, it is not depthwise but plain convolution
    return conv_func(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, groups=in_planes, **kwargs)

# MR
class BasicBlock(nn.Module):
    def __init__(self, conv_func, inplanes, planes, stride=1,
                 downsample=None, bnaff=True, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv0 = dw3x3(conv_func, inplanes, inplanes, stride=stride, **kwargs)
        self.bn0 = nn.BatchNorm2d(inplanes, affine=bnaff)
        self.relu0 = nn.ReLU(inplace=True)
        self.conv1 = conv1x1(conv_func, inplanes, planes, **kwargs)
        self.bn1 = nn.BatchNorm2d(planes, affine=bnaff)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv0(x)
        out = self.bn0(out)
        out = self.relu0(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu1(out)
        return out

# MR
class DS_CNN(nn.Module):
    def __init__(self, conv_func, num_classes=12, input_size=(49,10), bnaff=True, 
                    **kwargs):
        if 'abits' in kwargs:
            print('abits: {}'.format(kwargs['abits']))
        if 'wbits' in kwargs:
            print('wbits: {}'.format(kwargs['wbits']))
        self.conv_func = conv_func
        super().__init__()
        self.model = nn.Sequential(
            conv_func(1, 64, kernel_size=(10,4), stride=2, padding=1, bias=False, groups=1, **kwargs),
            nn.BatchNorm2d(64, affine=bnaff),
            #nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            BasicBlock(conv_func, 64, 64, 1, **kwargs),
            BasicBlock(conv_func, 64, 64, 1, **kwargs),
            BasicBlock(conv_func, 64, 64, 1, **kwargs),
            BasicBlock(conv_func, 64, 64, 1, **kwargs),
            nn.Dropout(0.4),
            nn.AvgPool2d(int(input_size[0]/2), int(input_size[1]/2)),
        )
        self.fc = nn.Linear(1, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def complexity_loss(self):
        size_product = []
        loss = 0
        for m in self.modules():
            if isinstance(m, self.conv_func):
                loss += m.complexity_loss()
                size_product += [m.size_product]
        normalizer = size_product[0].item()
        loss /= normalizer
        return loss

    def fetch_best_arch(self):
        sum_bitops, sum_bita, sum_bitw = 0, 0, 0
        sum_mixbitops, sum_mixbita, sum_mixbitw = 0, 0, 0
        layer_idx = 0
        best_arch = None
        for m in self.modules():
            if isinstance(m, self.conv_func):
                layer_arch, bitops, bita, bitw, mixbitops, mixbita, mixbitw = m.fetch_best_arch(layer_idx)
                if best_arch is None:
                    best_arch = layer_arch
                else:
                    for key in layer_arch.keys():
                        if key not in best_arch:
                            best_arch[key] = layer_arch[key]
                        else:
                            best_arch[key].append(layer_arch[key][0])
                sum_bitops += bitops
                sum_bita += bita
                sum_bitw += bitw
                sum_mixbitops += mixbitops
                sum_mixbita += mixbita
                sum_mixbitw += mixbitw
                layer_idx += 1
        return best_arch, sum_bitops, sum_bita, sum_bitw, sum_mixbitops, sum_mixbita, sum_mixbitw