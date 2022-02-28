import math
import numpy as np
import torch.nn as nn

from . import quant_module as qm

# MR
__all__ = [
    'mixdscnn_w0248a8_multiprec', 
    'mixdscnn_w248a8_multiprec', 
    'mixdscnn_w248a8_chan',
]

# MR
def conv1x1(conv_func, in_planes, out_planes, stride=1, **kwargs):
    "1x1 convolution with padding"
    return conv_func(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False, groups=1, **kwargs)
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
class BasicBlockGumbel(nn.Module):
    def __init__(self, conv_func, inplanes, planes, stride=1,
                 downsample=None, bnaff=True, **kwargs):
        super().__init__()
        self.conv0 = dw3x3(conv_func, inplanes, inplanes, stride=stride, **kwargs)
        self.bn0 = nn.BatchNorm2d(inplanes, affine=bnaff)
        #self.relu0 = nn.ReLU(inplace=True)
        self.conv1 = conv1x1(conv_func, inplanes, planes, **kwargs)
        self.bn1 = nn.BatchNorm2d(planes, affine=bnaff)
        #self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x, temp, is_hard):
        out, w_complexity0 = self.conv0(x, temp, is_hard)
        out = self.bn0(out)
        #out = self.relu0(out)
        out, w_complexity1 = self.conv1(out, temp, is_hard)
        out = self.bn1(out)
        #out = self.relu1(out)
        return out, w_complexity0 + w_complexity1

# MR
class Backbone(nn.Module):
    def __init__(self, conv_func, input_size, bnaff, **kwargs):
        super().__init__()
        self.input_layer = conv_func(1, 64, kernel_size=(10,4), stride=2, padding=(5,1), bias=False, groups=1, **kwargs)
        self.bn = nn.BatchNorm2d(64, affine=bnaff)
        self.dpout0 = nn.Dropout(0.2)
        self.bb_1 = BasicBlockGumbel(conv_func, 64, 64, 1, **kwargs)
        self.bb_2 = BasicBlockGumbel(conv_func, 64, 64, 1, **kwargs)
        self.bb_3 = BasicBlockGumbel(conv_func, 64, 64, 1, **kwargs)
        self.bb_4 = BasicBlockGumbel(conv_func, 64, 64, 1, **kwargs)
        self.dpout1 = nn.Dropout(0.4)
        self.pool = nn.AvgPool2d((int(input_size[0]/2), int(input_size[1]/2)))
    
    def forward(self, x, temp, is_hard):
        w_complexity = 0
        out, w_comp = self.input_layer(x, temp, is_hard)
        w_complexity += w_comp
        out = self.bn(out)
        out = self.dpout0(out)
        out, w_comp1 = self.bb_1(out, temp, is_hard)
        w_complexity += w_comp1
        out, w_comp2 = self.bb_2(out, temp, is_hard)
        w_complexity += w_comp2
        out, w_comp3 = self.bb_3(out, temp, is_hard)
        w_complexity += w_comp3
        out, w_comp4 = self.bb_4(out, temp, is_hard)
        w_complexity += w_comp4
        out = self.dpout1(out)
        out = self.pool(out)
        return out, w_complexity

# MR
class DS_CNN(nn.Module):
    def __init__(self, conv_func, search_fc=None, num_classes=12, input_size=(49,10), bnaff=True, 
                    **kwargs):
        if 'abits' in kwargs:
            print('abits: {}'.format(kwargs['abits']))
        if 'wbits' in kwargs:
            print('wbits: {}'.format(kwargs['wbits']))
        self.conv_func = conv_func
        self.search_types = ['fixed', 'mixed', 'multi']
        if search_fc in self.search_types:
            self.search_fc = search_fc
        else:
            self.search_fc = False
        super().__init__()
        #self.model = nn.Sequential(
        #    conv_func(1, 64, kernel_size=(10,4), stride=2, padding=1, bias=False, groups=1, **kwargs),
        #    nn.BatchNorm2d(64, affine=bnaff),
        #    #nn.ReLU(inplace=True),
        #    nn.Dropout(0.2),
        #    BasicBlock(conv_func, 64, 64, 1, **kwargs),
        #    BasicBlock(conv_func, 64, 64, 1, **kwargs),
        #    BasicBlock(conv_func, 64, 64, 1, **kwargs),
        #    BasicBlock(conv_func, 64, 64, 1, **kwargs),
        #    nn.Dropout(0.4),
        #    nn.AvgPool2d(int(input_size[0]/2), int(input_size[1]/2)),
        #)
        self.model = Backbone(conv_func, input_size, bnaff, **kwargs)
        if self.search_fc:
            self.fc = conv_func(64, num_classes, 
                        kernel_size=1, stride=1, padding=0, bias=True, groups=1, fc=self.search_fc, **kwargs)
        else:
            self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, temp, is_hard):
        x, w_comp0 = self.model(x, temp, is_hard)
        x = x if self.search_fc else x.view(x.size(0), -1)
        if self.search_fc:
            x, w_comp1 = self.fc(x, temp, is_hard)
            return x[:, :, 0, 0], w_comp0+w_comp1
        else:
            x = self.fc(x)
            return x, w_comp0

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

# MR
def mixdscnn_w0248a8_multiprec(**kwargs):
    return DS_CNN(qm.MultiPrecActivConv2d, search_fc='multi', wbits=[0, 2, 4, 8], abits=[8],
                  share_weight=True, **kwargs)

# MR
def mixdscnn_w248a8_multiprec(**kwargs):
    return DS_CNN(qm.MultiPrecActivConv2d, search_fc='multi', wbits=[2, 4, 8], abits=[8],
                  share_weight=True, **kwargs)

# MR
def mixdscnn_w248a8_chan(**kwargs):
    return DS_CNN(qm.MixActivChanConv2d, search_fc='mixed', wbits=[2, 4, 8], abits=[8],
                  share_weight=True, **kwargs)