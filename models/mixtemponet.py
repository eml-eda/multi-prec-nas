import copy
import numpy as np
import torch
import torch.nn as nn
import math
from . import quant_module_1d as qm1d

# MR
__all__ = [
    'mixtemponet_w248a8_multiprec', 
    'mixtemponet_w248a8_chan',
]

class TEMPONet(nn.Module):
    def __init__(self, conv_func, search_fc=None, num_classes=1, input_size=(256,),
                 bnaff=True, **kwargs):
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
        
        # Architecture:
        self.feature_extractor = FeatureExtractor(conv_func, input_size, **kwargs)
        self.classifier = Classifier(conv_func, self.search_fc, **kwargs)

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
        x, w_comp0 = self.feature_extractor(x, temp, is_hard)
        x = x.flatten(1).unsqueeze(-1)
        x, w_comp1 = self.classifier(x, temp, is_hard)
        #x = x if self.qtz_fc else x.view(x.size(0), -1)
        #x = self.fc(x)[:, :, 0, 0] if self.qtz_fc else self.fc(x)
        return x[:, :, 0], w_comp0+w_comp1

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

class FeatureExtractor(nn.Module):

    def __init__(self, conv_func, input_size, **kwargs):
        super().__init__()
        # pad = ((k-1)*dil+1)//2
        # First Stack
        self.tcb00 = TempConvBlock(conv_func, ch_in=4, ch_out=32, k=3, 
            dil=2, pad=2, stride=1, fix_qtz=True, **kwargs)
        self.tcb01 = TempConvBlock(conv_func, ch_in=32, ch_out=32, k=3, 
            dil=2, pad=2, stride=1, **kwargs)
        self.cb0 = ConvBlock(conv_func, ch_in=32, ch_out=64, k=5,
            dil=1, pad=2, stride=1, **kwargs)
        # Second Stack
        self.tcb10 = TempConvBlock(conv_func, ch_in=64, ch_out=64, k=3, 
            dil=4, pad=4, stride=1, **kwargs)
        self.tcb11 = TempConvBlock(conv_func, ch_in=64, ch_out=64, k=3, 
            dil=4, pad=4, stride=1, **kwargs)
        self.cb1 = ConvBlock(conv_func, ch_in=64, ch_out=128, k=5,
            dil=1, pad=2, stride=2, **kwargs)
        # Third Stack
        self.tcb20 = TempConvBlock(conv_func, ch_in=128, ch_out=128, k=3, 
            dil=8, pad=8, stride=1, **kwargs)
        self.tcb21 = TempConvBlock(conv_func, ch_in=128, ch_out=128, k=3, 
            dil=8, pad=8, stride=1, **kwargs)
        self.cb2 = ConvBlock(conv_func, ch_in=128, ch_out=128, k=5,
            dil=1, pad=4, stride=4, **kwargs)

    def forward(self, x, temp, is_hard):
        w_complexity = 0
        x, w_comp00 = self.tcb00(x, temp, is_hard)
        w_complexity += w_comp00
        x, w_comp01 = self.tcb01(x, temp, is_hard)
        w_complexity += w_comp01
        x, w_comp0 = self.cb0(x, temp, is_hard)
        w_complexity += w_comp0
        x, w_comp10 = self.tcb10(x, temp, is_hard)
        w_complexity += w_comp10
        x, w_comp11 = self.tcb11(x, temp, is_hard)
        w_complexity += w_comp11
        x, w_comp1 = self.cb1(x, temp, is_hard)
        w_complexity += w_comp1
        x, w_comp20 = self.tcb20(x, temp, is_hard)
        w_complexity += w_comp20
        x, w_comp21 = self.tcb21(x, temp, is_hard)
        w_complexity += w_comp21
        x, w_comp2 = self.cb2(x, temp, is_hard)
        w_complexity += w_comp2
        return x, w_complexity


class TempConvBlock(nn.Module):
    def __init__(self, conv_func, ch_in, ch_out, k, dil, pad, stride, fix_qtz=False, **kwargs):
        super().__init__()
        self.tcn = conv_func(ch_in, ch_out, kernel_size=k, 
            dilation=dil, stride=stride, padding=pad, bias=False, fix_qtz=fix_qtz, groups=1, **kwargs)
        self.bn = nn.BatchNorm1d(ch_out)

    def forward(self, x, temp, is_hard):
        x, w_comp = self.tcn(x, temp, is_hard)
        x = self.bn(x)
        return x, w_comp


class ConvBlock(nn.Module):
    def __init__(self, conv_func, ch_in, ch_out, k, dil, pad, stride, **kwargs):
        super().__init__()
        self.conv = conv_func(ch_in, ch_out, kernel_size=k, 
            dilation=dil, stride=stride, padding=pad, bias=False, groups=1, **kwargs)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm1d(ch_out)

    def forward(self, x, temp, is_hard):
        x, w_comp = self.conv(x, temp, is_hard)
        x = self.pool(x)
        x = self.bn(x)
        return x, w_comp


class Classifier(nn.Module):
    def __init__(self, conv_func, qtz_fc, **kwargs):
        super().__init__()
        self.fc0 = conv_func(512, 256,
            kernel_size=1, stride=1, padding=0, bias=False, groups=1, fc=qtz_fc, **kwargs)
        self.bn0 = nn.BatchNorm1d(256)
        self.fc1 = conv_func(256, 128,
            kernel_size=1, stride=1, padding=0, bias=False, groups=1, fc=qtz_fc, **kwargs)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = conv_func(128, 1,
            kernel_size=1, stride=1, padding=0, bias=True, groups=1, fc=qtz_fc, **kwargs)
    
    def forward(self, x, temp, is_hard):
        w_complexity = 0
        x, w_comp0 = self.fc0(x, temp, is_hard)
        w_complexity += w_comp0
        x = self.bn0(x)
        x, w_comp1 = self.fc1(x, temp, is_hard)
        w_complexity += w_comp1
        x = self.bn1(x)
        x, w_comp2 = self.fc2(x, temp, is_hard)
        w_complexity += w_comp2
        return x, w_complexity


# MR
def mixtemponet_w248a8_multiprec(**kwargs):
    return TEMPONet(qm1d.MultiPrecActivConv1d, search_fc='multi', wbits=[2, 4, 8], abits=[8],
                  share_weight=True, **kwargs)

# MR
def mixtemponet_w248a8_chan(**kwargs):
    return TEMPONet(qm1d.MixActivChanConv1d, search_fc='mixed', wbits=[2, 4, 8], abits=[8],
                  share_weight=True, **kwargs)