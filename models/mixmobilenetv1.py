import math
import numpy as np
import torch.nn as nn

from . import quant_module as qm

# AB
__all__ = [
    'mixmv1_w248a248_chan', 'mixmv1_w248a248_multiprec',
    'mixmobilenetv1_w0248a8_multiprec', 'mixmobilenetv1_w248a8_multiprec',
    'mixmobilenetv1_w248a8_chan',
]

# AB
def conv1x1(conv_func, in_planes, out_planes, stride=1, **kwargs):
    "1x1 convolution with padding"
    return conv_func(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False, groups = 1, **kwargs)
# AB
def dw3x3(conv_func, in_planes, out_planes, stride=1, **kwargs):
    "3x3 convolution dw with padding"
    #return conv_func(in_planes, out_planes, kernel_size=3, stride=stride,
    #                 padding=1, bias=False, groups=1, **kwargs)
    # ^ Previous implementation... I think is wrong because it is not true depthwise
    # If I set groups=1, it is not depthwise but plain convolution
    return conv_func(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, groups=in_planes, **kwargs)

# MR
def fc(conv_func, in_planes, out_planes, stride=1, groups=1, search_fc=None, **kwargs):
    "fc mapped to conv"
    return conv_func(in_planes, out_planes, kernel_size=1, groups=groups, stride=stride,
                     padding=0, bias=True, fc=search_fc, **kwargs)

# MR
class Backbone(nn.Module):
    def __init__(self, conv_func, input_size, bnaff, width_mult, **kwargs):
        super().__init__()
        self.input_layer = conv_func(3, make_divisible(32*width_mult), kernel_size=3, stride=2, padding=1, bias=False, groups=1, **kwargs)
        self.bn = nn.BatchNorm2d(make_divisible(32*width_mult), affine=bnaff)
        self.bb_1 = BasicBlockGumbel(conv_func, make_divisible(32*width_mult), make_divisible(64*width_mult), 1, **kwargs)
        self.bb_2 = BasicBlockGumbel(conv_func, make_divisible(64*width_mult), make_divisible(128*width_mult), 2, **kwargs)
        self.bb_3 = BasicBlockGumbel(conv_func, make_divisible(128*width_mult), make_divisible(128*width_mult), 1, **kwargs)
        self.bb_4 = BasicBlockGumbel(conv_func, make_divisible(128*width_mult), make_divisible(256*width_mult), 2, **kwargs)
        self.bb_5 = BasicBlockGumbel(conv_func, make_divisible(256*width_mult), make_divisible(256*width_mult), 1, **kwargs)
        self.bb_6 = BasicBlockGumbel(conv_func, make_divisible(256*width_mult), make_divisible(512*width_mult), 2, **kwargs)
        self.bb_7 = BasicBlockGumbel(conv_func, make_divisible(512*width_mult), make_divisible(512*width_mult), 1, **kwargs)
        self.bb_8 = BasicBlockGumbel(conv_func, make_divisible(512*width_mult), make_divisible(512*width_mult), 1, **kwargs)
        self.bb_9 = BasicBlockGumbel(conv_func, make_divisible(512*width_mult), make_divisible(512*width_mult), 1, **kwargs)
        self.bb_10 = BasicBlockGumbel(conv_func, make_divisible(512*width_mult), make_divisible(512*width_mult), 1, **kwargs)
        self.bb_11 = BasicBlockGumbel(conv_func, make_divisible(512*width_mult), make_divisible(512*width_mult), 1, **kwargs)
        self.bb_12 = BasicBlockGumbel(conv_func, make_divisible(512*width_mult), make_divisible(1024*width_mult), 2, **kwargs)
        self.bb_13 = BasicBlockGumbel(conv_func, make_divisible(1024*width_mult), make_divisible(1024*width_mult), 1, **kwargs)
        self.pool = nn.AvgPool2d(int(input_size/(2**5)))

    def forward(self, x, temp, is_hard):
        w_complexity = 0
        out, w_comp = self.input_layer(x, temp, is_hard)
        w_complexity += w_comp
        out = self.bn(out)
        out, w_comp1 = self.bb_1(out, temp, is_hard)
        w_complexity += w_comp1
        out, w_comp2 = self.bb_2(out, temp, is_hard)
        w_complexity += w_comp2
        out, w_comp3 = self.bb_3(out, temp, is_hard)
        w_complexity += w_comp3
        out, w_comp4 = self.bb_4(out, temp, is_hard)
        w_complexity += w_comp4
        out, w_comp5 = self.bb_5(out, temp, is_hard)
        w_complexity += w_comp5
        out, w_comp6 = self.bb_6(out, temp, is_hard)
        w_complexity += w_comp6
        out, w_comp7 = self.bb_7(out, temp, is_hard)
        w_complexity += w_comp7
        out, w_comp8 = self.bb_8(out, temp, is_hard)
        w_complexity += w_comp8
        out, w_comp9 = self.bb_9(out, temp, is_hard)
        w_complexity += w_comp9
        out, w_comp10 = self.bb_10(out, temp, is_hard)
        w_complexity += w_comp10
        out, w_comp11 = self.bb_11(out, temp, is_hard)
        w_complexity += w_comp11
        out, w_comp12 = self.bb_12(out, temp, is_hard)
        w_complexity += w_comp12
        out, w_comp13 = self.bb_13(out, temp, is_hard)
        w_complexity += w_comp13
        out = self.pool(out)
        return out, w_complexity

# AB
class BasicBlock(nn.Module):
    def __init__(self, conv_func, inplanes, planes, stride=1,
                 downsample=None, bnaff=True, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv0 = dw3x3(conv_func, inplanes, inplanes, stride=stride, **kwargs)
        self.bn0 = nn.BatchNorm2d(inplanes, affine=bnaff)
        #self.relu0 = nn.ReLU(inplace=True)
        self.conv1 = conv1x1(conv_func, inplanes, planes, **kwargs)
        self.bn1 = nn.BatchNorm2d(planes, affine=bnaff)
        #self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv0(x)
        out = self.bn0(out)
        #out = self.relu0(out)
        out = self.conv1(out)
        out = self.bn1(out)
        #out = self.relu1(out)
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

# AB
def make_divisible(x, divisible_by=8):
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)

# AB
class MobilenetV1(nn.Module):
    def __init__(self, conv_func, num_classes=10, input_size=32, width_mult=1.0,
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
        self.model = nn.Sequential(
            conv_func(3, make_divisible(32*width_mult), kernel_size=3, stride=2, padding=1, bias=False, groups=1, **kwargs),
            nn.BatchNorm2d(make_divisible(32*width_mult), affine=bnaff),
            nn.ReLU(inplace=True),
            BasicBlock(conv_func, make_divisible(32*width_mult), make_divisible(64*width_mult), 1, **kwargs),
            BasicBlock(conv_func, make_divisible(64*width_mult), make_divisible(128*width_mult), 2, **kwargs),
            BasicBlock(conv_func, make_divisible(128*width_mult), make_divisible(128*width_mult), 1, **kwargs),
            BasicBlock(conv_func, make_divisible(128*width_mult), make_divisible(256*width_mult), 2, **kwargs),
            BasicBlock(conv_func, make_divisible(256*width_mult), make_divisible(256*width_mult), 1, **kwargs),
            BasicBlock(conv_func, make_divisible(256*width_mult), make_divisible(512*width_mult), 2, **kwargs),
            BasicBlock(conv_func, make_divisible(512*width_mult), make_divisible(512*width_mult), 1, **kwargs),
            BasicBlock(conv_func, make_divisible(512*width_mult), make_divisible(512*width_mult), 1, **kwargs),
            BasicBlock(conv_func, make_divisible(512*width_mult), make_divisible(512*width_mult), 1, **kwargs),
            BasicBlock(conv_func, make_divisible(512*width_mult), make_divisible(512*width_mult), 1, **kwargs),
            BasicBlock(conv_func, make_divisible(512*width_mult), make_divisible(512*width_mult), 1, **kwargs),
            BasicBlock(conv_func, make_divisible(512*width_mult), make_divisible(1024*width_mult), 2, **kwargs),
            BasicBlock(conv_func, make_divisible(1024*width_mult), make_divisible(1024*width_mult), 1, **kwargs),
            nn.AvgPool2d(int(input_size/(2**5))),
        )
        if self.search_fc:
            self.fc = fc(make_divisible(1024*width_mult), num_classes, search_fc=self.search_fc, **kwargs)
        else:
            self.fc = nn.Linear(make_divisible(1024*width_mult), num_classes)
        #self.fc = nn.Linear(make_divisible(1024*width_mult), num_classes)

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

# MR
class TinyMLMobilenetV1(nn.Module):
    def __init__(self, conv_func, search_fc=None, num_classes=2, input_size=96, width_mult=.25,
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
        self.gumbel = kwargs.get('gumbel', False)
        #if not self.gumbel:
        #    self.model = nn.Sequential(
        #        conv_func(3, make_divisible(32*width_mult), kernel_size=3, stride=2, padding=1, bias=False, groups=1, **kwargs), # 1
        #        nn.BatchNorm2d(make_divisible(32*width_mult), affine=bnaff),
        #        nn.ReLU(inplace=True),
        #        BasicBlock(conv_func, make_divisible(32*width_mult), make_divisible(64*width_mult), 1, **kwargs), # 2
        #        BasicBlock(conv_func, make_divisible(64*width_mult), make_divisible(128*width_mult), 2, **kwargs), # 3
        #        BasicBlock(conv_func, make_divisible(128*width_mult), make_divisible(128*width_mult), 1, **kwargs), # 4
        #        BasicBlock(conv_func, make_divisible(128*width_mult), make_divisible(256*width_mult), 2, **kwargs), # 5
        #        BasicBlock(conv_func, make_divisible(256*width_mult), make_divisible(256*width_mult), 1, **kwargs), # 6
        #        BasicBlock(conv_func, make_divisible(256*width_mult), make_divisible(512*width_mult), 2, **kwargs), # 7
        #        BasicBlock(conv_func, make_divisible(512*width_mult), make_divisible(512*width_mult), 1, **kwargs), # 8
        #        BasicBlock(conv_func, make_divisible(512*width_mult), make_divisible(512*width_mult), 1, **kwargs), # 9
        #        BasicBlock(conv_func, make_divisible(512*width_mult), make_divisible(512*width_mult), 1, **kwargs), # 10
        #        BasicBlock(conv_func, make_divisible(512*width_mult), make_divisible(512*width_mult), 1, **kwargs), # 11
        #        BasicBlock(conv_func, make_divisible(512*width_mult), make_divisible(512*width_mult), 1, **kwargs), # 12
        #        BasicBlock(conv_func, make_divisible(512*width_mult), make_divisible(1024*width_mult), 2, **kwargs), # 13
        #        BasicBlock(conv_func, make_divisible(1024*width_mult), make_divisible(1024*width_mult), 1, **kwargs), # 14
        #        nn.AvgPool2d(int(input_size/(2**5))),
        #    )
        #else:
        self.model = Backbone(conv_func, input_size, bnaff, width_mult, **kwargs)
        if self.search_fc:
            self.fc = fc(conv_func, make_divisible(1024*width_mult), num_classes, 
                stride=1, groups=1, search_fc=self.search_fc, **kwargs)
        else:
            self.fc = nn.Linear(make_divisible(1024*width_mult), num_classes)

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
        #normalizer = size_product[0].item()
        #loss /= normalizer
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

# AB
def mixmv1_w248a248_chan(**kwargs):
    return MobilenetV1(qm.MixActivChanConv2d, wbits=[2,4,8], abits=[2,4,8],
                  share_weight=True, **kwargs)

# AB
def mixmv1_w248a248_multiprec(**kwargs):
    return MobilenetV1(qm.MultiPrecActivConv2d, wbits=[2,4,8], abits=[2,4,8],
                  share_weight=True, **kwargs)

# MR
def mixmobilenetv1_w0248a8_multiprec(**kwargs):
    return TinyMLMobilenetV1(qm.MultiPrecActivConv2d, search_fc='multi', wbits=[0, 2, 4, 8], abits=[8],
                  share_weight=True, **kwargs)

# MR
def mixmobilenetv1_w248a8_multiprec(**kwargs):
    return TinyMLMobilenetV1(qm.MultiPrecActivConv2d, search_fc='multi', wbits=[2, 4, 8], abits=[8],
                  share_weight=True, **kwargs)

# MR
def mixmobilenetv1_w248a8_chan(**kwargs):
    return TinyMLMobilenetV1(qm.MixActivChanConv2d, search_fc='mixed', wbits=[2, 4, 8], abits=[8],
                  share_weight=True, **kwargs)
