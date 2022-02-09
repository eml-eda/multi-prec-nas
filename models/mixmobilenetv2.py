import torch.nn as nn
import math
from . import quant_module as qm

# DJP
__all__ = [
    'mixmv2_w248a248_chan', 'mixmv2_w248a248_multiprec',
]


def conv3x3(conv_func, in_planes, out_planes, stride=1, **kwargs):
    "3x3 convolution with padding"
    return conv_func(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, **kwargs)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, conv_func, inplanes, planes, stride=1,
                 downsample=None, bnaff=True, **kwargs):
        super(BasicBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes, affine=bnaff)
        self.conv1 = conv3x3(conv_func, inplanes, planes, stride, **kwargs)
        self.bn1 = nn.BatchNorm2d(planes, affine=bnaff)
        self.conv2 = conv3x3(conv_func, planes, planes, **kwargs)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        out = self.bn0(x)
        if self.downsample is not None:
            residual = out
        else:
            residual = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual

        return out

class InvertedResidual(nn.Module):
    def __init__(self, conv_func, inp, oup, stride, expand_ratio, **kwargs):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                conv_func(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding =1, groups=hidden_dim, bias=False, **kwargs),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=False),
                # pw-linear
                conv_func(hidden_dim, oup, kernel_size=1, stride=1, padding =0, groups = 1, bias=False, **kwargs),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=False),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                conv_func(inp, hidden_dim, kernel_size=1, stride=1, padding =0, groups = 1, bias=False, **kwargs),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=False),
                # dw
                conv_func(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding =1, groups=hidden_dim, bias=False, **kwargs),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=False),
                # pw-linear
                conv_func(hidden_dim, oup, kernel_size=1, stride=1, padding =0, groups = 1, bias=False, **kwargs),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=False),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x+self.conv(x)
        else:
            return self.conv(x)

def conv_1x1_bn(conv_function, inp, oup, **kwargs):
    return nn.Sequential(
        conv_function(inp, oup, kernel_size=1, stride=1, padding =0, groups = 1, bias=False, **kwargs),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=False)
    )
def conv_bn(conv_function, inp, oup, stride, **kwargs):
    return nn.Sequential(
        conv_function(inp, oup, kernel_size=3, stride=stride, padding=1, groups = 1, bias=False, **kwargs),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )
# AB
def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)

class MobilenetV2(nn.Module):

    def __init__(self, block, conv_func, width_mult, num_classes=1000,
                 bnaff=True, **kwargs):
        if 'abits' in kwargs:
            print('abits: {}'.format(kwargs['abits']))
        if 'wbits' in kwargs:
            print('wbits: {}'.format(kwargs['wbits']))
        self.inplanes = 32
        self.conv_func = conv_func
        super(MobilenetV2, self).__init__()
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(self.conv_func, 3, self.inplanes, 2, **kwargs)]
        # building inverted residual blocks
        input_channel = self.inplanes
        input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!

        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(self.conv_func, input_channel, output_channel, s, expand_ratio=t, **kwargs))
                else:
                    self.features.append(block(self.conv_func, input_channel, output_channel, 1, expand_ratio=t, **kwargs))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(self.conv_func, input_channel, self.last_channel, **kwargs))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)
        self.pooling = nn.AvgPool2d(kernel_size=7, stride=7, padding=0)

        # building classifier
        self.classifier = nn.Linear(self.last_channel, num_classes)
        self.sig = nn.Sigmoid()


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
        x = self.features(x)
        x = self.pooling(x)
        x = x.flatten(1)
        x = self.classifier(x)
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

# AB
def mixmv2_w248a248_chan(**kwargs):
    return MobilenetV2(InvertedResidual, qm.MixActivChanConv2d, width_mult = 1.0, wbits=[2, 4, 8], abits=[2, 4, 8],
                  share_weight=True, **kwargs)

# AB
def mixmv2_w248a248_multiprec(**kwargs):
    return MobilenetV2(InvertedResidual, qm.MultiPrecActivConv2d, width_mult = 1.0,  wbits=[2, 4, 8], abits=[2, 4, 8],
                  share_weight=True, **kwargs)
