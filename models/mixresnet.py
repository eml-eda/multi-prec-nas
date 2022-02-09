import torch.nn as nn
import math
from . import quant_module as qm

# DJP
__all__ = [
    'mixres18_w248a248_chan', 'mixres18_w1234a234_chan',
    'mixres18_w248a248_multiprec', 'mixres18_w1234a234_multiprec', 'mixres18_w0248a248_multiprec',
    'mixres18_w1234a234', 'mixres50_w1234a234',
    'mixres18_w2345678a8_chan', 
    'mixres18_w248a8_chan', 'mixres18_w248a4_chan', 'mixres18_w248a2_chan',
    'mixres18_w0248a8_multiprec', 'mixres18_w0248a4_multiprec', 'mixres18_w0248a2_multiprec',
    'mixres18_w248a8_multiprec', 'mixres18_w248a4_multiprec', 'mixres18_w248a2_multiprec',
    'mixres18_w48a8_multiprec', 'mixres18_w08a8_multiprec', 'mixres18_w024a8_multiprec',
    'mixres8_w248a8_chan', 'mixres8_w0248a8_multiprec', 'mixres8_w248a8_multiprec',
]


def conv3x3(conv_func, in_planes, out_planes, stride=1, groups = 1, first_layer=False, **kwargs):
    "3x3 convolution with padding"
    if conv_func != nn.Conv2d:
        return conv_func(in_planes, out_planes, kernel_size=3, groups = groups, stride=stride,
                     padding=1, bias=False, first_layer=first_layer, **kwargs)
    else:
        return conv_func(in_planes, out_planes, kernel_size=3, groups = groups, stride=stride,
                     padding=1, bias=False, **kwargs)

def conv7x7(conv_func, in_planes, out_planes, stride=1, groups = 1, first_layer=False, **kwargs):
    "7x7 convolution with padding"
    if conv_func != nn.Conv2d:
        return conv_func(in_planes, out_planes, kernel_size=7, groups = groups, stride=stride,
                     padding=3, bias=False, first_layer=first_layer, **kwargs)
    else:
        return conv_func(in_planes, out_planes, kernel_size=7, groups = groups, stride=stride,
                     padding=3, bias=False, **kwargs)

# MR
def fc(conv_func, in_planes, out_planes, stride=1, groups = 1, search_fc=None, **kwargs):
    "fc mapped to conv"
    return conv_func(in_planes, out_planes, kernel_size=1, groups = groups, stride=stride,
                     padding=0, bias=False, fc=search_fc, **kwargs)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, conv_func, inplanes, planes, stride=1,
                 downsample=None, bnaff=True, **kwargs):
        super(BasicBlock, self).__init__()
        #self.bn0 = nn.BatchNorm2d(inplanes, affine=bnaff)
        self.conv1 = conv3x3(conv_func, inplanes, planes, stride, **kwargs)
        self.bn1 = nn.BatchNorm2d(planes, affine=bnaff)
        self.conv2 = conv3x3(conv_func, planes, planes, **kwargs)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        #out = self.bn0(x)
        if self.downsample is not None:
            residual = x
        else:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, conv_func, inplanes, planes, stride=1,
                 downsample=None, bnaff=True, **kwargs):
        super(Bottleneck, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes, affine=bnaff)
        self.conv1 = conv_func(
            inplanes, planes, kernel_size=1, groups = 1, bias=False, **kwargs)
        self.bn1 = nn.BatchNorm2d(planes, affine=bnaff)
        self.conv2 = conv_func(
            planes, planes, kernel_size=3, stride=stride, groups = 1, padding=1, bias=False, **kwargs)
        self.bn2 = nn.BatchNorm2d(planes, affine=bnaff)
        self.conv3 = conv_func(
            planes, planes * self.expansion, kernel_size=1, groups = 1, bias=False, **kwargs)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual

        return out


class ResNet(nn.Module):

    def __init__(self, block, conv_func, layers, search_fc=None, input_size=32, num_classes=1000,
                 bnaff=True, **kwargs):
        if 'abits' in kwargs:
            print('abits: {}'.format(kwargs['abits']))
        if 'wbits' in kwargs:
            print('wbits: {}'.format(kwargs['wbits']))
        self.inplanes = 64
        self.conv_func = conv_func
        self.search_types = ['fixed', 'mixed', 'multi']
        if search_fc in self.search_types:
            self.search_fc = search_fc
        else:
            self.search_fc = False
        self.num_classes = num_classes
        super(ResNet, self).__init__()
        #self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, groups = 1, padding=3, bias=False)
        #self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, groups = 1, padding=1, bias=False)
        if num_classes == 10:
            self.conv1 = conv3x3(conv_func, 3, 64, stride=1, groups=1, **kwargs)
        elif num_classes == 1000:
            self.conv1 = conv7x7(conv_func, 3, 64, stride=2, groups=1, **kwargs)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        if num_classes == 1000:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, conv_func, 64, layers[0], bnaff=bnaff, **kwargs)
        self.layer2 = self._make_layer(
            block, conv_func, 128, layers[1], stride=2, groups = 1, bnaff=bnaff, **kwargs)
        self.layer3 = self._make_layer(
            block, conv_func, 256, layers[2], stride=2, groups = 1, bnaff=bnaff, **kwargs)
        self.layer4 = self._make_layer(
            block, conv_func, 512, layers[3], stride=2, groups = 1, bnaff=bnaff, **kwargs)
        #self.avgpool = nn.AvgPool2d(int(input_size/(2**5)))
        if num_classes == 1000: # ImageNet
            self.avgpool = nn.AvgPool2d(kernel_size=7)
        else: # Cifar-10
            self.avgpool = nn.AvgPool2d(kernel_size=4)
        if self.search_fc:
            self.fc = fc(conv_func, 512 * block.expansion, num_classes, search_fc=self.search_fc, **kwargs)
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_layer(self, block, conv_func, planes, blocks, stride=1, groups = 1, bnaff=True, **kwargs):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv_func(self.inplanes, planes * block.expansion, kernel_size=1,
                          stride=stride,  groups = 1, bias=False, **kwargs),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(conv_func, self.inplanes, planes, stride, downsample, groups = 1, bnaff=bnaff, **kwargs))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(conv_func, self.inplanes, planes, groups = 1, bnaff=bnaff, **kwargs))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        if self.num_classes == 1000:
            x = self.maxpool(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.relu(x)
        #assert x.shape[2] == 7
        #assert x.shape[3] == 7
        x = self.avgpool(x)
        x = x if self.search_fc else x.view(x.size(0), -1)
        x = self.fc(x)[:, :, 0, 0] if self.search_fc else self.fc(x)

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

class TinyMLResNet(nn.Module):

    def __init__(self, block, conv_func, layers, search_fc=None, input_size = 32, num_classes=1000,
                 bnaff=True, **kwargs):
        if 'abits' in kwargs:
            print('abits: {}'.format(kwargs['abits']))
        if 'wbits' in kwargs:
            print('wbits: {}'.format(kwargs['wbits']))
        self.inplanes = 16
        self.conv_func = conv_func
        self.search_types = ['fixed', 'mixed', 'multi']
        if search_fc in self.search_types:
            self.search_fc = search_fc
        else:
            self.search_fc = False
        super().__init__()
        #self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, groups = 1, padding=3, bias=False)
        #self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, groups = 1, padding=1, bias=False)
        self.conv1 = conv3x3(conv_func, 3, 16, stride=1, groups=1, **kwargs)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, conv_func, 16, layers[0], bnaff=bnaff, **kwargs)
        self.layer2 = self._make_layer(
            block, conv_func, 32, layers[1], stride=2, groups = 1, bnaff=bnaff, **kwargs)
        self.layer3 = self._make_layer(
            block, conv_func, 64, layers[2], stride=2, groups = 1, bnaff=bnaff, **kwargs)
        #self.avgpool = nn.AvgPool2d(int(input_size/(2**5)))
        self.avgpool = nn.AvgPool2d(kernel_size=4)
        if self.search_fc:
            self.fc = fc(conv_func, 64 * block.expansion, num_classes, search_fc=self.search_fc, **kwargs)
        else:
            self.fc = nn.Linear(64 * block.expansion, num_classes, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_layer(self, block, conv_func, planes, blocks, stride=1, groups = 1, bnaff=True, **kwargs):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv_func(self.inplanes, planes * block.expansion, kernel_size=1,
                          stride=stride,  groups = 1, bias=False, **kwargs),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(conv_func, self.inplanes, planes, stride, downsample, groups = 1, bnaff=bnaff, **kwargs))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(conv_func, self.inplanes, planes, groups = 1, bnaff=bnaff, **kwargs))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        #x = self.maxpool(x)
        x = self.bn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(x)
        #assert x.shape[2] == 7
        #assert x.shape[3] == 7
        x = self.avgpool(x)
        x = x if self.search_fc else x.view(x.size(0), -1)
        x = self.fc(x)[:, :, 0, 0] if self.search_fc else self.fc(x)

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

# DJP
def mixres18_w1234a234_multiprec(**kwargs):
    return ResNet(BasicBlock, qm.MultiPrecActivConv2d, [2, 2, 2, 2], wbits=[1, 2, 3, 4], abits=[2, 3, 4],
                  share_weight=True, **kwargs)

# DJP
def mixres18_w1234a234_chan(**kwargs):
    return ResNet(BasicBlock, qm.MixActivChanConv2d, [2, 2, 2, 2], wbits=[1, 2, 3, 4], abits=[2, 3, 4],
                  share_weight=True, **kwargs)

# AB
def mixres18_w248a248_multiprec(**kwargs):
    return ResNet(BasicBlock, qm.MultiPrecActivConv2d, [2, 2, 2, 2], search_fc='multi', wbits=[2, 4, 8], abits=[2, 4, 8],
                  share_weight=True, **kwargs)

# MR
# search_fc: None or 'fixed' or 'mixed' or 'multi' 
def mixres18_w0248a248_multiprec(**kwargs):
    return ResNet(BasicBlock, qm.MultiPrecActivConv2d, [2, 2, 2, 2], search_fc='multi', wbits=[0, 2, 4, 8], abits=[2, 4, 8],
                  share_weight=True, **kwargs)

# AB
def mixres18_w248a248_chan(**kwargs):
    return ResNet(BasicBlock, qm.MixActivChanConv2d, [2, 2, 2, 2], wbits=[2, 4, 8], abits=[2, 4, 8],
                  share_weight=True, **kwargs)

# MR
def mixres18_w2345678a8_chan(**kwargs):
    return ResNet(BasicBlock, qm.MixActivChanConv2d, [2, 2, 2, 2], search_fc=True, wbits=[2, 3, 4, 5, 6, 7, 8], abits=[8],
                  share_weight=True, **kwargs)

# MR
def mixres18_w248a8_chan(**kwargs):
    return ResNet(BasicBlock, qm.MixActivChanConv2d, [2, 2, 2, 2], search_fc=True, wbits=[2, 4, 8], abits=[8],
                  share_weight=True, **kwargs)

# MR
def mixres18_w248a4_chan(**kwargs):
    return ResNet(BasicBlock, qm.MixActivChanConv2d, [2, 2, 2, 2], search_fc=True, wbits=[2, 4, 8], abits=[4],
                  share_weight=True, **kwargs)

# MR
def mixres18_w248a2_chan(**kwargs):
    return ResNet(BasicBlock, qm.MixActivChanConv2d, [2, 2, 2, 2], search_fc=True, wbits=[2, 4, 8], abits=[2],
                  share_weight=True, **kwargs)

def mixres18_w1234a234(**kwargs):
    return ResNet(BasicBlock, qm.MixActivConv2d, [2, 2, 2, 2], wbits=[1, 2, 3, 4], abits=[2, 3, 4],
                  share_weight=True, **kwargs)


def mixres50_w1234a234(**kwargs):
    return ResNet(Bottleneck, qm.MixActivConv2d, [3, 4, 6, 3], wbits=[1, 2, 3, 4], abits=[2, 3, 4],
                  share_weight=True, **kwargs)


# MR
def mixres8_w0248a8_multiprec(**kwargs):
    return TinyMLResNet(BasicBlock, qm.MultiPrecActivConv2d, [1, 1, 1], search_fc='multi', wbits=[0, 2, 4, 8], abits=[8],
                  share_weight=True, **kwargs)

# MR
def mixres8_w248a8_multiprec(**kwargs):
    return TinyMLResNet(BasicBlock, qm.MultiPrecActivConv2d, [1, 1, 1], search_fc='multi', wbits=[2, 4, 8], abits=[8],
                  share_weight=True, **kwargs)

# MR
def mixres8_w248a8_chan(**kwargs):
    return TinyMLResNet(BasicBlock, qm.MixActivChanConv2d, [1, 1, 1], search_fc='mixed', wbits=[2, 4, 8], abits=[8],
                  share_weight=True, **kwargs)

# MR
def mixres18_w0248a8_multiprec(**kwargs):
    return ResNet(BasicBlock, qm.MultiPrecActivConv2d, [2, 2, 2, 2], search_fc='multi', wbits=[0, 2, 4, 8], abits=[8],
                  share_weight=True, **kwargs)

# MR
def mixres18_w0248a4_multiprec(**kwargs):
    return ResNet(BasicBlock, qm.MultiPrecActivConv2d, [2, 2, 2, 2], search_fc='multi', wbits=[0, 2, 4, 8], abits=[4],
                  share_weight=True, **kwargs)

# MR
def mixres18_w0248a2_multiprec(**kwargs):
    return ResNet(BasicBlock, qm.MultiPrecActivConv2d, [2, 2, 2, 2], search_fc='multi', wbits=[0, 2, 4, 8], abits=[2],
                  share_weight=True, **kwargs)

# MR
def mixres18_w248a8_multiprec(**kwargs):
    return ResNet(BasicBlock, qm.MultiPrecActivConv2d, [2, 2, 2, 2], search_fc='multi', wbits=[2, 4, 8], abits=[8],
                  share_weight=True, **kwargs)

# MR
def mixres18_w248a4_multiprec(**kwargs):
    return ResNet(BasicBlock, qm.MultiPrecActivConv2d, [2, 2, 2, 2], search_fc='multi', wbits=[2, 4, 8], abits=[4],
                  share_weight=True, **kwargs)

# MR
def mixres18_w248a2_multiprec(**kwargs):
    return ResNet(BasicBlock, qm.MultiPrecActivConv2d, [2, 2, 2, 2], search_fc='multi', wbits=[2, 4, 8], abits=[2],
                  share_weight=True, **kwargs)

# MR
def mixres18_w48a8_multiprec(**kwargs):
    return ResNet(BasicBlock, qm.MultiPrecActivConv2d, [2, 2, 2, 2], search_fc='multi', wbits=[4, 8], abits=[8],
                  share_weight=True, **kwargs)

# MR
def mixres18_w08a8_multiprec(**kwargs):
    return ResNet(BasicBlock, qm.MultiPrecActivConv2d, [2, 2, 2, 2], search_fc='multi', wbits=[0, 8], abits=[8],
                  share_weight=True, **kwargs)

# MR
def mixres18_w024a8_multiprec(**kwargs):
    return ResNet(BasicBlock, qm.MultiPrecActivConv2d, [2, 2, 2, 2], search_fc='multi', wbits=[0, 2, 4], abits=[8],
                  share_weight=True, **kwargs)