import numpy as np
import torch
import torch.nn as nn
import math
from . import quant_module as qm

# MR
__all__ = [
   'quantdscnn_fp', 'quantdscnn_w8a8', 'quantdscnn_w4a8', 'quantdscnn_w2a8',
]

# MR
class BasicBlock(nn.Module):
    def __init__(self, conv_func, inplanes, planes, archws, archas, stride=1,
                 downsample=None, bnaff=True, **kwargs):
        super(BasicBlock, self).__init__()
        assert len(archws) == 2
        assert len(archas) == 2
        #self.conv0 = dw3x3(conv_func, inplanes, inplanes, stride=stride, **kwargs)
        self.conv0 = conv_func(inplanes, inplanes, archws[0], archas[0], kernel_size=3, stride=stride, 
                                padding=1, bias=False, groups=inplanes, **kwargs)
        self.bn0 = nn.BatchNorm2d(inplanes, affine=bnaff)
        #self.relu0 = nn.ReLU(inplace=True)
        #self.conv1 = conv1x1(conv_func, inplanes, planes, **kwargs)
        self.conv1 = conv_func(inplanes, planes, archws[1], archas[1], kernel_size=1, stride=1,
                                padding=0, bias=False, groups=1, **kwargs)
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

class DS_CNN(nn.Module):
    def __init__(self, conv_func, archws, archas, qtz_fc=None, num_classes=12, input_size=(49,10),
                 bnaff=True, **kwargs):
        print('archas: {}'.format(archas))
        print('archws: {}'.format(archws))
        self.conv_func = conv_func
        self.search_types = ['fixed', 'mixed', 'multi']
        if qtz_fc in self.search_types:
            self.qtz_fc = qtz_fc
        else:
            self.qtz_fc = False
        super().__init__()
        self.model = nn.Sequential(
            # first_layer=True removes the ReLU activation
            conv_func(1, 64, abits=archas[0], wbits=archws[0], 
                        kernel_size=(10,4), stride=2, padding=(5,1), bias=False, groups=1, first_layer=True, **kwargs), # 1
            #conv_func(1, 64, abits=archas[0], wbits=archws[0], 
            #            kernel_size=(10,4), stride=2, padding=(5,1), bias=False, groups=1, **kwargs), # 1
            nn.BatchNorm2d(64, affine=bnaff),
            #nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            BasicBlock(conv_func, 64, 64, archws[1:3], archas[1:3], stride=1, **kwargs), # 2
            BasicBlock(conv_func, 64, 64, archws[3:5], archas[3:5], stride=1, **kwargs), # 3
            BasicBlock(conv_func, 64, 64, archws[5:7], archas[5:7], stride=1, **kwargs), # 4
            BasicBlock(conv_func, 64, 64, archws[7:9], archas[7:9], stride=1, **kwargs), # 5
            nn.Dropout(0.4), 
            nn.AvgPool2d((int(input_size[0]/2), int(input_size[1]/2))),
        )
        if self.qtz_fc:
            self.fc = conv_func(64, num_classes, abits=archas[-1], wbits=archws[-1], 
                        kernel_size=1, stride=1, padding=0, bias=True, groups=1, fc=self.qtz_fc, **kwargs)
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

    def forward(self, x):
        x = self.model(x)
        #import pdb; pdb.set_trace()
        x = x if self.qtz_fc else x.view(x.size(0), -1)
        x = self.fc(x)[:, :, 0, 0] if self.qtz_fc else self.fc(x)
        return x
    
    def fetch_arch_info(self):
        sum_bitops, sum_bita, sum_bitw = 0, 0, 0
        peak_wbit = 0
        layer_idx = 0
        for layer_name, m in self.named_modules():
            if isinstance(m, self.conv_func):
                size_product = m.size_product.item()
                memory_size = m.memory_size.item()
                if isinstance(m, qm.QuantMultiPrecActivConv2d):
                    for name, params in m.state_dict().items():
                        full_name = name
                        name = name.split('.')[-1]
                        if name == 'alpha_activ':
                            abits = m.abits[torch.argmax(params, dim=0).item()]
                        elif name == 'alpha_weight':
                            wbits = [m.wbits[prec] for prec in torch.argmax(params, dim=0).tolist()]
                    wbit_eff = sum(wbits) / len(wbits)
                    bitops = size_product * abits * wbit_eff
                    bita = m.memory_size.item() * abits
                    bitw = m.param_size * wbit_eff
                    sum_bitops += bitops
                    sum_bita += bita
                    sum_bitw += bitw
                    peak_wbit = max(peak_wbit, bitw)
                    if peak_wbit == bitw:
                        peak_layer = layer_name
                #elif isinstance(m, qm.QuantMixActivChanConv2d):
                #    import pdb; pdb.set_trace()
                #    bitops = size_product * m.abits * m.wbit
                #    bita = m.memory_size.item() * m.abits
                #    bitw = m.param_size * m.wbit
                #    sum_bitops += bitops
                #    sum_bita += bita
                #    sum_bitw += bitw
                #    peak_wbit = max(peak_wbit, bitw)
                #    if peak_wbit == bitw:
                #        peak_layer = layer_name
                else:
                    bitops = size_product * m.abits * m.wbit
                    bita = m.memory_size.item() * m.abits
                    bitw = m.param_size * m.wbit
                    #weight_shape = list(m.conv.weight.shape)
                    #print('idx {} with shape {}, bitops: {:.3f}M * {} * {}, memory: {:.3f}K * {}, '
                    #      'param: {:.3f}M * {}'.format(layer_idx, weight_shape, size_product, m.abit,
                    #                                   m.wbit, memory_size, m.abit, m.param_size, m.wbit))
                    sum_bitops += bitops
                    sum_bita += bita
                    sum_bitw += bitw
                    peak_wbit = max(peak_wbit, bitw)
                    if peak_wbit == bitw:
                        peak_layer = layer_name
                    layer_idx += 1
        return sum_bitops, sum_bita, sum_bitw, peak_layer, peak_wbit

# MR
def quantdscnn_fp(arch_cfg_path, **kwargs):
    # This precisions can be whatever
    archas, archws = [8] * 9, [8] * 9
    #assert len(archas) == 10
    #assert len(archws) == 10
    return DS_CNN(qm.FpConv2d, archws, archas, **kwargs)

# MR
def quantdscnn_w8a8(arch_cfg_path, **kwargs):
    # This precisions can be whatever
    archas, archws = [8] * 9, [8] * 9
    #assert len(archas) == 10
    #assert len(archws) == 10
    return DS_CNN(qm.QuantMixActivChanConv2d, archws, archas, qtz_fc='fixed', **kwargs)

# MR
def quantdscnn_w4a8(arch_cfg_path, **kwargs):
    # This precisions can be whatever
    archas, archws = [8] * 9, [4] * 9
    #assert len(archas) == 10
    #assert len(archws) == 10
    return DS_CNN(qm.QuantMixActivChanConv2d, archws, archas, qtz_fc='fixed', **kwargs)

# MR
def quantdscnn_w2a8(arch_cfg_path, **kwargs):
    # This precisions can be whatever
    archas, archws = [8] * 9, [2] * 9
    #assert len(archas) == 10
    #assert len(archws) == 10
    return DS_CNN(qm.QuantMixActivChanConv2d, archws, archas, qtz_fc='fixed', **kwargs)